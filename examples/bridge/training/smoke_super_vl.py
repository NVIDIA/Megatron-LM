# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Offline, reduced-size construction probe for the experimental Super VL port.

This is not a pretrained quality or full-scale performance test. Run from the
MLM checkout with ``python -m torch.distributed.run --nproc-per-node=1``.
"""

import argparse
import inspect
import json
import logging
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from transformers import AutoConfig, NemotronHConfig, PretrainedConfig, PreTrainedTokenizerFast

from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.nemotron_super_vl.bridge import Nemotron35SuperVLBridge
from megatron.bridge.models.nemotron_super_vl.config import Nemotron35SuperVLModelConfig
from megatron.core import parallel_state
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

logger = logging.getLogger(__name__)


def language_config() -> PretrainedConfig:
    """Tiny Nemotron-H with attention, Mamba and MoE blocks, not a model preset."""
    return PretrainedConfig(
        architectures=["NemotronHForCausalLM"],
        torch_dtype="bfloat16",
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        layer_norm_epsilon=1e-6,
        vocab_size=256,
        max_position_embeddings=32,
        hybrid_override_pattern="M*E",
        mamba_head_dim=32,
        mamba_num_heads=4,
        n_groups=2,
        ssm_state_size=16,
        moe_intermediate_size=128,
        moe_shared_expert_intermediate_size=256,
        n_routed_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=3,
        tie_word_embeddings=False,
    )


def main() -> None:
    """Check local dispatch/serialization and run a real text forward/backward."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vision", action="store_true", help="Also construct the full RADIO encoder"
    )
    parser.add_argument(
        "--hf-fixture-output",
        type=Path,
        help="Save the reduced random text checkpoint for CLI comparison",
    )
    parser.add_argument(
        "--vl-reference-config",
        type=Path,
        help="Local trusted Super VL config/code artifacts for a reduced-language VL fixture",
    )
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    assert Path(inspect.getfile(AutoBridge)).is_relative_to(root / "megatron/bridge")
    text = language_config()
    text.num_nextn_predict_layers = 1
    text.mtp_hybrid_override_pattern = "*E"
    vl = PretrainedConfig(
        architectures=["NemotronH_Omni_Reasoning_V3"],
        llm_config=text,
        vision_config=PretrainedConfig(separate_video_embedder=False),
        sound_config=None,
        img_context_token_id=18,
        projector_hidden_size=256,
        video_temporal_patch_size=1,
    )
    if options.vl_reference_config is not None:
        vl = AutoConfig.from_pretrained(options.vl_reference_config, trust_remote_code=True)
        vl.llm_config = NemotronHConfig(**text.to_dict())
        vl.img_context_token_id = 18
        vl.projector_hidden_size = 256
        # This is a random fixture, not an export of the metadata-only source.
        vl.name_or_path = ""
    vl_bridge = AutoBridge.from_hf_config(vl)
    assert isinstance(vl_bridge._model_bridge, Nemotron35SuperVLBridge)
    vl_config = vl_bridge.get_model_config()
    assert isinstance(vl_config, Nemotron35SuperVLModelConfig)
    vl_config.finalize()
    restored = Nemotron35SuperVLModelConfig.from_dict(vl_config.as_dict())
    assert restored.as_dict() == vl_config.as_dict()
    assert restored.get_builder_cls() is vl_config.get_builder_cls()
    logger.warning("Super VL config dispatch and serialization PASS")
    assert vl_bridge._model_bridge.megatron_to_hf_config(vl_config)["hybrid_override_pattern"]

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(1234)
    try:
        bridge = AutoBridge.from_hf_config(text)
        config = bridge.get_model_config()
        config.transformer.moe_token_dispatcher_type = "allgather"
        config.transformer.moe_shared_expert_overlap = False
        config.transformer.moe_grouped_gemm = False
        # This probe deliberately skips DDP, which normally allocates main_grad.
        config.transformer.gradient_accumulation_fusion = False
        config.transformer.use_cpu_initialization = False
        models = bridge.get_model(config, load_weights=False, wrap_with_ddp=False)
        assert len(models) == 1
        tokens = torch.randint(0, 256, (1, 16), device="cuda")
        positions = torch.arange(16, device="cuda").unsqueeze(0)
        output = models[0](tokens, positions, None, labels=tokens)
        loss = output.float().mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None for p in models[0].parameters())
        logger.warning("Text Mamba/attention/MoE forward/backward PASS: loss=%s", loss.item())
        with TemporaryDirectory(prefix="mlm-super-vl-roundtrip-") as directory:
            bridge.save_megatron_model(models, directory)
            restored_models = bridge.load_megatron_model(directory)
            expected = dict(models[0].named_parameters())
            actual = dict(restored_models[0].named_parameters())
            assert expected.keys() == actual.keys()
            for name, tensor in expected.items():
                torch.testing.assert_close(tensor, actual[name], rtol=0, atol=0)
            logger.warning(
                "Native weights-only checkpoint roundtrip PASS (%s parameters)", len(expected)
            )
        with TemporaryDirectory(prefix="mlm-super-vl-hf-") as directory:
            exported = dict(bridge.export_hf_weights(models, cpu=True, show_progress=False))
            assert exported
            if options.hf_fixture_output is not None:
                from tokenizers import Tokenizer
                from tokenizers.models import WordLevel

                options.hf_fixture_output.mkdir(parents=True, exist_ok=False)
                fixture_config = NemotronHConfig(**text.to_dict())
                fixture_config.architectures = ["NemotronHForCausalLM"]
                fixture_config.save_pretrained(options.hf_fixture_output)
                tokenizer = PreTrainedTokenizerFast(
                    tokenizer_object=Tokenizer(
                        WordLevel({f"token{i}": i for i in range(256)}, unk_token="token0")
                    ),
                    unk_token="token0",
                    bos_token="token1",
                    eos_token="token2",
                    pad_token="token3",
                )
                tokenizer.save_pretrained(options.hf_fixture_output)
                save_file(
                    {name: value.contiguous() for name, value in exported.items()},
                    str(options.hf_fixture_output / "model.safetensors"),
                )
                logger.warning("Saved reduced random HF fixture: %s", options.hf_fixture_output)
            # Exercise the actual Super VL namespace projection from a local
            # safetensors checkpoint, including exclusion of a media-only weight.
            tensors = {
                "language_model." + name: value.contiguous() for name, value in exported.items()
            }
            tensors["vision_model.port_probe"] = torch.zeros(1)
            save_file(tensors, str(Path(directory) / "model.safetensors"))
            source = PreTrainedCausalLM(directory)
            source.config = vl
            projected = vl_bridge._model_bridge.text_only_pretrained(source)
            imported_bridge = AutoBridge(projected)
            imported_config = imported_bridge.get_model_config()
            imported_config.transformer.moe_shared_expert_overlap = False
            imported_config.transformer.moe_token_dispatcher_type = "allgather"
            imported_config.transformer.moe_grouped_gemm = False
            imported_config.transformer.gradient_accumulation_fusion = False
            imported_models = imported_bridge.get_model(imported_config, wrap_with_ddp=False)
            actual = dict(imported_models[0].named_parameters())
            assert expected.keys() == actual.keys()
            for name, tensor in expected.items():
                torch.testing.assert_close(tensor, actual[name], rtol=0, atol=0)
            logger.warning("Super VL text-only HF export/import roundtrip PASS")
        if options.vision:
            vl_config.transformer.moe_shared_expert_overlap = False
            vl_config.transformer.moe_token_dispatcher_type = "allgather"
            vl_config.transformer.moe_grouped_gemm = False
            vl_config.transformer.gradient_accumulation_fusion = False
            vl_models = vl_bridge.get_model(vl_config, load_weights=False, wrap_with_ddp=False)
            vl_models[0].eval()
            vl_tokens = torch.randint(20, 256, (1, 16), device="cuda")
            vl_tokens[:, 0] = vl_config.image_token_index
            with torch.no_grad():
                output = vl_models[0](
                    vl_tokens,
                    positions,
                    None,
                    images=torch.randn(1, 3, 32, 32, device="cuda", dtype=torch.bfloat16),
                    imgs_sizes=torch.tensor([[32, 32]], device="cuda"),
                    num_frames=[1],
                )
            assert torch.isfinite(output).all()
            logger.warning(
                "Super VL RADIO + projector + tiny language forward PASS: %s", tuple(output.shape)
            )
            if options.hf_fixture_output is not None and options.vl_reference_config is not None:
                vl_output = options.hf_fixture_output.with_name(
                    options.hf_fixture_output.name + "-vl"
                )
                vl_output.mkdir(parents=True, exist_ok=False)
                for artifact in options.vl_reference_config.glob("*.py"):
                    shutil.copy2(artifact, vl_output / artifact.name)
                vl.save_pretrained(vl_output)
                tokenizer.save_pretrained(vl_output)
                vl_weights = dict(
                    vl_bridge.export_hf_weights(vl_models, cpu=True, show_progress=False)
                )
                save_file(
                    {name: value.contiguous() for name, value in vl_weights.items()},
                    str(vl_output / "model.safetensors"),
                )
                # Match the indexed public checkpoint layout: Super VL export
                # adds its deterministic RADIO buffer through this index.
                (vl_output / "model.safetensors.index.json").write_text(
                    json.dumps(
                        {
                            "metadata": {
                                "total_size": sum(
                                    value.numel() * value.element_size()
                                    for value in vl_weights.values()
                                )
                            },
                            "weight_map": {name: "model.safetensors" for name in vl_weights},
                        }
                    )
                )
                logger.warning("Saved reduced-language VL HF fixture: %s", vl_output)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
