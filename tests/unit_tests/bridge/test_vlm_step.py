# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared VLM-step contracts, independent of the vision encoder backend."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.timers import Timers
from megatron.training.forward_steps import vlm_step
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def step_runtime(run_config, monkeypatch, request):
    tp, cp = getattr(request, "param", (1, 1))
    Utils.initialize_model_parallel(tp, 1, context_parallel_size=cp)
    run_config.model.transformer.tensor_model_parallel_size = tp
    run_config.model.transformer.context_parallel_size = cp
    model_parallel_cuda_manual_seed(1234)
    monkeypatch.setattr(vlm_step, "get_timers", lambda: Timers(0, "minmax"))
    run_config.rerun_state_machine.check_for_nan_in_loss = False
    run_config.rerun_state_machine.check_for_spiky_loss = False
    yield run_config
    Utils.destroy_model_parallel()


def _batch():
    return {
        "input_ids": torch.arange(16).view(1, 16),
        "labels": torch.arange(1, 17).view(1, 16),
        "loss_mask": torch.ones(1, 16),
        "visual_inputs": {
            "pixel_values": torch.randn(1, 3, 8, 8),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        },
    }


class _HFVisionWithMegatronDecoder(torch.nn.Module):
    """Small real HF encoder/Core decoder composition, not a production model port."""

    def __init__(self, config):
        super().__init__()
        from transformers import SiglipVisionConfig, SiglipVisionModel

        self.pre_process = self.post_process = True
        self.vision = SiglipVisionModel(
            SiglipVisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                image_size=8,
                patch_size=4,
            )
        )
        self.language_model = GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=128,
            max_sequence_length=16,
            position_embedding_type="rope",
        )

    def forward(self, input_ids, position_ids, attention_mask, labels, loss_mask, pixel_values):
        embeddings = self.language_model.embedding(input_ids, position_ids)
        visual = self.vision(pixel_values).last_hidden_state.mean(dim=1)
        embeddings = embeddings + visual.unsqueeze(0)
        output = self.language_model(
            input_ids, position_ids, attention_mask, decoder_input=embeddings, labels=labels
        )
        return output, loss_mask


def test_hf_encoder_and_megatron_decoder_backward(step_runtime):
    model = _HFVisionWithMegatronDecoder(step_runtime.model.transformer).cuda()
    output, loss_func = vlm_step.forward_step(iter([_batch()]), model)
    loss, count, _ = loss_func(output)
    assert torch.isfinite(loss) and count == 16
    loss.backward()
    for module in (model.vision, model.language_model):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g) > 0 for g in grads)


@pytest.mark.parametrize(
    "step_runtime,packed",
    [((1, 1), False), ((1, 1), True), ((2, 1), False), ((1, 2), True)],
    indirect=["step_runtime"],
)
def test_qwen35_megatron_encoder_and_hybrid_decoder_backward(step_runtime, packed):
    from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLBridge
    from tests.unit_tests.bridge.qwen_vl.test_qwen35_vl_bridge import _qwen35_vl_config

    reference = _qwen35_vl_config(
        moe=False, num_hidden_layers=2, full_attention_interval=2, mtp_num_hidden_layers=0
    )
    reference.image_token_id = 121
    reference.video_token_id = 123
    reference.vision_start_token_id = 120
    reference.vision_end_token_id = 122
    reference.vision_config.deepstack_visual_indexes = []
    cfg = Qwen35VLBridge().hf_config_to_model_config(reference)
    cfg.transformer.tensor_model_parallel_size = (
        step_runtime.model.transformer.tensor_model_parallel_size
    )
    cfg.transformer.context_parallel_size = step_runtime.model.transformer.context_parallel_size
    cfg.transformer.sequence_parallel = cfg.transformer.tensor_model_parallel_size > 1
    cfg.transformer.gdn_kernel_backend = "fla"
    cfg.transformer.max_seq_length_per_sample = 16
    cfg.transformer.cross_entropy_loss_fusion = False
    # This exercises the step directly, without DDP's main-gradient buffers.
    cfg.transformer.gradient_accumulation_fusion = False
    cfg.finalize()
    step_runtime.model = cfg
    model = (
        cfg.get_builder_cls()(cfg)
        .build_model(ProcessGroupCollection.use_mpu_process_groups())
        .cuda()
    )
    sample = _batch()
    sample["input_ids"][0, :3] = torch.tensor([120, 121, 122])
    patch = cfg.vision_config.patch_size
    sample["visual_inputs"]["pixel_values"] = torch.randn(
        4, 3 * cfg.vision_config.temporal_patch_size * patch * patch, dtype=torch.bfloat16
    )
    if packed:
        sample.update(cu_seqlens_q=torch.tensor([0, 8, 16], dtype=torch.int32), max_seqlen_q=8)
    output, loss_func = vlm_step.forward_step(iter([sample]), model)
    loss, count, _ = loss_func(output)
    assert torch.isfinite(loss) and count == 16 // cfg.transformer.context_parallel_size
    loss.backward()
    for module in (model.vision_model, model.language_model):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        maximum = torch.stack([g.float().abs().max() for g in grads]).max()
        # A CP rank whose shard contains no image tokens can have zero local
        # encoder gradients. Normal DDP reduces these replicas across CP.
        torch.distributed.all_reduce(
            maximum, op=torch.distributed.ReduceOp.MAX, group=model.pg_collection.cp
        )
        assert maximum > 0


def test_middle_stage_keeps_metadata_without_mutating_batch():
    sample = _batch()
    original = sample["visual_inputs"]
    sample["visual_inputs"] = SimpleNamespace(normalized_for_model=lambda: original)
    batch = vlm_step.get_batch(iter([sample]), pre_process=False, post_process=False, use_mtp=False)
    assert "pixel_values" not in batch["visual_inputs"]
    assert batch["visual_inputs"]["image_grid_thw"].is_cuda
    assert batch["input_ids"].is_cuda
    assert batch["labels"] is None and batch["loss_mask"] is None
    assert "pixel_values" in original and not original["pixel_values"].is_cuda


def test_model_returned_mask_controls_loss(step_runtime):
    class PartitioningModel(torch.nn.Module):
        pre_process = post_process = True

        def forward(self, input_ids, position_ids, attention_mask, labels, loss_mask, **visual):
            # Models can return a local CP shard; the step must not reuse the full mask.
            return torch.ones(1, 8, device="cuda"), loss_mask[:, :8] * 0.5

    output, loss_func = vlm_step.forward_step(iter([_batch()]), PartitioningModel())
    loss, count, _ = loss_func(output)
    assert loss == 4 and count == 4


def test_missing_model_parallel_iterator_is_explicit():
    with pytest.raises(ValueError, match="every model-parallel rank"):
        vlm_step.get_batch(None, pre_process=True, post_process=True, use_mtp=False)
