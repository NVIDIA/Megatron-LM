# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Two-GPU VERL connector coverage for an optional downstream workflow."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

VERL_EXAMPLE_ROOT = Path(__file__).resolve().parents[2] / "examples" / "verl"
if str(VERL_EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_EXAMPLE_ROOT))

pytestmark = [
    pytest.mark.gpus(2),
    pytest.mark.env(CUDA_DEVICE_MAX_CONNECTIONS="1"),
    pytest.mark.timeout(seconds=3600),
]


def _init_dist_or_skip():
    import os

    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MLite VERL CP smoke.")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Run with torchrun so CP ranks are available.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    if dist.get_world_size() != 2:
        pytest.skip("MLite VERL CP smoke requires exactly 2 ranks.")
    return torch.device("cuda", local_rank)


def _write_kimi_config(path) -> None:
    config = {
        "model_type": "deepseek_v3",
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": 128,
        "intermediate_size": 96,
        "moe_intermediate_size": 16,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "n_group": 2,
        "topk_group": 1,
        "first_k_dense_replace": 1,
        "q_lora_rank": 16,
        "kv_lora_rank": 12,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 8,
        "v_head_dim": 8,
        "max_position_embeddings": 128,
        "rope_theta": 10000.0,
        "rope_scaling": {
            "type": "yarn",
            "factor": 1.0,
            "original_max_position_embeddings": 128,
            "beta_fast": 1.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _write_glm5_config(path) -> None:
    config = {
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 2,
        "hidden_size": 128,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "head_dim": 256,
        "vocab_size": 32,
        "max_position_embeddings": 64,
        "initializer_range": 0.002,
        "q_lora_rank": 16,
        "kv_lora_rank": 512,
        "qk_head_dim": 256,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "index_head_dim": 128,
        "index_n_heads": 32,
        "index_topk": 512,
        "intermediate_size": 20,
        "moe_intermediate_size": 6,
        "first_k_dense_replace": 1,
        "n_routed_experts": 3,
        "n_shared_experts": 1,
        "num_experts_per_tok": 3,
        "num_nextn_predict_layers": 1,
        "mlp_layer_types": ["dense", "dense"],
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _write_deepseek_v4_config(path) -> None:
    config = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 2,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "vocab_size": 128,
        "max_position_embeddings": 128,
        "initializer_range": 0.02,
        "q_lora_rank": 64,
        "qk_rope_head_dim": 16,
        "o_lora_rank": 64,
        "o_groups": 4,
        "index_head_dim": 16,
        "index_n_heads": 4,
        "index_topk": 4,
        "moe_intermediate_size": 32,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "num_hash_layers": 1,
        "num_nextn_predict_layers": 0,
        "compress_ratios": [4, 4],
        "compress_rope_theta": 160000.0,
        "hc_mult": 2,
        "hc_eps": 1e-6,
        "hc_sinkhorn_iters": 4,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "sliding_window": 128,
        "swiglu_limit": 10.0,
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.0,
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _optimizer_config() -> SimpleNamespace:
    return SimpleNamespace(
        optimizer="adam",
        lr=1e-6,
        min_lr=None,
        min_lr_ratio=None,
        clip_grad=1.0,
        weight_decay=0.0,
        lr_warmup_steps_ratio=0.0,
        total_training_steps=1,
        lr_warmup_steps=0,
        lr_warmup_init=0.0,
        lr_decay_steps=None,
        lr_decay_style="constant",
        weight_decay_incr_style="constant",
        lr_wsd_decay_style="exponential",
        lr_wsd_decay_steps=None,
        use_checkpoint_opt_param_scheduler=False,
        betas=(0.9, 0.95),
        override_optimizer_config={},
    )


@pytest.mark.parametrize(
    ("model_name", "model_type", "write_config", "vocab_size", "lengths"),
    [
        ("kimi_k2", "deepseek_v3", _write_kimi_config, 128, [5, 7, 9]),
        ("glm5", "glm_moe_dsa", _write_glm5_config, 32, [16, 20, 24]),
        ("deepseek_v4", "deepseek_v4", _write_deepseek_v4_config, 128, [16, 20, 24]),
    ],
)
def test_mlite_engine_runtime_thd_cp_uses_typed_packed_batch(
    tmp_path, model_name, model_type, write_config, vocab_size, lengths
):
    torch = pytest.importorskip("torch")
    dist = pytest.importorskip("torch.distributed")
    TensorDict = pytest.importorskip("tensordict").TensorDict
    from verl_mlite.compat import apply_runtime_patches

    apply_runtime_patches()
    from verl_mlite.engine.config import MegatronLiteEngineConfig
    from verl_mlite.engine.mlite_engine import MegatronLiteEngine

    from megatron.lite.runtime.contracts import PackedBatch

    device = _init_dist_or_skip()
    world = dist.get_world_size()
    rank = dist.get_rank()
    hf_path = tmp_path / f"tiny-{model_name}"
    write_config(hf_path)

    engine = MegatronLiteEngine(
        model_config=SimpleNamespace(
            local_path=str(hf_path),
            hf_config={"model_type": model_type},
            mtp=None,
        ),
        engine_config=MegatronLiteEngineConfig(
            model_name=model_name,
            cp=world,
            impl_cfg={
                "use_thd": True,
                "optimizer": None,
                "deterministic": False,
                "mtp_enable": False,
            },
            use_fused_kernels=False,
        ),
        optimizer_config=_optimizer_config(),
        checkpoint_config={},
    )
    original_build_config = engine._build_mlite_config

    def _build_config_without_loading_weights(self):
        config = original_build_config()
        config.load_hf_weights = False
        return config

    engine._build_mlite_config = MethodType(_build_config_without_loading_weights, engine)
    engine.initialize()
    engine.optimizer_zero_grad()

    input_ids = torch.nested.as_nested_tensor(
        [
            torch.randint(0, vocab_size, (length,), device=device, dtype=torch.long)
            for length in lengths
        ],
        layout=torch.jagged,
    )
    loss_mask = torch.nested.as_nested_tensor(
        [torch.ones(length, device=device, dtype=torch.float32) for length in lengths],
        layout=torch.jagged,
    )
    micro_batch = TensorDict(
        {"input_ids": input_ids, "loss_mask": loss_mask},
        batch_size=[len(lengths)],
        device=device,
    )

    runtime_batch = engine._make_runtime_batch(micro_batch)
    loss_context = engine._make_runtime_loss_context(micro_batch, loss_scale=1.0)
    assert isinstance(runtime_batch, PackedBatch)
    # Connector batch is model-agnostic: true seq lengths, no padding, no extras.
    assert runtime_batch.seq_lens.tolist() == lengths
    assert int(runtime_batch.input_ids.numel()) == sum(lengths)
    assert not runtime_batch.extras

    with engine.train_mode():
        result = engine.runtime.forward_backward(
            engine.handle,
            iter([(runtime_batch, loss_context)]),
            loss_fn=None,
            num_microbatches=1,
            forward_only=False,
        )

    assert torch.isfinite(result.model_output.loss)
    assert result.model_output.log_probs is not None
    grad_norm = torch.zeros((), dtype=torch.float32, device=device)
    for param in engine.module.parameters():
        if param.grad is not None:
            grad_norm = grad_norm + param.grad.detach().float().norm()
    dist.all_reduce(grad_norm, op=dist.ReduceOp.SUM)
    assert torch.isfinite(grad_norm)
    assert grad_norm.item() > 0.0

    verl_output = engine._build_verl_model_output(
        raw_output={"log_probs": result.model_output.log_probs},
        runtime_batch=runtime_batch,
    )
    nested_log_probs = verl_output["log_probs"]
    assert [int(x) for x in nested_log_probs.offsets().diff().cpu()] == lengths

    if rank == 0:
        print(
            "NON_SKIP_VERL_MLITE_RUNTIME_THD_CP_SMOKE_PASSED "
            f"model={model_name} "
            f"world_size={world} lengths={lengths} "
            f"loss={float(result.model_output.loss.detach().item()):.6e} "
            f"grad_norm_sum={float(grad_norm.item()):.6e}"
        )
