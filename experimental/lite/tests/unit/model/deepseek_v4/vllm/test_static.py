from __future__ import annotations

import pytest
import torch

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.lite.model import (
    DeepseekV4Layer as LiteDeepseekV4Layer,
    DeepseekV4Model as LiteDeepseekV4Model,
)
from megatron.lite.model.deepseek_v4.lite.moe import (
    DeepseekV4MoE as LiteDeepseekV4MoE,
)
from megatron.lite.model.deepseek_v4.vllm.model import (
    DeepseekV4Layer,
    DeepseekV4Model,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.attention.module import (
    VLLMAttention,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.moe import DeepseekV4MoE


def _tiny_config() -> DeepseekV4Config:
    return DeepseekV4Config(
        vocab_size=32,
        hidden_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=4,
        head_dim=4,
        qk_rope_head_dim=2,
        q_lora_rank=8,
        o_lora_rank=4,
        o_groups=2,
        n_routed_experts=4,
        n_shared_experts=0,
        num_experts_per_tok=2,
        num_hash_layers=1,
        hc_mult=2,
        num_nextn_predict_layers=0,
    )


def test_vllm_path_reuses_lite_model_containers() -> None:
    assert issubclass(DeepseekV4Model, LiteDeepseekV4Model)
    assert issubclass(DeepseekV4Layer, LiteDeepseekV4Layer)
    assert issubclass(DeepseekV4MoE, LiteDeepseekV4MoE)


def test_cp1_has_no_boundary_projection() -> None:
    attention = VLLMAttention.__new__(VLLMAttention)
    local_k = torch.randn(7, 1, 8)
    result = attention._project_boundary_k(
        torch.empty(0, 16),
        local_k,
        torch.tensor([0, 7], dtype=torch.int32),
        global_start=0,
        cos_sin_cache=torch.empty(0),
    )

    assert result.shape == (0, 1, 8)
    assert result.untyped_storage().data_ptr() == local_k.untyped_storage().data_ptr()


@pytest.mark.gpus(1)
def test_small_model_state_dict_preserves_release_master_dtypes() -> None:
    model = DeepseekV4Model(_tiny_config())
    floating = {
        name: value.dtype
        for name, value in model.state_dict().items()
        if value.is_floating_point()
    }
    assert floating
    assert set(floating.values()) == {torch.bfloat16, torch.float32}
    fp32_suffixes = (
        ".fn",
        ".base",
        ".scale",
        ".hc_fn",
        ".hc_base",
        ".hc_scale",
        ".sinks",
        ".ape",
        ".expert_bias",
    )
    assert {
        name for name, dtype in floating.items() if dtype == torch.float32
    } == {name for name in floating if name.endswith(fp32_suffixes)}
    assert sum(value.numel() for value in model.state_dict().values()) < 100_000


@pytest.mark.gpus(1)
def test_deployment_weight_cache_policy_reaches_attention_and_moe() -> None:
    for enabled in (False, True):
        model = DeepseekV4Model(
            _tiny_config(), cache_deployment_weights=enabled
        )
        layer = model.layers["0"]
        attention = layer.self_attn
        assert attention.fused_linear.cache_weight is enabled
        assert attention.q_linear.cache_weight is enabled
        assert attention.indexer_q_linear.cache_weight is enabled
        assert layer.mlp.shared_gate_up_fp8.cache_weight is enabled
        assert layer.mlp.shared_down_fp8.cache_weight is enabled


def test_static_suite_never_constructs_release_dimensions() -> None:
    config = _tiny_config()
    assert config.vocab_size <= 32
    assert config.hidden_size <= 16
    assert config.moe_intermediate_size <= 8
    assert config.num_hidden_layers == 1
    assert config.n_routed_experts <= 4
