# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch.nn.functional as F

from megatron.core.transformer.transformer_config import TransformerConfig


def _make_overlap_config(mtp_num_layers: int | None) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        expert_model_parallel_size=2,
        moe_token_dispatcher_type="alltoall",
        overlap_moe_expert_parallel_comm=True,
        bf16=True,
        mtp_num_layers=mtp_num_layers,
    )


@pytest.mark.parametrize("mtp_num_layers", [None, 0, 1])
def test_ep_a2a_overlap_accepts_supported_mtp_layer_counts(mtp_num_layers: int | None):
    config = _make_overlap_config(mtp_num_layers)

    assert config.mtp_num_layers == mtp_num_layers


@pytest.mark.parametrize("mtp_num_layers", [-1, 2])
def test_ep_a2a_overlap_rejects_unsupported_mtp_layer_counts(mtp_num_layers: int):
    with pytest.raises(AssertionError, match="MTP supports at most one layer"):
        _make_overlap_config(mtp_num_layers)


def test_gdp_num_householder_defaults_to_three():
    config = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=4)

    assert config.gdp_num_householder == 3


def test_gdp_num_householder_accepts_positive_values():
    config = TransformerConfig(
        num_layers=1, hidden_size=128, num_attention_heads=4, gdp_num_householder=5
    )

    assert config.gdp_num_householder == 5


@pytest.mark.parametrize("num_householder", [0, -1])
def test_gdp_num_householder_rejects_non_positive_values(num_householder: int):
    with pytest.raises(ValueError, match="gdp_num_householder must be positive"):
        TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            gdp_num_householder=num_householder,
        )


def _fused_moe_config(**overrides) -> TransformerConfig:
    kwargs = dict(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=4,
        num_moe_experts=2,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="ncclep",
        moe_grouped_gemm=True,
        use_transformer_engine_op_fuser=True,
        moe_single_grouped_weight=True,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        bf16=True,
        moe_use_transformer_engine_fused_moe=True,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_fused_moe_config_enables_grouped_tensor():
    config = _fused_moe_config()

    assert config.moe_use_grouped_tensor


@pytest.mark.parametrize(
    "override,error",
    [
        ({"moe_token_dispatcher_type": "alltoall"}, "moe_token_dispatcher_type='flex'"),
        ({"moe_flex_dispatcher_backend": "deepep"}, "moe_flex_dispatcher_backend='ncclep'"),
        ({"moe_single_grouped_weight": False}, "moe_single_grouped_weight=True"),
        ({"moe_shared_expert_overlap": True}, "moe_shared_expert_overlap"),
        ({"cuda_graph_impl": "local"}, "CUDA graphs"),
        ({"moe_paged_stash": True}, "moe_paged_stash"),
        ({"fp4": "nvfp4"}, "does not support FP4"),
    ],
)
def test_fused_moe_config_rejects_incompatible_modes(override, error):
    with pytest.raises(ValueError, match=error):
        _fused_moe_config(**override)
