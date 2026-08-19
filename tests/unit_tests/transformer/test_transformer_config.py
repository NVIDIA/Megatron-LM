# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.transformer.enums import CudaGraphModule
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


@pytest.mark.parametrize(
    "removed_field",
    [
        "moe_shortcut_output_norm",
        "moe_shortcut_tied_norm",
        "moe_shortcut_untied_norm",
        "moe_shortcut_post_norm",
        "moe_shortcut_scalar_gate",
        "moe_shortcut_vector_gate",
        "moe_use_norm_before_up_proj",
    ],
)
def test_removed_shortcut_options_are_not_config_fields(removed_field: str):
    assert removed_field not in TransformerConfig.__dataclass_fields__


def test_shortcut_rejects_full_activation_recomputation():
    with pytest.raises(ValueError, match="moe_shortcut_connection is not supported"):
        TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            num_moe_experts=1,
            moe_router_topk=1,
            moe_shortcut_connection=True,
            recompute_granularity="full",
        )


def test_shortcut_block_cuda_graph_scope():
    config = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        moe_router_topk=1,
        moe_router_pre_softmax=True,
        moe_shortcut_connection=True,
        moe_shortcut_parallel=True,
        cuda_graph_impl="local",
        cuda_graph_modules=["shortcut_block"],
    )

    assert config.cuda_graph_modules == [CudaGraphModule.shortcut_block]


def test_shortcut_cuda_graphs_require_shortcut_block_scope():
    with pytest.raises(AssertionError, match="cuda_graph_modules=\\['shortcut_block'\\]"):
        TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            num_moe_experts=2,
            moe_router_topk=1,
            moe_router_pre_softmax=True,
            moe_shortcut_connection=True,
            moe_shortcut_parallel=True,
            cuda_graph_impl="local",
            cuda_graph_modules=["attn", "moe_router"],
        )


@pytest.mark.parametrize("num_householder", [0, -1])
def test_gdp_num_householder_rejects_non_positive_values(num_householder: int):
    with pytest.raises(ValueError, match="gdp_num_householder must be positive"):
        TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            gdp_num_householder=num_householder,
        )
