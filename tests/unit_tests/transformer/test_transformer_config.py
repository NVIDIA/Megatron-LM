# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch
from torch.nn import functional as F

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


def test_batch_invariant_backend_rejects_unknown_value_at_construction():
    # Programmatic construction bypasses argparse's Literal choices, so
    # __post_init__ must catch typos before model init.
    with pytest.raises(AssertionError, match="Unknown batch_invariant_backend"):
        TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            batch_invariant_mode=True,
            batch_invariant_backend="te-native",
        )


def test_gdp_num_householder_defaults_to_three():
    config = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=4)

    assert config.gdp_num_householder == 3


def test_gdp_num_householder_accepts_positive_values():
    config = TransformerConfig(
        num_layers=1, hidden_size=128, num_attention_heads=4, gdp_num_householder=5
    )

    assert config.gdp_num_householder == 5


def _make_replica_hybridep_config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        expert_model_parallel_size=2,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="replica_hybridep",
        moe_grouped_gemm=True,
        moe_router_dtype="fp32",
        use_transformer_engine_op_fuser=True,
        gradient_accumulation_fusion=True,
        add_bias_linear=False,
        activation_func=F.silu,
        gated_linear_unit=True,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_replica_hybridep_allows_moe_cuda_graph_without_drop_padding():
    config = _make_replica_hybridep_config(
        cuda_graph_impl="local",
        cuda_graph_modules=["moe"],
    )

    assert config.moe_flex_dispatcher_backend == "replica_hybridep"
    assert config.moe_expert_rank_capacity_factor == 1.0
    assert config.moe_single_grouped_weight is False
    assert config.grad_reduce_in_bf16 is False


def test_replica_hybridep_bf16_grad_reduce_requires_ddp_fp32_accumulation():
    with pytest.raises(ValueError, match="ddp-reduce-scatter-with-fp32-accumulation"):
        _make_replica_hybridep_config(grad_reduce_in_bf16=True)


def test_replica_hybridep_bf16_grad_reduce_requires_gtp_fp32_accumulation_with_gtp():
    with pytest.raises(ValueError, match="gtp-remat-reduce-scatter-with-fp32-accumulation"):
        _make_replica_hybridep_config(
            grad_reduce_in_bf16=True,
            ddp_reduce_scatter_with_fp32_accumulation=True,
            expert_tensor_parallel_num_weight_shards=2,
        )


@pytest.mark.parametrize("expert_gtp", [False, True])
def test_replica_hybridep_accepts_bf16_grad_reduce_with_fp32_accumulation(expert_gtp):
    config = _make_replica_hybridep_config(
        grad_reduce_in_bf16=True,
        ddp_reduce_scatter_with_fp32_accumulation=True,
        expert_tensor_parallel_num_weight_shards=2 if expert_gtp else 1,
        gtp_remat_reduce_scatter_with_fp32_accumulation=expert_gtp,
    )

    assert config.grad_reduce_in_bf16 is True


def test_replica_hybridep_allows_native_mxfp8_and_router_padding():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        expert_model_parallel_size=2,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="replica_hybridep",
        moe_grouped_gemm=True,
        moe_router_dtype="fp32",
        moe_router_padding_for_quantization=True,
        use_transformer_engine_op_fuser=True,
        gradient_accumulation_fusion=True,
        add_bias_linear=False,
        activation_func=F.silu,
        gated_linear_unit=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        fp8="e4m3",
        fp8_recipe="mxfp8",
        fp8_param=True,
    )

    assert config.fp8 == "e4m3"
    assert config.fp8_recipe == "mxfp8"
    assert config.fp8_param
    assert config.moe_router_padding_for_quantization


@pytest.mark.parametrize(
    ("fp8", "fp8_recipe", "fp8_param"),
    [("e4m3", "mxfp8", False), ("e4m3", "tensorwise", True), ("hybrid", "mxfp8", True)],
)
def test_replica_hybridep_rejects_unsupported_fp8_parameter_storage(fp8, fp8_recipe, fp8_param):
    with pytest.raises(ValueError, match="MXFP8 E4M3 with native FP8 parameters"):
        TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            num_moe_experts=2,
            expert_model_parallel_size=2,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="replica_hybridep",
            moe_grouped_gemm=True,
            moe_router_dtype="fp32",
            use_transformer_engine_op_fuser=True,
            gradient_accumulation_fusion=True,
            add_bias_linear=False,
            activation_func=F.silu,
            gated_linear_unit=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            fp8=fp8,
            fp8_recipe=fp8_recipe,
            fp8_param=fp8_param,
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


def _make_mxfp8_wire_config(**overrides) -> TransformerConfig:
    kwargs = dict(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        expert_model_parallel_size=2,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="ncclep",
        moe_grouped_gemm=True,
        use_transformer_engine_op_fuser=True,
        moe_dispatch_fwd_dtype='mxfp8',
        moe_combine_bwd_dtype='mxfp8',
        bf16=True,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_mxfp8_wire_dtypes_accept_valid_ncclep_config():
    config = _make_mxfp8_wire_config()

    assert config.moe_dispatch_fwd_dtype == 'mxfp8'
    assert config.moe_combine_bwd_dtype == 'mxfp8'


def test_mxfp8_wire_dtypes_accept_a2a_overlap():
    # The 1F1B a2a overlap schedule only moves/stages the dispatch output as an opaque block,
    # which the plain-tensor MXFP8 carrier survives; the combination is deliberately allowed.
    config = _make_mxfp8_wire_config(overlap_moe_expert_parallel_comm=True)

    assert config.overlap_moe_expert_parallel_comm


@pytest.mark.parametrize(
    "overrides",
    [
        dict(moe_flex_dispatcher_backend="hybridep"),
        dict(moe_token_dispatcher_type="alltoall", moe_flex_dispatcher_backend=None),
    ],
)
def test_mxfp8_wire_dtypes_reject_non_ncclep_dispatcher(overrides):
    with pytest.raises(ValueError, match="require the 'ncclep' flex"):
        _make_mxfp8_wire_config(**overrides)


@pytest.mark.parametrize(
    "overrides", [dict(use_transformer_engine_op_fuser=False), dict(moe_grouped_gemm=False)]
)
def test_mxfp8_wire_dtypes_require_op_fuser_grouped_gemm(overrides):
    with pytest.raises(ValueError, match="require BOTH"):
        _make_mxfp8_wire_config(**overrides)
