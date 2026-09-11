# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GDP training and inference declarations; importing them loads no kernels."""

from megatron.core.ops.kernel_metadata import (
    Dependency,
    Determinism,
    DeterminismResult,
    KernelMetadata,
)
from megatron.core.ops.ssm.common.kernel_metadata import check_causal_conv_determinism

GDP_CONV = KernelMetadata(
    name="gdp.causal_conv1d_fn",
    requires=(
        Dependency(
            requirement="causal-conv1d>=1.4.0",
            module="causal_conv1d",
            symbols=("causal_conv1d_fn",),
        ),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="Convolution forward/backward is not certified here. Deterministic backward "
        "requires causal-conv1d >= 1.6.0 and its deterministic reduction to be enabled.",
    ),
    contract="megatron.core.ops.ssm.gdp",
    determinism_check=check_causal_conv_determinism,
)
GDP_FLA = KernelMetadata(
    name="gdp.fla.chunk_gated_delta_product",
    requires=(
        Dependency(
            requirement="flash-linear-attention",
            module="fla.ops.gated_delta_product",
            symbols=("chunk_gated_delta_product",),
        ),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="GDP training forward/backward has not been audited for bit-exact repeatability.",
    ),
    contract="megatron.core.ops.ssm.gdp",
)
GDP_CUTEDSL = KernelMetadata(
    name="gdp.cutedsl.chunk_gated_delta_product",
    requires=(
        Dependency(
            requirement="gdp-attn", module="gdp_attn", symbols=("chunk_gated_delta_product",)
        ),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="GDP training forward/backward has not been audited for bit-exact repeatability.",
    ),
    contract="megatron.core.ops.ssm.gdp",
)
GDP_FLA_CP = KernelMetadata(
    name="gdp.fla.chunkwise_context_parallel",
    requires=(
        Dependency(requirement="triton", module="triton"),
        Dependency(
            requirement="flash-linear-attention",
            module="fla.ops.common.chunk_delta_h",
            symbols=("chunk_gated_delta_rule_bwd_dhu", "chunk_gated_delta_rule_fwd_h"),
        ),
        Dependency(
            requirement="flash-linear-attention",
            module="fla.ops.common.chunk_o",
            symbols=("chunk_bwd_dqkwg", "chunk_bwd_dv_local"),
        ),
        Dependency(
            requirement="flash-linear-attention",
            module="fla.ops.common.chunk_scaled_dot_kkt",
            symbols=("chunk_scaled_dot_kkt_fwd",),
        ),
        Dependency(
            requirement="flash-linear-attention",
            module="fla.ops.cp.chunk_delta_h",
            symbols=(
                "merge_fwd_bwd_kernel",
                "pre_process_bwd_kernel_merged",
                "pre_process_fwd_kernel_merged",
            ),
        ),
        Dependency(
            requirement="flash-linear-attention",
            module="fla.ops.gated_delta_product.chunk_deltaproduct_h",
            symbols=("chunk_gated_delta_product_fwd_h",),
        ),
        Dependency(
            requirement="flash-linear-attention",
            module="fla.ops.gated_delta_product.chunk_deltaproduct_o",
            symbols=("chunk_gated_delta_product_fwd_o",),
        ),
        Dependency(
            requirement="flash-linear-attention",
            module="fla.ops.gated_delta_rule.wy_fast",
            symbols=("prepare_wy_repr_bwd", "recompute_w_u_fwd"),
        ),
        Dependency(
            requirement="flash-linear-attention",
            module="fla.ops.utils",
            symbols=("chunk_local_cumsum", "solve_tril"),
        ),
        Dependency(
            requirement="flash-linear-attention",
            module="fla.ops.utils.constant",
            symbols=("RCP_LN2",),
        ),
        Dependency(
            requirement="flash-linear-attention",
            module="fla.ops.utils.index",
            symbols=("prepare_chunk_indices",),
        ),
        Dependency(
            requirement="flash-linear-attention",
            module="fla.utils",
            symbols=("autocast_custom_bwd", "autocast_custom_fwd", "input_guard", "tensor_cache"),
        ),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="GDP training forward/backward has not been audited for bit-exact repeatability.",
    ),
    contract="megatron.core.ops.ssm.gdp",
)
GDP_CUTEDSL_CP = KernelMetadata(
    name="gdp.cutedsl.chunkwise_context_parallel",
    requires=(
        Dependency(
            requirement="gdp-attn",
            module="gdp_attn",
            symbols=(
                "cp_forward_prepare",
                "cp_forward_apply",
                "cp_backward_prepare",
                "cp_backward_apply",
            ),
        ),
        Dependency(
            requirement="gdp-attn",
            module="gdp_attn.chunk_gated_delta_product",
            symbols=("GdpCpBackwardContext", "GdpCpForwardLocalContext", "GdpCpSavedContext"),
        ),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="GDP training forward/backward has not been audited for bit-exact repeatability.",
    ),
    contract="megatron.core.ops.ssm.gdp",
)
GDP_PREFILL = KernelMetadata(
    name="gdp.triton.chunk_gated_delta_product_varlen",
    requires=(Dependency(requirement="triton", module="triton"),),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="Forward-only dynamic prefill/decode. Caller-owned states and autotuning settings "
        "affect execution; no cross-batch or cross-device repeatability guarantee is declared.",
    ),
    contract="megatron.core.ops.ssm.gdp",
)
GDP_DECODE = KernelMetadata(
    name="gdp.triton.fused_recurrent_gated_delta_rule_update",
    requires=(Dependency(requirement="triton", module="triton"),),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="Forward-only dynamic prefill/decode. Caller-owned states and autotuning settings "
        "affect execution; no cross-batch or cross-device repeatability guarantee is declared.",
    ),
    contract="megatron.core.ops.ssm.gdp",
)
GDP_PREPARE = KernelMetadata(
    name="gdp.triton.gdp_decode_prepare",
    requires=(
        Dependency(
            requirement="triton>=3.0",
            module="triton.language.extra.libdevice",
            symbols=("exp", "log1p", "div_rn"),
        ),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="Forward-only dynamic prefill/decode. Caller-owned states and autotuning settings "
        "affect execution; no cross-batch or cross-device repeatability guarantee is declared.",
    ),
    contract="megatron.core.ops.ssm.gdp",
)
GDP_L2NORM = KernelMetadata(
    name="gdp.fla.l2_norm",
    requires=(
        Dependency(
            requirement="flash-linear-attention", module="fla.modules.l2norm", symbols=("l2_norm",)
        ),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="GDP training forward/backward has not been audited for bit-exact repeatability.",
    ),
    contract="megatron.core.ops.ssm.gdp",
)

KERNELS = (
    GDP_CONV,
    GDP_FLA,
    GDP_CUTEDSL,
    GDP_FLA_CP,
    GDP_CUTEDSL_CP,
    GDP_PREFILL,
    GDP_DECODE,
    GDP_PREPARE,
    GDP_L2NORM,
)
