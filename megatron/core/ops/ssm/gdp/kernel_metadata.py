# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GDP training and inference declarations; importing them loads no kernels."""

from dataclasses import replace

from megatron.core.ops.kernel_metadata import (
    Dependency,
    Determinism,
    DeterminismResult,
    KernelMetadata,
)
from megatron.core.ops.ssm.common.kernel_metadata import CAUSAL_CONV

_CONTRACT = "megatron.core.ops.ssm.gdp"
_TRAINING = DeterminismResult(
    Determinism.UNKNOWN,
    "GDP training forward/backward has not been audited for bit-exact repeatability.",
)
_INFERENCE = DeterminismResult(
    Determinism.UNKNOWN,
    "Forward-only dynamic prefill/decode. Caller-owned states and autotuning settings affect "
    "execution; no cross-batch or cross-device repeatability guarantee is declared.",
)
GDP_CONV = replace(
    CAUSAL_CONV,
    name="gdp.causal_conv1d_fn",
    requires=(Dependency("causal-conv1d>=1.4.0", "causal_conv1d", ("causal_conv1d_fn",)),),
    contract=_CONTRACT,
)
GDP_FLA = KernelMetadata(
    name="gdp.fla.chunk_gated_delta_product",
    requires=(
        Dependency(
            "flash-linear-attention", "fla.ops.gated_delta_product", ("chunk_gated_delta_product",)
        ),
    ),
    determinism=_TRAINING,
    contract=_CONTRACT,
)
GDP_CUTEDSL = KernelMetadata(
    name="gdp.cutedsl.chunk_gated_delta_product",
    requires=(Dependency("gdp-attn", "gdp_attn", ("chunk_gated_delta_product",)),),
    determinism=_TRAINING,
    contract=_CONTRACT,
)
GDP_FLA_CP = KernelMetadata(
    name="gdp.fla.chunkwise_context_parallel",
    requires=(
        Dependency("triton", "triton"),
        Dependency(
            "flash-linear-attention",
            "fla.ops.common.chunk_delta_h",
            ("chunk_gated_delta_rule_bwd_dhu", "chunk_gated_delta_rule_fwd_h"),
        ),
        Dependency(
            "flash-linear-attention",
            "fla.ops.common.chunk_o",
            ("chunk_bwd_dqkwg", "chunk_bwd_dv_local"),
        ),
        Dependency(
            "flash-linear-attention",
            "fla.ops.common.chunk_scaled_dot_kkt",
            ("chunk_scaled_dot_kkt_fwd",),
        ),
        Dependency(
            "flash-linear-attention",
            "fla.ops.cp.chunk_delta_h",
            (
                "merge_fwd_bwd_kernel",
                "pre_process_bwd_kernel_merged",
                "pre_process_fwd_kernel_merged",
            ),
        ),
        Dependency(
            "flash-linear-attention",
            "fla.ops.gated_delta_product.chunk_deltaproduct_h",
            ("chunk_gated_delta_product_fwd_h",),
        ),
        Dependency(
            "flash-linear-attention",
            "fla.ops.gated_delta_product.chunk_deltaproduct_o",
            ("chunk_gated_delta_product_fwd_o",),
        ),
        Dependency(
            "flash-linear-attention",
            "fla.ops.gated_delta_rule.wy_fast",
            ("prepare_wy_repr_bwd", "recompute_w_u_fwd"),
        ),
        Dependency("flash-linear-attention", "fla.ops.utils", ("chunk_local_cumsum", "solve_tril")),
        Dependency("flash-linear-attention", "fla.ops.utils.constant", ("RCP_LN2",)),
        Dependency("flash-linear-attention", "fla.ops.utils.index", ("prepare_chunk_indices",)),
        Dependency(
            "flash-linear-attention",
            "fla.utils",
            ("autocast_custom_bwd", "autocast_custom_fwd", "input_guard", "tensor_cache"),
        ),
    ),
    determinism=_TRAINING,
    contract=_CONTRACT,
)
GDP_CUTEDSL_CP = KernelMetadata(
    name="gdp.cutedsl.chunkwise_context_parallel",
    requires=(
        Dependency(
            "gdp-attn",
            "gdp_attn",
            ("cp_forward_prepare", "cp_forward_apply", "cp_backward_prepare", "cp_backward_apply"),
        ),
        Dependency(
            "gdp-attn",
            "gdp_attn.chunk_gated_delta_product",
            ("GdpCpBackwardContext", "GdpCpForwardLocalContext", "GdpCpSavedContext"),
        ),
    ),
    determinism=_TRAINING,
    contract=_CONTRACT,
)
GDP_PREFILL = KernelMetadata(
    name="gdp.triton.chunk_gated_delta_product_varlen",
    requires=(Dependency("triton", "triton"),),
    determinism=_INFERENCE,
    contract=_CONTRACT,
)
GDP_DECODE = KernelMetadata(
    name="gdp.triton.fused_recurrent_gated_delta_rule_update",
    requires=GDP_PREFILL.requires,
    determinism=_INFERENCE,
    contract=_CONTRACT,
)
GDP_PREPARE = KernelMetadata(
    name="gdp.triton.gdp_decode_prepare",
    requires=(
        Dependency("triton>=3.0", "triton.language.extra.libdevice", ("exp", "log1p", "div_rn")),
    ),
    determinism=_INFERENCE,
    contract=_CONTRACT,
)
GDP_L2NORM = KernelMetadata(
    name="gdp.fla.l2_norm",
    requires=(Dependency("flash-linear-attention", "fla.modules.l2norm", ("l2_norm",)),),
    determinism=_TRAINING,
    contract=_CONTRACT,
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
