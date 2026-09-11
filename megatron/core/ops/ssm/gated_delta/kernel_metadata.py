# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dependency and determinism declarations for GDN/GDN2 entry points."""

from megatron.core.ops.kernel_metadata import (
    Dependency,
    Determinism,
    DeterminismResult,
    KernelMetadata,
)

GDN_TORCH = KernelMetadata(
    name="gdn.torch_chunk_gated_delta_rule",
    requires=(
        Dependency(
            requirement="flash-linear-attention",
            module="fla.modules.l2norm",
            symbols=("l2norm",),
            feature="qk_l2norm",
        ),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="Existing deterministic-mode recurrence, for unpacked inputs. Bit-exact "
        "forward/backward has not been audited across dtypes and devices; optional Q/K "
        "normalization uses FLA.",
    ),
    contract="megatron.core.ops.ssm.gated_delta",
)
GDN2_TORCH = KernelMetadata(
    name="gdn2.torch_chunk_gdn2",
    requires=(
        Dependency(
            requirement="flash-linear-attention",
            module="fla.modules.l2norm",
            symbols=("l2norm",),
            feature="qk_l2norm",
        ),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="Existing deterministic-mode recurrence, for unpacked inputs. Bit-exact "
        "forward/backward has not been audited across dtypes and devices; optional Q/K "
        "normalization uses FLA.",
    ),
    contract="megatron.core.ops.ssm.gated_delta",
)
GDN_FLA = KernelMetadata(
    name="gdn.fla.chunk_gated_delta_rule",
    requires=(
        Dependency(
            requirement="flash-linear-attention",
            module="fla.ops.gated_delta_rule",
            symbols=("chunk_gated_delta_rule",),
        ),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="FLA chunk recurrence repeatability has not been audited here. GDN/GDN2 select "
        "the Torch reference for deterministic_mode; that selection alone does not classify "
        "the FLA kernel.",
    ),
    contract="megatron.core.ops.ssm.gated_delta",
)
GDN2_FLA = KernelMetadata(
    name="gdn2.fla.chunk_gdn2",
    requires=(
        Dependency(
            requirement="fla-core>=0.5.1", module="fla.ops.gdn2.chunk", symbols=("chunk_gdn2",)
        ),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="FLA chunk recurrence repeatability has not been audited here. GDN/GDN2 select "
        "the Torch reference for deterministic_mode; that selection alone does not classify "
        "the FLA kernel.",
    ),
    contract="megatron.core.ops.ssm.gated_delta",
)
GDN_RECURRENT = KernelMetadata(
    name="gdn.fla.fused_recurrent_gated_delta_rule",
    requires=(
        Dependency(
            requirement="flash-linear-attention",
            module="fla.ops.gated_delta_rule",
            symbols=("fused_recurrent_gated_delta_rule",),
        ),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="FLA auxiliary and recurrent entry points have not been audited.",
    ),
    contract="megatron.core.ops.ssm.gated_delta",
)
FLA_CONV = KernelMetadata(
    name="gdn.fla.causal_conv1d",
    requires=(
        Dependency(
            requirement="flash-linear-attention",
            module="fla.modules.convolution",
            symbols=("causal_conv1d",),
        ),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="FLA auxiliary and recurrent entry points have not been audited.",
    ),
    contract="megatron.core.ops.ssm.gated_delta",
)
FLA_CONV_UPDATE = KernelMetadata(
    name="gdn.fla.causal_conv1d_update",
    requires=(
        Dependency(
            requirement="flash-linear-attention",
            module="fla.modules.convolution",
            symbols=("causal_conv1d_update",),
        ),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="FLA auxiliary and recurrent entry points have not been audited.",
    ),
    contract="megatron.core.ops.ssm.gated_delta",
)
FLA_L2NORM = KernelMetadata(
    name="gdn.fla.l2norm",
    requires=(
        Dependency(
            requirement="flash-linear-attention", module="fla.modules.l2norm", symbols=("l2norm",)
        ),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="FLA auxiliary and recurrent entry points have not been audited.",
    ),
    contract="megatron.core.ops.ssm.gated_delta",
)

KERNELS = (
    GDN_TORCH,
    GDN2_TORCH,
    GDN_FLA,
    GDN2_FLA,
    GDN_RECURRENT,
    FLA_CONV,
    FLA_CONV_UPDATE,
    FLA_L2NORM,
)
