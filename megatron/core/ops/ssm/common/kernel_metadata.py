# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Declarations for shared convolution and state-extraction entry points."""

from megatron.core.ops.kernel_metadata import (
    Dependency,
    Determinism,
    DeterminismResult,
    KernelMetadata,
)


def check_causal_conv_determinism() -> DeterminismResult:
    """Apply the convolution library's version and reduction-mode requirements."""
    from megatron.core.ops.ssm.common.causal_conv1d_cp import assert_causal_conv1d_deterministic

    try:
        assert_causal_conv1d_deterministic(deterministic_mode=True)
    except AssertionError as exc:
        return DeterminismResult(Determinism.NONDETERMINISTIC, str(exc))
    return DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="Convolution forward/backward is not certified here. Deterministic backward "
        "requires causal-conv1d >= 1.6.0 and its deterministic reduction to be enabled.",
    )


CAUSAL_CONV = KernelMetadata(
    name="ssm.causal_conv1d_fn",
    requires=(
        Dependency(
            requirement="causal-conv1d", module="causal_conv1d", symbols=("causal_conv1d_fn",)
        ),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="Convolution forward/backward is not certified here. Deterministic backward "
        "requires causal-conv1d >= 1.6.0 and its deterministic reduction to be enabled.",
    ),
    contract="megatron.core.ops.ssm.common",
    determinism_check=check_causal_conv_determinism,
)
CAUSAL_CONV_CP = KernelMetadata(
    name="ssm.causal_conv1d_cp",
    requires=(
        Dependency(
            requirement="causal-conv1d", module="causal_conv1d", symbols=("causal_conv1d_fn",)
        ),
        # Combining seq_idx with initial_states requires the packed-CP implementation.
        Dependency(requirement="causal-conv1d>=1.7.0", module="causal_conv1d", feature="packed"),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="Convolution forward/backward is not certified here. Deterministic backward "
        "requires causal-conv1d >= 1.6.0 and its deterministic reduction to be enabled.",
    ),
    contract="megatron.core.ops.ssm.common",
    determinism_check=check_causal_conv_determinism,
)
CAUSAL_CONV_CUDA_UPDATE = KernelMetadata(
    name="ssm.causal_conv1d_update_cuda",
    requires=(
        Dependency(
            requirement="causal-conv1d", module="causal_conv1d", symbols=("causal_conv1d_update",)
        ),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="Forward-only state mutation/extraction. Callers own buffers and indices; "
        "autotuning, aliasing and indexed writes are not covered by a general repeatability "
        "guarantee.",
    ),
    contract="megatron.core.ops.ssm.common",
)
CAUSAL_CONV_TRITON_UPDATE = KernelMetadata(
    name="ssm.triton.causal_conv1d_update",
    requires=(Dependency(requirement="triton", module="triton"),),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="Forward-only state mutation/extraction. Callers own buffers and indices; "
        "autotuning, aliasing and indexed writes are not covered by a general repeatability "
        "guarantee.",
    ),
    contract="megatron.core.ops.ssm.common",
)
CAUSAL_CONV_VARLEN = KernelMetadata(
    name="ssm.triton.causal_conv1d_varlen_fn",
    requires=(Dependency(requirement="triton", module="triton"),),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="Forward-only state mutation/extraction. Callers own buffers and indices; "
        "autotuning, aliasing and indexed writes are not covered by a general repeatability "
        "guarantee.",
    ),
    contract="megatron.core.ops.ssm.common",
)
CAUSAL_CONV_CARRY = KernelMetadata(
    name="ssm.triton.causal_conv1d_varlen_carry_states",
    requires=(Dependency(requirement="triton", module="triton"),),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="Forward-only state mutation/extraction. Callers own buffers and indices; "
        "autotuning, aliasing and indexed writes are not covered by a general repeatability "
        "guarantee.",
    ),
    contract="megatron.core.ops.ssm.common",
)
SCATTER_SSM = KernelMetadata(
    name="ssm.triton.scatter_intermediate_ssm",
    requires=(Dependency(requirement="triton", module="triton"),),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="Forward-only state mutation/extraction. Callers own buffers and indices; "
        "autotuning, aliasing and indexed writes are not covered by a general repeatability "
        "guarantee.",
    ),
    contract="megatron.core.ops.ssm.common",
)
THD_PARTITION = KernelMetadata(
    name="ssm.te.thd_get_partitioned_indices",
    requires=(
        Dependency(requirement="transformer-engine>=1.10.0", module="transformer_engine"),
        Dependency(
            requirement="transformer-engine",
            module="transformer_engine_torch",
            symbols=("thd_get_partitioned_indices",),
        ),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="Packed-sequence CP index generation has not been audited for repeatability.",
    ),
    contract="megatron.core.ops.ssm.common",
)
SCATTER_CONV = KernelMetadata(
    name="ssm.triton.scatter_intermediate_conv",
    requires=(Dependency(requirement="triton", module="triton"),),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="Forward-only state mutation/extraction. Callers own buffers and indices; "
        "autotuning, aliasing and indexed writes are not covered by a general repeatability "
        "guarantee.",
    ),
    contract="megatron.core.ops.ssm.common",
)

KERNELS = (
    THD_PARTITION,
    CAUSAL_CONV,
    CAUSAL_CONV_CP,
    CAUSAL_CONV_CUDA_UPDATE,
    CAUSAL_CONV_TRITON_UPDATE,
    CAUSAL_CONV_VARLEN,
    CAUSAL_CONV_CARRY,
    SCATTER_SSM,
    SCATTER_CONV,
)
