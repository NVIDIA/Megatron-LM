# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Declarations for shared convolution and state-extraction entry points."""

from dataclasses import replace

from megatron.core.ops.kernel_metadata import (
    Dependency,
    Determinism,
    DeterminismResult,
    KernelMetadata,
)

_CONTRACT = "megatron.core.ops.ssm.common"
_CONV = DeterminismResult(
    Determinism.UNKNOWN,
    "Convolution forward/backward is not certified here. Deterministic backward requires "
    "causal-conv1d >= 1.6.0 and its deterministic reduction to be enabled.",
)
_INFERENCE = DeterminismResult(
    Determinism.UNKNOWN,
    "Forward-only state mutation/extraction. Callers own buffers and indices; autotuning, "
    "aliasing and indexed writes are not covered by a general repeatability guarantee.",
)


def _conv_determinism() -> DeterminismResult:
    from megatron.core.ops.ssm.common.causal_conv1d_cp import assert_causal_conv1d_deterministic

    try:
        assert_causal_conv1d_deterministic(deterministic_mode=True)
    except AssertionError as exc:
        return DeterminismResult(Determinism.NONDETERMINISTIC, str(exc))
    return _CONV


CAUSAL_CONV = KernelMetadata(
    name="ssm.causal_conv1d_fn",
    requires=(Dependency("causal-conv1d", "causal_conv1d", ("causal_conv1d_fn",)),),
    determinism=_CONV,
    contract=_CONTRACT,
    determinism_check=_conv_determinism,
)
CAUSAL_CONV_CP = replace(
    CAUSAL_CONV,
    name="ssm.causal_conv1d_cp",
    requires=(
        *CAUSAL_CONV.requires,
        # Combining seq_idx with initial_states requires the packed-CP implementation.
        Dependency("causal-conv1d>=1.7.0", "causal_conv1d", feature="packed"),
    ),
)
CAUSAL_CONV_CUDA_UPDATE = KernelMetadata(
    name="ssm.causal_conv1d_update_cuda",
    requires=(Dependency("causal-conv1d", "causal_conv1d", ("causal_conv1d_update",)),),
    determinism=_INFERENCE,
    contract=_CONTRACT,
)
CAUSAL_CONV_TRITON_UPDATE = KernelMetadata(
    name="ssm.triton.causal_conv1d_update",
    requires=(Dependency("triton", "triton"),),
    determinism=_INFERENCE,
    contract=_CONTRACT,
)
CAUSAL_CONV_VARLEN = KernelMetadata(
    name="ssm.triton.causal_conv1d_varlen_fn",
    requires=CAUSAL_CONV_TRITON_UPDATE.requires,
    determinism=_INFERENCE,
    contract=_CONTRACT,
)
CAUSAL_CONV_CARRY = KernelMetadata(
    name="ssm.triton.causal_conv1d_varlen_carry_states",
    requires=CAUSAL_CONV_TRITON_UPDATE.requires,
    determinism=_INFERENCE,
    contract=_CONTRACT,
)
SCATTER_SSM = KernelMetadata(
    name="ssm.triton.scatter_intermediate_ssm",
    requires=CAUSAL_CONV_TRITON_UPDATE.requires,
    determinism=_INFERENCE,
    contract=_CONTRACT,
)
THD_PARTITION = KernelMetadata(
    name="ssm.te.thd_get_partitioned_indices",
    requires=(
        Dependency("transformer-engine>=1.10.0", "transformer_engine"),
        Dependency(
            "transformer-engine", "transformer_engine_torch", ("thd_get_partitioned_indices",)
        ),
    ),
    determinism=DeterminismResult(
        Determinism.UNKNOWN,
        "Packed-sequence CP index generation has not been audited for repeatability.",
    ),
    contract=_CONTRACT,
)
SCATTER_CONV = KernelMetadata(
    name="ssm.triton.scatter_intermediate_conv",
    requires=CAUSAL_CONV_TRITON_UPDATE.requires,
    determinism=_INFERENCE,
    contract=_CONTRACT,
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
