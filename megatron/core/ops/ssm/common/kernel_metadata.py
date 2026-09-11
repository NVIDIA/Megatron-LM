# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Declarations for shared convolution and state-extraction entry points."""

import os

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
    import torch

    from megatron.core.utils import is_causal_conv1d_min_version

    env = os.environ.get("CAUSAL_CONV1D_DETERMINISTIC", "")
    enabled = (
        env.startswith("1")
        if env[:1] in ("0", "1")
        else torch.are_deterministic_algorithms_enabled()
    )
    if not enabled:
        return DeterminismResult(
            Determinism.NONDETERMINISTIC,
            "causal-conv1d's deterministic backward reduction is disabled.",
        )
    if not is_causal_conv1d_min_version("1.6.0"):
        return DeterminismResult(
            Determinism.NONDETERMINISTIC,
            "causal-conv1d < 1.6.0 lacks the deterministic backward reduction.",
        )
    return _CONV


CAUSAL_CONV = KernelMetadata(
    name="ssm.causal_conv1d_fn",
    requires=(Dependency("causal-conv1d", "causal_conv1d", ("causal_conv1d_fn",)),),
    determinism=_CONV,
    contract=_CONTRACT,
    determinism_check=_conv_determinism,
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
SCATTER_CONV = KernelMetadata(
    name="ssm.triton.scatter_intermediate_conv",
    requires=CAUSAL_CONV_TRITON_UPDATE.requires,
    determinism=_INFERENCE,
    contract=_CONTRACT,
)

KERNELS = (
    CAUSAL_CONV,
    CAUSAL_CONV_CUDA_UPDATE,
    CAUSAL_CONV_TRITON_UPDATE,
    CAUSAL_CONV_VARLEN,
    CAUSAL_CONV_CARRY,
    SCATTER_SSM,
    SCATTER_CONV,
)
