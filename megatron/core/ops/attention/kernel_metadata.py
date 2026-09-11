# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared fused RoPE requirements for the MLA and DeepSeek attention operations."""

from megatron.core.ops.kernel_metadata import (
    Dependency,
    Determinism,
    DeterminismResult,
    KernelMetadata,
)

_ROPE_MODULE = "megatron.core.fusions.fused_mla_yarn_rope_apply"
_ROPE_DETERMINISM = DeterminismResult(
    Determinism.UNKNOWN, "Fused RoPE forward/backward repeatability has not been audited."
)
MLA_ROPE = KernelMetadata(
    name="mla.triton.fused_rope",
    requires=(
        Dependency("triton", "triton"),
        Dependency("megatron-core", _ROPE_MODULE, ("fused_apply_mla_rope_for_q",)),
    ),
    determinism=_ROPE_DETERMINISM,
    contract="megatron.core.ops.attention",
)
DSV4_ROPE = KernelMetadata(
    name="dsv4.triton.fused_rope",
    requires=(
        Dependency("triton", "triton"),
        Dependency(
            "megatron-core", _ROPE_MODULE, ("fused_mla_rope_inplace", "fused_mla_rope_out_of_place")
        ),
    ),
    determinism=_ROPE_DETERMINISM,
    contract="megatron.core.ops.attention",
)
KERNELS = (MLA_ROPE, DSV4_ROPE)
