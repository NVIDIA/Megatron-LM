# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Phase-specific Mamba2 declarations, independent of model state and CP."""

from megatron.core.ops.kernel_metadata import (
    Dependency,
    Determinism,
    DeterminismResult,
    KernelMetadata,
)

MAMBA_SCAN = KernelMetadata(
    name="mamba2.mamba_ssm.mamba_chunk_scan_combined",
    requires=(
        Dependency(
            requirement="mamba-ssm",
            module="mamba_ssm.ops.triton.ssd_combined",
            symbols=("mamba_chunk_scan_combined",),
        ),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="mamba-ssm training forward/backward is not certified here. The caller's "
        "causal-conv1d version and deterministic-reduction guards must still be applied.",
    ),
    contract="megatron.core.ops.ssm.mamba2",
)
MAMBA_SPLIT_SCAN = KernelMetadata(
    name="mamba2.mamba_ssm.mamba_split_conv1d_scan_combined",
    requires=(
        Dependency(
            requirement="mamba-ssm",
            module="mamba_ssm.ops.triton.ssd_combined",
            symbols=("mamba_split_conv1d_scan_combined",),
        ),
        Dependency(
            requirement="causal-conv1d", module="causal_conv1d", symbols=("causal_conv1d_fn",)
        ),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="mamba-ssm training forward/backward is not certified here. The caller's "
        "causal-conv1d version and deterministic-reduction guards must still be applied.",
    ),
    contract="megatron.core.ops.ssm.mamba2",
)
MAMBA_NORM = KernelMetadata(
    name="mamba2.mamba_ssm.RMSNormGated",
    requires=(
        Dependency(
            requirement="mamba-ssm",
            module="mamba_ssm.ops.triton.layernorm_gated",
            symbols=("RMSNorm",),
        ),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="mamba-ssm training forward/backward is not certified here. The caller's "
        "causal-conv1d version and deterministic-reduction guards must still be applied.",
    ),
    contract="megatron.core.ops.ssm.mamba2",
)
MAMBA_PREFILL = KernelMetadata(
    name="mamba2.triton.mamba_chunk_scan_combined_varlen",
    requires=(Dependency(requirement="triton", module="triton"),),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="Forward-only SSD/state update. Existing deterministic autotuning and "
        "batch-invariant decode controls remain required; no broader bit-exact guarantee "
        "is declared.",
    ),
    contract="megatron.core.ops.ssm.mamba2",
)
MAMBA_DECODE = KernelMetadata(
    name="mamba2.triton.selective_state_update",
    requires=(Dependency(requirement="triton", module="triton"),),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="Forward-only SSD/state update. Existing deterministic autotuning and "
        "batch-invariant decode controls remain required; no broader bit-exact guarantee "
        "is declared.",
    ),
    contract="megatron.core.ops.ssm.mamba2",
)
MAMBA_BATCH_INVARIANT = KernelMetadata(
    name="mamba2.triton.batch_invariant_decode_buffered_scan",
    requires=(Dependency(requirement="triton", module="triton"),),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="Forward-only SSD/state update. Existing deterministic autotuning and "
        "batch-invariant decode controls remain required; no broader bit-exact guarantee "
        "is declared.",
    ),
    contract="megatron.core.ops.ssm.mamba2",
)

KERNELS = (
    MAMBA_SCAN,
    MAMBA_SPLIT_SCAN,
    MAMBA_NORM,
    MAMBA_PREFILL,
    MAMBA_DECODE,
    MAMBA_BATCH_INVARIANT,
)
