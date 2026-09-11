# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Phase-specific Mamba2 declarations, independent of model state and CP."""

from megatron.core.ops.kernel_metadata import (
    Dependency,
    Determinism,
    DeterminismResult,
    KernelMetadata,
)

_CONTRACT = "megatron.core.ops.ssm.mamba2"
_TRAINING = DeterminismResult(
    Determinism.UNKNOWN,
    "mamba-ssm training forward/backward is not certified here. The caller's causal-conv1d "
    "version and deterministic-reduction guards must still be applied.",
)
_INFERENCE = DeterminismResult(
    Determinism.UNKNOWN,
    "Forward-only SSD/state update. Existing deterministic autotuning and batch-invariant "
    "decode controls remain required; no broader bit-exact guarantee is declared.",
)
MAMBA_SCAN = KernelMetadata(
    name="mamba2.mamba_ssm.mamba_chunk_scan_combined",
    requires=(
        Dependency(
            "mamba-ssm", "mamba_ssm.ops.triton.ssd_combined", ("mamba_chunk_scan_combined",)
        ),
    ),
    determinism=_TRAINING,
    contract=_CONTRACT,
)
MAMBA_SPLIT_SCAN = KernelMetadata(
    name="mamba2.mamba_ssm.mamba_split_conv1d_scan_combined",
    requires=(
        Dependency(
            "mamba-ssm", "mamba_ssm.ops.triton.ssd_combined", ("mamba_split_conv1d_scan_combined",)
        ),
        Dependency("causal-conv1d", "causal_conv1d", ("causal_conv1d_fn",)),
    ),
    determinism=_TRAINING,
    contract=_CONTRACT,
)
MAMBA_NORM = KernelMetadata(
    name="mamba2.mamba_ssm.RMSNormGated",
    requires=(Dependency("mamba-ssm", "mamba_ssm.ops.triton.layernorm_gated", ("RMSNorm",)),),
    determinism=_TRAINING,
    contract=_CONTRACT,
)
MAMBA_PREFILL = KernelMetadata(
    name="mamba2.triton.mamba_chunk_scan_combined_varlen",
    requires=(Dependency("triton", "triton"),),
    determinism=_INFERENCE,
    contract=_CONTRACT,
)
MAMBA_DECODE = KernelMetadata(
    name="mamba2.triton.selective_state_update",
    requires=MAMBA_PREFILL.requires,
    determinism=_INFERENCE,
    contract=_CONTRACT,
)
MAMBA_BATCH_INVARIANT = KernelMetadata(
    name="mamba2.triton.batch_invariant_decode_buffered_scan",
    requires=MAMBA_PREFILL.requires,
    determinism=_INFERENCE,
    contract=_CONTRACT,
)

KERNELS = (
    MAMBA_SCAN,
    MAMBA_SPLIT_SCAN,
    MAMBA_NORM,
    MAMBA_PREFILL,
    MAMBA_DECODE,
    MAMBA_BATCH_INVARIANT,
)
