# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CSA's eager kernel declarations, shared with the legacy determinism mapping."""

from megatron.core.ops.kernel_metadata import Determinism, DeterminismResult, KernelMetadata

CSA_ATTENTION = KernelMetadata(
    name="csa.unfused_sparse_attention",
    requires=(),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="CUDA reductions and indexed accumulation have not been audited for "
        "bit-exact repeatability.",
    ),
    contract="megatron.core.ops.attention.csa",
)
CSA_LSE = KernelMetadata(
    name="csa.non_compressed_lse",
    requires=(),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="CUDA reductions and indexed accumulation have not been audited for "
        "bit-exact repeatability.",
    ),
    contract="megatron.core.ops.attention.csa",
)
CSA_POOLING = KernelMetadata(
    name="csa.compressor_pooling",
    requires=(),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="CUDA reductions and indexed accumulation have not been audited for "
        "bit-exact repeatability.",
    ),
    contract="megatron.core.ops.attention.csa",
)

KERNELS = (CSA_ATTENTION, CSA_LSE, CSA_POOLING)

CSA_OPERATION_DETERMINISM = {
    kernel.name.removeprefix("csa."): kernel.determinism.status.value for kernel in KERNELS
}
