# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CSA's eager kernel declarations, shared with the legacy determinism mapping."""

from megatron.core.ops.kernel_metadata import Determinism, DeterminismResult, KernelMetadata

_CONTRACT = "megatron.core.ops.attention.csa"
_DETERMINISM = DeterminismResult(
    Determinism.UNKNOWN,
    "CUDA reductions and indexed accumulation have not been audited for bit-exact repeatability.",
)
CSA_ATTENTION = KernelMetadata(
    name="csa.unfused_sparse_attention", requires=(), determinism=_DETERMINISM, contract=_CONTRACT
)
CSA_LSE = KernelMetadata(
    name="csa.non_compressed_lse", requires=(), determinism=_DETERMINISM, contract=_CONTRACT
)
CSA_POOLING = KernelMetadata(
    name="csa.compressor_pooling", requires=(), determinism=_DETERMINISM, contract=_CONTRACT
)

KERNELS = (CSA_ATTENTION, CSA_LSE, CSA_POOLING)

CSA_OPERATION_DETERMINISM = {
    kernel.name.removeprefix("csa."): kernel.determinism.status.value for kernel in KERNELS
}
