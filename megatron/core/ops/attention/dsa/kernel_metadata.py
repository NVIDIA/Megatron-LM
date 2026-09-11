# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-hook DSA declarations; unsupported runtime inputs still use reference fallback."""

from megatron.core.ops.kernel_metadata import (
    Dependency,
    Determinism,
    DeterminismResult,
    KernelMetadata,
)

_CONTRACT = "megatron.core.ops.attention.dsa"
_DETERMINISM = DeterminismResult(
    Determinism.UNKNOWN,
    "DSA top-k tie handling, attention and indexer-loss forward/backward have not been "
    "audited for bit-exact repeatability across supported shapes, dtypes and devices.",
)
_TILELANG = (Dependency("tilelang", "tilelang"), Dependency("triton", "triton"))
_CUDNN_INDEXER = Dependency(
    "nvidia-cudnn-frontend", "cudnn", ("DSA.indexer_top_k_wrapper", "DSA.indexer_forward_wrapper")
)
_CUDNN_LOSS = Dependency(
    "nvidia-cudnn-frontend",
    "cudnn",
    (
        "DSA.indexer_backward_wrapper",
        "DSA.dense_indexer_backward_wrapper",
        "DSA.sparse_attn_score_recompute_wrapper",
        "DSA.dense_attn_score_recompute_wrapper",
    ),
)
_CUDNN_ATTENTION = Dependency(
    "nvidia-cudnn-frontend", "cudnn", ("DSA.sparse_attention_backward_wrapper",)
)
_FLASH_MLA = Dependency("flash-mla", "flash_mla", ("flash_mla_sparse_fwd",))

DSA_REFERENCE = KernelMetadata(
    name="dsa.unfused_dsa_fn", requires=(), determinism=_DETERMINISM, contract=_CONTRACT
)
DSA_INDEXER_REFERENCE = KernelMetadata(
    name="dsa.FusedDSAIndexerLoss", requires=(), determinism=_DETERMINISM, contract=_CONTRACT
)
HADAMARD_ROTATION = KernelMetadata(
    name="dsa.hadamard.rotate_activation",
    requires=(
        Dependency("fast-hadamard-transform", "fast_hadamard_transform", ("hadamard_transform",)),
    ),
    determinism=DeterminismResult(
        Determinism.UNKNOWN,
        "BF16 indexer rotation; native Hadamard forward/backward has not been audited "
        "for bit-exact repeatability.",
    ),
    contract=_CONTRACT,
)
TILELANG_TOPK = KernelMetadata(
    name="dsa.tilelang.run_fused_qk_topk",
    requires=(
        *_TILELANG,
        Dependency(
            "megatron-core",
            "megatron.core.ops.attention.dsa.kernels.indexer",
            ("lighting_indexer", "lighting_indexer_indices"),
        ),
    ),
    determinism=_DETERMINISM,
    contract=_CONTRACT,
)
TILELANG_LOSS = KernelMetadata(
    name="dsa.tilelang.run_fused_qk_topk_with_loss",
    requires=(
        *TILELANG_TOPK.requires,
        Dependency(
            "megatron-core",
            "megatron.core.ops.attention.dsa.kernels.tilelang_indexer_loss",
            ("SparseIndexerKLLoss", "sparse_indexer_target_interface"),
        ),
    ),
    determinism=_DETERMINISM,
    contract=_CONTRACT,
)
TILELANG_ATTENTION = KernelMetadata(
    name="dsa.tilelang.run_fused_absorbed_sparse_attention",
    requires=(
        *_TILELANG,
        Dependency(
            "megatron-core", "megatron.core.ops.attention.dsa.kernels.sparse_mla", ("SparseMLA",)
        ),
    ),
    determinism=_DETERMINISM,
    contract=_CONTRACT,
)
CUDNN_TOPK = KernelMetadata(
    name="dsa.cudnn.run_fused_qk_topk",
    requires=(_CUDNN_INDEXER,),
    determinism=_DETERMINISM,
    contract=_CONTRACT,
)
CUDNN_LOSS = KernelMetadata(
    name="dsa.cudnn.run_fused_qk_topk_with_loss",
    requires=(_CUDNN_INDEXER, _CUDNN_LOSS),
    determinism=_DETERMINISM,
    contract=_CONTRACT,
)
CUDNN_ATTENTION = KernelMetadata(
    name="dsa.cudnn.run_fused_absorbed_sparse_attention",
    requires=(_CUDNN_ATTENTION, _FLASH_MLA),
    determinism=_DETERMINISM,
    contract=_CONTRACT,
)
CUDNN_FULL = KernelMetadata(
    name="dsa.cudnn.run_fused_dsa_attention",
    requires=(_CUDNN_INDEXER, _CUDNN_LOSS, _CUDNN_ATTENTION, _FLASH_MLA),
    determinism=_DETERMINISM,
    contract=_CONTRACT,
)

KERNELS = (
    DSA_REFERENCE,
    DSA_INDEXER_REFERENCE,
    HADAMARD_ROTATION,
    TILELANG_TOPK,
    TILELANG_LOSS,
    TILELANG_ATTENTION,
    CUDNN_TOPK,
    CUDNN_LOSS,
    CUDNN_ATTENTION,
    CUDNN_FULL,
)
