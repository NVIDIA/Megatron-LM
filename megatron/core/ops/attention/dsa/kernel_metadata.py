# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-hook DSA declarations; unsupported runtime inputs still use reference fallback."""

from megatron.core.ops.kernel_metadata import (
    Dependency,
    Determinism,
    DeterminismResult,
    KernelMetadata,
)

DSA_REFERENCE = KernelMetadata(
    name="dsa.unfused_dsa_fn",
    requires=(),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="DSA top-k tie handling, attention and indexer-loss forward/backward have not been "
        "audited for bit-exact repeatability across supported shapes, dtypes and devices.",
    ),
    contract="megatron.core.ops.attention.dsa",
)
DSA_INDEXER_REFERENCE = KernelMetadata(
    name="dsa.FusedDSAIndexerLoss",
    requires=(),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="DSA top-k tie handling, attention and indexer-loss forward/backward have not been "
        "audited for bit-exact repeatability across supported shapes, dtypes and devices.",
    ),
    contract="megatron.core.ops.attention.dsa",
)
HADAMARD_ROTATION = KernelMetadata(
    name="dsa.hadamard.rotate_activation",
    requires=(
        Dependency(
            requirement="fast-hadamard-transform",
            module="fast_hadamard_transform",
            symbols=("hadamard_transform",),
        ),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="BF16 indexer rotation; native Hadamard forward/backward has not been audited "
        "for bit-exact repeatability.",
    ),
    contract="megatron.core.ops.attention.dsa",
)
TILELANG_TOPK = KernelMetadata(
    name="dsa.tilelang.run_fused_qk_topk",
    requires=(
        Dependency(requirement="tilelang", module="tilelang"),
        Dependency(requirement="triton", module="triton"),
        Dependency(
            requirement="megatron-core",
            module="megatron.core.ops.attention.dsa.kernels.indexer",
            symbols=("lighting_indexer", "lighting_indexer_indices"),
        ),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="DSA top-k tie handling, attention and indexer-loss forward/backward have not been "
        "audited for bit-exact repeatability across supported shapes, dtypes and devices.",
    ),
    contract="megatron.core.ops.attention.dsa",
)
TILELANG_LOSS = KernelMetadata(
    name="dsa.tilelang.run_fused_qk_topk_with_loss",
    requires=(
        Dependency(requirement="tilelang", module="tilelang"),
        Dependency(requirement="triton", module="triton"),
        Dependency(
            requirement="megatron-core",
            module="megatron.core.ops.attention.dsa.kernels.indexer",
            symbols=("lighting_indexer", "lighting_indexer_indices"),
        ),
        Dependency(
            requirement="megatron-core",
            module="megatron.core.ops.attention.dsa.kernels.tilelang_indexer_loss",
            symbols=("SparseIndexerKLLoss", "sparse_indexer_target_interface"),
        ),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="DSA top-k tie handling, attention and indexer-loss forward/backward have not been "
        "audited for bit-exact repeatability across supported shapes, dtypes and devices.",
    ),
    contract="megatron.core.ops.attention.dsa",
)
TILELANG_ATTENTION = KernelMetadata(
    name="dsa.tilelang.run_fused_absorbed_sparse_attention",
    requires=(
        Dependency(requirement="tilelang", module="tilelang"),
        Dependency(requirement="triton", module="triton"),
        Dependency(
            requirement="megatron-core",
            module="megatron.core.ops.attention.dsa.kernels.sparse_mla",
            symbols=("SparseMLA",),
        ),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="DSA top-k tie handling, attention and indexer-loss forward/backward have not been "
        "audited for bit-exact repeatability across supported shapes, dtypes and devices.",
    ),
    contract="megatron.core.ops.attention.dsa",
)
CUDNN_TOPK = KernelMetadata(
    name="dsa.cudnn.run_fused_qk_topk",
    requires=(
        Dependency(
            requirement="nvidia-cudnn-frontend",
            module="cudnn",
            symbols=("DSA.indexer_top_k_wrapper", "DSA.indexer_forward_wrapper"),
        ),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="DSA top-k tie handling, attention and indexer-loss forward/backward have not been "
        "audited for bit-exact repeatability across supported shapes, dtypes and devices.",
    ),
    contract="megatron.core.ops.attention.dsa",
)
CUDNN_LOSS = KernelMetadata(
    name="dsa.cudnn.run_fused_qk_topk_with_loss",
    requires=(
        Dependency(
            requirement="nvidia-cudnn-frontend",
            module="cudnn",
            symbols=("DSA.indexer_top_k_wrapper", "DSA.indexer_forward_wrapper"),
        ),
        Dependency(
            requirement="nvidia-cudnn-frontend",
            module="cudnn",
            symbols=(
                "DSA.indexer_backward_wrapper",
                "DSA.dense_indexer_backward_wrapper",
                "DSA.sparse_attn_score_recompute_wrapper",
                "DSA.dense_attn_score_recompute_wrapper",
            ),
        ),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="DSA top-k tie handling, attention and indexer-loss forward/backward have not been "
        "audited for bit-exact repeatability across supported shapes, dtypes and devices.",
    ),
    contract="megatron.core.ops.attention.dsa",
)
CUDNN_ATTENTION = KernelMetadata(
    name="dsa.cudnn.run_fused_absorbed_sparse_attention",
    requires=(
        Dependency(
            requirement="nvidia-cudnn-frontend",
            module="cudnn",
            symbols=("DSA.sparse_attention_backward_wrapper",),
        ),
        Dependency(requirement="flash-mla", module="flash_mla", symbols=("flash_mla_sparse_fwd",)),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="DSA top-k tie handling, attention and indexer-loss forward/backward have not been "
        "audited for bit-exact repeatability across supported shapes, dtypes and devices.",
    ),
    contract="megatron.core.ops.attention.dsa",
)
CUDNN_FULL = KernelMetadata(
    name="dsa.cudnn.run_fused_dsa_attention",
    requires=(
        Dependency(
            requirement="nvidia-cudnn-frontend",
            module="cudnn",
            symbols=("DSA.indexer_top_k_wrapper", "DSA.indexer_forward_wrapper"),
        ),
        Dependency(
            requirement="nvidia-cudnn-frontend",
            module="cudnn",
            symbols=(
                "DSA.indexer_backward_wrapper",
                "DSA.dense_indexer_backward_wrapper",
                "DSA.sparse_attn_score_recompute_wrapper",
                "DSA.dense_attn_score_recompute_wrapper",
            ),
        ),
        Dependency(
            requirement="nvidia-cudnn-frontend",
            module="cudnn",
            symbols=("DSA.sparse_attention_backward_wrapper",),
        ),
        Dependency(requirement="flash-mla", module="flash_mla", symbols=("flash_mla_sparse_fwd",)),
    ),
    determinism=DeterminismResult(
        status=Determinism.UNKNOWN,
        reason="DSA top-k tie handling, attention and indexer-loss forward/backward have not been "
        "audited for bit-exact repeatability across supported shapes, dtypes and devices.",
    ),
    contract="megatron.core.ops.attention.dsa",
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
