# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Construction-time selection for detached KPool score postprocessing."""

from megatron.core.ops._backends import require
from megatron.core.ops.attention.dsa import KPoolScorePostprocess
from megatron.core.ops.attention.dsa.reference import torch_kpool_score_postprocess


def select_kpool_score_postprocess(backend: str, num_heads: int) -> KPoolScorePostprocess:
    """Bind the requested score backend; unsupported explicit choices never fall back."""
    if backend == "torch":
        return torch_kpool_score_postprocess
    if backend != "triton":
        raise ValueError("dsa_indexer_kpool_score_backend must be torch or triton")
    if num_heads != 32:
        raise ValueError("Triton KPool score postprocessing requires 32 indexer heads")
    return require(
        "megatron.core.ops.attention.dsa.kpool_score_kernels",
        "triton_kpool_score_postprocess",
        needed_by="Triton KPool score postprocessing",
    ).triton_kpool_score_postprocess
