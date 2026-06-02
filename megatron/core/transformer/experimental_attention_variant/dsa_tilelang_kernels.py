# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""TileLang backend hooks for optional fused DeepSeek sparse attention kernels."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.experimental_attention_variant.ops import tilelang_dsa


def run_fused_qk_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    index_topk: int,
    starts: torch.Tensor,
    ends: torch.Tensor,
    block_size: int,
    use_relu: bool = True,
) -> Optional[torch.Tensor]:
    """Run fused TileLang indexer and return top-k indices."""
    return tilelang_dsa.run_fused_qk_topk(
        q, k, weights, index_topk, starts, ends, block_size, use_relu
    )


def run_fused_qk_topk_with_loss(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    index_topk: int,
    starts: torch.Tensor,
    ends: torch.Tensor,
    block_size: int,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    pg_collection: ProcessGroupCollection,
    query_valid_rows: Optional[torch.Tensor] = None,
    calculate_per_token_loss: bool = False,
    use_relu: bool = True,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Run fused TileLang indexer and sparse indexer loss."""
    return tilelang_dsa.run_fused_qk_topk_with_loss(
        q=q,
        k=k,
        weights=weights,
        index_topk=index_topk,
        starts=starts,
        ends=ends,
        block_size=block_size,
        query=query,
        key=key,
        softmax_scale=softmax_scale,
        loss_coeff=loss_coeff,
        pg_collection=pg_collection,
        query_valid_rows=query_valid_rows,
        calculate_per_token_loss=calculate_per_token_loss,
        use_relu=use_relu,
    )


def run_fused_absorbed_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    v_channels: int,
) -> Optional[torch.Tensor]:
    """Run fused TileLang SparseMLA for absorbed DSA sparse attention."""
    return tilelang_dsa.run_fused_absorbed_sparse_attention(
        query, key, topk_indices, softmax_scale, v_channels
    )


__all__ = [
    "run_fused_absorbed_sparse_attention",
    "run_fused_qk_topk",
    "run_fused_qk_topk_with_loss",
]
