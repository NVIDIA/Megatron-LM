# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from .tilelang_indexer_bwd import HAVE_TILELANG as HAVE_TILELANG_INDEXER_BWD
from .tilelang_indexer_bwd import indexer_bwd_interface
from .tilelang_indexer_fwd import HAVE_TILELANG as HAVE_TILELANG_INDEXER_FWD
from .tilelang_indexer_fwd import indexer_fwd_interface

HAVE_TILELANG_INDEXER = HAVE_TILELANG_INDEXER_BWD and HAVE_TILELANG_INDEXER_FWD


def pytorch_extract_topk_scores(logits, topk_indices, dim=-1):
    """Gather top-k logits and mask invalid (-1) entries with -inf."""
    if logits.size(dim) == 0:
        return torch.full(
            topk_indices.shape, float("-inf"), dtype=logits.dtype, device=logits.device
        )
    valid_mask = (topk_indices >= 0) & (topk_indices < logits.size(dim))
    safe_indices = topk_indices.clamp(min=0, max=logits.size(dim) - 1).to(torch.int64)
    scores = torch.gather(logits, dim=dim, index=safe_indices)
    scores = torch.where(valid_mask, scores, float("-inf"))
    return scores


def _select_topk_from_logits(
    logits: torch.Tensor, topk: int, mask_invalid: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select top-k scores and int32 indices from indexer logits."""
    effective_topk = min(topk, logits.size(-1))
    if effective_topk > 0:
        topk_scores, topk_indices = torch.topk(logits, effective_topk, dim=-1, sorted=False)
        topk_indices = topk_indices.to(torch.int32)
        if mask_invalid:
            topk_indices = topk_indices.masked_fill(topk_scores == -torch.inf, -1)
        return topk_scores, topk_indices

    empty_shape = logits.shape[:-1] + (0,)
    topk_scores = torch.empty(empty_shape, dtype=logits.dtype, device=logits.device)
    topk_indices = torch.empty(empty_shape, dtype=torch.int32, device=logits.device)
    return topk_scores, topk_indices


class IndexerFunction(torch.autograd.Function):  # pragma: no cover
    """Autograd wrapper for fused tilelang indexer forward/backward."""

    @staticmethod
    def forward(
        ctx,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        topk: int,
        topk_indices: torch.Tensor | None = None,
        use_relu: bool = True,
    ):
        """Run fused indexer forward and optionally select top-k indices."""
        logits = indexer_fwd_interface(
            index_q,
            index_k,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            clean_logits=True,
            use_relu=use_relu,
        )
        if topk_indices is None:
            index_score, topk_indices = _select_topk_from_logits(logits, topk)
        else:
            index_score = pytorch_extract_topk_scores(logits, topk_indices)

        ctx.save_for_backward(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices)
        ctx.use_relu = use_relu
        return index_score, topk_indices

    @staticmethod
    def backward(ctx, grad_scores, grad_indices):
        """Propagate gradients through fused indexer outputs."""
        index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices = ctx.saved_tensors
        grad_q, grad_w, grad_k = indexer_bwd_interface(
            index_q, weights, index_k, topk_indices, grad_scores, use_relu=ctx.use_relu
        )
        return grad_q, grad_k, grad_w, None, None, None, None, None


def lighting_indexer(  # pragma: no cover
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk: int,
    topk_indices: torch.Tensor | None = None,
    use_relu: bool = True,
):
    """Compute indexer top-k scores/indices via the custom autograd function."""
    return IndexerFunction.apply(
        index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk, topk_indices, use_relu
    )


def lighting_indexer_indices(  # pragma: no cover
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk: int,
    use_relu: bool = True,
):
    """Compute TileLang indexer top-k indices without score/autograd bookkeeping."""
    with torch.no_grad():
        logits = indexer_fwd_interface(
            index_q,
            index_k,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            clean_logits=True,
            use_relu=use_relu,
        )
        _, topk_indices = _select_topk_from_logits(logits, topk, mask_invalid=False)
    return topk_indices


if not HAVE_TILELANG_INDEXER:
    IndexerFunction = None
    lighting_indexer = None
    lighting_indexer_indices = None
