# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Compiled row masking, KL reduction and CP normalization for packed CSA."""

from __future__ import annotations

import torch
from torch import Tensor

_CLIP_PROB_MIN = torch.finfo(torch.float32).tiny


def _sparse_kl_loss_impl(
    target: Tensor,
    predict: Tensor,
    topk_indices: Tensor,
    loss_coeff: float,
    calculate_per_token_loss: bool,
    loss_divisor: int | float | Tensor | None,
) -> Tensor:
    """Compute sparse indexer KL with all rowwise glue kept in one graph."""
    target_clamped = target.clamp(min=_CLIP_PROB_MIN)
    predict_clamped = predict.clamp(min=_CLIP_PROB_MIN)
    kl_per_row = (target_clamped * (torch.log(target_clamped) - torch.log(predict_clamped))).sum(
        dim=-1
    )

    row_valid = (topk_indices >= 0).any(dim=-1)
    kl_per_row = torch.where(row_valid, kl_per_row, torch.zeros_like(kl_per_row))
    loss = kl_per_row.sum() if calculate_per_token_loss else kl_per_row.mean()
    loss = loss_coeff * loss
    if loss_divisor is not None:
        loss = loss / loss_divisor
    return loss


@torch.compile(fullgraph=True)
def _compiled_sparse_kl_loss(
    target: Tensor,
    predict: Tensor,
    topk_indices: Tensor,
    loss_coeff: float,
    calculate_per_token_loss: bool,
    loss_divisor: int | float | Tensor | None,
) -> Tensor:
    """Compile the sparse KL reductions into CUDA pointwise/reduction kernels."""
    return _sparse_kl_loss_impl(
        target, predict, topk_indices, loss_coeff, calculate_per_token_loss, loss_divisor
    )


def sparse_kl_loss(
    target: Tensor,
    predict: Tensor,
    topk_indices: Tensor,
    loss_coeff: float,
    calculate_per_token_loss: bool = False,
    loss_divisor: int | float | Tensor | None = None,
) -> Tensor:
    """Compute the sparse CSA indexer KL loss.

    CUDA inputs use a compiled full graph so clamp, log, row validity, row
    reduction, scaling, and an optional global divisor do not materialize as
    separate eager kernels. CPU inputs retain the eager implementation for
    reference tests and environments without CUDA.

    Args:
        target: Attention probabilities over selected positions.
        predict: Indexer probabilities with the same shape as ``target``.
        topk_indices: Selected indices; a row is valid when any entry is nonnegative.
        loss_coeff: Multiplicative loss coefficient.
        calculate_per_token_loss: Return a row sum instead of the default row mean.
        loss_divisor: Optional divisor applied after ``loss_coeff``. This folds
            context-parallel loss normalization into the compiled reduction.

    Returns:
        A scalar loss tensor.
    """
    implementation = _compiled_sparse_kl_loss if target.is_cuda else _sparse_kl_loss_impl
    return implementation(
        target, predict, topk_indices, loss_coeff, calculate_per_token_loss, loss_divisor
    )


def _dense_kl_loss_impl(
    attn_score: Tensor,
    attn_l1norm: Tensor,
    index_score: Tensor,
    index_lse: Tensor,
    loss_coeff: float,
    calculate_per_token_loss: bool,
) -> Tensor:
    """Reduce dense teacher KL without keeping the rowwise pointwise intermediates."""
    eps = _CLIP_PROB_MIN
    # row_valid: rows with at least one un-masked KV position.
    row_valid = (attn_l1norm > eps) & torch.isfinite(index_lse)

    # Safe denoms: replace invalid rows with a finite value so target /
    # log-predict don't produce NaN; the row mask zeroes their KL below.
    safe_l1 = attn_l1norm.clamp(min=eps)
    safe_lse = torch.where(row_valid, index_lse, torch.zeros_like(index_lse))

    target = attn_score / safe_l1.unsqueeze(-1)
    target_clamped = target.clamp(min=eps)
    # Per-position validity: the indexer-score kernel emits -inf at
    # ratio-masked positions; those contribute 0 to KL by the
    # ``0 · log(0/p) = 0`` convention. Without this gate, the eps-clamp
    # on target makes the term ``eps · (log eps - (-inf)) = +inf``.
    position_valid = torch.isfinite(index_score)
    safe_index_score = torch.where(position_valid, index_score, torch.zeros_like(index_score))
    log_predict = safe_index_score - safe_lse.unsqueeze(-1)

    kl_terms = target_clamped * (torch.log(target_clamped) - log_predict)
    kl_terms = torch.where(position_valid, kl_terms, torch.zeros_like(kl_terms))
    kl_per_row = kl_terms.sum(dim=-1)  # (B, S_q)
    kl_per_row = torch.where(row_valid, kl_per_row, torch.zeros_like(kl_per_row))
    loss = kl_per_row.sum() if calculate_per_token_loss else kl_per_row.mean()
    return loss_coeff * loss


_compiled_dense_kl_loss = torch.compile(_dense_kl_loss_impl, fullgraph=True)


def dense_kl_loss(
    attn_score: Tensor,
    attn_l1norm: Tensor,
    index_score: Tensor,
    index_lse: Tensor,
    loss_coeff: float,
    calculate_per_token_loss: bool = False,
) -> Tensor:
    """Compute packed dense KL in a fused CUDA graph or the CPU reference path."""
    implementation = _compiled_dense_kl_loss if attn_score.is_cuda else _dense_kl_loss_impl
    return implementation(
        attn_score, attn_l1norm, index_score, index_lse, loss_coeff, calculate_per_token_loss
    )


__all__ = ["sparse_kl_loss", "dense_kl_loss"]
