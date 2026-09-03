# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compiled post-processing kernels for CSA indexer losses."""

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


__all__ = ["sparse_kl_loss"]
