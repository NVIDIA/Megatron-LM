# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compiled post-processing kernels for CSA indexer losses."""

from __future__ import annotations

import torch
from torch import Tensor

_CLIP_PROB_MIN = torch.finfo(torch.float32).tiny
_FLOAT32_MIN = torch.finfo(torch.float32).min

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _prepare_sparse_loss_kernel(
        indexer_scores,
        topk_indices,
        physical_indices,
        padding_row_mask,
        predict,
        sanitized_topk_indices,
        sanitized_physical_indices,
        stride_scores_row,
        stride_scores_col,
        stride_topk_row,
        stride_topk_col,
        stride_physical_row,
        stride_physical_col,
        stride_predict_row,
        stride_predict_col,
        stride_sanitized_topk_row,
        stride_sanitized_topk_col,
        stride_sanitized_physical_row,
        stride_sanitized_physical_col,
        SCORE_WIDTH: tl.constexpr,
        TOPK_WIDTH: tl.constexpr,
        HAS_PADDING: tl.constexpr,
        HAS_PHYSICAL_INDICES: tl.constexpr,
        BLOCK_TOPK: tl.constexpr,
    ):
        """Gather selected scores and form one masked softmax row."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_TOPK)
        col_mask = cols < TOPK_WIDTH
        indices = tl.load(
            topk_indices + row * stride_topk_row + cols * stride_topk_col, mask=col_mask, other=-1
        )

        row_is_padding = False
        if HAS_PADDING:
            row_is_padding = tl.load(padding_row_mask + row).to(tl.int1)
            tl.store(
                sanitized_topk_indices
                + row * stride_sanitized_topk_row
                + cols * stride_sanitized_topk_col,
                tl.where(row_is_padding, -1, indices),
                mask=col_mask,
            )
            if HAS_PHYSICAL_INDICES:
                physical = tl.load(
                    physical_indices + row * stride_physical_row + cols * stride_physical_col,
                    mask=col_mask,
                    other=-1,
                )
                tl.store(
                    sanitized_physical_indices
                    + row * stride_sanitized_physical_row
                    + cols * stride_sanitized_physical_col,
                    tl.where(row_is_padding, -1, physical),
                    mask=col_mask,
                )

        selected = col_mask & (indices >= 0) & (indices < SCORE_WIDTH)
        if HAS_PADDING:
            selected = selected & ~row_is_padding
        safe_indices = tl.where(selected, indices, 0)
        gathered_scores = tl.load(
            indexer_scores + row * stride_scores_row + safe_indices * stride_scores_col,
            mask=selected,
            other=0.0,
        ).to(tl.float32)

        # Match the eager path exactly for logical invalid slots: using the
        # finite fp32 minimum makes an all-invalid row softmax uniform instead
        # of NaN. Lanes outside TOPK_WIDTH remain -inf so they do not enter the
        # reduction when BLOCK_TOPK is rounded up.
        gathered_scores = tl.where(selected, gathered_scores, -3.4028234663852886e38)
        gathered_scores = tl.where(col_mask, gathered_scores, -float("inf"))
        row_max = tl.max(gathered_scores, axis=0)
        numerator = tl.exp(gathered_scores - row_max)
        denominator = tl.sum(numerator, axis=0)
        tl.store(
            predict + row * stride_predict_row + cols * stride_predict_col,
            numerator / denominator,
            mask=col_mask,
        )


def _prepare_sparse_loss_fallback(
    indexer_scores: Tensor,
    topk_indices: Tensor,
    padding_row_mask: Tensor | None = None,
    physical_indices: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Eager reference for sparse-loss score selection and padding masks."""
    sanitized_topk_indices = topk_indices
    sanitized_physical_indices = physical_indices
    if padding_row_mask is not None:
        row_mask = padding_row_mask.unsqueeze(-1)
        sanitized_topk_indices = topk_indices.masked_fill(row_mask, -1)
        if physical_indices is not None:
            sanitized_physical_indices = physical_indices.masked_fill(row_mask, -1)

    safe_indices = sanitized_topk_indices.clamp(min=0).long()
    gathered_scores = torch.gather(indexer_scores, dim=-1, index=safe_indices)
    gathered_scores = torch.where(sanitized_topk_indices >= 0, gathered_scores, _FLOAT32_MIN)
    predict = torch.softmax(gathered_scores, dim=-1)
    return predict, sanitized_topk_indices, sanitized_physical_indices


def prepare_sparse_loss(
    indexer_scores: Tensor,
    topk_indices: Tensor,
    padding_row_mask: Tensor | None = None,
    physical_indices: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Fuse sparse-loss score gather, padding invalidation, and softmax.

    The tensors may use either packed ``(total_q, width)`` or batched
    ``(..., width)`` layouts. All leading dimensions are flattened into rows
    for the CUDA kernel and restored without a device copy. When a padding
    mask is supplied, sanitized index tensors are returned for the teacher and
    indexer-backward consumers; otherwise the original index tensors are
    returned without copying.

    Args:
        indexer_scores: Full indexer scores with shape ``(..., score_width)``.
        topk_indices: Selected score indices with shape ``(..., topk_width)``.
        padding_row_mask: Optional mask over the leading dimensions.
        physical_indices: Optional physical selected indices matching
            ``topk_indices``.

    Returns:
        The fp32 selected-score softmax, loss-sanitized logical indices, and
        optional loss-sanitized physical indices.
    """
    if indexer_scores.ndim < 2 or topk_indices.ndim != indexer_scores.ndim:
        raise ValueError("indexer_scores and topk_indices must have matching rank >= 2")
    if indexer_scores.shape[:-1] != topk_indices.shape[:-1]:
        raise ValueError("indexer_scores and topk_indices must have equal leading dimensions")
    if topk_indices.numel() == 0 or indexer_scores.shape[-1] == 0:
        return _prepare_sparse_loss_fallback(
            indexer_scores, topk_indices, padding_row_mask, physical_indices
        )
    if padding_row_mask is not None and padding_row_mask.shape != topk_indices.shape[:-1]:
        raise ValueError("padding_row_mask must match the index tensors' leading dimensions")
    if physical_indices is not None and physical_indices.shape != topk_indices.shape:
        raise ValueError("physical_indices must have the same shape as topk_indices")

    topk_width = topk_indices.shape[-1]
    use_triton = (
        _TRITON_AVAILABLE
        and indexer_scores.is_cuda
        and topk_indices.is_cuda
        and (padding_row_mask is None or padding_row_mask.is_cuda)
        and (physical_indices is None or physical_indices.is_cuda)
        and triton.next_power_of_2(topk_width) <= 65536
    )
    if not use_triton:
        return _prepare_sparse_loss_fallback(
            indexer_scores, topk_indices, padding_row_mask, physical_indices
        )

    tensors = [topk_indices]
    if padding_row_mask is not None:
        tensors.append(padding_row_mask)
    if physical_indices is not None:
        tensors.append(physical_indices)
    if any(tensor.device != indexer_scores.device for tensor in tensors):
        raise ValueError("sparse-loss preparation tensors must share a CUDA device")

    row_count = topk_indices.numel() // topk_width
    score_width = indexer_scores.shape[-1]
    scores_flat = indexer_scores.reshape(row_count, score_width)
    topk_flat = topk_indices.reshape(row_count, topk_width)
    predict = torch.empty(topk_indices.shape, dtype=torch.float32, device=topk_indices.device)
    predict_flat = predict.reshape(row_count, topk_width)

    if padding_row_mask is None:
        padding_flat = topk_flat
        sanitized_topk_indices = topk_indices
        sanitized_topk_flat = topk_flat
    else:
        padding_flat = padding_row_mask.reshape(row_count)
        sanitized_topk_indices = torch.empty_like(topk_indices)
        sanitized_topk_flat = sanitized_topk_indices.reshape(row_count, topk_width)

    if physical_indices is None:
        physical_flat = topk_flat
        sanitized_physical_indices = None
        sanitized_physical_flat = topk_flat
    else:
        physical_flat = physical_indices.reshape(row_count, topk_width)
        if padding_row_mask is None:
            sanitized_physical_indices = physical_indices
            sanitized_physical_flat = physical_flat
        else:
            sanitized_physical_indices = torch.empty_like(physical_indices)
            sanitized_physical_flat = sanitized_physical_indices.reshape(row_count, topk_width)

    block_topk = max(16, triton.next_power_of_2(topk_width))
    num_warps = 8 if block_topk >= 1024 else 4
    with torch.cuda.device(indexer_scores.device):
        _prepare_sparse_loss_kernel[(row_count,)](
            scores_flat,
            topk_flat,
            physical_flat,
            padding_flat,
            predict_flat,
            sanitized_topk_flat,
            sanitized_physical_flat,
            scores_flat.stride(0),
            scores_flat.stride(1),
            topk_flat.stride(0),
            topk_flat.stride(1),
            physical_flat.stride(0),
            physical_flat.stride(1),
            predict_flat.stride(0),
            predict_flat.stride(1),
            sanitized_topk_flat.stride(0),
            sanitized_topk_flat.stride(1),
            sanitized_physical_flat.stride(0),
            sanitized_physical_flat.stride(1),
            SCORE_WIDTH=score_width,
            TOPK_WIDTH=topk_width,
            HAS_PADDING=padding_row_mask is not None,
            HAS_PHYSICAL_INDICES=physical_indices is not None,
            BLOCK_TOPK=block_topk,
            num_warps=num_warps,
        )
    return predict, sanitized_topk_indices, sanitized_physical_indices


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


__all__ = ["prepare_sparse_loss", "sparse_kl_loss"]
