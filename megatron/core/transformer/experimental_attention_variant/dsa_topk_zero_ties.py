# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Lowest-index exact-zero ties for the deterministic DSA indexer top-k.

cuDNN's radix top-k ranks scores by their IEEE-754 bit pattern, so ``+0.0`` ranks above
``-0.0``. The deterministic top-k contract compares values, under which both zeros are the same
cutoff tie. DSA indexer scores ``sum_h w_h * relu(q . k)`` with negative head weights can
produce both signs of zero, so a row whose top-k cutoff value is zero may need a different set
of zeros than the radix order picks.

:func:`select_lowest_index_zero_ties` returns, for every row with a positive quota ``q``, the
``q`` lowest column indices in ``[0, row_limit)`` whose score compares equal to zero, ascending.
All other slots hold ``INT32_MAX``. Rows with a zero quota cost nothing on CUDA: the Triton
kernel returns before reading them, and no step synchronizes with the host, so the selection
can be captured in a CUDA graph. Other devices (and CUDA without Triton) use a fixed-shape
PyTorch implementation with the same result.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_TRITON = False

ZERO_TIE_SENTINEL = torch.iinfo(torch.int32).max
_TRITON_BLOCK = 1024
_TORCH_MAX_BYTES = 512 * 1024 * 1024

if HAVE_TRITON:

    @triton.jit
    def _lowest_index_zero_ties_kernel(
        scores_ptr,
        score_row_stride,
        row_limits_ptr,
        quotas_ptr,
        out_ptr,
        out_row_stride,
        BLOCK: tl.constexpr,
    ):
        """One program per row: scan ``[0, limit)`` in order until ``quota`` zeros are found."""
        row = tl.program_id(0).to(tl.int64)
        quota = tl.load(quotas_ptr + row)
        if quota > 0:
            limit = tl.load(row_limits_ptr + row)
            row_scores = scores_ptr + row * score_row_stride
            row_out = out_ptr + row * out_row_stride
            found = tl.full((), 0, tl.int32)
            start = tl.full((), 0, tl.int32)
            while (found < quota) & (start < limit):
                columns = start + tl.arange(0, BLOCK)
                in_row = columns < limit
                values = tl.load(row_scores + columns, mask=in_row, other=1.0)
                is_zero = (values == 0.0) & in_row
                zero_count = is_zero.to(tl.int32)
                slots = found + tl.cumsum(zero_count, axis=0) - 1
                tl.store(row_out + slots, columns, mask=is_zero & (slots < quota))
                found += tl.sum(zero_count, axis=0)
                start += BLOCK


def _check_inputs(scores: Tensor, row_limits: Tensor, quotas: Tensor, topk_k: int) -> None:
    if scores.ndim != 2:
        raise RuntimeError(f"zero-tie selection expects 2-D scores, got {tuple(scores.shape)}")
    n_rows = scores.size(0)
    if row_limits.shape != (n_rows,) or quotas.shape != (n_rows,):
        raise RuntimeError(f"zero-tie row limits and quotas must have shape ({n_rows},)")
    if row_limits.dtype != torch.int32 or quotas.dtype != torch.int32:
        raise RuntimeError("zero-tie row limits and quotas must be int32")
    if topk_k <= 0:
        raise RuntimeError(f"zero-tie selection needs a positive top-k, got {topk_k}")


def _select_lowest_index_zero_ties_triton(
    scores: Tensor, row_limits: Tensor, quotas: Tensor, topk_k: int
) -> Tensor:
    """Triton implementation; reads only the rows whose quota is positive."""
    n_rows = scores.size(0)
    out = torch.full((n_rows, topk_k), ZERO_TIE_SENTINEL, dtype=torch.int32, device=scores.device)
    if n_rows == 0 or scores.size(1) == 0:
        return out
    if scores.stride(1) != 1:
        scores = scores.contiguous()
    _lowest_index_zero_ties_kernel[(n_rows,)](
        scores,
        scores.stride(0),
        row_limits.contiguous(),
        quotas.contiguous(),
        out,
        out.stride(0),
        BLOCK=_TRITON_BLOCK,
        num_warps=4,
    )
    return out


def _select_lowest_index_zero_ties_torch(
    scores: Tensor, row_limits: Tensor, quotas: Tensor, topk_k: int
) -> Tensor:
    """Fixed-shape PyTorch implementation (row chunks bound the full-row temporaries)."""
    n_rows, sk = scores.shape
    out = torch.full((n_rows, topk_k), ZERO_TIE_SENTINEL, dtype=torch.int32, device=scores.device)
    if n_rows == 0 or sk == 0:
        return out
    columns = torch.arange(sk, device=scores.device, dtype=torch.int32).view(1, sk)
    targets = torch.arange(1, topk_k + 1, device=scores.device, dtype=torch.int32).view(1, topk_k)
    # Per row: the zero mask, its int32 prefix count, and comparison temporaries.
    chunk_rows = max(1, _TORCH_MAX_BYTES // (16 * sk + 16 * topk_k))
    for row_start in range(0, n_rows, chunk_rows):
        row_end = min(row_start + chunk_rows, n_rows)
        limits = row_limits[row_start:row_end].view(-1, 1)
        chunk_quotas = quotas[row_start:row_end].view(-1, 1)
        is_zero = (scores[row_start:row_end] == 0) & (columns < limits) & (chunk_quotas > 0)
        zero_rank = torch.cumsum(is_zero, dim=1, dtype=torch.int32)
        slots = torch.searchsorted(
            zero_rank, targets.expand(row_end - row_start, topk_k).contiguous(), out_int32=True
        )
        keep = (targets <= chunk_quotas) & (slots < sk)
        out[row_start:row_end] = torch.where(keep, slots, torch.full_like(slots, ZERO_TIE_SENTINEL))
    return out


def select_lowest_index_zero_ties(
    scores: Tensor, row_limits: Tensor, quotas: Tensor, topk_k: int
) -> Tensor:
    """Return the ``quotas[r]`` lowest in-row column indices whose score equals zero.

    Args:
        scores: ``[rows, columns]`` scores. Only ``[0, row_limits[r])`` of row ``r`` is read.
        row_limits: ``[rows]`` int32 valid lengths, already clamped to ``[0, columns]``.
        quotas: ``[rows]`` int32 number of zeros to return per row, at most ``topk_k``; rows
            with quota 0 are not read.
        topk_k: Width of the result.

    Returns:
        ``[rows, topk_k]`` int32 column indices in ascending order, ``INT32_MAX`` in unused
        slots. ``-0.0`` and ``+0.0`` both count as zero; NaN does not.
    """
    _check_inputs(scores, row_limits, quotas, topk_k)
    if scores.is_cuda and HAVE_TRITON:
        return _select_lowest_index_zero_ties_triton(scores, row_limits, quotas, topk_k)
    return _select_lowest_index_zero_ties_torch(scores, row_limits, quotas, topk_k)
