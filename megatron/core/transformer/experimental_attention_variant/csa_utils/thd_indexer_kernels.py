# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fused Triton glue kernels for the packed-THD CSA indexer."""

from __future__ import annotations

import torch
from torch import Tensor

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
    def _dsv4_thd_build_seq_lens_kernel(
        cu_seqlens_q,
        cu_seqlens_kv,
        q_causal_offsets,
        seq_lens,
        total_q,
        NUM_SEQUENCES: tl.constexpr,
        RATIO: tl.constexpr,
        HAS_CAUSAL_OFFSETS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        """Map packed query rows to their causal compressed-K lengths."""
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        launch_mask = rows < total_q
        owner_found = tl.zeros((BLOCK_ROWS,), dtype=tl.int1)
        owner_q_start = tl.zeros((BLOCK_ROWS,), dtype=tl.int32)
        owner_kv_len = tl.zeros((BLOCK_ROWS,), dtype=tl.int32)
        owner_causal_offset = tl.zeros((BLOCK_ROWS,), dtype=tl.int32)

        for seq in range(NUM_SEQUENCES):
            q_start = tl.load(cu_seqlens_q + seq)
            q_end = tl.load(cu_seqlens_q + seq + 1)
            owns_row = launch_mask & (rows >= q_start) & (rows < q_end)
            owner_found = owner_found | owns_row
            owner_q_start = tl.where(owns_row, q_start, owner_q_start)
            kv_len = tl.load(cu_seqlens_kv + seq + 1) - tl.load(cu_seqlens_kv + seq)
            owner_kv_len = tl.where(owns_row, kv_len, owner_kv_len)
            if HAS_CAUSAL_OFFSETS:
                causal_offset = tl.load(q_causal_offsets + seq)
                owner_causal_offset = tl.where(owns_row, causal_offset, owner_causal_offset)

        position = rows - owner_q_start + owner_causal_offset
        visible = (position + 1) // RATIO
        visible = tl.minimum(visible, owner_kv_len)
        visible = tl.where(owner_found, visible, 0)
        tl.store(seq_lens + rows, visible, mask=launch_mask)

    @triton.jit
    def _dsv4_thd_sanitize_topk_kernel(
        candidate_indices,
        scores,
        seq_lens,
        sanitized_indices,
        topk_length,
        stride_candidate_row,
        stride_candidate_col,
        stride_scores_row,
        stride_scores_col,
        stride_sanitized_row,
        stride_sanitized_col,
        CANDIDATE_WIDTH: tl.constexpr,
        OUTPUT_WIDTH: tl.constexpr,
        SCORE_WIDTH: tl.constexpr,
        BLOCK_TOPK: tl.constexpr,
    ):
        """Validate one row of top-k ids and reduce its valid length."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_TOPK)
        output_mask = cols < OUTPUT_WIDTH
        candidate_mask = cols < CANDIDATE_WIDTH
        candidates = tl.load(
            candidate_indices + row * stride_candidate_row + cols * stride_candidate_col,
            mask=candidate_mask,
            other=-1,
        )
        seq_len = tl.load(seq_lens + row)
        index_valid = (
            candidate_mask & (candidates >= 0) & (candidates < seq_len) & (candidates < SCORE_WIDTH)
        )
        safe_candidates = tl.where(index_valid, candidates, 0)
        selected_scores = tl.load(
            scores + row * stride_scores_row + safe_candidates * stride_scores_col,
            mask=index_valid,
            other=-float("inf"),
        ).to(tl.float32)
        score_valid = (selected_scores > -float("inf")) & (selected_scores < float("inf"))
        valid = index_valid & score_valid
        tl.store(
            sanitized_indices + row * stride_sanitized_row + cols * stride_sanitized_col,
            tl.where(valid, candidates, -1),
            mask=output_mask,
        )
        tl.store(topk_length + row, tl.sum(valid.to(tl.int32), axis=0))


def _build_seq_lens_fallback(
    cu_seqlens_q: Tensor,
    cu_seqlens_kv: Tensor,
    total_q: int,
    ratio: int,
    q_causal_offsets: Tensor | None,
) -> Tensor:
    """Eager reference for environments where the Triton kernel is unavailable."""
    row_idx = torch.arange(total_q, device=cu_seqlens_q.device, dtype=torch.int32)
    row_batch_ids = torch.bucketize(row_idx, cu_seqlens_q[1:], right=True).clamp(
        max=cu_seqlens_q.shape[0] - 2
    )
    row_valid = row_idx < cu_seqlens_q[-1]
    pos_in_seq = row_idx - cu_seqlens_q[row_batch_ids]
    if q_causal_offsets is not None:
        pos_in_seq = pos_in_seq + q_causal_offsets[row_batch_ids]
    pos_in_seq = torch.where(row_valid, pos_in_seq, torch.zeros_like(pos_in_seq))
    seqlen_kv_per_row = (cu_seqlens_kv[1:] - cu_seqlens_kv[:-1])[row_batch_ids]
    seq_lens = ((pos_in_seq + 1) // ratio).clamp(max=seqlen_kv_per_row).to(torch.int32).contiguous()
    return torch.where(row_valid, seq_lens, torch.zeros_like(seq_lens))


def build_seq_lens(
    cu_seqlens_q: Tensor,
    cu_seqlens_kv: Tensor,
    total_q: int,
    ratio: int,
    q_causal_offsets: Tensor | None = None,
) -> Tensor:
    """Build per-row valid compressed-K lengths for a packed THD indexer.

    Rows outside the last packed sequence, including CUDA-graph tail rows, receive
    length zero. ``q_causal_offsets`` shifts the position within each sequence.
    """
    total_q, ratio = int(total_q), int(ratio)
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    if cu_seqlens_q.ndim != 1 or cu_seqlens_kv.ndim != 1:
        raise ValueError("cu_seqlens_q and cu_seqlens_kv must be one-dimensional")
    if cu_seqlens_q.shape != cu_seqlens_kv.shape or cu_seqlens_q.shape[0] < 2:
        raise ValueError("cu_seqlens_q and cu_seqlens_kv must describe the same sequences")
    if q_causal_offsets is not None and q_causal_offsets.shape != (cu_seqlens_q.shape[0] - 1,):
        raise ValueError("q_causal_offsets must contain one value per packed sequence")

    if not _TRITON_AVAILABLE or not cu_seqlens_q.is_cuda:
        return _build_seq_lens_fallback(
            cu_seqlens_q, cu_seqlens_kv, total_q, ratio, q_causal_offsets
        )
    if not cu_seqlens_kv.is_cuda or (q_causal_offsets is not None and not q_causal_offsets.is_cuda):
        raise ValueError("packed indexer metadata must reside on the same CUDA device")
    if cu_seqlens_kv.device != cu_seqlens_q.device or (
        q_causal_offsets is not None and q_causal_offsets.device != cu_seqlens_q.device
    ):
        raise ValueError("packed indexer metadata must reside on the same CUDA device")
    if not cu_seqlens_q.is_contiguous() or not cu_seqlens_kv.is_contiguous():
        raise ValueError("packed indexer cumulative lengths must be contiguous")
    if q_causal_offsets is not None and not q_causal_offsets.is_contiguous():
        raise ValueError("q_causal_offsets must be contiguous")

    seq_lens = torch.empty((total_q,), dtype=torch.int32, device=cu_seqlens_q.device)
    if total_q == 0:
        return seq_lens
    offsets = q_causal_offsets if q_causal_offsets is not None else cu_seqlens_q
    block_rows = 256
    with torch.cuda.device(cu_seqlens_q.device):
        _dsv4_thd_build_seq_lens_kernel[(triton.cdiv(total_q, block_rows),)](
            cu_seqlens_q,
            cu_seqlens_kv,
            offsets,
            seq_lens,
            total_q,
            NUM_SEQUENCES=cu_seqlens_q.shape[0] - 1,
            RATIO=ratio,
            HAS_CAUSAL_OFFSETS=q_causal_offsets is not None,
            BLOCK_ROWS=block_rows,
            num_warps=4,
        )
    return seq_lens


def _sanitize_topk_fallback(
    candidate_indices: Tensor, scores: Tensor, seq_lens: Tensor, output_width: int | None = None
) -> tuple[Tensor, Tensor]:
    """Eager reference for environments where the Triton kernel is unavailable."""
    if output_width is not None and output_width > candidate_indices.shape[1]:
        padding = torch.full(
            (candidate_indices.shape[0], output_width - candidate_indices.shape[1]),
            -1,
            dtype=candidate_indices.dtype,
            device=candidate_indices.device,
        )
        candidate_indices = torch.cat((candidate_indices, padding), dim=-1)
    score_width = scores.shape[1]
    row_valid = (candidate_indices >= 0) & (candidate_indices < seq_lens.unsqueeze(1))
    sanitized_indices = candidate_indices.masked_fill(~row_valid, -1)
    safe_topk = sanitized_indices.clamp(min=0, max=score_width - 1).to(torch.long)
    selected_scores = torch.gather(scores, dim=-1, index=safe_topk)
    selected_valid = (
        (sanitized_indices >= 0)
        & (sanitized_indices < score_width)
        & torch.isfinite(selected_scores)
    )
    sanitized_indices = sanitized_indices.masked_fill(~selected_valid, -1)
    return sanitized_indices, (sanitized_indices >= 0).sum(dim=-1).int()


def sanitize_topk(
    candidate_indices: Tensor, scores: Tensor, seq_lens: Tensor, output_width: int | None = None
) -> tuple[Tensor, Tensor]:
    """Validate packed-THD top-k ids and compute valid counts in one launch.

    An id is retained only when it is inside both the row's causal length and
    the score width and its selected score is finite. Invalid ids become ``-1``;
    ``output_width`` also pads the selected rows without a separate launch.
    """
    if candidate_indices.ndim != 2 or scores.ndim != 2 or seq_lens.ndim != 1:
        raise ValueError("candidate_indices, scores, and seq_lens must be 2-D, 2-D, and 1-D")
    if candidate_indices.shape[0] != scores.shape[0] or seq_lens.shape[0] != scores.shape[0]:
        raise ValueError("top-k candidates, scores, and sequence lengths must have equal rows")
    if scores.shape[1] == 0:
        raise ValueError("scores must contain at least one column")

    candidate_width = candidate_indices.shape[1]
    output_width = candidate_width if output_width is None else int(output_width)
    if output_width < candidate_width:
        raise ValueError("output_width cannot be smaller than the candidate width")
    if (
        not _TRITON_AVAILABLE
        or not candidate_indices.is_cuda
        or output_width == 0
        or triton.next_power_of_2(output_width) > 65536
    ):
        return _sanitize_topk_fallback(candidate_indices, scores, seq_lens, output_width)
    if not scores.is_cuda or not seq_lens.is_cuda:
        raise ValueError("top-k candidates, scores, and sequence lengths must share a CUDA device")
    if candidate_indices.device != scores.device or seq_lens.device != scores.device:
        raise ValueError("top-k candidates, scores, and sequence lengths must share a CUDA device")

    sanitized_indices = torch.empty(
        (candidate_indices.shape[0], output_width),
        dtype=candidate_indices.dtype,
        device=candidate_indices.device,
    )
    topk_length = torch.empty(
        (candidate_indices.shape[0],), dtype=torch.int32, device=scores.device
    )
    if candidate_indices.shape[0] == 0:
        return sanitized_indices, topk_length
    block_topk = max(16, triton.next_power_of_2(output_width))
    num_warps = 8 if block_topk >= 1024 else 4
    with torch.cuda.device(candidate_indices.device):
        _dsv4_thd_sanitize_topk_kernel[(candidate_indices.shape[0],)](
            candidate_indices,
            scores,
            seq_lens,
            sanitized_indices,
            topk_length,
            candidate_indices.stride(0),
            candidate_indices.stride(1),
            scores.stride(0),
            scores.stride(1),
            sanitized_indices.stride(0),
            sanitized_indices.stride(1),
            CANDIDATE_WIDTH=candidate_width,
            OUTPUT_WIDTH=output_width,
            SCORE_WIDTH=scores.shape[1],
            BLOCK_TOPK=block_topk,
            num_warps=num_warps,
        )
    return sanitized_indices, topk_length
