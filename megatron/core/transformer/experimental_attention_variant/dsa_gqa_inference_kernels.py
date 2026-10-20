# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Triton kernels for DSA-over-GQA inference: indexer scores and sparse attention.

Every query token of a step, decode or prefill, is a row: row ``t`` belongs to active request
``row_requests[t]`` and attends among that request's cached keys ``[0, row_lengths[t])`` (its
causal prefix, itself included), routed to the indexer's top-k of them. The same kernels thus
serve decode-only, prefill, and mixed steps, with launch sizes fixed by the padded batch, so
every step type can run as a CUDA graph.

Indexer scores: programs take blocks of consecutive rows of one request (``build_row_metadata``)
and stride over the request's paged keys up to the block's longest row, loading each key tile
once for the whole block and scoring it on tensor cores.

Sparse attention: one program handles every query head of one KV group for one slice of a row's
selected slots, so each selected K/V row is read once per group rather than once per head, and
QK and PV run on tensor cores. A second kernel merges the slices with their running maxima and
sums (flash decoding).
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAVE_TRITON = False

LOG2E = 1.4426950408889634
# Attention: selected slots per inner tile, and the smallest query-head tile tensor cores accept.
BLOCK_N = 128
MIN_HEAD_TILE = 16
# Attention programs to aim for per SM across a decode step.
ATTENTION_PROGRAMS_PER_SM = 1
# Indexer scores: rows per block (tensor cores need at least 16), keys per tile, and score
# programs to aim for per SM across a step.
SCORE_ROW_BLOCK = 32
SCORE_BLOCK_T = 128
SCORE_PROGRAMS_PER_SM = 8


if HAVE_TRITON:

    # Integers that change from step to step are not specialized (Triton otherwise compiles
    # separate variants for values equal to 1 or divisible by 16); only model constants are
    # constexpr, so a kernel compiles once per model.
    @triton.jit(do_not_specialize=["stride_scores", "width"])
    def _score_kernel(
        query,
        key_cache,
        block_table,
        row_lengths,
        block_starts,
        block_rows,
        block_requests,
        scores,
        stride_q,
        stride_k_page,
        stride_k_pos,
        stride_table,
        stride_scores,
        width,
        max_pages,
        num_pages,
        INDEX_DIM: tl.constexpr,
        INDEX_DIM_TILE: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
        BLOCK_T: tl.constexpr,
        FILL_INVALID: tl.constexpr,
    ):
        worker = tl.program_id(0)
        block = tl.program_id(1)
        workers = tl.num_programs(0)
        num_rows = tl.load(block_rows + block)
        request = tl.load(block_requests + block).to(tl.int64)
        row_in_block = tl.arange(0, ROW_BLOCK)
        row_valid = row_in_block < num_rows
        rows = (tl.load(block_starts + block) + row_in_block).to(tl.int64)
        lengths = tl.load(row_lengths + rows, mask=row_valid, other=0)
        lengths = tl.minimum(lengths, width)
        dims = tl.arange(0, INDEX_DIM_TILE)
        dim_valid = dims < INDEX_DIM
        q = tl.load(
            query + rows[:, None] * stride_q + dims[None, :],
            mask=row_valid[:, None] & dim_valid[None, :],
            other=0.0,
        )
        longest = tl.max(lengths, axis=0)
        # Positions below `limit` are written: whole rows when invalid positions are filled
        # with -inf, else only each row's keys. Empty (padding) blocks write nothing.
        limit = tl.where(num_rows > 0, width, 0) if FILL_INVALID else longest
        for start in range(worker * BLOCK_T, limit, workers * BLOCK_T):
            positions = start + tl.arange(0, BLOCK_T)
            logical_page = positions // PAGE_SIZE
            key_valid = (positions < longest) & (logical_page < max_pages)
            page = tl.load(
                block_table + request * stride_table + logical_page, mask=key_valid, other=-1
            )
            key_valid = key_valid & (page >= 0) & (page < num_pages)
            page = tl.where(key_valid, page, 0).to(tl.int64)
            k = tl.load(
                key_cache
                + page[:, None] * stride_k_page
                + (positions % PAGE_SIZE)[:, None] * stride_k_pos
                + dims[None, :],
                mask=key_valid[:, None] & dim_valid[None, :],
                other=0.0,
            )
            score = tl.dot(q, tl.trans(k))
            in_row = positions[None, :] < lengths[:, None]
            score = tl.where(in_row & key_valid[None, :], score, float("-inf"))
            store = row_valid[:, None] & (positions[None, :] < limit)
            if not FILL_INVALID:
                store = store & in_row
            tl.store(scores + rows[:, None] * stride_scores + positions[None, :], score, mask=store)

    @triton.jit(do_not_specialize=["stride_topk", "topk", "slice_len", "num_slices"])
    def _partial_kernel(
        query,
        key_cache,
        value_cache,
        block_table,
        row_requests,
        topk_indices,
        row_lengths,
        partial_out,
        partial_max,
        partial_sum,
        stride_q_req,
        stride_q_head,
        stride_k_page,
        stride_k_pos,
        stride_k_group,
        stride_v_page,
        stride_v_pos,
        stride_v_group,
        stride_table,
        stride_topk,
        num_pages,
        max_pages,
        topk,
        slice_len,
        num_slices,
        scale_log2,
        HEADS_PER_GROUP: tl.constexpr,
        HEAD_TILE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HEAD_DIM_TILE: tl.constexpr,
        VALUE_DIM: tl.constexpr,
        VALUE_DIM_TILE: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        group = tl.program_id(1)
        piece = tl.program_id(2)
        num_groups = tl.num_programs(1)
        # Padding rows have no request and length 0.
        request = tl.maximum(tl.load(row_requests + row), 0).to(tl.int64)

        head_in_group = tl.arange(0, HEAD_TILE)
        head_valid = head_in_group < HEADS_PER_GROUP
        heads = group * HEADS_PER_GROUP + head_in_group
        dims = tl.arange(0, HEAD_DIM_TILE)
        dim_valid = dims < HEAD_DIM
        value_dims = tl.arange(0, VALUE_DIM_TILE)
        value_dim_valid = value_dims < VALUE_DIM
        q = tl.load(
            query
            + row.to(tl.int64) * stride_q_req
            + heads[:, None] * stride_q_head
            + dims[None, :],
            mask=head_valid[:, None] & dim_valid[None, :],
            other=0.0,
        )
        length = tl.load(row_lengths + row)

        running_max = tl.full((HEAD_TILE,), float("-inf"), tl.float32)
        running_sum = tl.zeros((HEAD_TILE,), tl.float32)
        acc = tl.zeros((HEAD_TILE, VALUE_DIM_TILE), tl.float32)
        first_slot = piece * slice_len
        for start in range(0, slice_len, BLOCK):
            slots = first_slot + start + tl.arange(0, BLOCK)
            # The last slice may be short when the slice count does not divide top-k.
            in_slice = (start + tl.arange(0, BLOCK) < slice_len) & (slots < topk)
            position = tl.load(
                topk_indices + row.to(tl.int64) * stride_topk + slots, mask=in_slice, other=-1
            )
            valid = (position >= 0) & (position < length)
            safe_position = tl.where(valid, position, 0)
            logical_page = safe_position // PAGE_SIZE
            valid = valid & (logical_page < max_pages)
            page = tl.load(
                block_table + request * stride_table + logical_page, mask=valid, other=-1
            )
            valid = valid & (page >= 0) & (page < num_pages)
            page = tl.where(valid, page, 0).to(tl.int64)
            offset = safe_position % PAGE_SIZE
            k = tl.load(
                key_cache
                + page[:, None] * stride_k_page
                + offset[:, None] * stride_k_pos
                + group * stride_k_group
                + dims[None, :],
                mask=valid[:, None] & dim_valid[None, :],
                other=0.0,
            )
            scores = tl.dot(q, tl.trans(k)) * scale_log2
            scores = tl.where(valid[None, :], scores, float("-inf"))
            new_max = tl.maximum(running_max, tl.max(scores, axis=1))
            # Rows with no valid key yet keep a finite reference so exp2 never sees inf - inf.
            reference = tl.where(new_max == float("-inf"), 0.0, new_max)
            p = tl.exp2(scores - reference[:, None])
            alpha = tl.exp2(running_max - reference)
            running_sum = running_sum * alpha + tl.sum(p, axis=1)
            v = tl.load(
                value_cache
                + page[:, None] * stride_v_page
                + offset[:, None] * stride_v_pos
                + group * stride_v_group
                + value_dims[None, :],
                mask=valid[:, None] & value_dim_valid[None, :],
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            running_max = new_max

        # Partial rows are [row, slice, head], so a group's heads are contiguous.
        rows = ((row * num_slices + piece) * num_groups + group) * HEADS_PER_GROUP
        rows = rows.to(tl.int64) + head_in_group
        tl.store(
            partial_out + rows[:, None] * VALUE_DIM + value_dims[None, :],
            acc,
            mask=head_valid[:, None] & value_dim_valid[None, :],
        )
        tl.store(partial_max + rows, running_max, mask=head_valid)
        tl.store(partial_sum + rows, running_sum, mask=head_valid)

    @triton.jit(do_not_specialize=["num_slices"])
    def _combine_kernel(
        partial_out,
        partial_max,
        partial_sum,
        output,
        stride_o_req,
        stride_o_head,
        num_heads,
        num_slices,
        VALUE_DIM: tl.constexpr,
        VALUE_DIM_TILE: tl.constexpr,
    ):
        row = tl.program_id(0)
        head = tl.program_id(1)
        value_dims = tl.arange(0, VALUE_DIM_TILE)
        value_dim_valid = value_dims < VALUE_DIM
        running_max = float("-inf")
        running_sum = 0.0
        acc = tl.zeros((VALUE_DIM_TILE,), tl.float32)
        for piece in range(0, num_slices):
            partial = (row * num_slices + piece).to(tl.int64) * num_heads + head
            slice_max = tl.load(partial_max + partial)
            slice_sum = tl.load(partial_sum + partial)
            slice_out = tl.load(
                partial_out + partial * VALUE_DIM + value_dims, mask=value_dim_valid, other=0.0
            )
            new_max = tl.maximum(running_max, slice_max)
            reference = tl.where(new_max == float("-inf"), 0.0, new_max)
            alpha = tl.exp2(running_max - reference)
            beta = tl.exp2(slice_max - reference)
            running_sum = running_sum * alpha + slice_sum * beta
            acc = acc * alpha + slice_out * beta
            running_max = new_max
        # A row with no valid key (graph padding) gets zeros.
        has_keys = running_sum > 0
        result = tl.where(has_keys, acc / tl.where(has_keys, running_sum, 1.0), 0.0)
        tl.store(
            output + row.to(tl.int64) * stride_o_req + head * stride_o_head + value_dims,
            result.to(output.dtype.element_ty),
            mask=value_dim_valid,
        )


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _num_slices(num_rows: int, num_groups: int, topk: int, num_sms: int) -> int:
    """Power-of-two slice count near ATTENTION_PROGRAMS_PER_SM per SM, each at least one tile."""
    target = max(1, (ATTENTION_PROGRAMS_PER_SM * num_sms) // max(1, num_rows * num_groups))
    slices = 1
    while slices * 2 <= target and topk // (slices * 2) >= BLOCK_N:
        slices *= 2
    return slices


def build_row_metadata(
    query_lengths: torch.Tensor, kv_lengths: torch.Tensor, row_block: int = SCORE_ROW_BLOCK
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rows and score blocks for requests that each add ``query_lengths[r]`` query tokens.

    Request ``r``'s tokens are rows ``cu[r] .. cu[r+1]-1`` (``cu`` the cumulative query
    lengths) and the last of them sits at position ``kv_lengths[r] - 1``. Each request's rows
    are split into blocks of at most ``row_block`` rows.

    Args:
        query_lengths: Int CPU tensor ``[requests]`` of new query tokens per request.
        kv_lengths: Int CPU tensor ``[requests]`` of keys per request, the new ones included.
        row_block: Rows per score block.

    Returns:
        ``(row_requests, row_lengths, block_starts, block_rows, block_requests)``, int32 CPU
        tensors over ``sum(query_lengths)`` rows and the blocks.
    """
    query_lengths = query_lengths.to(torch.int64)
    kv_lengths = kv_lengths.to(torch.int64)
    num_requests = query_lengths.numel()
    num_rows = int(query_lengths.sum())
    request_ids = torch.arange(num_requests)
    first_row = torch.cumsum(query_lengths, 0) - query_lengths
    row_requests = torch.repeat_interleave(request_ids, query_lengths)
    within = torch.arange(num_rows) - first_row[row_requests]
    row_lengths = (kv_lengths - query_lengths)[row_requests] + within + 1
    blocks_per_request = (query_lengths + row_block - 1) // row_block
    block_requests = torch.repeat_interleave(request_ids, blocks_per_request)
    first_block = torch.cumsum(blocks_per_request, 0) - blocks_per_request
    block_in_request = torch.arange(int(blocks_per_request.sum())) - first_block[block_requests]
    block_starts = first_row[block_requests] + block_in_request * row_block
    block_rows = torch.clamp(
        query_lengths[block_requests] - block_in_request * row_block, max=row_block
    )
    return tuple(
        tensor.to(torch.int32)
        for tensor in (row_requests, row_lengths, block_starts, block_rows, block_requests)
    )


def _require_cuda_on(device: torch.device, tensors) -> None:
    if any(not tensor.is_cuda or tensor.device != device for tensor in tensors):
        raise ValueError("All DSA inference kernel tensors must be on the same CUDA device.")


def _require_int32_vector(name: str, tensor: torch.Tensor, length: Optional[int] = None) -> None:
    if tensor.dtype != torch.int32 or tensor.dim() != 1 or not tensor.is_contiguous():
        raise TypeError(f"{name} must be a contiguous int32 vector.")
    if length is not None and tensor.numel() != length:
        raise ValueError(f"{name} must have {length} entries, got {tensor.numel()}.")


def _require_block_table(block_table: torch.Tensor) -> None:
    if block_table.dtype != torch.int32 or block_table.dim() != 2 or block_table.stride(-1) != 1:
        raise TypeError("block_table must be int32 [requests,max_pages] with dense rows.")


@torch.no_grad()
def dsa_gqa_indexer_scores(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    row_lengths: torch.Tensor,
    block_starts: torch.Tensor,
    block_rows: torch.Tensor,
    block_requests: torch.Tensor,
    *,
    out: torch.Tensor,
    fill_invalid: bool,
) -> None:
    """Score each row's cached indexer keys against the row's indexer query.

    Args:
        query: BF16 indexer queries ``[rows,index_dim]``.
        key_cache: Paged BF16 indexer keys ``[num_pages,page_size,index_dim]``.
        block_table: Int32 physical-page table ``[requests,max_pages]``.
        row_lengths: Int32 ``[rows]``; row ``t`` scores keys ``[0, row_lengths[t])``.
        block_starts, block_rows, block_requests: Int32 score blocks (``build_row_metadata``);
            a block with 0 rows is skipped.
        out: Caller-owned FP32 scores ``[rows,width]``; ``out[t, j]`` is the dot product of
            query ``t`` with its request's key ``j`` for ``j < min(row_lengths[t], width)``.
        fill_invalid: Write negative infinity at positions at or past each row's length.
            Without it those positions are left unwritten, for a top-k that reads only the
            first ``row_lengths[t]`` scores of row ``t``. Rows in no block (padding) are
            never written.
    """
    if not HAVE_TRITON:
        raise RuntimeError("Triton is required for the DSA indexer kernel.")
    _require_cuda_on(
        query.device,
        (key_cache, block_table, row_lengths, block_starts, block_rows, block_requests, out),
    )
    if query.dtype != torch.bfloat16 or query.dim() != 2 or query.stride(-1) != 1:
        raise TypeError("query must be BF16 [rows,index_dim] with dense rows.")
    num_rows, index_dim = query.shape
    if (
        key_cache.dtype != torch.bfloat16
        or key_cache.dim() != 3
        or key_cache.size(2) != index_dim
        or key_cache.stride(-1) != 1
    ):
        raise TypeError("key_cache must be BF16 [pages,page_size,index_dim] matching query.")
    _require_block_table(block_table)
    _require_int32_vector("row_lengths", row_lengths, num_rows)
    num_blocks = block_starts.numel()
    for name, tensor in (
        ("block_starts", block_starts),
        ("block_rows", block_rows),
        ("block_requests", block_requests),
    ):
        _require_int32_vector(name, tensor, num_blocks)
    if (
        out.dtype != torch.float32
        or out.dim() != 2
        or out.size(0) != num_rows
        or out.stride(-1) != 1
    ):
        raise TypeError("out must be FP32 [rows,width] with dense rows.")
    width = out.size(1)
    if num_blocks == 0 or width == 0:
        return
    tiles = triton.cdiv(width, SCORE_BLOCK_T)
    workers = max(1, min(tiles, SCORE_PROGRAMS_PER_SM * _num_sms(query.device.index) // num_blocks))
    _score_kernel[(workers, num_blocks)](
        query,
        key_cache,
        block_table,
        row_lengths,
        block_starts,
        block_rows,
        block_requests,
        out,
        query.stride(0),
        key_cache.stride(0),
        key_cache.stride(1),
        block_table.stride(0),
        out.stride(0),
        width,
        block_table.size(1),
        key_cache.size(0),
        INDEX_DIM=index_dim,
        INDEX_DIM_TILE=max(MIN_HEAD_TILE, triton.next_power_of_2(index_dim)),
        PAGE_SIZE=key_cache.size(1),
        ROW_BLOCK=SCORE_ROW_BLOCK,
        BLOCK_T=SCORE_BLOCK_T,
        FILL_INVALID=fill_invalid,
        num_warps=4,
        num_stages=2,
    )


@torch.no_grad()
def dsa_gqa_sparse_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    row_requests: torch.Tensor,
    row_lengths: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    *,
    out: torch.Tensor,
) -> None:
    """Attend each row's query heads to its selected paged keys and values.

    Args:
        query: BF16 ``[rows,query_heads,head_dim]``; the row stride may be pitched.
        key_cache: Paged BF16 keys ``[num_pages,page_size,query_groups,head_dim]``.
        value_cache: Paged BF16 values ``[num_pages,page_size,query_groups,value_head_dim]``.
        block_table: Int32 physical-page table ``[requests,max_pages]``.
        row_requests: Int32 ``[rows]`` block-table row of each row; -1 for padding rows.
        row_lengths: Int32 ``[rows]`` valid key length of each row; 0 for padding rows.
        topk_indices: Int32 logical key positions ``[rows,topk]``. Entries that are negative
            or not below the row's length are skipped; rows may be pitched.
        softmax_scale: Attention score scale.
        out: Caller-owned BF16 output ``[rows,query_heads,value_head_dim]``. A row with no
            valid selected key gets zeros.
    """
    if not HAVE_TRITON:
        raise RuntimeError("Triton is required for the DSA sparse attention kernel.")
    _require_cuda_on(
        query.device,
        (key_cache, value_cache, block_table, row_requests, row_lengths, topk_indices, out),
    )
    if query.dtype != torch.bfloat16 or query.dim() != 3 or query.stride(-1) != 1:
        raise TypeError("query must be BF16 [rows,heads,head_dim] with contiguous head rows.")
    num_rows, num_query_heads, head_dim = query.shape
    for name, cache in (("key_cache", key_cache), ("value_cache", value_cache)):
        if cache.dtype != torch.bfloat16 or cache.dim() != 4 or cache.stride(-1) != 1:
            raise TypeError(f"{name} must be BF16 [pages,page_size,groups,dim], last dim dense.")
    if key_cache.shape[:3] != value_cache.shape[:3] or key_cache.size(3) != head_dim:
        raise ValueError("Key/value cache geometry must agree and key head_dim must match query.")
    num_pages, page_size, num_groups, _ = key_cache.shape
    if num_query_heads % num_groups != 0:
        raise ValueError("Query heads must be divisible by the number of KV groups.")
    _require_block_table(block_table)
    _require_int32_vector("row_requests", row_requests, num_rows)
    _require_int32_vector("row_lengths", row_lengths, num_rows)
    if (
        topk_indices.dtype != torch.int32
        or topk_indices.dim() != 2
        or topk_indices.size(0) != num_rows
        or topk_indices.size(1) == 0
        or topk_indices.stride(-1) != 1
    ):
        raise TypeError("topk_indices must be int32 [rows,topk] with dense rows.")
    value_dim = value_cache.size(-1)
    if (
        out.dtype != torch.bfloat16
        or out.shape != (num_rows, num_query_heads, value_dim)
        or out.stride(-1) != 1
    ):
        raise TypeError("out must be BF16 [rows,heads,value_head_dim] with dense rows.")
    if num_rows == 0:
        return
    topk = topk_indices.size(1)
    heads_per_group = num_query_heads // num_groups
    slices = _num_slices(num_rows, num_groups, topk, _num_sms(query.device.index))
    slice_len = triton.cdiv(topk, slices)
    partial_rows = num_rows * slices * num_query_heads
    partial_out = torch.empty(partial_rows, value_dim, dtype=torch.float32, device=query.device)
    partial_max = torch.empty(partial_rows, dtype=torch.float32, device=query.device)
    partial_sum = torch.empty(partial_rows, dtype=torch.float32, device=query.device)
    _partial_kernel[(num_rows, num_groups, slices)](
        query,
        key_cache,
        value_cache,
        block_table,
        row_requests,
        topk_indices,
        row_lengths,
        partial_out,
        partial_max,
        partial_sum,
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        block_table.stride(0),
        topk_indices.stride(0),
        num_pages,
        block_table.size(1),
        topk,
        slice_len,
        slices,
        softmax_scale * LOG2E,
        HEADS_PER_GROUP=heads_per_group,
        HEAD_TILE=max(MIN_HEAD_TILE, triton.next_power_of_2(heads_per_group)),
        HEAD_DIM=head_dim,
        HEAD_DIM_TILE=triton.next_power_of_2(head_dim),
        VALUE_DIM=value_dim,
        VALUE_DIM_TILE=triton.next_power_of_2(value_dim),
        PAGE_SIZE=page_size,
        BLOCK=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )
    _combine_kernel[(num_rows, num_query_heads)](
        partial_out,
        partial_max,
        partial_sum,
        out,
        out.stride(0),
        out.stride(1),
        num_query_heads,
        slices,
        VALUE_DIM=value_dim,
        VALUE_DIM_TILE=triton.next_power_of_2(value_dim),
        num_warps=4,
    )


__all__ = [
    "HAVE_TRITON",
    "SCORE_ROW_BLOCK",
    "build_row_metadata",
    "dsa_gqa_indexer_scores",
    "dsa_gqa_sparse_attention",
]
