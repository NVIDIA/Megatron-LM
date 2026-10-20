# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Triton paged decode kernels for DSA over GQA: indexer scores and sparse attention.

Indexer scores: each request's programs stride over its cached indexer keys up to its length,
streaming page-contiguous key tiles and dotting them with the request's indexer query on tensor
cores. The launch size depends only on the batch and the score width, so a CUDA graph replays
with any lengths, and the work follows the real lengths.

Sparse attention: each decode request attends to its top-k selected cached keys. One program
handles every query head of one KV group for one slice of the request's selected slots, so each
selected K/V row is read once per group rather than once per head, and QK and PV run on tensor
cores. A second kernel merges the slices with their running maxima and sums (flash decoding). The
slice count depends only on the batch size.
"""

from __future__ import annotations

import functools

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
# Indexer scores: keys per tile, and score programs to aim for per SM across a decode step.
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
        context_lengths,
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
        BLOCK_T: tl.constexpr,
        FILL_INVALID: tl.constexpr,
        QUERY_COLUMNS: tl.constexpr,
    ):
        worker = tl.program_id(0)
        request = tl.program_id(1)
        workers = tl.num_programs(0)
        dims = tl.arange(0, INDEX_DIM_TILE)
        dim_valid = dims < INDEX_DIM
        q = tl.load(query + request.to(tl.int64) * stride_q + dims, mask=dim_valid, other=0.0)
        # Tensor cores need an N of at least 16: column 0 holds the query, the rest zeros.
        columns = tl.arange(0, QUERY_COLUMNS)
        q_matrix = tl.where(
            columns[None, :] == 0, q[:, None], tl.zeros((INDEX_DIM_TILE, QUERY_COLUMNS), q.dtype)
        )
        length = tl.load(context_lengths + request)
        # Positions below `limit` are written: the whole row when invalid positions are
        # filled with -inf, else only the request's keys.
        limit = width if FILL_INVALID else tl.minimum(length, width)
        for start in range(worker * BLOCK_T, limit, workers * BLOCK_T):
            positions = start + tl.arange(0, BLOCK_T)
            valid = positions < length
            logical_page = positions // PAGE_SIZE
            valid = valid & (logical_page < max_pages)
            page = tl.load(
                block_table + request.to(tl.int64) * stride_table + logical_page,
                mask=valid,
                other=-1,
            )
            valid = valid & (page >= 0) & (page < num_pages)
            page = tl.where(valid, page, 0).to(tl.int64)
            k = tl.load(
                key_cache
                + page[:, None] * stride_k_page
                + (positions % PAGE_SIZE)[:, None] * stride_k_pos
                + dims[None, :],
                mask=valid[:, None] & dim_valid[None, :],
                other=0.0,
            )
            score = tl.sum(tl.dot(k, q_matrix), axis=1)
            score = tl.where(valid, score, float("-inf"))
            tl.store(
                scores + request.to(tl.int64) * stride_scores + positions,
                score,
                mask=positions < limit,
            )

    @triton.jit(do_not_specialize=["stride_topk", "topk", "slice_len", "num_slices"])
    def _partial_kernel(
        query,
        key_cache,
        value_cache,
        block_table,
        topk_indices,
        context_lengths,
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
        request = tl.program_id(0)
        group = tl.program_id(1)
        piece = tl.program_id(2)
        num_groups = tl.num_programs(1)

        head_in_group = tl.arange(0, HEAD_TILE)
        head_valid = head_in_group < HEADS_PER_GROUP
        heads = group * HEADS_PER_GROUP + head_in_group
        dims = tl.arange(0, HEAD_DIM_TILE)
        dim_valid = dims < HEAD_DIM
        value_dims = tl.arange(0, VALUE_DIM_TILE)
        value_dim_valid = value_dims < VALUE_DIM
        q = tl.load(
            query
            + request.to(tl.int64) * stride_q_req
            + heads[:, None] * stride_q_head
            + dims[None, :],
            mask=head_valid[:, None] & dim_valid[None, :],
            other=0.0,
        )
        length = tl.load(context_lengths + request)

        running_max = tl.full((HEAD_TILE,), float("-inf"), tl.float32)
        running_sum = tl.zeros((HEAD_TILE,), tl.float32)
        acc = tl.zeros((HEAD_TILE, VALUE_DIM_TILE), tl.float32)
        first_slot = piece * slice_len
        for start in range(0, slice_len, BLOCK):
            slots = first_slot + start + tl.arange(0, BLOCK)
            # The last slice may be short when the slice count does not divide top-k.
            in_slice = (start + tl.arange(0, BLOCK) < slice_len) & (slots < topk)
            position = tl.load(
                topk_indices + request.to(tl.int64) * stride_topk + slots, mask=in_slice, other=-1
            )
            valid = (position >= 0) & (position < length)
            safe_position = tl.where(valid, position, 0)
            logical_page = safe_position // PAGE_SIZE
            valid = valid & (logical_page < max_pages)
            page = tl.load(
                block_table + request.to(tl.int64) * stride_table + logical_page,
                mask=valid,
                other=-1,
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

        # Partial rows are [request, slice, head], so a group's heads are contiguous.
        rows = ((request * num_slices + piece) * num_groups + group) * HEADS_PER_GROUP
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
        request = tl.program_id(0)
        head = tl.program_id(1)
        value_dims = tl.arange(0, VALUE_DIM_TILE)
        value_dim_valid = value_dims < VALUE_DIM
        running_max = float("-inf")
        running_sum = 0.0
        acc = tl.zeros((VALUE_DIM_TILE,), tl.float32)
        for piece in range(0, num_slices):
            row = (request * num_slices + piece).to(tl.int64) * num_heads + head
            slice_max = tl.load(partial_max + row)
            slice_sum = tl.load(partial_sum + row)
            slice_out = tl.load(
                partial_out + row * VALUE_DIM + value_dims, mask=value_dim_valid, other=0.0
            )
            new_max = tl.maximum(running_max, slice_max)
            reference = tl.where(new_max == float("-inf"), 0.0, new_max)
            alpha = tl.exp2(running_max - reference)
            beta = tl.exp2(slice_max - reference)
            running_sum = running_sum * alpha + slice_sum * beta
            acc = acc * alpha + slice_out * beta
            running_max = new_max
        # A request with no valid key (graph padding) gets zeros.
        has_keys = running_sum > 0
        result = tl.where(has_keys, acc / tl.where(has_keys, running_sum, 1.0), 0.0)
        tl.store(
            output + request.to(tl.int64) * stride_o_req + head * stride_o_head + value_dims,
            result.to(output.dtype.element_ty),
            mask=value_dim_valid,
        )


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _num_slices(num_requests: int, num_groups: int, topk: int, num_sms: int) -> int:
    """Power-of-two slice count near ATTENTION_PROGRAMS_PER_SM per SM, each at least one tile."""
    target = max(1, (ATTENTION_PROGRAMS_PER_SM * num_sms) // max(1, num_requests * num_groups))
    slices = 1
    while slices * 2 <= target and topk // (slices * 2) >= BLOCK_N:
        slices *= 2
    return slices


def _validate_score(query, key_cache, block_table, context_lengths, output):
    if not HAVE_TRITON:
        raise RuntimeError("Triton is required for the DSA-GQA decode indexer kernel.")
    tensors = (query, key_cache, block_table, context_lengths, output)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("All decode indexer tensors must be CUDA tensors.")
    if any(tensor.device != query.device for tensor in tensors[1:]):
        raise ValueError("All decode indexer tensors must be on the same CUDA device.")
    if query.dtype != torch.bfloat16 or query.dim() != 2 or query.stride(-1) != 1:
        raise TypeError("query must be BF16 [batch,index_dim] with dense rows.")
    index_dim = query.size(1)
    if (
        key_cache.dtype != torch.bfloat16
        or key_cache.dim() != 3
        or key_cache.size(2) != index_dim
        or key_cache.stride(-1) != 1
    ):
        raise TypeError("key_cache must be BF16 [pages,page_size,index_dim] matching query.")
    if block_table.dtype != torch.int32 or block_table.dim() != 2 or block_table.stride(-1) != 1:
        raise TypeError("block_table must be int32 [batch,max_pages] with dense rows.")
    if block_table.size(0) != query.size(0):
        raise ValueError("block_table batch dimension must match query.")
    if (
        context_lengths.dtype != torch.int32
        or context_lengths.shape != (query.size(0),)
        or not context_lengths.is_contiguous()
    ):
        raise TypeError("context_lengths must be contiguous int32 with shape [batch].")
    if (
        output.dtype != torch.float32
        or output.dim() != 2
        or output.size(0) != query.size(0)
        or output.stride(-1) != 1
    ):
        raise TypeError("out must be FP32 [batch,max_context_length] with dense rows.")


@torch.no_grad()
def dsa_gqa_decode_indexer_score(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lengths: torch.Tensor,
    *,
    out: torch.Tensor,
    fill_invalid: bool,
) -> None:
    """Score each decode request's cached indexer keys against its indexer query.

    Args:
        query: BF16 indexer queries ``[batch,index_dim]``.
        key_cache: Paged BF16 indexer keys ``[num_pages,page_size,index_dim]``.
        block_table: Int32 physical-page table ``[batch,max_pages]``.
        context_lengths: Int32 valid key lengths ``[batch]``.
        out: Caller-owned FP32 scores ``[batch,max_context_length]``; ``out[r, t]`` is the
            dot product of query ``r`` with its key ``t`` for ``t < context_lengths[r]``.
        fill_invalid: Write negative infinity at positions at or past each request's
            length. Without it those positions are left unwritten, for a top-k that
            reads only the first ``context_lengths[r]`` scores of row ``r``.
    """
    _validate_score(query, key_cache, block_table, context_lengths, out)
    batch_size, width = out.shape
    if batch_size == 0 or width == 0:
        return
    tiles = triton.cdiv(width, SCORE_BLOCK_T)
    workers = max(1, min(tiles, SCORE_PROGRAMS_PER_SM * _num_sms(query.device.index) // batch_size))
    _score_kernel[(workers, batch_size)](
        query,
        key_cache,
        block_table,
        context_lengths,
        out,
        query.stride(0),
        key_cache.stride(0),
        key_cache.stride(1),
        block_table.stride(0),
        out.stride(0),
        width,
        block_table.size(1),
        key_cache.size(0),
        INDEX_DIM=query.size(1),
        INDEX_DIM_TILE=max(MIN_HEAD_TILE, triton.next_power_of_2(query.size(1))),
        PAGE_SIZE=key_cache.size(1),
        BLOCK_T=SCORE_BLOCK_T,
        FILL_INVALID=fill_invalid,
        QUERY_COLUMNS=MIN_HEAD_TILE,
        num_warps=4,
        num_stages=2,
    )


def _validate(query, key_cache, value_cache, block_table, topk_indices, context_lengths, output):
    if not HAVE_TRITON:
        raise RuntimeError("Triton is required for the DSA-GQA decode attention kernel.")
    tensors = (query, key_cache, value_cache, block_table, topk_indices, context_lengths, output)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("All decode attention tensors must be CUDA tensors.")
    if any(tensor.device != query.device for tensor in tensors[1:]):
        raise ValueError("All decode attention tensors must be on the same CUDA device.")
    if query.dtype != torch.bfloat16 or query.dim() != 3 or query.stride(-1) != 1:
        raise TypeError("query must be BF16 [batch,heads,head_dim] with contiguous head rows.")
    batch_size, num_query_heads, query_head_dim = query.shape
    for name, cache in (("key_cache", key_cache), ("value_cache", value_cache)):
        if cache.dtype != torch.bfloat16 or cache.dim() != 4 or cache.stride(-1) != 1:
            raise TypeError(f"{name} must be BF16 [pages,page_size,groups,dim], last dim dense.")
    if key_cache.shape[:3] != value_cache.shape[:3] or key_cache.size(3) != query_head_dim:
        raise ValueError("Key/value cache geometry must agree and key head_dim must match query.")
    num_query_groups = key_cache.size(2)
    if num_query_heads % num_query_groups != 0:
        raise ValueError("Query heads must be divisible by the number of KV groups.")
    if block_table.dtype != torch.int32 or block_table.dim() != 2 or block_table.stride(-1) != 1:
        raise TypeError("block_table must be int32 [batch,max_pages] with dense rows.")
    if block_table.size(0) != batch_size:
        raise ValueError("block_table batch dimension must match query.")
    if (
        topk_indices.dtype != torch.int32
        or topk_indices.dim() != 2
        or topk_indices.size(0) != batch_size
        or topk_indices.size(1) == 0
        or topk_indices.stride(-1) != 1
    ):
        raise TypeError("topk_indices must be int32 [batch,topk] with dense rows.")
    if context_lengths.dtype != torch.int32 or context_lengths.shape != (batch_size,):
        raise TypeError("context_lengths must be contiguous int32 with shape [batch].")
    if not context_lengths.is_contiguous():
        raise ValueError("context_lengths must be contiguous.")
    if (
        output.dtype != torch.bfloat16
        or output.shape != (batch_size, num_query_heads, value_cache.size(3))
        or output.stride(-1) != 1
    ):
        raise TypeError("out must be BF16 [batch,heads,value_head_dim] with dense rows.")


@torch.no_grad()
def dsa_gqa_decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    topk_indices: torch.Tensor,
    context_lengths: torch.Tensor,
    softmax_scale: float,
    *,
    out: torch.Tensor,
) -> None:
    """Attend each decode request's query heads to its selected paged keys and values.

    Args:
        query: BF16 ``[batch,query_heads,head_dim]``; the token stride may be pitched.
        key_cache: Paged BF16 keys ``[num_pages,page_size,query_groups,head_dim]``.
        value_cache: Paged BF16 values ``[num_pages,page_size,query_groups,value_head_dim]``.
        block_table: Int32 physical-page table ``[batch,max_pages]``.
        topk_indices: Int32 logical key positions ``[batch,topk]``. Entries that are negative
            or not below the request's length are skipped; rows may be pitched.
        context_lengths: Int32 valid key lengths ``[batch]``.
        softmax_scale: Attention score scale.
        out: Caller-owned BF16 output ``[batch,query_heads,value_head_dim]``. A request with
            no valid selected key gets zeros.
    """
    _validate(query, key_cache, value_cache, block_table, topk_indices, context_lengths, out)
    num_requests, num_query_heads, head_dim = query.shape
    if num_requests == 0:
        return
    num_pages, page_size, num_groups, _ = key_cache.shape
    value_dim = value_cache.size(-1)
    topk = topk_indices.size(1)
    heads_per_group = num_query_heads // num_groups
    slices = _num_slices(num_requests, num_groups, topk, _num_sms(query.device.index))
    slice_len = triton.cdiv(topk, slices)
    partial_rows = num_requests * slices * num_query_heads
    partial_out = torch.empty(partial_rows, value_dim, dtype=torch.float32, device=query.device)
    partial_max = torch.empty(partial_rows, dtype=torch.float32, device=query.device)
    partial_sum = torch.empty(partial_rows, dtype=torch.float32, device=query.device)
    _partial_kernel[(num_requests, num_groups, slices)](
        query,
        key_cache,
        value_cache,
        block_table,
        topk_indices,
        context_lengths,
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
    _combine_kernel[(num_requests, num_query_heads)](
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


__all__ = ["HAVE_TRITON", "dsa_gqa_decode_attention", "dsa_gqa_decode_indexer_score"]
