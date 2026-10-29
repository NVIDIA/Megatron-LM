# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Gluon TMA/tcgen05 scoring and Triton split-KV attention for DSA decode."""

from __future__ import annotations

import triton
import triton.language as tl
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tcgen05_mma,
)
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma

LOG2E = 1.4426950408889634
BLOCK_N = 128
ATTENTION_PROGRAMS_PER_SM = 1


@g.jit
def score_tcgen(
    Qdesc,
    Kdesc,
    Table,
    Lengths,
    Requests,
    Out,
    stride_table: gl.constexpr,
    WIDTH: gl.constexpr,
    PAGE: gl.constexpr,
    WORKERS: gl.constexpr,
    STAGES: gl.constexpr,
):
    """Score a paged indexer row with staged TMA loads and tcgen05 MMA."""
    row = gl.program_id(0)
    worker = gl.program_id(1)
    request = gl.load(Requests + row)
    n = gl.minimum(gl.load(Lengths + row), WIDTH)
    tiles = gl.cdiv(gl.cdiv(n, 128), WORKERS)
    first = worker * tiles
    count = gl.maximum(gl.minimum(tiles, gl.cdiv(n, 128) - first), 0)
    qbuf = gl.allocate_shared_memory(Qdesc.dtype, Qdesc.block_type.shape, Qdesc.layout)
    keys = gl.allocate_shared_memory(Kdesc.dtype, [STAGES] + Kdesc.block_type.shape, Kdesc.layout)
    qbar = mbarrier.allocate_mbarrier()
    done = mbarrier.allocate_mbarrier()
    mbarrier.init(done, count=1)
    bars = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    mbarrier.init(qbar, count=1)
    for init_slot in gl.static_range(STAGES):
        mbarrier.init(bars.index(init_slot), count=1)
    mbarrier.expect(qbar, Qdesc.block_type.nbytes)
    tma.async_load(Qdesc, [row, 0], qbar, qbuf)
    for prefetch_slot in gl.static_range(STAGES):
        start = (first + prefetch_slot) * 128
        page = gl.load(
            Table + request * stride_table + start // PAGE, mask=prefetch_slot < count, other=0
        )
        mbarrier.expect(
            bars.index(prefetch_slot), Kdesc.block_type.nbytes, pred=prefetch_slot < count
        )
        tma.async_load(
            Kdesc,
            [page * PAGE + start % PAGE, 0],
            bars.index(prefetch_slot),
            keys.index(prefetch_slot),
            pred=prefetch_slot < count,
        )
    mbarrier.wait(qbar, 0)
    acc = allocate_tensor_memory(gl.float32, [128, 8], TensorMemoryLayout([128, 8], col_stride=1))
    for step in range(count):
        i = step % STAGES
        phase = step // STAGES % 2
        mbarrier.wait(bars.index(i), phase)
        tcgen05_mma(keys.index(i), qbuf.permute((1, 0)), acc, use_acc=False, mbarriers=[done])
        mbarrier.wait(done, step % 2)
        # Explicit completion protects both the TMEM read and reuse of the key slot.
        values = acc.load()
        layout: gl.constexpr = values.type.layout
        rows = (first + step) * 128 + gl.arange(0, 128, layout=gl.SliceLayout(1, layout))
        cols = gl.arange(0, 8, layout=gl.SliceLayout(0, layout))
        gl.store(
            Out + row * WIDTH + rows[:, None] + cols[None, :] * 0,
            values,
            mask=(rows[:, None] < n) & (cols[None, :] == 0),
        )
        next_step = step + STAGES
        start = (first + next_step) * 128
        page = gl.load(
            Table + request * stride_table + start // PAGE, mask=next_step < count, other=0
        )
        mbarrier.expect(bars.index(i), Kdesc.block_type.nbytes, pred=next_step < count)
        tma.async_load(
            Kdesc,
            [page * PAGE + start % PAGE, 0],
            bars.index(i),
            keys.index(i),
            pred=next_step < count,
        )
    mbarrier.invalidate(done)
    mbarrier.invalidate(qbar)
    for free_slot in gl.static_range(STAGES):
        mbarrier.invalidate(bars.index(free_slot))


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
        query + row.to(tl.int64) * stride_q_req + heads[:, None] * stride_q_head + dims[None, :],
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
        page = tl.load(block_table + request * stride_table + logical_page, mask=valid, other=-1)
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


def _num_slices(num_rows: int, num_groups: int, topk: int, num_sms: int) -> int:
    """Power-of-two slice count near ATTENTION_PROGRAMS_PER_SM per SM, each at least one tile."""
    target = max(1, (ATTENTION_PROGRAMS_PER_SM * num_sms) // max(1, num_rows * num_groups))
    slices = 1
    while slices * 2 <= target and topk // (slices * 2) >= BLOCK_N:
        slices *= 2
    return slices
