# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Gluon scorer using the same N=8 tcgen05 geometry as the CuTe decode scorer."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tcgen05_mma,
)
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma


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
