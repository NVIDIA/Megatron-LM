# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fused backward of a seam: one persistent kernel plus the final gradient reductions.

Groups of CTAs span the width (256 columns each) and walk 16-row tiles, pipelining reader
projections, predecessor/carry TMA transfers and the residual/update/controller gradients.
"""

from typing import NamedTuple

import torch
import triton
import triton.language as tl
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout, SwizzledSharedLayout
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, mbarrier, tma

from .kernel_utils import (
    FLAGSHIPS,
    _coefficient,
    _descriptor,
    _fma2,
    _mul2,
    qualified_device,
    reduce_jobs,
    validate_num_streams,
)

# Triton renamed this API in 3.9; both names use the same TMA operation.
_tma_load = tma.async_load if hasattr(tma, "async_load") else tma.async_copy_global_to_shared

_GRADIENT_LAYOUT = gl.constexpr(BlockedLayout([1, 4], [4, 8], [1, 4], [1, 0]))


class _BackwardInputs(NamedTuple):
    coefficients: object
    rstd: tuple
    gammas: tuple
    carry: object
    reader_grads: tuple


class _BackwardOutputs(NamedTuple):
    residual_grad: object
    update_grad: object
    controller_partials: object
    projection: object  # tagged handoff words (see the protocol below)
    gamma_partials: tuple
    epochs: object  # admission and row-tile counters


class _Geometry(NamedTuple):
    M: int
    H: int
    STREAMS: int
    HAS_RETENTION: bool


class _BackwardConfig(NamedTuple):
    """Projection-reduction layout, grid, register cap and Super one-reader early carry load."""

    PROJECTION_ROWS_PER_WARP: int
    PROJECTION_VECTOR: int
    TARGET_CTAS: int
    MAXNREG: int
    EARLY_CARRY: bool = False


def get_backward_knobs(m, h, streams, readers=1, *, grid="auto"):
    """Select the launch knobs for one shape; an explicit grid overrides the auto grid."""
    production = streams == 3 and (m, h) in FLAGSHIPS
    rows = 2 if h == 4096 and streams not in (1, 4) else 4
    vector = 4 if (h <= 3072 and not (h == 3072 and streams > 4)) or streams == 4 else 2
    if streams == 3 and not production:
        rows, vector = 8, 2 if h % 2048 else 4
    budget = 456 if streams <= 3 else 288
    if grid == "auto":
        participants = triton.cdiv(h, 256)
        grid = min(triton.cdiv(m, 16), max(1, budget // participants)) * participants
    return _BackwardConfig(
        PROJECTION_ROWS_PER_WARP=rows,
        PROJECTION_VECTOR=vector,
        TARGET_CTAS=grid,
        EARLY_CARRY=production and h == 4096 and readers == 1,
        # Six streams have less shared-memory residency; allow registers to avoid spills.
        MAXNREG=255 if streams == 6 else 168,
    )


@torch.no_grad()
def backward(
    saved,
    previous,
    dys,
    carry=None,
    *,
    columns=None,
    save_columns=False,
    config="auto",
    on_columns_ready=None,
):
    """Gradients of one seam from its ``SavedWrite`` and predecessor residual.

    Returns ``(grads, columns)``: the residual gradient, one update gradient per branch update
    (a shortcut add's share one tensor), the controller gradients (write, retain, one read per
    reader) and one RMSNorm weight gradient per reader. ``save_columns`` (a bool for reader
    zero, or one per reader) also returns columnwise MXFP8 reader operands, reusing any given
    in ``columns``; ``on_columns_ready`` receives them before the fused kernel launches, so
    dependent weight GEMMs overlap it. ``config`` may set an explicit grid (tests).
    """
    n, streams, updates = saved.reader_count, saved.num_streams, saved.update_count
    m, h = saved.update.shape
    validate_num_streams(streams)
    if m <= 0 or m % 32 or h < 32 or h > 16384 or h % 32:
        raise ValueError(
            "shortcut backward requires positive rows divisible by 32 and H=32..16384, step 32"
        )
    if n not in (1, 2) or updates not in (1, 2) or len(dys) != n:
        raise ValueError("backward requires one or two readers (one dy each) and updates")
    if previous.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("backward requires an FP32 or BF16 predecessor residual")
    if carry is None:
        carry = torch.zeros_like(previous)
    if carry.dtype != previous.dtype:
        raise ValueError("the residual gradient must have the residual's dtype")
    dys = tuple(torch.zeros_like(saved.update) if dy is None else dy for dy in dys)
    if not qualified_device(torch.cuda.get_device_properties(previous.device)):
        raise ValueError("shortcut backward requires a qualified GB300")
    nt = triton.cdiv(h, 256)
    if config != "auto" and (type(config) is not int or config <= 0 or config % nt):
        raise ValueError("shortcut grid must contain positive whole participant groups")
    routes = save_columns
    routes = (routes,) + (False,) * (n - 1) if isinstance(routes, bool) else tuple(routes)
    columns = (None,) * n if columns is None else tuple(columns)
    if len(routes) != n or len(columns) != n:
        raise ValueError("shortcut backward requires one column route and operand per reader")
    missing = tuple(needed and column is None for needed, column in zip(routes, columns))
    if any(missing):
        from .replay_kernel import replay_interval

        generated = tuple(
            replay_interval(previous, (saved,), prepare_columns=need, reader=r)[1][0]
            for r, need in enumerate(missing)
        )
        columns = tuple(old if old is not None else new for old, new in zip(columns, generated))
    if on_columns_ready is not None:
        on_columns_ready(columns)
    knobs = get_backward_knobs(m, h, streams, n, grid=config)
    grid = knobs.TARGET_CTAS
    # Outputs and scratch belong to this invocation; a shortcut add's updates share one gradient.
    empty = saved.coefficients.new_empty
    du = empty((m, h), dtype=torch.bfloat16)
    grads = (
        empty((m, streams * h), dtype=previous.dtype),
        *((du,) * updates),
        *(empty((size,)) if size else None for size in saved.controller_sizes),
        *(empty((h,)) for _ in range(n)),
    )
    p = triton.cdiv(m, 16) * triton.cdiv(h, 128) * 8
    gp = tuple(empty((triton.cdiv(m, 16), h)) for _ in range(n))
    planes = (1 + n + saved.has_retention) * streams
    controllers = empty((planes + n, p))
    chunks = empty((planes + n, triton.cdiv(p, 1024)))

    groups = grid // nt
    tiles = triton.cdiv(m, 16)
    if max(grid, tiles + 2 * groups, nt * (tiles + 1)) >= 2**31:
        raise ValueError("arrival counter exceeds signed 32-bit capacity")
    # Per row tile: a 16-word header (arrival hint, next row tile) and one projection word per
    # (reader, row, column tile); see the handoff protocol below.
    block = 16 + n * 16 * nt
    if tiles * block + 16 >= 2**31:
        raise ValueError("handoff workspace exceeds signed 32-bit indexing")
    epochs = empty((2,), dtype=torch.int32)
    pp = empty((tiles * block,), dtype=torch.int64)

    slice_shape = (16, 128)
    kw = dict(
        inputs=_BackwardInputs(saved.coefficients, saved.rstd, saved.gammas, carry, dys),
        GEOMETRY=_Geometry(m, h, streams, saved.has_retention),
        CONFIG=_BackwardConfig(*(gl.constexpr(value) for value in knobs)),
        outputs=_BackwardOutputs(grads[0], grads[1], controllers, pp, gp, epochs),
        CARRY_DESC=_descriptor(carry, slice_shape),
        R_DESC=_descriptor(previous, slice_shape),
        DY0_DESC=_descriptor(dys[0] if streams < 6 else None, slice_shape),
        DY1_DESC=_descriptor(dys[1] if streams < 6 and n == 2 else None, slice_shape),
        U_DESC=_descriptor(saved.update, slice_shape),
        maxnreg=knobs.MAXNREG,
        num_warps=4,
        enable_fp_fusion=True,
    )
    # Reset on every invocation (also under graph replay) so every handoff word starts at tag 0.
    epochs.zero_()
    pp.zero_()
    _backward[(grid,)](**kw)

    # Reduce controller chunks, then all gamma and controller gradients.
    retention, two = saved.has_retention, n == 2

    def plane(index):
        return chunks[index * streams : (index + 1) * streams]

    def coefficients(index):
        return saved.coefficients[index * streams : (index + 1) * streams]

    raw_b = planes  # Each reader's raw b row follows every controller plane.
    beta0, beta1 = coefficients(2), coefficients(3) if two else None
    write, retain = coefficients(0), coefficients(1) if retention else None
    # Per controller (write, retention, then each reader's read): its partial plane and the
    # raw rows/coefficients its final reduction combines (see _reduce_jobs).
    kinds = (
        (0, (chunks[raw_b], beta0, chunks[raw_b + 1] if two else None, beta1, True)),
        (1, (plane(2), beta0, plane(3) if two else None, beta1) if retention else None),
        *((1 + retention + r, (chunks[raw_b + r], write, retain)) for r in range(n)),
    )
    final = [(p, g, 0, 8, 0, None) for p, g in zip(gp, grads[-n:])]
    for kind, (output, (index, combo)) in enumerate(
        zip(grads[1 + updates : 3 + updates + n], kinds)
    ):
        if output is not None:
            final.append((plane(index), output, 1, 1, 0, coefficients(4 + kind), combo))
    reduce_jobs([(controllers, chunks, 1, 1, 1024, None)])
    reduce_jobs(final)
    return grads, columns


# Fence-free cross-CTA handoff (projection partials, next row tile): each value travels in one
# aligned 64-bit word ``value_bits | (row_tile + 1) << 32``, written once per launch (one owning
# group per tile, no reuse) after the host zeroed it. The producer's st.relaxed.gpu.b64 and the
# consumers' strong 64-bit loads are morally strong, so seeing the tag implies seeing its value.
# No other data crosses CTAs, so no fence (no MEMBAR.GPU wait for the tile's dR/du stores) is
# needed. Arrival counts and L2 warm-ups only pace the wait: every thread checks the tag of
# each copy it sums (layouts may replicate words, so a CTA-wide count would see one replica).
# Progress comes from ticket admission (co-resident groups) and eventually visible stores.


@g.jit
def _publish_word(pointer, value, tag, OWNER_MASK: gl.constexpr):
    """Store value|tag<<32 once from threads whose (tid & OWNER_MASK) == 0."""
    return gl.inline_asm_elementwise(
        """
        {
            .reg .pred owner;
            .reg .u32 tid;
            .reg .b64 word;
            mov.u32 tid,%tid.x;
            and.b32 tid,tid,$4;
            setp.eq.u32 owner,tid,0;
            mov.b64 word,{$2,$3};
            @owner st.relaxed.gpu.global.b64 [$1],word;
            mov.u32 $0,0;
        }
        """,
        "=r,l,r,r,n",
        [pointer, value, tag, OWNER_MASK],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@g.jit
def _arrive_hint(pointer):
    """Count this CTA's publication with one relaxed reduction (no fence)."""
    return gl.inline_asm_elementwise(
        """
        {
            .reg .pred leader;
            .reg .u32 tid;
            mov.u32 tid,%tid.x;
            setp.eq.u32 leader,tid,0;
            @leader red.relaxed.gpu.global.add.u32 [$1],1;
            mov.u32 $0,0;
        }
        """,
        "=r,l",
        [pointer],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@g.jit
def _await_hint(pointer, expected):
    """Spin until the arrival hint reaches ``expected`` (pacing only; words are verified)."""
    return gl.inline_asm_elementwise(
        """
        {
            .reg .pred ready;
            .reg .u32 count;
        WAIT:
            ld.relaxed.gpu.global.u32 count,[$1];
            setp.ge.u32 ready,count,$2;
            @!ready bra WAIT;
            mov.u32 $0,count;
        }
        """,
        "=r,l,n",
        [pointer, expected],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@g.jit
def _verify_word(word, pointer, tag):
    """Return word once its high half equals tag, re-polling this thread's copy."""
    return gl.inline_asm_elementwise(
        """
        {
            .reg .pred ready;
            .reg .b32 value,seen;
            .reg .b64 word;
            mov.b64 word,$1;
        CHECK:
            mov.b64 {value,seen},word;
            setp.eq.u32 ready,seen,$3;
            @ready bra DONE;
            ld.relaxed.gpu.global.b64 word,[$2];
            bra CHECK;
        DONE:
            mov.b64 $0,word;
        }
        """,
        "=l,l,l,r",
        [word, pointer, tag],
        dtype=gl.int64,
        is_pure=False,
        pack=1,
    )


@g.jit
def _bulk_prefetch_l2(pointer, valid, BYTES: gl.constexpr, THREADS: gl.constexpr):
    """Warm BYTES at pointer in L2 from each valid thread below THREADS (a hint only)."""
    return gl.inline_asm_elementwise(
        """
        {
            .reg .pred p,owner;
            .reg .u32 tid;
            mov.u32 tid,%tid.x;
            setp.lt.u32 owner,tid,$4;
            setp.ne.u32 p,$2,0;
            and.pred p,p,owner;
            @p cp.async.bulk.prefetch.L2.global [$1],$3;
            mov.u32 $0,0;
        }
        """,
        "=r,l,r,n,n",
        [pointer, valid, BYTES, THREADS],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@g.jit
def _widen_bf16(x):
    # Exact widening to FP32; BF16 bits become the upper half of the FP32 word.
    if x.dtype == gl.bfloat16:
        return (x.to(gl.uint16, bitcast=True).to(gl.uint32) << 16).to(gl.float32, bitcast=True)
    return x.to(gl.float32)


@g.jit
def _load_dy(
    reader_grads, ds, reader: gl.constexpr, part: gl.constexpr, offset, mask, WIDEN: gl.constexpr
):
    """A reader's FP32 dy slice from global memory or staged BF16 (WIDEN: by bit shift).

    Both conversions are exact, but they schedule (and so contract) differently.
    """
    if ds is None:
        dy = gl.load(reader_grads[reader] + offset, mask, 0).to(gl.float32)
    elif WIDEN:
        dy = _widen_bf16(ds.slice(reader * 32 + part * 16, 16).load(_GRADIENT_LAYOUT))
    else:
        dy = ds.slice(reader * 32 + part * 16, 16).load(_GRADIENT_LAYOUT).to(gl.float32)
    return dy


@g.jit
def _thread_pairs(value):
    """Eight per-thread (128, 2) column pairs of a 16x128 slice, in _store_controller order."""
    owned = gl.permute(gl.reshape(value, (4, 4, 4, 8, 4)), (2, 1, 3, 0, 4))
    # Per-thread index a*4 + e = (a_hi, a_lo, e_hi, e_lo); e_lo pairs adjacent columns.
    t = gl.reshape(owned, (128, 2, 2, 2, 2))
    e_hi0, e_hi1 = gl.split(gl.permute(t, (0, 1, 2, 4, 3)))
    pairs = ()
    for half in gl.static_range(2):
        rest = e_hi0 if half == 0 else e_hi1
        a_lo0, a_lo1 = gl.split(gl.permute(rest, (0, 1, 3, 2)))
        for quarter in gl.static_range(2):
            a_hi0, a_hi1 = gl.split(gl.permute(a_lo0 if quarter == 0 else a_lo1, (0, 2, 1)))
            pairs += (a_hi0, a_hi1)
    return pairs


@g.jit
def _store_controller(destination, a, b, valid, PACKED: gl.constexpr):
    """Reduce each 16x128 slice of a * b into eight fixed lane-group controller partials;
    PACKED accumulates each thread's products as one packed FMA chain."""
    if PACKED:
        pa = _thread_pairs(a)
        pb = _thread_pairs(b)
        acc = _mul2(pa[0], pb[0])
        for k in gl.static_range(1, 8):
            acc = _fma2(pa[k], pb[k], acc)
        per_thread = gl.sum(acc, 1)
    else:
        owned = gl.permute(gl.reshape(_mul2(a, b), (4, 4, 4, 8, 4)), (2, 1, 3, 0, 4))
        per_thread = gl.sum(gl.reshape(owned, (128, 16)), 1)
    total = gl.sum(gl.reshape(per_thread, (8, 16)), 1)
    gl.store(destination + gl.arange(0, 8, layout=total.type.layout), total, valid)


@g.jit
def _partial_offset(reader, rr, cc, NT: gl.constexpr):
    TILE: gl.constexpr = min(16, NT & -NT)
    return reader * 16 * NT + (cc // TILE * 16 + rr) * TILE + cc % TILE


@g.jit
def _prefetch(P, rp, cp, GEOMETRY: gl.constexpr):
    """Warm this tile's carry rows (every stream's 256 columns) in L2."""
    m: gl.constexpr = GEOMETRY.M
    h: gl.constexpr = GEOMETRY.H
    streams: gl.constexpr = GEOMETRY.STREAMS
    ROWS: gl.constexpr = 16
    ELEMENT: gl.constexpr = P.dtype.element_ty.primitive_bitwidth // 8
    ROW_BYTES: gl.constexpr = 256 * ELEMENT
    i = gl.arange(0, ROWS, layout=gl.constexpr(BlockedLayout([1], [32], [4], [0])))
    rr = rp * 16 + i
    cc = cp * 256
    base = P + rr * (streams * h) + cc
    for stream in gl.static_range(streams):
        valid = rr < m
        if h % 256:
            valid &= cc + 256 <= h
        _bulk_prefetch_l2(base + stream * h, valid.to(gl.int32), ROW_BYTES, ROWS)
        if h % 256:
            # A short final tile keeps a single 128-column hint inside the stream.
            tail = (rr < m) & (cc + 128 <= h) & (cc + 256 > h)
            _bulk_prefetch_l2(base + stream * h, tail.to(gl.int32), 128 * ELEMENT, ROWS)


@g.jit
def _copy_tiles(plan, row_tile, barrier, part: gl.constexpr):
    """Copy column half ``part`` of each planned slice into its rows of the [32, 128] tile."""
    for i in gl.static_range(len(plan)):
        descriptor, tile, column = plan[i]
        tile = tile.slice(part * 16, 16)
        _tma_load(descriptor, [row_tile * 16, column + part * 128], barrier, tile)


@g.jit
def _refill_slice(
    s: gl.constexpr,
    carry_copies,
    residual_copies,
    barriers,
    copy_barrier,
    row_tile,
    next_rp,
    part: gl.constexpr,
):
    """Refill consumed predecessor slice s with carry s + 1, or the next tile's last stream."""
    if s + 1 < len(residual_copies):
        bits: gl.constexpr = carry_copies[s][0].dtype.primitive_bitwidth
        mbarrier.expect(barriers[s], 16 * 128 * bits // 8)
        _copy_tiles((carry_copies[s],), row_tile, barriers[s], part)
    else:
        _copy_tiles((residual_copies[s],), next_rp, copy_barrier, part)


@g.jit
def _load_residual(residuals, cols, part, stream: gl.constexpr, h: gl.constexpr):
    # A BF16 residual is widened exactly; every product and sum below runs in FP32.
    value = _widen_bf16(residuals[stream].slice(part * 16, 16).load(_GRADIENT_LAYOUT))
    if h % 256:
        value = gl.where(cols < h, value, 0)
    return value


@g.jit
def _backward(
    inputs,
    outputs,
    GEOMETRY: gl.constexpr,
    CONFIG: gl.constexpr,
    CARRY_DESC,
    R_DESC,
    DY0_DESC,
    DY1_DESC,
    U_DESC,
):
    """Persistent 16-row tiles in width-spanning CTA groups; ticket admission (each group
    reserves its first tile) keeps partial groups from starving. Prefetches refill consumed
    shared slices (one copy phase per tile); carry streams reuse consumed predecessor slices."""
    m: gl.constexpr = GEOMETRY.M
    h: gl.constexpr = GEOMETRY.H
    streams: gl.constexpr = GEOMETRY.STREAMS
    retention: gl.constexpr = GEOMETRY.HAS_RETENTION
    GRID: gl.constexpr = CONFIG.TARGET_CTAS
    NREADERS: gl.constexpr = len(inputs.reader_grads)
    TWO: gl.constexpr = NREADERS == 2
    ctas_per_group: gl.constexpr = gl.cdiv(h, 256)
    L: gl.constexpr = _GRADIENT_LAYOUT
    residuals = ()
    for stream in gl.static_range(streams):
        residuals += (gl.allocate_shared_memory(R_DESC.dtype, [32, 128], R_DESC.layout),)
    # Staged unscaled BF16 dy; six streams skip it to keep two-CTA residency.
    STAGE_DY: gl.constexpr = DY0_DESC is not None
    ds = None
    if STAGE_DY:
        ds = gl.allocate_shared_memory(DY0_DESC.dtype, [NREADERS * 32, 128], DY0_DESC.layout)
    update_shared = gl.allocate_shared_memory(U_DESC.dtype, [32, 128], U_DESC.layout)

    HALF_BYTES: gl.constexpr = (  # one column half of a tile's TMA copies
        16
        * 128
        * (
            R_DESC.dtype.primitive_bitwidth // 8 * streams
            + 2 * NREADERS * STAGE_DY
            + U_DESC.dtype.primitive_bitwidth // 8
        )
    )
    copy_deps = ((ds,) if STAGE_DY else ()) + (update_shared,) + residuals
    # Carry s > 0 lands on barrier s - 1, the last two complete each half's copy phase; one
    # allocation keeps them all reserved (a dead barrier was once reused for staged gamma).
    barrier_storage = gl.allocate_shared_memory(
        gl.int64, [streams + 1, 1], mbarrier.MBarrierLayout()
    )
    barriers = ()
    for index in gl.static_range(streams + 1):
        mbarrier.init(barrier_storage.index(index), count=1)
        barriers += (barrier_storage.index(index),)
    gl.barrier()  # every init before any shared write
    copy_barrier = barriers[streams - 1]
    copy_barriers = (copy_barrier, barriers[streams])
    LINEAR_SHARED: gl.constexpr = SwizzledSharedLayout(1, 1, 1, [0])
    # Shared projection storage avoids register spills on Nano/Ultra.
    local_projection = gl.allocate_shared_memory(gl.float32, [NREADERS, 16], LINEAR_SHARED)

    GROUPS: gl.constexpr = GRID // ctas_per_group
    total_tiles: gl.constexpr = gl.cdiv(m, 16)
    cta_ticket_counter = outputs.epochs
    row_tile_counter = cta_ticket_counter + 1
    BLOCK: gl.constexpr = 16 + NREADERS * 16 * ctas_per_group
    # Tickets are taken only once resident.
    bid = gl.atomic_add(cta_ticket_counter, 1, sem="relaxed")
    column_tile, group = bid % ctas_per_group, bid // ctas_per_group

    rr0 = gl.arange(0, 16, layout=gl.constexpr(SliceLayout(1, L)))
    cc0 = gl.arange(0, 128, layout=gl.constexpr(SliceLayout(0, L)))
    cc = column_tile * 256 + cc0
    residual_copies, reader_copies, carry_copies = (), (), ()
    for stream in gl.static_range(streams):
        column = stream * h + column_tile * 256
        residual_copies += ((R_DESC, residuals[stream], column),)
        if stream > 0:
            # carry_copies[s - 1] lands carry stream s in consumed predecessor slice s - 1.
            carry_copies += ((CARRY_DESC, residuals[stream - 1], column),)
    if STAGE_DY:
        for reader in gl.static_range(NREADERS):
            reader_copies += (
                ((DY0_DESC, DY1_DESC)[reader], ds.slice(reader * 32, 32), column_tile * 256),
            )
    reader_copies += ((U_DESC, update_shared, column_tile * 256),)
    COEFFICIENTS: gl.constexpr = triton.next_power_of_2(4 * streams)
    cs = gl.allocate_shared_memory(gl.float32, [COEFFICIENTS], LINEAR_SHARED)
    ci = gl.arange(0, COEFFICIENTS, layout=gl.constexpr(BlockedLayout([1], [32], [4], [0])))
    gs = gl.allocate_shared_memory(gl.float32, [NREADERS * 256], LINEAR_SHARED)
    gamma_cc = column_tile * 256 + gl.arange(0, 256, layout=gl.constexpr(SliceLayout(0, L)))
    for reader in gl.static_range(NREADERS):
        saved_gamma = gl.load(inputs.gammas[reader] + gamma_cc, gamma_cc < h, 0)
        gs.slice(reader * 256, 256).store(saved_gamma.to(gl.float32))
    coefficients = gl.load(inputs.coefficients + ci, ci < 4 * streams, 0)
    cs.store(coefficients)
    row_tile = group
    iteration = 0

    # Bootstrap only active groups; gradient slices prefetch subsequent inputs.
    if row_tile < total_tiles:
        gl.barrier()
        mbarrier.expect(copy_barriers[0], HALF_BYTES)
        mbarrier.expect(copy_barriers[1], HALF_BYTES)
        for part in gl.static_range(2):
            _copy_tiles(residual_copies + reader_copies, row_tile, copy_barriers[part], part)

    # Each iteration completes 16 rows in two 16x128 column halves.
    while tl.condition(row_tile < total_tiles, disable_licm=True):
        tag = row_tile + 1
        handoff = outputs.projection + row_tile * BLOCK
        arrival_hint = handoff  # int32 counter in the low half of header word 0.
        next_tile_word = handoff + 1
        projection_words = handoff + 16
        rr = row_tile * 16 + rr0

        # The group leader claims the next row tile, publishes it in this tile's word and
        # warms the claimed tile's block in L2 (its zeroed lines are long evicted by now).
        if column_tile == 0:
            claimed = GROUPS + gl.atomic_add(row_tile_counter, 1, sem="relaxed")
            _publish_word(next_tile_word, claimed, tag, 127)
            _bulk_prefetch_l2(
                outputs.projection + claimed * BLOCK,
                (claimed < total_tiles).to(gl.int32),
                BLOCK * 8,
                1,
            )

        # Wait for column half zero only; half one is awaited inside the reader phase.
        mbarrier.wait(copy_barrier, iteration & 1, deps=copy_deps)
        gl.barrier()
        fence_async_shared()

        # Reader phase: x = sum_s beta t R_s + q u, z = r x, p = sum(dy gamma z) / H per reader.
        rs = ()
        for reader in gl.static_range(NREADERS):
            rs += (gl.load(inputs.rstd[reader] + rr, rr < m, 0),)
        alphas, mixes = ((),) * NREADERS, (0.0,) * NREADERS
        for s in gl.static_range(streams):
            retain = _coefficient(cs, streams + s) if retention else 1.0
            write = _coefficient(cs, s)
            new_alphas, new_mixes = (), ()
            for reader in gl.static_range(NREADERS):
                beta = _coefficient(cs, (2 + reader) * streams + s)
                new_alphas += (alphas[reader] + (beta * retain,),)
                new_mixes += (gl.fma(beta, write, mixes[reader]),)
            alphas, mixes = new_alphas, new_mixes
        projections = (gl.full((16, 128), 0, gl.float32, L),) * NREADERS
        for part in gl.static_range(2):
            if part == 1:
                mbarrier.wait(copy_barriers[1], iteration & 1, deps=copy_deps)
            cols = (cc + part * 128)[None, :]
            mask = rr[:, None] < m
            if h % 256:
                mask &= cols < h
            u = _widen_bf16(update_shared.slice(part * 16, 16).load(L))
            xs = (gl.full((16, 128), 0, gl.float32, L),) * NREADERS
            for s in gl.static_range(streams):
                rv = _load_residual(residuals, cols, part, s, h)
                new_xs = ()
                for reader in gl.static_range(NREADERS):
                    new_xs += (_fma2(alphas[reader][s], rv, xs[reader]),)
                xs = new_xs
            new_projections = ()
            for reader in gl.static_range(NREADERS):
                z = _fma2(mixes[reader], u, xs[reader]) * rs[reader][:, None]
                dy = _load_dy(
                    inputs.reader_grads, ds, reader, part, rr[:, None] * h + cols, mask, False
                )
                gamma = gs.slice(reader * 256 + part * 128, 128).load(
                    gl.constexpr(SliceLayout(0, L))
                )
                new_projections += (projections[reader] + (dy * gamma[None, :]) * z,)
                # Gamma partials: sum over this tile's 16 rows of dy * z.
                gamma_partials = outputs.gamma_partials[reader] + row_tile * h + cc + part * 128
                gl.store(
                    gamma_partials, gl.sum(dy * z, 0), (row_tile * 16 < m) & (cc + part * 128 < h)
                )
            projections = new_projections
        # Publish this CTA's projection partials. In SliceLayout(1, L) each row sum is
        # replicated over tid bits 0-2 and 5-6; replica zero publishes.
        for reader in gl.static_range(NREADERS):
            _publish_word(
                projection_words + _partial_offset(reader, rr0, column_tile, ctas_per_group),
                (gl.sum(projections[reader], 1) / h).to(gl.int32, bitcast=True),
                tag,
                7 | (3 << 5),
            )

        gl.barrier()
        _arrive_hint(arrival_hint)
        _prefetch(inputs.carry, row_tile, column_tile, GEOMETRY)
        # Super one-reader issues carry zero while the projection is reduced.
        if CONFIG.EARLY_CARRY:
            early_rows = rr[:, None]
            early_cols = cc[None, :]
            early_mask = early_rows < m
            if h % 256:
                early_mask &= early_cols < h
            early_incoming_zero = _widen_bf16(
                gl.load(inputs.carry + early_rows * (streams * h) + early_cols, early_mask, 0)
            )

        # Finish the projection: sum the width participants' partials.
        projection_rows: gl.constexpr = CONFIG.PROJECTION_ROWS_PER_WARP
        PL: gl.constexpr = BlockedLayout(
            [1, CONFIG.PROJECTION_VECTOR], [projection_rows, 32 // projection_rows], [4, 1], [1, 0]
        )
        projection_cols: gl.constexpr = triton.next_power_of_2(ctas_per_group)
        pcc = gl.arange(0, projection_cols, layout=gl.constexpr(SliceLayout(0, PL)))
        prr = gl.arange(0, 16, layout=gl.constexpr(SliceLayout(1, PL)))
        partial_valid = (prr[:, None] >= 0) & (pcc[None, :] < ctas_per_group)
        # Padding columns read the zero word, expect tag 0 and contribute +0.
        partial_tag = gl.where(partial_valid, tag, 0)
        _await_hint(arrival_hint, ctas_per_group)
        # ld.volatile is relaxed.sys: morally strong with the producers' stores.
        next_word = gl.load(next_tile_word, volatile=True)
        words0 = projection_words + _partial_offset(0, prr[:, None], pcc[None, :], ctas_per_group)
        words1 = words0 + 16 * ctas_per_group  # Second reader's words follow the first's.
        w0 = gl.load(words0, partial_valid, 0, volatile=True)
        w1 = gl.load(words1, partial_valid, 0, volatile=True) if TWO else w0
        next_rp = _verify_word(next_word, next_tile_word, tag).to(gl.int32)
        # Each thread verifies its own copies (the layout may replicate words across threads).
        w0 = _verify_word(w0, words0, partial_tag)
        w1 = _verify_word(w1, words1, partial_tag) if TWO else w0
        for reader in gl.static_range(NREADERS):
            partial = (w1 if reader else w0).to(gl.int32).to(gl.float32, bitcast=True)
            local_projection.index(reader).store(gl.sum(partial, 1))
        gl.barrier()

        if next_rp < total_tiles:
            # Arm the whole next tile now (each column half receives every predecessor stream
            # plus its dy/U slice, HALF_BYTES); gradient slices issue it as buffers free up.
            mbarrier.expect(copy_barriers[0], HALF_BYTES)
            mbarrier.expect(copy_barriers[1], HALF_BYTES)

        # Gradient phase: dx = r * (dz - z * p); per stream dR = t * dR + sum alpha * dx and du.
        # Two readers start transfers first and reduce controllers as packed FMA chains; one
        # reader releases each slice after its controller reductions.
        OUTPUT_CACHE: gl.constexpr = ".cs" if not TWO else ""
        HELD: gl.constexpr = min(2, streams)  # predecessor slices read before the first barrier
        prefetch_active = next_rp < total_tiles
        stride = gl.cdiv(m, 16) * gl.cdiv(h, 128) * 8  # between controller partial planes
        for part in gl.static_range(2):
            cols = (cc + part * 128)[None, :]
            rows = rr[:, None]
            mask = rows < m
            if h % 256:
                mask &= cols < h
            offset = rows * h + cols
            controller_row = (
                outputs.controller_partials
                + (row_tile * gl.cdiv(h, 128) + column_tile * 2 + part) * 8
            )
            u_rows = controller_row + (1 + retention + NREADERS) * streams * stride
            valid = (row_tile * 16 < m) & (column_tile * 2 + part < gl.cdiv(h, 128))
            refill = (carry_copies, residual_copies, barriers, copy_barriers[part], row_tile)
            if TWO:
                # One dR row pointer per slice; stream offsets become store immediates.
                residual_grad_rows = outputs.residual_grad + (rows * (streams * h) + cols)

            joint_coefficients = ()
            for i in gl.static_range(4 * streams):
                joint_coefficients += (_coefficient(cs, i),)

            u = _widen_bf16(update_shared.slice(part * 16, 16).load(L))
            if CONFIG.EARLY_CARRY and part == 0:
                incoming_zero = (
                    early_incoming_zero  # pylint: disable=possibly-used-before-assignment
                )
            elif TWO and m % 16 == 0 and h % 256 == 0:
                # Every row and column of a scheduled tile is in range.
                incoming_zero = _widen_bf16(gl.load(inputs.carry + rows * (streams * h) + cols))
            else:
                incoming_zero = _widen_bf16(
                    gl.load(inputs.carry + rows * (streams * h) + cols, mask, 0)
                )
            dxs = ()
            update_mixes = ()
            for reader in gl.static_range(NREADERS):
                dy = _load_dy(inputs.reader_grads, ds, reader, part, offset, mask, TWO)
                gamma = gs.slice(reader * 256 + part * 128, 128).load(
                    gl.constexpr(SliceLayout(0, L))
                )[None, :]
                p = gl.gather(
                    local_projection.index(reader).load(gl.constexpr(SliceLayout(1, L))), rr0, 0
                )[:, None]
                r = gl.gather(rs[reader], rr0, 0)[:, None]
                # Packed multiplies round every product exactly like scalar ones.
                dz = _mul2(dy, gamma)
                x = gl.full((16, 128), 0, gl.float32, L)
                update_mix = 0.0
                for s in gl.static_range(streams):
                    R_prev = _load_residual(residuals, cols, part, s, h)
                    beta = joint_coefficients[(2 + reader) * streams + s]
                    retain = joint_coefficients[streams + s] if retention else 1.0
                    x = _fma2(beta * retain, R_prev, x)
                    update_mix = gl.fma(beta, joint_coefficients[s], update_mix)
                x = _fma2(update_mix, u, x)
                dxs += (_mul2(_fma2(_mul2(x, -r), p, dz), r),)
                update_mixes += (update_mix,)

            if not TWO:
                for reader in gl.static_range(NREADERS):
                    _store_controller(u_rows + reader * stride, u, dxs[reader], valid, False)
            du = gl.full((16, 128), 0, gl.float32, L)

            if TWO:
                # Held slices stay in registers, so their refills and the next dy/U start now.
                held = ()
                for first in gl.static_range(HELD):
                    held += (_load_residual(residuals, cols, part, first, h),)
                if streams > 1 or prefetch_active:
                    gl.barrier()
                    fence_async_shared()
                    for first in gl.static_range(HELD):
                        if first + 1 < streams or prefetch_active:
                            _refill_slice(first, *refill, next_rp, part)
                    if prefetch_active:
                        _copy_tiles(reader_copies, next_rp, copy_barriers[part], part)

            for s in gl.static_range(streams):
                if TWO and s < HELD:
                    R_prev_s = held[s]
                else:
                    R_prev_s = _load_residual(residuals, cols, part, s, h)
                # Stream s's own release: before its controller reductions for two readers past
                # HELD and for EARLY_CARRY, after them otherwise.
                if ((TWO and s >= HELD) or CONFIG.EARLY_CARRY) and (
                    s + 1 < streams or prefetch_active
                ):
                    gl.barrier()
                    fence_async_shared()
                    _refill_slice(s, *refill, next_rp, part)
                w = joint_coefficients[s]
                t = joint_coefficients[streams + s] if retention else 1.0
                controller = controller_row + s * stride
                if TWO and s == 0:
                    for reader in gl.static_range(NREADERS):
                        _store_controller(u_rows + reader * stride, u, dxs[reader], valid, True)
                # Raw a[r, s] = sum(Rprev_s * dx_r); the final reduction forms t * a + w * b.
                for reader in gl.static_range(NREADERS):
                    reads = controller + (1 + retention + reader) * streams * stride
                    _store_controller(reads, R_prev_s, dxs[reader], valid, TWO)
                if not TWO and not CONFIG.EARLY_CARRY and (s + 1 < streams or prefetch_active):
                    gl.barrier()
                    fence_async_shared()
                    _refill_slice(s, *refill, next_rp, part)

                wide = rows * (streams * h) + s * h + cols
                if s == 0:
                    incoming_dR_s = incoming_zero
                else:
                    tile = carry_copies[s - 1][1]
                    mbarrier.wait(barriers[s - 1], part & 1, deps=(tile,))
                    incoming_dR_s = _widen_bf16(tile.slice(part * 16, 16).load(L))
                if TWO and s > 0 and prefetch_active:
                    # Release the consumed carry slice to the next tile's predecessor s - 1.
                    gl.barrier()
                    fence_async_shared()
                    _copy_tiles((residual_copies[s - 1],), next_rp, copy_barriers[part], part)
                if retention:
                    _store_controller(
                        controller + streams * stride, incoming_dR_s, R_prev_s, valid, TWO
                    )
                if h % 256:
                    incoming_dR_s = gl.where(cols < h, incoming_dR_s, 0)
                # Differentiate the rounded folded coefficients alpha = FP32(beta * t).
                d_previous = _mul2(t, incoming_dR_s) if retention else incoming_dR_s
                for reader in gl.static_range(NREADERS):
                    alpha = joint_coefficients[(2 + reader) * streams + s] * t
                    d_previous = _fma2(alpha, dxs[reader], d_previous)

                if not TWO:
                    gl.barrier()
                    fence_async_shared()
                    # Start the next tile into the consumed column half.
                    if prefetch_active:
                        pending = reader_copies if s == 0 else (residual_copies[s - 1],)
                        _copy_tiles(pending, next_rp, copy_barriers[part], part)

                if TWO:
                    dR_pointer = residual_grad_rows + s * h
                else:
                    dR_pointer = outputs.residual_grad + wide
                d_previous = d_previous.to(outputs.residual_grad.dtype.element_ty)  # RN for BF16
                gl.store(dR_pointer, d_previous, mask, cache_modifier=OUTPUT_CACHE)
                # du accumulates the carry here and the readers' q * dx after the streams.
                du = _fma2(w, incoming_dR_s, du)
                _store_controller(controller, incoming_dR_s, u, valid, TWO)

            for reader in gl.static_range(NREADERS):
                du = _fma2(update_mixes[reader], dxs[reader], du)

            gl.store(outputs.update_grad + offset, du, mask, cache_modifier=OUTPUT_CACHE)

        row_tile = next_rp
        iteration += 1
