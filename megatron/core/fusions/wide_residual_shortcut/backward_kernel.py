# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Backward kernels for the aligned folded-reader forward graph.

Read backward for configuration, allocation, launches, and final gradient reductions.
_backward pipelines reader projections, predecessor reads and gradients within
width-spanning groups, reusing shared slices as their consumers finish.
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
    _descriptor,
    _fma2,
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


class _ReaderPartials(NamedTuple):
    projection: object
    gamma: tuple


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


# Launch keys whose compile and residency check already passed.
_RESIDENCY = {}


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
    prepare_wgrad=None,
    on_columns_ready=None,
):
    """Execute backward; both updates of a shortcut add share one gradient tensor.

    Returns ``(grads, columns)``: grads holds the residual gradient, one update gradient per
    branch update, the controller gradients (write, retain, then one read per reader) and one
    RMSNorm weight gradient per reader. Only the reader count shapes the kernel.

    Explicit config supports correctness/stress tests. on_columns_ready submits
    dependent weight GEMMs before the fused kernel, which never reads the columns.
    """
    n, streams, updates = saved.reader_count, saved.num_streams, saved.update_count
    m, h = saved.update.shape
    validate_num_streams(streams)
    if m <= 0 or m % 32 or h < 32 or h > 16384 or h % 32:
        raise ValueError(
            "shortcut backward requires positive rows divisible by 32 and H=32..16384, step 32"
        )
    if n not in (1, 2):
        raise ValueError("backward requires one or two readers")
    if updates not in (1, 2):
        raise ValueError("backward requires one update, or two summed by a shortcut add")
    if len(dys) != n:
        raise ValueError("backward requires one gradient slot per reader")
    if previous.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("backward requires an FP32 or BF16 predecessor residual")
    if carry is None:
        carry = torch.zeros_like(previous)
    if carry.dtype != previous.dtype:
        raise ValueError("the residual gradient must have the residual's dtype")
    dys = tuple(torch.zeros_like(saved.update) if dy is None else dy for dy in dys)
    if not qualified_device(torch.cuda.get_device_properties(previous.device)):
        raise ValueError("shortcut backward requires a qualified GB300")
    # Resolve the column routes and build any missing column operands.
    routes = prepare_wgrad
    if routes is None:
        routes = save_columns if config == "auto" else False
    grid = config
    nt = triton.cdiv(h, 256)
    if grid != "auto" and (type(grid) is not int or grid <= 0 or grid % nt):
        raise ValueError("shortcut grid must contain positive whole participant groups")
    routes = (routes,) + (False,) * (n - 1) if isinstance(routes, bool) else routes
    if len(routes) != n:
        raise ValueError("shortcut backward requires one column route per reader")
    columns = (None,) * n if columns is None else tuple(columns)
    if len(columns) != n:
        raise ValueError("one prebuilt column operand per reader is required")
    missing = tuple(needed and column is None for needed, column in zip(routes, columns))
    if any(missing):
        from .replay_kernel import replay_columns

        generated = replay_columns(previous, saved, readers=missing)
        columns = tuple(old if old is not None else new for old, new in zip(columns, generated))
    # The columns are complete and the kernel does not read them: submit the dependent
    # weight GEMMs (and their reduce-scatters) first so communication overlaps the kernel.
    if on_columns_ready is not None:
        on_columns_ready(columns)
    knobs = get_backward_knobs(m, h, streams, n, grid=grid)
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
    # Write-once handoff words (see _publish_word): one block per row tile, a
    # 128-byte header (arrival hint, next-row-tile word) and one projection word per
    # (reader, row, column tile). Every tagged word is written at most once per
    # launch; each hint counts its row tile's producers. The admission and row-tile
    # counters are the two int32 scheduler words.
    block = 16 + n * 16 * nt
    if tiles * block + 16 >= 2**31:
        raise ValueError("handoff workspace exceeds signed 32-bit indexing")
    # Separate allocations keep each workspace an ordinary invocation-owned tensor.
    epochs = empty((2,), dtype=torch.int32)
    pp = empty((tiles * block,), dtype=torch.int64)

    slice_shape = (16, 128)
    kw = dict(
        inputs=_BackwardInputs(saved.coefficients, saved.rstd, saved.gammas, carry, dys),
        GEOMETRY=_Geometry(m, h, streams, saved.has_retention),
        CONFIG=_BackwardConfig(*(gl.constexpr(value) for value in knobs)),
        partials=_ReaderPartials(pp, gp),
        outputs=_BackwardOutputs(grads[0], grads[1], controllers),
        scheduler=epochs,
        CARRY_DESC=_descriptor(carry, slice_shape),
        R_DESC=_descriptor(previous, slice_shape),
        DY0_DESC=_descriptor(dys[0] if streams < 6 else None, slice_shape),
        DY1_DESC=_descriptor(dys[1] if streams < 6 and n == 2 else None, slice_shape),
        U_DESC=_descriptor(saved.update, slice_shape),
        maxnreg=knobs.MAXNREG,
        num_warps=4,
        enable_fp_fusion=True,
    )
    # Compile and check residency once per specialization: the knobs carry the grid
    # and every launch option; pointer-alignment variants share the shared-memory size.
    key = (previous.device, previous.dtype, carry.dtype, saved.update.dtype)
    key += (*(dy.dtype for dy in dys), m, h, streams, saved.has_retention, knobs)
    if key not in _RESIDENCY:
        kernel = _backward.warmup(**kw, grid=(grid,))
        kernel._init_handles()
        # Triton's cluster query assumes 128 threads, matching num_warps=4.
        # Ticket admission needs one complete participant group, not the grid.
        limit = triton.runtime.driver.active.utils.cuOccupancyMaxActiveClusters(
            kernel.function, kernel.metadata.shared, 1
        )
        if nt > limit:
            raise ValueError(f"required residency {nt} exceeds capacity: {limit}")
        _RESIDENCY[key] = limit
    # Reset on every invocation, including graph replay: every handoff word must
    # start with tag 0 so an old launch's word can never satisfy a current poll.
    # Stream order makes this memset visible to the whole kernel.
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
    # Per controller (write, retention, then each reader's read): its partial plane
    # and the raw rows/coefficients its final reduction combines. The raw write
    # partial is sum(u * incoming_s); reader terms use the raw b rows only here.
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


# Backward arithmetic, partial reductions, and pipeline scheduling.


# Fence-free cross-CTA handoff (projection partials and the next row tile).
#
# Each handoff value travels in one naturally aligned 64-bit word:
#     word = value_bits | tag << 32,  tag = row_tile + 1 (never 0).
# Memory-model argument (PTX ISA memory consistency model):
# 1. Write-once: words are indexed by row tile, and each row tile is processed by
#    exactly one group (initial tile = group index, later tiles from the atomic
#    row-tile counter), so every word has at most one producer store per launch and
#    is never reused within a launch: no write-after-read hazard, no ring buffer.
# 2. Initial value: the host zeroes all words before every launch (also captured in
#    CUDA graphs); stream order makes tag 0 the only prior value any thread sees.
# 3. Atomicity: the producer's st.relaxed.gpu.b64 and the consumer's strong loads
#    (ld.relaxed.gpu.b64, or ld.volatile = ld.relaxed.sys, possibly as a vector whose
#    64-bit elements are each a scalar access) are morally strong: both strong,
#    scopes containing both CTAs, same naturally aligned 8 bytes. So each load
#    returns either the zero word or the complete produced word; seeing the
#    expected tag implies seeing the value stored with it.
# 4. No other data crosses CTAs inside this kernel (dR, du, gamma and controller
#    partials are consumed by later launches), so no release/acquire fence is
#    needed. In particular, no MEMBAR.GPU waits for the previous tile's dR/du stores.
# 5. The per-row-tile arrival count (relaxed red.add, no fence) and the L2 warm-up
#    of each claimed row tile's block are performance hints only: the hint is not
#    ordered after the words, so consumers still check the tag of every register
#    copy they sum, reloading any word that lags. Correctness never depends on them.
#    A layout that replicates words across threads must check each thread's own
#    copies: a CTA-wide reduction counts only one replica, and every replica stores
#    its row sums (see the projection reduction in _backward).
# 6. Progress: strong stores become visible eventually, and a group's CTAs are
#    co-resident by ticket admission.


@g.jit
def _publish_word(pointer, value, tag, OWNER_MASK: gl.constexpr):
    """Store value|tag<<32 once from threads whose (tid & OWNER_MASK) == 0."""
    return gl.inline_asm_elementwise(
        """
        { // Begin write-once handoff publication.
            .reg .pred owner; // Select one thread per distinct replicated value.
            .reg .u32 tid; // CTA thread index.
            .reg .b64 word; // Tagged 64-bit handoff word.
            mov.u32 tid,%tid.x; // Read this thread's index.
            and.b32 tid,tid,$4; // Keep the bits that index layout replicas.
            setp.eq.u32 owner,tid,0; // Replica zero owns the store.
            mov.b64 word,{$2,$3}; // Low half: value bits; high half: nonzero tag.
            @owner st.relaxed.gpu.global.b64 [$1],word; // Single-copy-atomic publish, no fence.
            mov.u32 $0,0; // Provide the required dummy output.
        } // End write-once handoff publication.
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
        { // Begin arrival hint.
            .reg .pred leader; // One thread per CTA counts the arrival.
            .reg .u32 tid; // CTA thread index.
            mov.u32 tid,%tid.x; // Read this thread's index.
            setp.eq.u32 leader,tid,0; // Thread zero is the leader.
            @leader red.relaxed.gpu.global.add.u32 [$1],1; // Hint only: orders nothing.
            mov.u32 $0,0; // Provide the required dummy output.
        } // End arrival hint.
        """,
        "=r,l",
        [pointer],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@g.jit
def _await_hint(pointer, expected):
    """Spin until the arrival hint reaches expected. Performance only: the tagged
    words are verified afterwards, so a hint seen before its CTA's words is safe."""
    return gl.inline_asm_elementwise(
        """
        { // Begin arrival-hint poll; labels are local to this scope.
            .reg .pred ready; // Whether every producer CTA has arrived.
            .reg .u32 count; // Observed arrival count.
        WAIT: // Poll one scalar per warp.
            ld.relaxed.gpu.global.u32 count,[$1]; // Strong GPU-scope load of the hint.
            setp.ge.u32 ready,count,$2; // Compare with the participant count.
            @!ready bra WAIT; // Keep polling until all producers arrived.
            mov.u32 $0,count; // Return the observed count.
        } // End arrival-hint poll.
        """,
        "=r,l,n",
        [pointer, expected],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@g.jit
def _verify_word(word, pointer, tag):
    """Return word once its high half equals tag, re-polling only this thread's lagging word.

    Every thread checks its own copies: the consumer layout may replicate a word across
    warps, and a CTA-wide reduction counts only one replica of each element."""
    return gl.inline_asm_elementwise(
        """
        { // Begin per-thread handoff verification; labels are local to this scope.
            .reg .pred ready; // Whether the observed tag is the expected one.
            .reg .b32 value,seen; // Low and high halves of the observed word.
            .reg .b64 word; // Most recently observed word.
            mov.b64 word,$1; // Start from the vector load (usually ready).
        CHECK: // Re-examine the most recent observation.
            mov.b64 {value,seen},word; // Split the single-copy-atomic word.
            setp.eq.u32 ready,seen,$3; // The producer's tag implies its value.
            @ready bra DONE; // Use the value stored together with the tag.
            ld.relaxed.gpu.global.b64 word,[$2]; // Re-poll this word with a strong load.
            bra CHECK; // Check the new observation.
        DONE: // The word is the producer's word for this tag.
            mov.b64 $0,word; // Return the verified word.
        } // End per-thread handoff verification.
        """,
        "=l,l,l,r",
        [word, pointer, tag],
        dtype=gl.int64,
        is_pure=False,
        pack=1,
    )


@g.jit
def _missing_words(w0, w1, tag, TWO: gl.constexpr):
    """Count words (both readers, whole CTA) whose tag is not yet the expected one."""
    missing = gl.where((w0 >> 32).to(gl.int32) == tag, 0, 1)
    if TWO:
        missing += gl.where((w1 >> 32).to(gl.int32) == tag, 0, 1)
    return gl.sum(gl.sum(missing, 1), 0)


@g.jit
def _bulk_prefetch_l2(pointer, valid, BYTES: gl.constexpr, THREADS: gl.constexpr):
    """Warm BYTES at pointer in L2 from each valid thread below THREADS (a hint only)."""
    return gl.inline_asm_elementwise(
        """
        { // Begin L2 bulk prefetch.
            .reg .pred p,owner; // Declare validity and ownership predicates.
            .reg .u32 tid; // Declare the CTA thread index.
            mov.u32 tid,%tid.x; // Read this thread's index.
            setp.lt.u32 owner,tid,$4; // Select the issuing threads.
            setp.ne.u32 p,$2,0; // Check the validity mask.
            and.pred p,p,owner; // Require both validity and ownership.
            @p cp.async.bulk.prefetch.L2.global [$1],$3; // Warm the range in L2.
            mov.u32 $0,0; // Provide the required dummy output.
        } // End L2 bulk prefetch.
        """,
        "=r,l,r,n,n",
        [pointer, valid, BYTES, THREADS],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@g.jit
def _await_word(word, pointer, tag):
    """Return the low 32 bits of the word once its high half equals tag."""
    return gl.inline_asm_elementwise(
        """
        { // Begin handoff poll; labels are local to this scope.
            .reg .pred ready; // Whether the observed tag is the expected one.
            .reg .b32 value,seen; // Low and high halves of the observed word.
            .reg .b64 word; // Most recently observed word.
            mov.b64 word,$1; // Start from the optimistic load (usually ready).
        POLL: // Re-examine the most recent observation.
            mov.b64 {value,seen},word; // Split the single-copy-atomic word.
            setp.eq.u32 ready,seen,$3; // The producer's tag implies its value.
            @ready bra DONE; // Use the value stored together with the tag.
            ld.relaxed.gpu.global.b64 word,[$2]; // Re-poll with a strong GPU-scope load.
            bra POLL; // Check the new observation.
        DONE: // The value is the producer's value for this tag.
            mov.b32 $0,value; // Return the value bits.
        } // End handoff poll.
        """,
        "=r,l,l,r",
        [word, pointer, tag],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@g.jit
def _mul2(a, b):
    # Pair adjacent independent raw-dot products; preserve each FP32 RN result.
    return gl.inline_asm_elementwise(
        """
        {
            .reg .b64 a,b,r;
            mov.b64 a,{$2,$3};
            mov.b64 b,{$4,$5};
            mul.rn.f32x2 r,a,b;
            mov.b64 {$0,$1},r;
        }
        """,
        "=f,=f,f,f,f,f",
        [a, b],
        dtype=gl.float32,
        is_pure=True,
        pack=2,
    )


@g.jit
def _store_controller(destination, value, valid):
    """Reduce each 16×128 slice into eight fixed lane-group controller partials."""
    owned = gl.permute(gl.reshape(value, (4, 4, 4, 8, 4)), (2, 1, 3, 0, 4))
    per_thread = gl.sum(gl.reshape(owned, (128, 16)), 1)
    total = gl.sum(gl.reshape(per_thread, (8, 16)), 1)
    gl.store(destination + gl.arange(0, 8, layout=total.type.layout), total, valid)


@g.jit
def _widen_bf16(x):
    # Exact widening to FP32; BF16 bits become the upper half of the FP32 word.
    if x.dtype == gl.bfloat16:
        return (x.to(gl.uint16, bitcast=True).to(gl.uint32) << 16).to(gl.float32, bitcast=True)
    return x.to(gl.float32)


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
def _store_controller_dot(destination, a, b, valid):
    """_store_controller(destination, a * b, valid) with one packed FMA chain per thread.

    Each thread accumulates its 16 products in two column lanes (eight fma.rn.f32x2) and
    adds the lanes; the lane-group reduction and the eight partial slots are unchanged.
    """
    pa = _thread_pairs(a)
    pb = _thread_pairs(b)
    acc = _mul2(pa[0], pb[0])
    for k in gl.static_range(1, 8):
        acc = _fma2(pa[k], pb[k], acc)
    per_thread = gl.sum(acc, 1)
    total = gl.sum(gl.reshape(per_thread, (8, 16)), 1)
    gl.store(destination + gl.arange(0, 8, layout=total.type.layout), total, valid)


@g.jit
def _coefficient(cs, index):
    indices = gl.full((1,), index, gl.int32, gl.constexpr(BlockedLayout([1], [32], [4], [0])))
    return gl.sum(cs.gather(indices, 0), 0)


@g.jit
def _partial_offset(reader, rr, cc, NT: gl.constexpr):
    TILE: gl.constexpr = min(16, NT & -NT)
    return reader * 16 * NT + (cc // TILE * 16 + rr) * TILE + cc % TILE


@g.jit
def _prefetch(P, rp, cp, GEOMETRY: gl.constexpr):
    """Warm this tile's carry rows in L2 without replacing any TMA wait."""
    m: gl.constexpr = GEOMETRY.M
    h: gl.constexpr = GEOMETRY.H
    streams: gl.constexpr = GEOMETRY.STREAMS
    # One thread per row warms the row's 256 columns of each stream.
    ROWS: gl.constexpr = 16
    ELEMENT: gl.constexpr = P.dtype.element_ty.primitive_bitwidth // 8
    ROW_BYTES: gl.constexpr = 256 * ELEMENT
    i = gl.arange(0, ROWS, layout=gl.constexpr(BlockedLayout([1], [32], [4], [0])))
    rr = rp * 16 + i
    cc = cp * 256
    base = P + rr * (streams * h) + cc
    # Stream zero is loaded directly, without a bootstrap transfer, so warm every stream.
    for stream in gl.static_range(streams):
        valid = rr < m
        if h % 256:
            # Keep prefetch hints within the logical residual stream.
            valid &= cc + 256 <= h
        _bulk_prefetch_l2(base + stream * h, valid.to(gl.int32), ROW_BYTES, ROWS)
        if h % 256:
            # Preserve the original single 128-column hint at a short final tile.
            # Full rows use one 1024-byte request instead of two 512-byte requests.
            tail = (rr < m) & (cc + 128 <= h) & (cc + 256 > h)
            _bulk_prefetch_l2(base + stream * h, tail.to(gl.int32), 128 * ELEMENT, ROWS)


@g.jit
def _copy_tiles(plan, row_tile, barrier, part: gl.constexpr):
    """Copy column half part of each slice into its rows of the [32, 128] storage."""
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
def _reader_partials(
    inputs,
    partials,
    residuals,
    ds,
    update_shared,
    gs,
    cs,
    projection_words,
    tag,
    rs0,
    rs1,
    row_tile,
    column_tile,
    cc,
    GEOMETRY: gl.constexpr,
    ctas_per_group: gl.constexpr,
    half_barrier,
    phase,
    deps,
):
    """Projection and gamma partials of one or two readers, one 16x128 half at a time.

    The caller has waited for half zero; half one's transfer (the last one issued by
    the previous tile) completes on half_barrier while half zero is reduced.

    Each half is consumed directly in the gradient layout L (no logical [16,256]
    reshape or layout conversion). Every R/U element is read from shared memory
    once and feeds both readers' FMAs in forward order:
        x_r = sum_s fma(beta_r[s] * t[s], R_s, x_r);  x_r = fma(q_r, u, x_r)
    Projection: each thread accumulates dz*z over both halves, then one row sum.
    Gamma partials sum dy*z over the 16 rows of each half.

    With one reader, reader one's loads alias reader zero's (rs1 = rs0, slice zero of
    dy and gamma) and its stores are skipped, so every reader-one value is dead and
    compiled out: the single reader keeps exactly its own FMA order.
    """
    TWO: gl.constexpr = len(inputs.reader_grads) == 2
    m: gl.constexpr = GEOMETRY.M
    h: gl.constexpr = GEOMETRY.H
    streams: gl.constexpr = GEOMETRY.STREAMS
    retention: gl.constexpr = GEOMETRY.HAS_RETENTION
    L: gl.constexpr = _GRADIENT_LAYOUT
    # Folded reader coefficients alpha_r[s] = FP32(beta_r[s] * t[s]) and the rounded
    # update mix q_r = sum_fma(beta_r[s] * w[s]), as in forward and the gradient phase.
    alpha0, alpha1 = (), ()
    mix0 = 0.0
    mix1 = 0.0
    for s in gl.static_range(streams):
        retain = _coefficient(cs, streams + s) if retention else 1.0
        write = _coefficient(cs, s)
        beta0 = _coefficient(cs, 2 * streams + s)
        beta1 = _coefficient(cs, 3 * streams + s)
        alpha0 += (beta0 * retain,)
        alpha1 += (beta1 * retain,)
        mix0 = gl.fma(beta0, write, mix0)
        mix1 = gl.fma(beta1, write, mix1)
    kr = gl.arange(0, 16, layout=gl.constexpr(SliceLayout(1, L)))
    rows = (row_tile * 16 + kr)[:, None]
    r0 = rs0[:, None]
    r1 = rs1[:, None]
    projection0 = gl.full((16, 128), 0, gl.float32, L)
    projection1 = gl.full((16, 128), 0, gl.float32, L)
    for part in gl.static_range(2):
        if part == 1:
            mbarrier.wait(half_barrier, phase, deps=deps)
        cols = (cc + part * 128)[None, :]
        mask = rows < m
        if h % 256:
            mask &= cols < h
        offset = rows * h + cols
        u = update_shared.slice(part * 16, 16).load(L).to(gl.float32)
        x0 = gl.full((16, 128), 0, gl.float32, L)
        x1 = gl.full((16, 128), 0, gl.float32, L)
        for s in gl.static_range(streams):
            rv = _load_residual(residuals, cols, part, s, h)
            x0 = _fma2(alpha0[s], rv, x0)
            x1 = _fma2(alpha1[s], rv, x1)
        x0 = _fma2(mix0, u, x0)
        x1 = _fma2(mix1, u, x1)
        # z[j] = r * x[j]
        z0 = x0 * r0
        z1 = x1 * r1
        if ds is None:
            dy0 = gl.load(inputs.reader_grads[0] + offset, mask, 0).to(gl.float32)
            dy1 = gl.load(inputs.reader_grads[-1] + offset, mask, 0).to(gl.float32)
        else:
            dy0 = ds.slice(part * 16, 16).load(L).to(gl.float32)
            dy1 = ds.slice(TWO * 32 + part * 16, 16).load(L).to(gl.float32)
        gamma0 = gs.slice(part * 128, 128).load(gl.constexpr(SliceLayout(0, L)))[None, :]
        gamma1 = gs.slice(TWO * 256 + part * 128, 128).load(gl.constexpr(SliceLayout(0, L)))
        gamma1 = gamma1[None, :]
        # dz[j] = dy[j] * gamma[j]; accumulate dz*z per element across both halves.
        projection0 += (dy0 * gamma0) * z0
        projection1 += (dy1 * gamma1) * z1
        # Gamma partials: sum over this tile's 16 rows of dy*z.
        gamma_mask = row_tile * 16 < m
        if h % 256:
            gamma_mask &= cc + part * 128 < h
        gamma_cols = row_tile * h + cc + part * 128
        gl.store(partials.gamma[0] + gamma_cols, gl.sum(dy0 * z0, 0), gamma_mask)
        if TWO:
            gl.store(partials.gamma[1] + gamma_cols, gl.sum(dy1 * z1, 0), gamma_mask)
    # Projection partials p_r = sum_j(dz[j] * z[j]) / H over this CTA's 256 columns.
    # Publish as write-once tagged words (see _publish_word). In SliceLayout(1, L) each
    # row's sum is replicated over the 8 column lanes (tid bits 0-2) and the 4 column
    # warps (tid bits 5-6); replica zero (tid & 0x67 == 0) publishes.
    OWNER_MASK: gl.constexpr = 7 | (3 << 5)
    # Within the row tile's block, each reader owns 16 x NT words.
    _publish_word(
        projection_words + _partial_offset(0, kr, column_tile, ctas_per_group),
        (gl.sum(projection0, 1) / h).to(gl.int32, bitcast=True),
        tag,
        OWNER_MASK,
    )
    if TWO:
        _publish_word(
            projection_words + _partial_offset(1, kr, column_tile, ctas_per_group),
            (gl.sum(projection1, 1) / h).to(gl.int32, bitcast=True),
            tag,
            OWNER_MASK,
        )


@g.jit
def _backward(
    inputs,
    partials,
    outputs,
    scheduler,
    GEOMETRY: gl.constexpr,
    CONFIG: gl.constexpr,
    CARRY_DESC,
    R_DESC,
    DY0_DESC,
    DY1_DESC,
    U_DESC,
):
    """Pipeline projections, predecessor reads and gradients within full-width CTA groups.

    Resident CTAs take admission tickets; one group must fit residency. Each group
    reserves its first tile, so partial groups cannot starve earlier complete ones.
    Readers consume 16x256 tiles; gradients consume two 16x128 column halves.
    R/dy/U prefetch replaces only dead slices, with one copy-barrier phase per tile
    and no transfer after a group's final tile. Carry streams have separate barriers
    and reuse preceding residual tiles once their consumers finish.

    inputs holds the coefficients, saved norm statistics, carry and incoming gradients.
    Partials, outputs and scheduler arguments carry invocation-owned workspaces.
    """
    m: gl.constexpr = GEOMETRY.M
    h: gl.constexpr = GEOMETRY.H
    streams: gl.constexpr = GEOMETRY.STREAMS
    retention: gl.constexpr = GEOMETRY.HAS_RETENTION
    GRID: gl.constexpr = CONFIG.TARGET_CTAS
    NREADERS: gl.constexpr = len(inputs.reader_grads)
    TWO: gl.constexpr = NREADERS == 2
    # Admission tickets form groups spanning H. One whole group must fit in
    # residency; independent groups need not be resident together.
    ctas_per_group: gl.constexpr = gl.cdiv(h, 256)
    L: gl.constexpr = _GRADIENT_LAYOUT
    residuals = ()
    for stream in gl.static_range(streams):
        residuals += (gl.allocate_shared_memory(R_DESC.dtype, [32, 128], R_DESC.layout),)
    # Stage unscaled BF16 dy; each reader/gradient phase applies gamma.
    # Six-stream cases omit dy staging to preserve shared-memory headroom
    # and two-CTA residency, including wide row groups.
    STAGE_DY: gl.constexpr = DY0_DESC is not None
    ds = None
    if STAGE_DY:
        ds = gl.allocate_shared_memory(DY0_DESC.dtype, [NREADERS * 32, 128], DY0_DESC.layout)
    # Stage U; reuse each staged slice only after its gradient load has finished.
    update_shared = gl.allocate_shared_memory(U_DESC.dtype, [32, 128], U_DESC.layout)

    COPY_BYTES: gl.constexpr = (
        32
        * 128
        * (
            R_DESC.dtype.primitive_bitwidth // 8 * streams
            + 2 * NREADERS * STAGE_DY
            + U_DESC.dtype.primitive_bitwidth // 8
        )
    )
    copy_deps = ((ds,) if STAGE_DY else ()) + (update_shared,) + residuals
    # Stream zero's incoming carry is loaded directly into registers. Each later carry
    # stream s has barrier s - 1, permitting independent slice transfers; the last two
    # barriers complete each tile's copy phase per physical column half.
    # One allocation keeps every barrier reserved for the whole kernel: a barrier the
    # allocator saw as dead after mbarrier.init had its bytes reused for staged gamma,
    # and a late init write (warp 0 delayed under co-resident kernels) clobbered it.
    barrier_storage = gl.allocate_shared_memory(
        gl.int64, [streams + 1, 1], mbarrier.MBarrierLayout()
    )
    barriers = ()
    for index in gl.static_range(streams + 1):
        mbarrier.init(barrier_storage.index(index), count=1)
        barriers += (barrier_storage.index(index),)
    # Order every init before any warp writes shared memory.
    gl.barrier()
    copy_barrier = barriers[streams - 1]
    copy_barriers = (copy_barrier, barriers[streams])
    HALF_BYTES: gl.constexpr = COPY_BYTES // 2
    LINEAR_SHARED: gl.constexpr = SwizzledSharedLayout(1, 1, 1, [0])
    # Shared projection storage avoids register spills on Nano/Ultra.
    local_projection = gl.allocate_shared_memory(gl.float32, [NREADERS, 16], LINEAR_SHARED)

    # Each CTA owns 256 columns; a group spans the complete width.
    GROUPS: gl.constexpr = GRID // ctas_per_group
    # Each group stops after its own last tile and issues no next-tile transfer
    # on it, so no TMA operation outlives its storage.
    total_tiles: gl.constexpr = gl.cdiv(m, 16)
    cta_ticket_counter = scheduler
    row_tile_counter = cta_ticket_counter + 1
    # Write-once tagged words: one 128-byte-aligned block of int64 words per row
    # tile (header: arrival hint, next row tile; then projection partials).
    BLOCK: gl.constexpr = 16 + NREADERS * 16 * ctas_per_group
    # Assign tickets only after a CTA becomes resident. A partially admitted
    # group cannot prevent earlier complete groups from making progress.
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
        saved_gamma = (
            gl.load(inputs.gammas[reader] + gamma_cc, gamma_cc < h, 0)
            if h % 256
            else gl.load(inputs.gammas[reader] + gamma_cc)
        )
        gs.slice(reader * 256, 256).store(saved_gamma.to(gl.float32))
    coefficients = gl.load(inputs.coefficients + ci, ci < 4 * streams, 0)
    cs.store(coefficients)
    # Reserve each ticket group's first tile before dynamic work allocation.
    # A partial group holds only this tile while complete groups can finish.
    row_tile = group
    iteration = 0

    # Bootstrap only active groups; gradient slices prefetch subsequent inputs.
    # Publish barrier initialization before issuing the first R/dy/U transfers.
    if row_tile < total_tiles:
        gl.barrier()
        mbarrier.expect(copy_barriers[0], HALF_BYTES)
        mbarrier.expect(copy_barriers[1], HALF_BYTES)
        for part in gl.static_range(2):
            _copy_tiles(residual_copies + reader_copies, row_tile, copy_barriers[part], part)

    # Each iteration completes 16 rows in two physical 16x128 column halves.
    while tl.condition(row_tile < total_tiles, disable_licm=True):
        # Handoff words of this row tile carry tag row_tile + 1 (see _publish_word).
        tag = row_tile + 1
        handoff = partials.projection + row_tile * BLOCK
        arrival_hint = handoff  # int32 counter in the low half of header word 0.
        next_tile_word = handoff + 1
        projection_words = handoff + 16
        rr = row_tile * 16 + rr0

        # The leader of each row group claims the group's next row tile and publishes
        # it in this row tile's write-once word; the group polls it after the projection.
        # It also warms the claimed tile's block in L2: the memset's zero lines are long
        # evicted by then, and partial-sector writes/atomics on them would miss to DRAM
        # on the per-tile critical path.
        if column_tile == 0:
            claimed = GROUPS + gl.atomic_add(row_tile_counter, 1, sem="relaxed")
            _publish_word(next_tile_word, claimed, tag, 127)
            _bulk_prefetch_l2(
                partials.projection + claimed * BLOCK,
                (claimed < total_tiles).to(gl.int32),
                BLOCK * 8,
                1,
            )

        # Wait here for column half zero only; half one is awaited inside the reader.
        mbarrier.wait(copy_barrier, iteration & 1, deps=copy_deps)
        # The previous iteration consumed the second slice before its
        # existing CTA reuse barrier. Start slice zero ahead of reader stage.
        gl.barrier()
        fence_async_shared()

        # Compute the input gradient to the norm, dx:
        # 1. x[j]  = sum_s(beta[s] * R[s, j])
        # 2. z[j]  = r * x[j]
        # 3. dz[j] = dy[j] * gamma[j]
        # 4. p     = sum_j(dz[j] * z[j]) / H
        # 5. dx[j] = r * (dz[j] - z[j] * p)

        # Load inverse RMS values saved during forward.
        rs0 = gl.load(inputs.rstd[0] + rr, rr < m, 0)
        rs1 = gl.load(inputs.rstd[1] + rr, rr < m, 0) if TWO else rs0
        # Both readers share each R/U load; see _reader_partials.
        _reader_partials(
            inputs,
            partials,
            residuals,
            ds,
            update_shared,
            gs,
            cs,
            projection_words,
            tag,
            rs0,
            rs1,
            row_tile,
            column_tile,
            cc,
            GEOMETRY,
            ctas_per_group,
            copy_barriers[1],
            iteration & 1,
            copy_deps,
        )

        # Projection partials are already published as tagged words: no release
        # fence, so this CTA never waits here for its dR/du stores. The relaxed
        # arrival hint (after every warp issued its stores) only paces the polls.
        gl.barrier()
        _arrive_hint(arrival_hint)
        _prefetch(inputs.carry, row_tile, column_tile, GEOMETRY)
        # Super one-reader issues carry zero while the independent projection is reduced.
        if CONFIG.EARLY_CARRY:
            early_rows = rr[:, None]
            early_cols = cc[None, :]
            early_mask = early_rows < m
            if h % 256:
                early_mask &= early_cols < h
            early_incoming_zero = _widen_bf16(
                gl.load(inputs.carry + early_rows * (streams * h) + early_cols, early_mask, 0)
            )

        # Finish the projection reduction.
        projection_rows: gl.constexpr = CONFIG.PROJECTION_ROWS_PER_WARP
        PL: gl.constexpr = BlockedLayout(
            [1, CONFIG.PROJECTION_VECTOR], [projection_rows, 32 // projection_rows], [4, 1], [1, 0]
        )
        projection_cols: gl.constexpr = triton.next_power_of_2(ctas_per_group)
        pcc = gl.arange(0, projection_cols, layout=gl.constexpr(SliceLayout(0, PL)))
        # Wait cheaply for all width participants, then load the words with strong
        # loads (ld.volatile is relaxed.sys: morally strong with the producers'
        # st.relaxed.gpu, and each 64-bit vector element is single-copy atomic) and
        # verify every tag the thread sums; reload only if a word lagged.
        prr = gl.arange(0, 16, layout=gl.constexpr(SliceLayout(1, PL)))
        partial_valid = (prr[:, None] >= 0) & (pcc[None, :] < ctas_per_group)
        # Padding columns read the zero word, expect tag 0 and contribute +0.
        partial_tag = gl.where(partial_valid, tag, 0)
        _await_hint(arrival_hint, ctas_per_group)
        next_word = gl.load(next_tile_word, volatile=True)
        words0 = projection_words + _partial_offset(0, prr[:, None], pcc[None, :], ctas_per_group)
        words1 = words0 + 16 * ctas_per_group  # Second reader's words follow the first's.
        w0 = gl.load(words0, partial_valid, 0, volatile=True)
        w1 = gl.load(words1, partial_valid, 0, volatile=True) if TWO else w0
        next_rp = _await_word(next_word, next_tile_word, tag)
        # PL replicates words across threads when it spans more than 16 x projection_cols.
        # A CTA-wide count sees only one replica of each word, so a replicated layout
        # verifies every thread's own copies; an exact tiling keeps the cheap count.
        REPLICATED: gl.constexpr = (projection_rows * 4 > 16) or (
            (32 // projection_rows) * CONFIG.PROJECTION_VECTOR > projection_cols
        )
        if REPLICATED:
            w0 = _verify_word(w0, words0, partial_tag)
            if TWO:
                w1 = _verify_word(w1, words1, partial_tag)
            else:
                w1 = w0
        else:
            missing = _missing_words(w0, w1, partial_tag, TWO)
            while missing != 0:
                w0 = gl.load(words0, partial_valid, 0, volatile=True)
                w1 = gl.load(words1, partial_valid, 0, volatile=True) if TWO else w0
                missing = _missing_words(w0, w1, partial_tag, TWO)
        for reader in gl.static_range(NREADERS):
            # The low half of a verified word is the producer's FP32 partial.
            partial = (w1 if reader else w0).to(gl.int32).to(gl.float32, bitcast=True)
            # Sum the width participants' partials into the full projection.
            local_projection.index(reader).store(gl.sum(partial, 1))
        gl.barrier()

        if next_rp < total_tiles:
            # Gradient slices issue the next inputs as shared buffers become free.
            # Account for the entire next tile here, then wait next loop.
            # Do not arm or issue this transfer on the group's final tile.
            # Every next-tile copy issued while gradient part p runs targets column half p
            # and arrives on copy_barriers[p]; in both the one-reader order and the two-reader
            # (PAIRED) transfer-first order, each half receives all predecessor streams
            # plus its dy/U slice, exactly HALF_BYTES.
            mbarrier.expect(copy_barriers[0], HALF_BYTES)
            mbarrier.expect(copy_barriers[1], HALF_BYTES)

        # Shared memory at the start of the gradient phase:
        #     residuals[s]     predecessor stream s, two 16x128 column halves
        #     ds, gs           reader gradients and gamma (dy may be unstaged)
        #     cs               write, retention, and reader coefficients
        #     local_projection full-width p for each reader and token
        # Registers hold rs0/rs1 for all 16 rows; stream zero's carry is loaded directly.
        OUTPUT_CACHE: gl.constexpr = ".cs" if not TWO else ""
        # Two-reader gradient schedule. Each part reads its shared slices once, then
        # one barrier releases the carry targets and the next dy/U slice; afterwards one
        # barrier per stream releases the consumed carry together with the next stream's
        # predecessor slice. Controller reductions run from registers after those transfers,
        # as packed per-thread FMA chains, so they do not delay them.
        PAIRED: gl.constexpr = TWO
        # Predecessor slices read before the part's first barrier.
        HELD: gl.constexpr = min(2, streams)
        prefetch_active = next_rp < total_tiles
        coefficient_stride = gl.cdiv(m, 16) * gl.cdiv(h, 128) * 8
        # Now calculate the grad through the residuals, use 16-row slices to limit
        # live values and overlap transfers into shared slices after their last use.
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
            valid = (row_tile * 16 < m) & (column_tile * 2 + part < gl.cdiv(h, 128))
            refill = (carry_copies, residual_copies, barriers, copy_barriers[part], row_tile)
            if PAIRED:
                # One dR row pointer per slice; stream offsets become store immediates.
                residual_grad_rows = outputs.residual_grad + (rows * (streams * h) + cols)

            # Keep only scalar coefficients live across this gradient slice.
            joint_coefficients = ()
            for i in gl.static_range(4 * streams):
                joint_coefficients += (_coefficient(cs, i),)

            # Load the branch update and initialize its gradient. Accumulate du
            # across residual streams while each stream's carry arrives.
            if PAIRED:
                u = _widen_bf16(update_shared.slice(part * 16, 16).load(L))
            else:
                u = update_shared.slice(part * 16, 16).load(L).to(gl.float32)
            if CONFIG.EARLY_CARRY and part == 0:
                # Assigned above under the same compile-time EARLY_CARRY guard.
                incoming_zero = (
                    early_incoming_zero  # pylint: disable=possibly-used-before-assignment
                )
            elif PAIRED and m % 16 == 0 and h % 256 == 0:
                # Every row and column of a scheduled tile is in range.
                incoming_zero = _widen_bf16(gl.load(inputs.carry + rows * (streams * h) + cols))
            else:
                incoming_zero = _widen_bf16(
                    gl.load(inputs.carry + rows * (streams * h) + cols, mask, 0)
                )
            # The dx section computes dx for each reader. Inputs:
            #     Rprev[s,j]  Predecessor residual stream s
            #     beta[s]   This reader's mixing coefficients
            #     dy[j]     This reader's incoming gradient
            #     gamma[j]  This reader's RMSNorm weight
            #     r         Saved inverse RMS for this token
            #     p         Full-width projection already computed
            # 1. dz[j] = dy[j] * gamma[j]
            # 2. x[j] = sum_s(alpha[s] * Rprev[s,j]) + q * u[j]
            # 3. z[j] = r * x[j]
            # 4. dx[j] = r * (dz[j] - z[j] * p)
            # Both readers share each Rprev[s] register load.
            dxs = ()
            update_mixes = ()
            for reader in gl.static_range(NREADERS):
                if ds is None:
                    dy = gl.load(inputs.reader_grads[reader] + offset, mask, 0).to(gl.float32)
                elif PAIRED:
                    dy = _widen_bf16(ds.slice(reader * 32 + part * 16, 16).load(L))
                else:
                    dy = ds.slice(reader * 32 + part * 16, 16).load(L).to(gl.float32)
                gamma = gs.slice(reader * 256 + part * 128, 128).load(
                    gl.constexpr(SliceLayout(0, L))
                )[None, :]
                p = gl.gather(
                    local_projection.index(reader).load(gl.constexpr(SliceLayout(1, L))), rr0, 0
                )[:, None]
                r = gl.gather(rs0 if reader == 0 else rs1, rr0, 0)[:, None]
                # Packed multiplies (PAIRED) round every product exactly like scalar ones.
                dz = _mul2(dy, gamma) if PAIRED else dy * gamma
                x = gl.full((16, 128), 0, gl.float32, L)
                update_mix = 0.0
                for s in gl.static_range(streams):
                    R_prev = _load_residual(residuals, cols, part, s, h)
                    beta = joint_coefficients[(2 + reader) * streams + s]
                    retain = joint_coefficients[streams + s] if retention else 1.0
                    x = _fma2(beta * retain, R_prev, x)
                    update_mix = gl.fma(beta, joint_coefficients[s], update_mix)
                x = _fma2(update_mix, u, x)
                if PAIRED:
                    dxs += (_mul2(_fma2(_mul2(x, -r), p, dz), r),)
                else:
                    dxs += (_fma2(-(x * r), p, dz) * r,)
                update_mixes += (update_mix,)

            if not PAIRED:
                for reader in gl.static_range(NREADERS):
                    _store_controller(
                        controller_row
                        + ((1 + retention + NREADERS) * streams + reader) * coefficient_stride,
                        _mul2(u, dxs[reader]),
                        valid,
                    )
            du = gl.full((16, 128), 0, gl.float32, L)

            if PAIRED:
                # Held Rprev slices stay in registers, so their transfers (carry s+1 or the
                # next tile's last predecessor) and the next dy/U slice start together.
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
                        # The prologue was the last reader of this dy/U slice.
                        _copy_tiles(reader_copies, next_rp, copy_barriers[part], part)

            # Calculate for each stream a 16 row gradient
            for s in gl.static_range(streams):
                if PAIRED and s < HELD:
                    R_prev_s = held[s]
                else:
                    R_prev_s = _load_residual(residuals, cols, part, s, h)
                # Stream s's own release: before its controller reductions for PAIRED
                # streams past HELD and for EARLY_CARRY, after them otherwise.
                if ((PAIRED and s >= HELD) or CONFIG.EARLY_CARRY) and (
                    s + 1 < streams or prefetch_active
                ):
                    gl.barrier()
                    fence_async_shared()
                    _refill_slice(s, *refill, next_rp, part)
                w = joint_coefficients[s]
                t = joint_coefficients[streams + s] if retention else 1.0
                controller = controller_row + s * coefficient_stride
                # Store raw a[r,s] = sum(Rprev_s * dx_r). The final reduction
                # combines t*a + w*b without materializing any current residual.
                if PAIRED:
                    # The transfers above are already in flight; reduce from registers.
                    if s == 0:
                        for reader in gl.static_range(NREADERS):
                            _store_controller_dot(
                                controller_row
                                + ((1 + retention + NREADERS) * streams + reader)
                                * coefficient_stride,
                                u,
                                dxs[reader],
                                valid,
                            )
                    for reader in gl.static_range(NREADERS):
                        _store_controller_dot(
                            controller + (1 + retention + reader) * streams * coefficient_stride,
                            R_prev_s,
                            dxs[reader],
                            valid,
                        )
                else:
                    for reader in gl.static_range(NREADERS):
                        _store_controller(
                            controller + (1 + retention + reader) * streams * coefficient_stride,
                            _mul2(R_prev_s, dxs[reader]),
                            valid,
                        )
                if not PAIRED and not CONFIG.EARLY_CARRY and (s + 1 < streams or prefetch_active):
                    gl.barrier()
                    fence_async_shared()
                    _refill_slice(s, *refill, next_rp, part)

                wide = rows * (streams * h) + s * h + cols

                # Calculate the upstream residual gradient
                if s == 0:
                    incoming_dR_s = incoming_zero
                else:
                    tile = carry_copies[s - 1][1]
                    mbarrier.wait(barriers[s - 1], part & 1, deps=(tile,))
                    # Load the downstream residual gradient after its transfer.
                    incoming_dR_s = _widen_bf16(tile.slice(part * 16, 16).load(L))
                if PAIRED and s > 0 and prefetch_active:
                    # Release the consumed carry slice to the next tile's predecessor s-1.
                    gl.barrier()
                    fence_async_shared()
                    _copy_tiles((residual_copies[s - 1],), next_rp, copy_barriers[part], part)
                if retention:
                    if PAIRED:
                        _store_controller_dot(
                            controller + streams * coefficient_stride,
                            incoming_dR_s,
                            R_prev_s,
                            valid,
                        )
                    else:
                        _store_controller(
                            controller + streams * coefficient_stride,
                            _mul2(incoming_dR_s, R_prev_s),
                            valid,
                        )
                if h % 256:
                    incoming_dR_s = gl.where(cols < h, incoming_dR_s, 0)
                # Differentiate the actual rounded folded reader coefficients.
                # alpha = FP32(beta * t), rather than scaling a combined dR.
                if PAIRED and retention:
                    d_previous = _mul2(t, incoming_dR_s)  # Same rounding as t * dR.
                else:
                    d_previous = t * incoming_dR_s
                for reader in gl.static_range(NREADERS):
                    alpha = joint_coefficients[(2 + reader) * streams + s] * t
                    d_previous = _fma2(alpha, dxs[reader], d_previous)

                if not PAIRED:
                    gl.barrier()
                    fence_async_shared()
                    # Start the next tile into the consumed physical column half.
                    if prefetch_active:
                        pending = reader_copies if s == 0 else (residual_copies[s - 1],)
                        _copy_tiles(pending, next_rp, copy_barriers[part], part)

                if PAIRED:
                    dR_pointer = residual_grad_rows + s * h
                else:
                    dR_pointer = outputs.residual_grad + wide
                d_previous = d_previous.to(outputs.residual_grad.dtype.element_ty)  # RN for BF16
                gl.store(dR_pointer, d_previous, mask, cache_modifier=OUTPUT_CACHE)
                # Reader update terms use rounded q = sum_fma(beta * w).
                # Accumulate only incoming carry here; add q * dx after streams.
                du = _fma2(w, incoming_dR_s, du)
                if PAIRED:
                    _store_controller_dot(controller, incoming_dR_s, u, valid)
                else:
                    _store_controller(controller, _mul2(incoming_dR_s, u), valid)

            for reader in gl.static_range(NREADERS):
                du = _fma2(update_mixes[reader], dxs[reader], du)

            gl.store(outputs.update_grad + offset, du, mask, cache_modifier=OUTPUT_CACHE)

        row_tile = next_rp
        iteration += 1
