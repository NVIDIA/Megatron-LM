# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fused backward of a seam: one persistent kernel plus the final gradient reductions.

Groups of CTAs span the width (256 columns each) and walk 16-row tiles, pipelining reader
projections, predecessor/carry TMA transfers and the residual/update/controller gradients.
"""

from itertools import product
from typing import NamedTuple

import torch
import triton
import triton.language as tl
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout, SwizzledSharedLayout
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, mbarrier, tma

from .configs import choose_backward_layout, get_backward_knobs, qualified_device
from .kernel_utils import (
    KERNEL_FORMAT_TAGS,
    OUTPUT_FORMATS,
    _coefficient,
    _descriptor,
    _e8m0_scale,
    _fma2,
    _mul2,
    _round_bf16,
    _volatile_asm,
    _widen_bf16,
    allocate_operand,
    cdiv,
    named_jit,
    reduce_jobs,
    validate_num_streams,
)

# Triton renamed this API in 3.9; both names use the same TMA operation.
_tma_load = tma.async_load if hasattr(tma, "async_load") else tma.async_copy_global_to_shared

_GRADIENT_LAYOUT = gl.constexpr(BlockedLayout([1, 4], [4, 8], [1, 4], [1, 0]))
_ROWS = gl.constexpr(SliceLayout(1, _GRADIENT_LAYOUT))  # one value per slice row
_COLUMNS = gl.constexpr(SliceLayout(0, _GRADIENT_LAYOUT))  # one value per slice column


class _BackwardInputs(NamedTuple):
    coefficients: object
    rstd: tuple
    gammas: tuple
    carry: object
    reader_grads: tuple  # per reader, one or two consumer gradients


class _BackwardOutputs(NamedTuple):
    residual_grad: object
    update_grad: object  # BF16 du, or None when only MXFP8 du is emitted
    # MXFP8 du for the producer Linears' dgrad/wgrad: (rowwise data, rowwise scales, columnwise
    # data, columnwise scales), or None; and where an even row tile's BF16 du waits for its pair:
    # None in shared memory, else one 4096-value global slot per CTA.
    update_mxfp8: object
    update_stash: object
    controller_partials: object
    projection: object  # tagged handoff words (see the protocol below)
    gamma_partials: tuple
    epochs: object  # admission and row-tile counters


class _Geometry(NamedTuple):
    M: int
    H: int
    STREAMS: int
    HAS_RETENTION: bool


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
    main_grads=None,
    update_formats=None,
):
    """Gradients of one seam from its ``SavedWrite`` and predecessor residual.

    ``dys`` holds one gradient per reader, or a pair for a reader with two consumers (summed in
    FP32 in the kernel). Returns ``(grads, columns)``: the residual gradient, one update gradient
    per branch update (a shortcut add's share one tensor per format), the controller gradients
    (write, retain, one read per reader) and one RMSNorm weight gradient per reader.
    ``save_columns`` (a bool for reader zero, or one per reader) also returns columnwise MXFP8
    reader operands, reusing any given in ``columns``; ``on_columns_ready`` receives them before
    the fused kernel launches, so dependent weight GEMMs overlap it. ``config`` may set an
    explicit grid (tests).
    ``main_grads`` (one entry per controller and RMSNorm weight gradient, in return order) may
    give ``(main_grad, parameter dtype)``: that gradient is accumulated into ``main_grad`` as
    autograd's cast to the parameter dtype and DDP's add would, and returned as None.
    ``update_formats`` gives each update's gradient format: "bf16" (default) or "mxfp8", the
    rowwise and columnwise MXFP8 operand its producer Linear's dgrad and wgrad take (E4M3, TE's
    GEMM-swizzled scales), quantized from the BF16 gradient as TE would.
    """
    n, streams, updates = len(saved.gammas), saved.num_streams, saved.update_count
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
    gradients = []
    extra = None
    for index, dy in enumerate(dys):
        if isinstance(dy, tuple) and len(dy) != 2:
            raise ValueError("a reader takes one gradient or a pair")
        first, second = dy if isinstance(dy, tuple) else (dy, None)
        if second is not None:
            if index != n - 1 or first is None:
                raise ValueError("only the last reader may take a second gradient, after its first")
            if (
                second.shape != first.shape
                or second.dtype != first.dtype
                or not second.is_contiguous()
            ):
                raise ValueError("a reader's second gradient must match its first, contiguous")
            extra = second
        if first is None:
            first = torch.zeros_like(saved.update, dtype=torch.bfloat16)
        gradients.append((first,) if second is None else (first, second))
    gradients = tuple(gradients)
    if not qualified_device(torch.cuda.get_device_properties(previous.device)):
        raise ValueError("shortcut backward requires a qualified GB300")
    formats = ("bf16",) * updates if update_formats is None else tuple(update_formats)
    if len(formats) != updates or any(f not in ("bf16", "mxfp8") for f in formats):
        raise ValueError("update_formats needs one 'bf16' or 'mxfp8' entry per update")
    # Accumulated parameter gradients: round to the parameter dtype (2) or not (1), then add.
    targets = (None,) * (2 + 2 * n) if main_grads is None else tuple(main_grads)
    if len(targets) != 2 + 2 * n:
        raise ValueError("main_grads needs one entry per controller and norm weight gradient")
    if any(t is not None and t[1] not in (torch.bfloat16, torch.float32) for t in targets):
        raise ValueError("accumulated gradients need a BF16 or FP32 parameter")
    accumulate = tuple(0 if t is None else 2 if t[1] == torch.bfloat16 else 1 for t in targets)
    nt = cdiv(h, 256)
    if config != "auto" and (type(config) is not int or config <= 0 or config % nt):
        raise ValueError("shortcut grid must contain positive whole participant groups")
    routes = (
        (save_columns,) + (False,) * (n - 1)
        if isinstance(save_columns, bool)
        else tuple(save_columns)
    )
    columns = (None,) * n if columns is None else tuple(columns)
    if len(routes) != n or len(columns) != n:
        raise ValueError("shortcut backward requires one column route and operand per reader")
    missing = tuple(needed and column is None for needed, column in zip(routes, columns))
    if any(missing):
        from .replay_kernel import replay_interval

        # Replay materializes every missing reader together.
        generated = replay_interval(previous, (saved,), prepare_columns=(missing,))[1][0]
        columns = tuple(old if old is not None else new for old, new in zip(columns, generated))
    if on_columns_ready is not None:
        on_columns_ready(columns)
    bf16 = previous.dtype == torch.bfloat16
    grid, maxnreg, spare = get_backward_knobs(
        m, h, streams, n, grid=config, bf16=bf16, second=extra is not None
    )
    planes = (1 + n + saved.has_retention) * streams
    shares = sum(_share_planes.fn(planes + n, k) for k in (0, 1)) * 512  # paired shares' bytes
    stash_shared, tma_store, paired = choose_backward_layout(
        spare, h, n, bf16=bf16, mxfp8="mxfp8" in formats, paired_shares=shares
    )
    # Outputs and scratch belong to this invocation; a shortcut add's updates share one gradient
    # (per format).
    empty = saved.coefficients.new_empty
    du = empty((m, h), dtype=torch.bfloat16) if "bf16" in formats else None
    du_mx, mx = None, None
    if "mxfp8" in formats:
        du_mx = allocate_operand((m, h), previous.device, columnwise=True)
        mx = (
            du_mx._rowwise_data,
            du_mx._rowwise_scale_inv,
            du_mx._columnwise_data,
            du_mx._columnwise_scale_inv,
        )
        if m % 128 or h % 128:
            for scale in mx[1::2]:  # padded scales stay zero, as the forward's
                scale.zero_()
    residual_grad = empty((m, streams * h), dtype=previous.dtype)
    sizes = saved.controller_sizes + (h,) * n
    parameters = tuple(
        (empty((size,)) if target is None else target[0]) if size else None
        for size, target in zip(sizes, targets)
    )
    # Eight partials per controller plane and (row tile, column tile or half).
    p = cdiv(m, 16) * cdiv(h, 256 if paired else 128) * 8
    gp = tuple(empty((cdiv(m, 16), h)) for _ in range(n))
    controllers_scratch = empty((planes + n, p))
    chunks = empty((planes + n, cdiv(p, 1024)))

    # Per row tile: a 16-word header (arrival hint, next row tile) and one projection word per
    # (reader, row, column tile); see the handoff protocol below. Its word count bounds every
    # 32-bit counter of the launch (tickets, claimed tiles, arrivals).
    tiles, block = cdiv(m, 16), 16 + n * 16 * nt
    if max(grid, tiles) * block + 16 >= 2**31:
        raise ValueError("handoff workspace and counters exceed signed 32-bit capacity")
    # One zeroed workspace: the handoff words, then the admission and row-tile counters.
    workspace = empty((tiles * block + 1,), dtype=torch.int64)
    pp, epochs = workspace[: tiles * block], workspace[tiles * block :].view(torch.int32)

    # The packed TMA tile stages BF16 reader gradients. Six streams or FP32 dy use global loads.
    staged = streams < 6 and all(group[0].dtype == torch.bfloat16 for group in gradients)
    # Reset on every invocation (also under graph replay) so every handoff word starts at tag 0.
    workspace.zero_()
    du_formats = "both" if len(set(formats)) == 2 else formats[0]
    BACKWARD_KERNELS[n, extra is not None, du_formats][(grid,)](
        inputs=_BackwardInputs(saved.coefficients, saved.rstd, saved.gammas, carry, gradients),
        GEOMETRY=_Geometry(m, h, streams, saved.has_retention),
        GRID=grid,
        PAIRED=paired,
        outputs=_BackwardOutputs(
            residual_grad,
            du,
            mx,
            None if stash_shared or mx is None else empty((grid * 4096,), dtype=torch.bfloat16),
            controllers_scratch,
            pp,
            gp,
            epochs,
        ),
        CARRY_DESC=_descriptor(carry),
        R_DESC=_descriptor(previous),
        # Triton's launcher requires tensor descriptors as top-level arguments.
        DY0_DESC=_descriptor(gradients[0][0] if staged else None),
        DY1_DESC=_descriptor(gradients[1][0] if staged and n == 2 else None),
        DY2_DESC=_descriptor(extra if staged else None),
        U_DESC=_descriptor(saved.update),
        DR_DESC=_descriptor(residual_grad if tma_store else None),
        DU_DESC=_descriptor(du if tma_store else None),
        maxnreg=maxnreg,
        num_warps=4,
        enable_fp_fusion=True,
    )

    # Per stream, A[r] = sum(previous * dx[r]). B[r] = sum(update * dx[r]) is shared.
    # The primary partials occupy one [streams, chunks] plane per controller.
    retention = saved.has_retention
    primary = chunks[:planes].view(1 + retention + n, streams, -1)
    a, b = primary[1 + retention :].unbind(), chunks[planes:].unbind()
    coefficients = saved.coefficients.view(8, streams)
    write, retain = coefficients[0], coefficients[1] if retention else None
    betas = coefficients[2 : 2 + n].unbind()
    # Write adds sum(beta * B), retention adds sum(beta * A), read forms t * A + w * B.
    controllers = (
        (primary[0], tuple(zip(b, betas)), None),
        (primary[1], tuple(zip(a, betas)), None),
        *((ar, ((br, write),), retain) for ar, br in zip(a, b)),
    )
    final = [
        (partial, out, 0, 8, 0, None, None, mode)
        for partial, out, mode in zip(gp, parameters[-n:], accumulate[-n:])
    ]
    for index, (source, terms, weight) in enumerate(controllers):
        if (out := parameters[index]) is not None:
            final.append(
                (source, out, 1, 1, 0, coefficients[4 + index], (terms, weight), accumulate[index])
            )
    reduce_jobs([(controllers_scratch, chunks, 1, 1, 1024, None, None, 0)])
    reduce_jobs(final)
    updates_out = (du if fmt == "bf16" else du_mx for fmt in formats)
    returned = (out if target is None else None for out, target in zip(parameters, targets))
    return (residual_grad, *updates_out, *returned), columns


# Fence-free cross-CTA handoff (projection partials, next row tile): each value travels in one
# aligned 64-bit word ``value_bits | (row_tile + 1) << 32``, written once per launch (one owning
# group per tile, no reuse) after the host zeroed it. The producer's st.relaxed.gpu.b64 and the
# consumers' strong 64-bit loads are morally strong, so seeing the tag implies seeing its value.
# No other data crosses CTAs, so no fence (no MEMBAR.GPU wait for the tile's dR/du stores) is
# needed. Arrival counts and L2 warm-ups only pace the wait: every thread checks the tag of
# each copy it sums (layouts may replicate words, so a CTA-wide count would see one replica).
# Progress comes from ticket admission (co-resident groups) and eventually visible stores.
#
# Inline PTX: the per-thread poll is what Gluon cannot express. A Gluon wait loop must reduce its
# condition across the CTA, so each check becomes a CTA-wide reduction; with relaxed
# gl.atomic_xchg/atomic_add publishing and such loops, the two-reader backward measured ~7%
# slower. The PTX also stores only from the owning replica (``OWNER_MASK``) with plain relaxed
# stores rather than atomic exchanges.


@g.jit
def _publish_word(pointer, value, tag, OWNER_MASK: gl.constexpr):
    """Store value|tag<<32 (st.relaxed.gpu, morally strong) once, from the threads whose
    (tid & OWNER_MASK) == 0: one replica of each word in the layout."""
    return _volatile_asm(
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
    )


@g.jit
def _arrive_hint(pointer):
    """Count this CTA's publication with one relaxed reduction from thread 0 (no fence)."""
    return _volatile_asm(
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
    )


@g.jit
def _await_hint(pointer, expected):
    """Spin on relaxed loads until the arrival hint reaches ``expected`` (pacing only; the
    words themselves are verified by ``_verify_word``)."""
    return _volatile_asm(
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
    )


@g.jit
def _verify_word(word, pointer, tag):
    """Return word once its high half equals tag, each thread re-polling only its own copy
    (no CTA-wide reduction per check)."""
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
def _gradient_slice(pointer, tile, start: gl.constexpr, offset, mask):
    if tile is None:
        value = gl.load(pointer + offset, mask, 0).to(gl.float32)
    else:
        value = _widen_bf16(tile.slice(start, 16).load(_GRADIENT_LAYOUT))
    return value


@g.jit
def _load_dy(inputs, ds, ds2, reader: gl.constexpr, part: gl.constexpr, offset, mask):
    """Sum the reader's consumer gradients in FP32."""
    grads = inputs.reader_grads[reader]
    dy = _gradient_slice(grads[0], ds, reader * 32 + part * 16, offset, mask)
    if len(grads) == 2:
        dy += _gradient_slice(grads[1], ds2, part * 16, offset, mask)
    return dy


@g.jit
def _thread_pairs(value):
    """Eight per-thread (128, 2) column pairs of a 16x128 slice, in _controller_partial order."""
    owned = gl.permute(gl.reshape(value, (4, 4, 4, 8, 4)), (2, 1, 3, 0, 4))
    # Per-thread index a*4 + e = (a_hi, a_lo, e_hi, e_lo); e_lo pairs adjacent columns. Peel
    # e_hi, a_lo, then a_hi off the last axis.
    halves = gl.split(gl.permute(gl.reshape(owned, (128, 2, 2, 2, 2)), (0, 4, 1, 2, 3)))
    pairs = ()
    for half in gl.static_range(2):
        quarters = gl.split(halves[half])
        for quarter in gl.static_range(2):
            pairs += gl.split(quarters[quarter])
    return pairs


@g.jit
def _controller_partial(a, b):
    """This thread's share of sum(a * b) over a 16x128 slice: its products accumulated as one
    packed FMA chain, then its pair summed; _store_partial reduces the shares into eight fixed
    lane-group controller partials (stores are deferred to the end of each stream)."""
    pa = _thread_pairs(a)
    pb = _thread_pairs(b)
    acc = _mul2(pa[0], pb[0])
    for k in gl.static_range(1, 8):
        acc = _fma2(pa[k], pb[k], acc)
    return gl.sum(acc, 1)


@g.jit
def _store_partial(destination, partial, valid):
    """Reduce _controller_partial shares into eight lane-group partials and store them."""
    total = gl.sum(gl.reshape(partial, (8, 16)), 1)
    gl.store(destination + gl.arange(0, 8, layout=total.type.layout), total, valid)


@triton.constexpr_function
def _share_planes(planes, buffer):
    """Planes in buffer 0 or 1 of a pair's controller shares (512 B per plane): two power-of-two
    buffers, the first the largest power of two below ``planes + 1``."""
    first = 1 << (planes.bit_length() - 1)  # integer-only: cheap as a host call (``.fn``)
    return first if buffer == 0 else 1 << max(planes - first - 1, 0).bit_length()


@g.jit
def _defer(queued, a, b, PLANE: gl.constexpr, at, PART: gl.constexpr):
    """Queue this thread's share of sum(a * b) for controller plane PLANE, stored at
    ``row + PLANE * stride`` for ``at = (row, stride, valid[, shares])``. With ``shares``,
    paired halves leave half zero's share in slot PLANE and queue the pair's sum from half one
    (``valid`` zeroes a half past H)."""
    share = _controller_partial(a, b)
    destination = at[0] + PLANE * at[1]
    if len(at) == 3:
        queued += ((destination, share),)
    else:
        share = gl.where(at[2], share, 0)
        FIRST: gl.constexpr = at[3][0].shape[0] // 128
        if PLANE < FIRST:
            slot = at[3][0].slice(PLANE * 128, 128)
        else:
            slot = at[3][1].slice((PLANE - FIRST) * 128, 128)
        if PART == 0:
            slot.store(share)
        else:
            queued += ((destination, slot.load(share.type.layout) + share),)
    return queued


@g.jit
def _pick(cs, scalars, INDEX: gl.constexpr):
    """Coefficient INDEX: a uniform scalar when loaded once (one reader), else the shared table."""
    return scalars[INDEX] if len(scalars) else _coefficient(cs, INDEX)


@g.jit
def _partial_offset(reader, rr, cc, NT: gl.constexpr):
    TILE: gl.constexpr = min(16, NT & -NT)
    return reader * 16 * NT + (cc // TILE * 16 + rr) * TILE + cc % TILE


@g.jit
def _prefetch_warp_rows(base, rows, ROW: gl.constexpr, STRIDE: gl.constexpr, BYTES: gl.constexpr):
    """Warm BYTES in L2 at ``base + (warp * 4 + ROW) * STRIDE`` from lane 0 of each of the four
    warps when that tile row is below ``rows`` (a hint). The address is uniform within a warp,
    so the bulk prefetch stays one instruction (per-thread addresses make the compiler loop).

    Inline PTX: Gluon has no bulk L2 prefetch (``cp.async.bulk.prefetch.L2``). Warming the next
    claimed tile measured 4-13% of the backward (e.g. hidden 10240: 512 -> 580 us without it).
    """
    return _volatile_asm(
        """
        {
            .reg .pred p, q;
            .reg .u32 tid, lane, row;
            .reg .u64 offset, address;
            mov.u32 tid,%tid.x;
            and.b32 lane,tid,31;
            shr.u32 row,tid,5;
            mad.lo.u32 row,row,4,$3;
            setp.lt.u32 q,row,$2;
            setp.eq.and.u32 p,lane,0,q;
            mul.wide.u32 offset,row,$4;
            add.s64 address,$1,offset;
            @p cp.async.bulk.prefetch.L2.global [address],$5;
            mov.u32 $0,0;
        }
        """,
        "=r,l,r,n,n,n",
        [base, rows, ROW, STRIDE, BYTES],
    )


@g.jit
def _prefetch(P, rp, cp, GEOMETRY: gl.constexpr):
    """Warm this tile's carry rows (every stream's 256 columns) in L2: each warp four rows (a
    width's partial last column tile gets no hint)."""
    h: gl.constexpr = GEOMETRY.H
    streams: gl.constexpr = GEOMETRY.STREAMS
    ELEMENT: gl.constexpr = P.dtype.element_ty.primitive_bitwidth // 8
    cc = cp * 256
    rows = gl.minimum(GEOMETRY.M - rp * 16, 16)
    if h % 256:
        rows = gl.where(cc + 256 <= h, rows, 0)
    for stream in gl.static_range(streams):
        base = P + rp * 16 * (streams * h) + stream * h + cc
        for r in gl.static_range(4):
            _prefetch_warp_rows(base, rows, r, streams * h * ELEMENT, 256 * ELEMENT)


@g.jit
def _in_range(rows, cols, GEOMETRY: gl.constexpr):
    """The slice elements inside the [M, H] tensor."""
    mask = rows < GEOMETRY.M
    if GEOMETRY.H % 256:
        mask &= cols < GEOMETRY.H
    return mask


@g.jit
def _load_carry_zero(carry, rows, cols, GEOMETRY: gl.constexpr, MASKED: gl.constexpr):
    """Carry stream zero of a slice, from global memory (later streams arrive by TMA)."""
    pointer = carry + rows * (GEOMETRY.STREAMS * GEOMETRY.H) + cols
    if MASKED:
        return _widen_bf16(gl.load(pointer, _in_range(rows, cols, GEOMETRY), 0))
    return _widen_bf16(gl.load(pointer))


@g.jit
def _tma_store(DESC, slot, value, row, column):
    """Store one 16x128 slice through a ring slot and a TMA bulk store (rows past M clip)."""
    slot.store(value)
    fence_async_shared()
    # The issuer's earlier bulk stores have read their slots (the other ring slot, written
    # next, among them) before the barrier releases any thread to write it.
    tma.store_wait(0)
    gl.barrier()
    tma.async_copy_shared_to_global(DESC, [row, column], slot)


@g.jit
def _emit_update_mxfp8(
    mx,
    stash,
    du,
    row_tile,
    column,
    PART: gl.constexpr,
    SHARED: gl.constexpr,
    M: gl.constexpr,
    H: gl.constexpr,
):
    """Quantize a 16x128 du slice (its BF16 rounding, as TE quantizes the BF16 gradient; column
    half PART, first column ``column``) to MXFP8 in TE's GEMM layouts. Rowwise 1x32 blocks now;
    columnwise 32x1 blocks span a row-tile pair, so an even tile stashes its BF16 slice (shared
    memory, or a per-CTA global slot each thread rereads itself) and the odd tile quantizes both.
    Row tiles are 16 aligned rows and slices 128 aligned columns, so every scale index
    (kernel_utils._swizzled_scale_index) splits into a per-tile scalar plus a per-thread constant.
    """
    row_data, row_scales, col_data, col_scales = mx
    L: gl.constexpr = du.type.layout
    EVEN: gl.constexpr = H % 256 == 0  # every slice column inside H (rows always are)
    r = gl.arange(0, 16, layout=gl.constexpr(SliceLayout(1, L)))[:, None]
    cc = gl.arange(0, 128, layout=gl.constexpr(SliceLayout(0, L)))
    c = cc[None, :]
    mask = (c + column < H) if not EVEN else None
    local = r * H + c
    base = row_tile * 16 * H + column
    y = _widen_bf16(_round_bf16(du))
    # Rowwise: a warp holds each 32-column block of four rows.
    blocks = gl.reshape(y, (16, 4, 32))
    exponent, bits = _e8m0_scale(gl.max(gl.abs(blocks), 2))
    blocks, block_scale = gl.broadcast(blocks, gl.expand_dims(bits.to(gl.float32, bitcast=True), 2))
    q = gl.reshape(
        _mul2(blocks, block_scale).to(gl.float8e4nv).to(gl.uint8, bitcast=True), (16, 128)
    )
    gl.store(row_data + base + local, gl.convert_layout(q, L), mask)
    SL: gl.constexpr = exponent.type.layout
    br = gl.arange(0, 16, layout=gl.constexpr(SliceLayout(1, SL)))[:, None]
    bb = gl.arange(0, 4, layout=gl.constexpr(SliceLayout(0, SL)))[None, :]
    # Scale index of row row_tile * 16 + br, block column // 32 + bb.
    row_scale = (row_tile // 8 * gl.cdiv(H, 128) + column // 128) * 512
    row_scale += row_tile % 2 * 256 + row_tile % 8 // 2 * 4
    gl.store(
        row_scales + row_scale + (br * 16 + bb),
        exponent.to(gl.uint8),
        (column + bb * 32 < H) if not EVEN else None,
    )
    # Columnwise: the pair's 32 rows of each column share one scale.
    if row_tile % 2 == 0:
        if SHARED:
            stash.slice(PART * 16, 16).store(y.to(gl.bfloat16))
        else:
            gl.store(stash + PART * 2048 + r * 128 + c, y.to(gl.bfloat16))
    else:
        if SHARED:
            first = _widen_bf16(stash.slice(PART * 16, 16).load(L))
        else:
            first = _widen_bf16(gl.load(stash + PART * 2048 + r * 128 + c))
        amax = gl.max(gl.maximum(gl.abs(first), gl.abs(y)), 0)  # one cross-lane reduction
        column_exponent, column_bits = _e8m0_scale(amax)
        first, column_scale = gl.broadcast(first, column_bits.to(gl.float32, bitcast=True)[None, :])
        for k in gl.static_range(2):
            value = first if k == 0 else y
            quantized = _mul2(value, column_scale).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
            gl.store(col_data + base - (1 - k) * 16 * H + local, quantized, mask)
        # Scale index of column column + cc, block row_tile // 2.
        pair = row_tile // 2
        col_scale = (column // 128 * gl.cdiv(M, 128) + pair // 4) * 512 + pair % 4
        gl.store(
            col_scales + col_scale + (cc % 32 * 16 + cc // 32 * 4),
            column_exponent.to(gl.uint8),
            (column + cc < H) if not EVEN else None,
        )


@g.jit
def _sync():
    """Order every thread's shared-memory reads before the TMA copies issued next."""
    gl.barrier()
    fence_async_shared()


@triton.constexpr_function
def _half_bytes(tile_types):
    """Bytes expected at a column half's TMA barrier across all copy descriptors."""
    return sum(
        tile.shape[0] // 2 * tile.shape[1] * tile.element_ty.primitive_bitwidth // 8
        for tile in tile_types
    )


@g.jit
def _copy_tiles(plan, row_tile, barrier, part: gl.constexpr):
    """Copy column half ``part`` of each planned slice into its rows of the [32, 128] tile."""
    for i in gl.static_range(len(plan)):
        descriptor, tile, column = plan[i]
        tile = tile.slice(part * 16, 16)
        _tma_load(descriptor, [row_tile * 16, column + part * 128], barrier, tile)


@g.jit
def _copy_carry(load, row, part: gl.constexpr):
    copy, barrier = load
    mbarrier.expect(barrier, 16 * 128 * copy[0].dtype.primitive_bitwidth // 8)
    _copy_tiles((copy,), row, barrier, part)


@g.jit
def _refill(carries, next_inputs, tile, part: gl.constexpr):
    """After a slice's last use, load the current carry and/or the next row's inputs.
    Each carry has a private barrier. All next-row inputs share the column half's barrier.
    """
    row, next_row, has_next, next_barrier = tile
    if len(carries) or (len(next_inputs) and has_next):
        _sync()
        for i in gl.static_range(len(carries)):
            _copy_carry(carries[i], row, part)
        if len(next_inputs) and has_next:
            _copy_tiles(next_inputs, next_row, next_barrier, part)


@g.jit
def _load_residual(residuals, cols, part, stream: gl.constexpr, h: gl.constexpr):
    # A BF16 residual is widened exactly; every product and sum below runs in FP32.
    value = _widen_bf16(residuals[stream].slice(part * 16, 16).load(_GRADIENT_LAYOUT))
    if h % 256:
        value = gl.where(cols < h, value, 0)
    return value


def _backward_body(
    inputs,
    outputs,
    GEOMETRY: gl.constexpr,
    GRID: gl.constexpr,
    PAIRED: gl.constexpr,
    CARRY_DESC,
    R_DESC,
    DY0_DESC,
    DY1_DESC,
    DY2_DESC,
    U_DESC,
    DR_DESC,
    DU_DESC,
):
    """Persistent 16-row tiles in width-spanning CTA groups; ticket admission (each group
    reserves its first tile) keeps partial groups from starving. Prefetches refill consumed
    shared slices (one copy phase per tile); carry streams reuse consumed predecessor slices."""
    m: gl.constexpr = GEOMETRY.M
    h: gl.constexpr = GEOMETRY.H
    streams: gl.constexpr = GEOMETRY.STREAMS
    retention: gl.constexpr = GEOMETRY.HAS_RETENTION
    NREADERS: gl.constexpr = len(inputs.reader_grads)
    TWO: gl.constexpr = NREADERS == 2
    ctas_per_group: gl.constexpr = gl.cdiv(h, 256)
    L: gl.constexpr = _GRADIENT_LAYOUT
    residuals = ()
    for stream in gl.static_range(streams):
        residuals += (gl.allocate_shared_memory(R_DESC.dtype, [32, 128], R_DESC.layout),)
    # Staged unscaled BF16 dy; six streams skip it to keep two-CTA residency.
    STAGE_DY: gl.constexpr = DY0_DESC is not None
    SECOND: gl.constexpr = DY2_DESC is not None
    ds = None
    if STAGE_DY:
        ds = gl.allocate_shared_memory(DY0_DESC.dtype, [NREADERS * 32, 128], DY0_DESC.layout)
    ds2 = None  # the last reader's second consumer gradient
    if SECOND:
        ds2 = gl.allocate_shared_memory(DY2_DESC.dtype, [32, 128], DY2_DESC.layout)
    update_shared = gl.allocate_shared_memory(U_DESC.dtype, [32, 128], U_DESC.layout)
    # Two readers' dR and any BF16 du leave through a two-slot ring, RING stores per column half
    # (every tile issues an even count, so consecutive stores alternate slots).
    TMA_STORE: gl.constexpr = DR_DESC is not None
    RING: gl.constexpr = streams + 1 if DU_DESC is not None else streams
    out_shared = None
    if TMA_STORE:
        out_shared = gl.allocate_shared_memory(DR_DESC.dtype, [32, 128], DR_DESC.layout)

    # One column half of a tile's TMA copies: every predecessor stream, the staged BF16 dys, U.
    copy_deps = ((ds,) if STAGE_DY else ()) + ((ds2,) if SECOND else ()) + (update_shared,)
    copy_deps += residuals
    HALF_BYTES: gl.constexpr = _half_bytes(copy_deps.type)
    # Carry s > 0 lands on barrier s - 1; the last two complete each half's copy phase.
    storage = gl.allocate_shared_memory(gl.int64, [streams + 1, 1], mbarrier.MBarrierLayout())
    barriers = ()
    for index in gl.static_range(streams + 1):
        mbarrier.init(storage.index(index), count=1)
        barriers += (storage.index(index),)
    gl.barrier()  # every init before any shared write
    copy_barrier = barriers[streams - 1]
    copy_barriers = (copy_barrier, barriers[streams])
    LINEAR_SHARED: gl.constexpr = SwizzledSharedLayout(1, 1, 1, [0])
    # Shared projection storage avoids register spills.
    local_projection = gl.allocate_shared_memory(gl.float32, [NREADERS, 16], LINEAR_SHARED)

    GROUPS: gl.constexpr = GRID // ctas_per_group
    total_tiles: gl.constexpr = gl.cdiv(m, 16)
    cta_ticket_counter = outputs.epochs
    row_tile_counter = cta_ticket_counter + 1
    BLOCK: gl.constexpr = 16 + NREADERS * 16 * ctas_per_group
    # Tickets are taken only once resident.
    bid = gl.atomic_add(cta_ticket_counter, 1, sem="relaxed")
    column_tile, group = bid % ctas_per_group, bid // ctas_per_group

    rr0 = gl.arange(0, 16, layout=_ROWS)
    cc = column_tile * 256 + gl.arange(0, 128, layout=_COLUMNS)
    residual_copies, reader_copies = (), ()
    for stream in gl.static_range(streams):
        residual_copies += ((R_DESC, residuals[stream], stream * h + column_tile * 256),)
    if STAGE_DY:
        for reader in gl.static_range(NREADERS):
            reader_copies += (
                ((DY0_DESC, DY1_DESC)[reader], ds.slice(reader * 32, 32), column_tile * 256),
            )
        if SECOND:
            reader_copies += ((DY2_DESC, ds2, column_tile * 256),)
    reader_copies += ((U_DESC, update_shared, column_tile * 256),)
    # One reader with a BF16 residual lands carries s > 0 in the dy and U slices, free once a
    # column half's gradient phase holds dy and U, so they are requested a whole half early.
    EARLY_CARRY: gl.constexpr = (
        not TWO
        and streams <= 3
        and STAGE_DY
        and R_DESC.dtype == gl.bfloat16
        and U_DESC.dtype == gl.bfloat16
    )
    carry_loads = ()
    for s in gl.static_range(streams - 1):
        target = reader_copies[s][1] if EARLY_CARRY else residuals[s]
        copy = (CARRY_DESC, target, (s + 1) * h + column_tile * 256)
        carry_loads += ((copy, barriers[s]),)
    gs = gl.allocate_shared_memory(gl.float32, [NREADERS * 256], LINEAR_SHARED)
    # Half zero's controller shares, one word per thread and plane (write, retention and read
    # per stream, then raw b); only the owning thread touches a word.
    # Two power-of-two buffers cover the planes (10 = 8 + 2 for three streams with retention).
    shares = None
    if PAIRED:
        PLANES: gl.constexpr = (2 + retention) * streams + 1
        shares = (
            gl.allocate_shared_memory(gl.float32, [_share_planes(PLANES, 0) * 128], LINEAR_SHARED),
            gl.allocate_shared_memory(gl.float32, [_share_planes(PLANES, 1) * 128], LINEAR_SHARED),
        )
    gamma_cc = column_tile * 256 + gl.arange(0, 256, layout=_COLUMNS)
    for reader in gl.static_range(NREADERS):
        saved_gamma = gl.load(inputs.gammas[reader] + gamma_cc, gamma_cc < h, 0)
        gs.slice(reader * 256, 256).store(saved_gamma.to(gl.float32))
    # One reader keeps the launch's coefficients as uniform scalars, loaded once (identical
    # FP32 values); two readers have no registers to spare and read a shared table.
    coefficient, cs = (), None
    if TWO:
        COEFFICIENTS: gl.constexpr = triton.next_power_of_2(4 * streams)
        cs = gl.allocate_shared_memory(gl.float32, [COEFFICIENTS], LINEAR_SHARED)
        ci = gl.arange(0, COEFFICIENTS, layout=gl.constexpr(BlockedLayout([1], [32], [4], [0])))
        cs.store(gl.load(inputs.coefficients + ci, ci < 4 * streams, 0))
    else:
        for i in gl.static_range(4 * streams):
            coefficient += (gl.load(inputs.coefficients + i),)
    # MXFP8 du's 32-row column blocks: a group takes row tiles in pairs (even, then odd).
    TILE_PAIRS: gl.constexpr = outputs.update_mxfp8 is not None
    STASH_SHARED: gl.constexpr = TILE_PAIRS and outputs.update_stash is None
    stash = None
    if STASH_SHARED:
        BF16_SHARED: gl.constexpr = gl.NVMMASharedLayout(64, 16, 2)
        stash = gl.allocate_shared_memory(gl.bfloat16, [32, 128], BF16_SHARED)
    row_tile = group * 2 if TILE_PAIRS else group
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
            if TILE_PAIRS:
                claimed = row_tile + 1
                if row_tile % 2:
                    claimed = (GROUPS + gl.atomic_add(row_tile_counter, 1, sem="relaxed")) * 2
            else:
                claimed = GROUPS + gl.atomic_add(row_tile_counter, 1, sem="relaxed")
            _publish_word(next_tile_word, claimed, tag, 127)
            block = outputs.projection + claimed * BLOCK
            _prefetch_warp_rows(block, (claimed < total_tiles).to(gl.int32), 0, 0, BLOCK * 8)

        # Wait for column half zero only; half one is awaited inside the reader phase.
        mbarrier.wait(copy_barrier, iteration & 1, deps=copy_deps)
        _sync()

        # Reader phase: x = sum_s beta t R_s + q u, z = r x, p = sum(dy gamma z) / H per reader.
        rs = ()
        for reader in gl.static_range(NREADERS):
            rs += (gl.load(inputs.rstd[reader] + rr, rr < m, 0),)
        alphas, mixes = ((),) * NREADERS, (0.0,) * NREADERS
        for s in gl.static_range(streams):
            retain = _pick(cs, coefficient, streams + s) if retention else 1.0
            write = _pick(cs, coefficient, s)
            new_alphas, new_mixes = (), ()
            for reader in gl.static_range(NREADERS):
                beta = _pick(cs, coefficient, (2 + reader) * streams + s)
                new_alphas += (alphas[reader] + (beta * retain,),)
                new_mixes += (gl.fma(beta, write, mixes[reader]),)
            alphas, mixes = new_alphas, new_mixes
        projections = (gl.full((16, 128), 0, gl.float32, L),) * NREADERS
        for part in gl.static_range(2):
            if part == 1:
                mbarrier.wait(copy_barriers[1], iteration & 1, deps=copy_deps)
            cols = (cc + part * 128)[None, :]
            mask = _in_range(rr[:, None], cols, GEOMETRY)
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
                z = _mul2(_fma2(mixes[reader], u, xs[reader]), rs[reader][:, None])
                dy = _load_dy(inputs, ds, ds2, reader, part, rr[:, None] * h + cols, mask)
                gamma = gs.slice(reader * 256 + part * 128, 128).load(_COLUMNS)
                new_projections += (_fma2(_mul2(dy, gamma[None, :]), z, projections[reader]),)
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
        # Finish the projection: sum the width participants' partials.
        PL: gl.constexpr = BlockedLayout([1, 2], [4, 8], [4, 1], [1, 0])
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
        # Two readers start transfers first; one reader releases each slice after its
        # controller reductions.
        OUTPUT_CACHE: gl.constexpr = ".cs" if not TWO else ""
        HELD: gl.constexpr = min(2, streams)  # predecessor slices read before the first barrier
        # Two readers skip the carry mask when every scheduled element is in range.
        MASKED: gl.constexpr = not TWO or h % 256 != 0
        prefetch_active = next_rp < total_tiles
        COLUMN_SLOTS: gl.constexpr = gl.cdiv(h, 256) if PAIRED else gl.cdiv(h, 128)
        stride = gl.cdiv(m, 16) * COLUMN_SLOTS * 8  # between controller partial planes
        # Planes: write and retention per stream, each reader's read per stream, then raw b.
        READ_PLANE: gl.constexpr = (1 + retention) * streams
        U_PLANE: gl.constexpr = (1 + retention + NREADERS) * streams
        for part in gl.static_range(2):
            cols = (cc + part * 128)[None, :]
            rows = rr[:, None]
            mask = _in_range(rows, cols, GEOMETRY)
            offset = rows * h + cols
            column_slot = column_tile if PAIRED else column_tile * 2 + part
            controller_row = (
                outputs.controller_partials + (row_tile * COLUMN_SLOTS + column_slot) * 8
            )
            valid = (row_tile * 16 < m) & (column_tile * 2 + part < gl.cdiv(h, 128))
            # Half one's paired store also covers a half zero whose half one lies past H.
            store_valid = (row_tile * 16 < m) if PAIRED else valid
            at = (controller_row, stride, valid if h % 256 else True)
            at += (shares,) if PAIRED else ()
            refill = (row_tile, next_rp, prefetch_active, copy_barriers[part])
            # One dR row pointer per slice; stream offsets become store immediates.
            residual_grad_rows = outputs.residual_grad + (rows * (streams * h) + cols)

            joint_coefficients = ()
            for i in gl.static_range(4 * streams):
                joint_coefficients += (_pick(cs, coefficient, i),)

            queued = ()
            u = _widen_bf16(update_shared.slice(part * 16, 16).load(L))
            incoming_zero = _load_carry_zero(inputs.carry, rows, cols, GEOMETRY, MASKED)
            dxs = ()
            update_mixes = ()
            for reader in gl.static_range(NREADERS):
                dy = _load_dy(inputs, ds, ds2, reader, part, offset, mask)
                gamma = gs.slice(reader * 256 + part * 128, 128).load(_COLUMNS)[None, :]
                p = gl.gather(local_projection.index(reader).load(_ROWS), rr0, 0)[:, None]
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

            if EARLY_CARRY:
                _refill(carry_loads, (), refill, part)
            if not TWO:
                for reader in gl.static_range(NREADERS):
                    queued = _defer(queued, u, dxs[reader], U_PLANE + reader, at, part)
            du = gl.full((16, 128), 0, gl.float32, L)

            if TWO:
                # Held slices stay in registers, so their refills and the next dy/U start now.
                held = ()
                for first in gl.static_range(HELD):
                    held += (_load_residual(residuals, cols, part, first, h),)
                released = residual_copies[-1:] if HELD == streams else ()
                _refill(carry_loads[:HELD], released + reader_copies, refill, part)

            for s in gl.static_range(streams):
                if TWO and s < HELD:
                    R_prev_s = held[s]
                else:
                    R_prev_s = _load_residual(residuals, cols, part, s, h)
                # Stream s's own release: before its controller reductions for two readers past
                # HELD, after them for one reader.
                carry = carry_loads[s : s + 1]
                last_slice = residual_copies[s : s + 1] if s + 1 == streams else ()
                if TWO and s >= HELD:
                    _refill(carry, last_slice, refill, part)
                w = joint_coefficients[s]
                t = joint_coefficients[streams + s] if retention else 1.0
                if TWO and s == 0:
                    for reader in gl.static_range(NREADERS):
                        queued = _defer(queued, u, dxs[reader], U_PLANE + reader, at, part)
                # Raw a[r, s] = sum(Rprev_s * dx_r); the final reduction forms t * a + w * b.
                for reader in gl.static_range(NREADERS):
                    queued = _defer(
                        queued, R_prev_s, dxs[reader], READ_PLANE + reader * streams + s, at, part
                    )
                if EARLY_CARRY and prefetch_active:
                    # Slice s goes straight to the next tile; so does the dy/U slice whose
                    # carry the previous stream consumed.
                    _sync()
                    _copy_tiles((residual_copies[s],), next_rp, copy_barriers[part], part)
                    if s > 1:
                        _copy_tiles((reader_copies[s - 2],), next_rp, copy_barriers[part], part)
                if not TWO and not EARLY_CARRY and (s + 1 < streams or prefetch_active):
                    _sync()
                    if s + 1 < streams:
                        _copy_carry(carry_loads[s], row_tile, part)
                    else:
                        _copy_tiles(last_slice, next_rp, copy_barriers[part], part)
                    if s > 0 and prefetch_active:
                        # The previous stream's consumed carry slice (or dy/U) for the next tile.
                        deferred = reader_copies if s == 1 else (residual_copies[s - 2],)
                        _copy_tiles(deferred, next_rp, copy_barriers[part], part)

                if s == 0:
                    incoming_dR_s = incoming_zero
                else:
                    copy, barrier = carry_loads[s - 1]
                    tile = copy[1]
                    mbarrier.wait(barrier, part & 1, deps=(tile,))
                    incoming_dR_s = _widen_bf16(tile.slice(part * 16, 16).load(L))
                if TWO and s > 0 and prefetch_active:
                    # Release the consumed carry slice to the next tile's predecessor s - 1.
                    _sync()
                    _copy_tiles((residual_copies[s - 1],), next_rp, copy_barriers[part], part)
                if retention:
                    queued = _defer(queued, incoming_dR_s, R_prev_s, streams + s, at, part)
                if h % 256:
                    incoming_dR_s = gl.where(cols < h, incoming_dR_s, 0)
                # Differentiate the rounded folded coefficients alpha = FP32(beta * t).
                d_previous = _mul2(t, incoming_dR_s) if retention else incoming_dR_s
                for reader in gl.static_range(NREADERS):
                    alpha = joint_coefficients[(2 + reader) * streams + s] * t
                    d_previous = _fma2(alpha, dxs[reader], d_previous)

                if not TWO and not EARLY_CARRY and s + 1 == streams and prefetch_active:
                    _sync()
                    # Start the next tile into the consumed column half (the last stream's slice;
                    # earlier ones went out at the following stream's release barrier).
                    pending = reader_copies if s == 0 else (residual_copies[s - 1],)
                    _copy_tiles(pending, next_rp, copy_barriers[part], part)

                d_previous = d_previous.to(outputs.residual_grad.dtype.element_ty)  # RN for BF16
                if TMA_STORE:
                    store_column = s * h + column_tile * 256 + part * 128
                    slot = out_shared.slice((part * RING + s) % 2 * 16, 16)
                    _tma_store(DR_DESC, slot, d_previous, row_tile * 16, store_column)
                else:
                    pointer = residual_grad_rows + s * h
                    gl.store(pointer, d_previous, mask, cache_modifier=OUTPUT_CACHE)
                # du accumulates the carry here and the readers' q * dx after the streams.
                du = _fma2(w, incoming_dR_s, du)
                queued = _defer(queued, incoming_dR_s, u, s, at, part)
                for k in gl.static_range(len(queued)):
                    _store_partial(queued[k][0], queued[k][1], store_valid)
                queued = ()

            if EARLY_CARRY:
                _refill((), reader_copies[max(0, streams - 2) :], refill, part)
            for reader in gl.static_range(NREADERS):
                du = _fma2(update_mixes[reader], dxs[reader], du)

            du_column = column_tile * 256 + part * 128
            if DU_DESC is not None:
                slot = out_shared.slice((part * RING + streams) % 2 * 16, 16)
                _tma_store(DU_DESC, slot, du.to(gl.bfloat16), row_tile * 16, du_column)
            elif outputs.update_grad is not None:
                gl.store(outputs.update_grad + offset, du, mask, cache_modifier=OUTPUT_CACHE)
            if TILE_PAIRS:  # a global stash slot is rebuilt here: kept live, it costs registers
                slot = stash if STASH_SHARED else outputs.update_stash + bid * 4096
                mx = outputs.update_mxfp8
                _emit_update_mxfp8(mx, slot, du, row_tile, du_column, part, STASH_SHARED, m, h)

        row_tile = next_rp
        iteration += 1
    if TMA_STORE:
        tma.store_wait(0)


def _backward_kernel_name(readers, second, du):
    """The profiler name of one variant: its readers, the gradients flowing into them (a reader
    with two consumers takes two) and the update gradient's formats (``KERNEL_FORMAT_TAGS``)."""
    return f"bwd_wr_writeread_{readers}reader_{readers + second}grad_{KERNEL_FORMAT_TAGS[du]}"


BACKWARD_KERNELS = {
    key: named_jit(_backward_body, _backward_kernel_name(*key))
    for key in product((1, 2), (False, True), OUTPUT_FORMATS)
}
