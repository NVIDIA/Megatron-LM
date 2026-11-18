# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Checkpoint forward replay and reader-column materialization.

Replayed residuals reproduce the forward's row/stream RN/FMA choices in FP32, rounded to the
residual's dtype as the forward stores them. MXFP8 columns encode the forward's folded FP32
reader value before output rounding. Callers own interval scheduling and buffer reuse.
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout

from .kernel_utils import (
    _e8m0_scale,
    _fma2,
    _forward_value,
    _load_f32x4,
    _mul2,
    _mul_rn,
    _round_bf16,
    _store_bf16x4,
    allocate_operand,
    operand_buffers,
)


@g.jit
def _bwd_wr_recompute(
    checkpoint,
    updates,
    coefficients,
    outputs,
    H: gl.constexpr,
    STREAMS: gl.constexpr,
    RETENTION: gl.constexpr,
    BLOCK: gl.constexpr,
):
    pid = gl.program_id(0)
    row = pid // gl.cdiv(H, BLOCK)
    col = (pid % gl.cdiv(H, BLOCK)) * BLOCK + gl.arange(
        0, BLOCK, layout=gl.constexpr(BlockedLayout([1], [32], [4], [0]))
    )
    mask = col < H
    values = ()
    for s in gl.static_range(STREAMS):
        values += (
            gl.load(checkpoint + row * STREAMS * H + s * H + col, mask, 0, cache_modifier=".cg").to(
                gl.float32
            ),
        )
    for step in gl.static_range(len(updates)):
        update = gl.load(updates[step] + row * H + col, mask, 0, cache_modifier=".cg").to(
            gl.float32
        )
        next_values = ()
        for s in gl.static_range(STREAMS):
            write = gl.load(coefficients[step] + s)
            retain = gl.load(coefficients[step] + STREAMS + s) if RETENTION[step] else 1.0
            value = _forward_value(values[s], update, write, retain)
            output = outputs[step]
            # The next transition reads the stored (rounded) residual, as the forward does.
            value = value.to(output.dtype.element_ty)
            gl.store(output + row * STREAMS * H + s * H + col, value, mask, cache_modifier=".cs")
            next_values += (value.to(gl.float32),)
        values = next_values


@g.jit
def _tile_value(previous, update, write, retain, RETAIN: gl.constexpr):
    """``_forward_value`` on a tile in packed f32x2 arithmetic. Bit-identical to it."""
    return _fma2(write, update, _mul2(retain, previous) if RETAIN else previous)


@g.jit
def _emit_columns(y, data, scales, index, mask, EVEN: gl.constexpr):
    """Columnwise MXFP8 of a (32, C) FP32 tile that is one 32-row block: E8M0 scale per column
    (at ``index`` of the compact [row block, column] scales). Padded rows are zero in ``y``;
    tiles cover whole 128 x 128 scale tiles, so padded scales are written too."""
    exponent, bits = _e8m0_scale(gl.max(gl.abs(y), 0))
    q = _mul2(y, gl.expand_dims(bits.to(gl.float32, bitcast=True), 0))
    q = q.to(gl.float8e4nv).to(gl.uint8, bitcast=True)
    if EVEN:
        gl.store(data, q)
    else:
        gl.store(data, q, mask)
    gl.store(scales + index, exponent.to(gl.uint8))


@g.jit
def _bwd_wr_recompute_mxfp8quant(
    checkpoint,
    updates,
    cs,
    rstd,
    gamma,
    outputs,
    data,
    scales,
    terminal,
    M: gl.constexpr,
    H: gl.constexpr,
    S: gl.constexpr,
    RETENTION: gl.constexpr,
    TERMINAL: gl.constexpr,
    READERS: gl.constexpr,
    FIRST: gl.constexpr,
    TABLE: gl.constexpr,
    VECTOR_COEFFICIENTS: gl.constexpr,
    VEC: gl.constexpr,
    THREADS_N: gl.constexpr,
    WARPS: gl.constexpr,
    AHEAD: gl.constexpr,
):
    """Replay transitions on 32-row tiles, emitting the columns of every reader in READERS[j]
    after transition j. ``rstd``, ``gamma`` and ``data`` hold one entry per emitted column
    operand, in transition order; FIRST[j] is transition j's first. ``scales`` holds every
    operand's compact [row block, padded column] scales, swizzled afterwards: one byte per
    column of a row block lands in a different 32-byte sector of TE's layout, and such partial
    sector writes cost far more than their bytes."""
    BLOCK_COLS: gl.constexpr = VEC * THREADS_N
    # Training shapes cover whole 128 x 128 scale tiles: no masks.
    EVEN: gl.constexpr = (M % 128 == 0) and (H % 128 == 0)
    L: gl.constexpr = gl.BlockedLayout([1, VEC], [32 // THREADS_N, THREADS_N], [WARPS, 1], [1, 0])
    columns: gl.constexpr = gl.cdiv(H, 128) * (128 // BLOCK_COLS)
    rowtile = gl.program_id(0) // columns
    coltile = gl.program_id(0) % columns
    lane = gl.arange(0, 32, layout=gl.constexpr(gl.SliceLayout(1, L)))
    rr = rowtile * 32 + lane
    cc = coltile * BLOCK_COLS + gl.arange(0, BLOCK_COLS, layout=gl.constexpr(gl.SliceLayout(0, L)))
    mask = (rr[:, None] < M) & (cc[None, :] < H)
    values = ()
    for s in gl.static_range(S):
        ptr = checkpoint + rr[:, None] * (S * H) + s * H + cc[None, :]
        values += ((gl.load(ptr) if EVEN else gl.load(ptr, mask, 0)).to(gl.float32),)
    N: gl.constexpr = len(updates)
    # Per-thread offsets every transition shares: [M, H] tiles (updates, column data), the
    # residual's first stream, and each column's swizzled scale.
    offsets = rr[:, None] * H + cc[None, :]
    residual_offsets = rr[:, None] * (S * H) + cc[None, :]
    PADDED_H: gl.constexpr = gl.cdiv(H, 128) * 128
    OPERAND: gl.constexpr = gl.cdiv(M, 128) * 4 * PADDED_H  # compact scale bytes per operand
    scale_index = rowtile * PADDED_H + cc
    # Updates load AHEAD transitions early, overlapping earlier transitions' stores.
    pending = ()
    for k in gl.static_range(AHEAD):
        if k < N:
            uptr = updates[k] + offsets
            pending += (gl.load(uptr) if EVEN else gl.load(uptr, mask, 0),)
    for j in gl.static_range(N):
        if AHEAD == 0:
            uptr = updates[j] + offsets
            pending = (gl.load(uptr) if EVEN else gl.load(uptr, mask, 0),)
        u = pending[0].to(gl.float32)
        rest = ()
        for k in gl.static_range(1, len(pending)):
            rest += (pending[k],)
        if j + AHEAD < N:
            uptr = updates[j + AHEAD] + offsets
            rest += (gl.load(uptr) if EVEN else gl.load(uptr, mask, 0),)
        pending = rest
        # The transition's [write, retain, read...] x S coefficient rows, four per vector load.
        table = ()
        if VECTOR_COEFFICIENTS:
            for k in gl.static_range(gl.cdiv(TABLE[j], 4)):
                table += _load_f32x4(cs[j] + 4 * k)
        else:
            for k in gl.static_range(TABLE[j]):
                table += (gl.load(cs[j] + k),)
        writes, retains = (), ()
        for s in gl.static_range(S):
            writes += (table[s],)
            retains += (table[S + s] if RETENTION[j] else 1.0,)
        for e in gl.static_range(len(READERS[j])):
            x = None
            q = 0.0
            for s in gl.static_range(S):
                beta = table[(2 + READERS[j][e]) * S + s]
                # Explicit rn multiply prevents contraction into neighboring recurrence.
                alpha = _mul_rn(beta, retains[s])
                x = _mul2(alpha, values[s]) if x is None else _fma2(alpha, values[s], x)
                q = gl.fma(beta, writes[s], q)
            x = _fma2(q, u, x)
            if EVEN:
                r = gl.load(rstd[FIRST[j] + e] + rr)
                weight = gl.load(gamma[FIRST[j] + e] + cc).to(gl.float32)
            else:
                r = gl.load(rstd[FIRST[j] + e] + rr, rr < M, 0)
                weight = gl.load(gamma[FIRST[j] + e] + cc, cc < H, 0).to(gl.float32)
            y = _mul2(_mul2(x, r[:, None]), weight[None, :])
            if not EVEN:
                y = gl.where(mask, y, 0.0)
            _emit_columns(
                y,
                data[FIRST[j] + e] + offsets,
                scales + (FIRST[j] + e) * OPERAND,
                scale_index,
                mask,
                EVEN,
            )
        if j < len(updates) - 1 or TERMINAL:
            next_values = ()
            for s in gl.static_range(S):
                value = _tile_value(values[s], u, writes[s], retains[s], RETENTION[j])
                target = outputs[j] if j < len(updates) - 1 else terminal
                tptr = target + s * H + residual_offsets
                if EVEN and VEC % 4 == 0 and target.dtype.element_ty == gl.bfloat16:
                    value = _store_bf16x4(tptr, value)
                else:
                    if target.dtype.element_ty == gl.bfloat16:
                        value = _round_bf16(value)
                    else:
                        value = value.to(target.dtype.element_ty)
                    if EVEN:
                        gl.store(tptr, value)
                    else:
                        gl.store(tptr, value, mask)
                    value = value.to(gl.float32)
                next_values += (value,)
            values = next_values


@triton.jit
def _swizzle_scales(compact, scales, M, PADDED_H: tl.constexpr):
    """Copy each operand's compact [row block, column] E8M0 scales (program axis 1) into TE's
    GEMM-ready layout: one program per 128 x 128 tile, whose 512 bytes it writes whole (byte
    t * 16 + g * 4 + i holds column g * 32 + t of row block i)."""
    tile = tl.program_id(0)
    operand = tl.program_id(1)
    blocks_n: tl.constexpr = PADDED_H // 128
    rb = tile // blocks_n
    cb = tile % blocks_n
    t = tl.arange(0, 32)[:, None]
    j = tl.arange(0, 16)[None, :]
    rows = tl.cdiv(M, 128) * 4
    source = compact + operand * rows * PADDED_H + (rb * 4 + j % 4) * PADDED_H
    value = tl.load(source + cb * 128 + j // 4 * 32 + t)
    offset = (cb * tl.cdiv(M, 128) + rb) * 512 + t * 16 + j
    for k in tl.static_range(len(scales)):
        if operand == k:
            tl.store(scales[k] + offset, value)


def _column_inputs(checkpoint, tapes, requested, columns):
    """Per tape, per reader: the columnwise operands (allocated unless supplied) and their data
    and scale buffers (None where not requested)."""
    m, h = tapes[0].update.shape
    if columns is None:
        columns = tuple(
            tuple(
                allocate_operand((m, h), checkpoint.device, columnwise=True) if need else None
                for need in needs
            )
            for needs in requested
        )
    else:
        columns = tuple(tuple(per_tape) for per_tape in columns)
        if len(columns) != len(tapes) or any(
            len(per_tape) != len(needs)
            or any((column is not None) != need for need, column in zip(needs, per_tape))
            for needs, per_tape in zip(requested, columns)
        ):
            raise ValueError("column entries must match requested transitions and readers")
    buffers = tuple(operand_buffers(per_tape, columnwise=True) for per_tape in columns)
    data = tuple(tuple(d) for d, _ in buffers)
    scales = tuple(tuple(q) for _, q in buffers)
    for needs, ds, qs in zip(requested, data, scales):
        for need, d, q in zip(needs, ds, qs):
            if need and (
                d is None
                or q is None
                or d.shape != (m, h)
                or q.numel() != triton.cdiv(m, 128) * triton.cdiv(h, 128) * 512
                or any(
                    t.dtype != torch.uint8 or t.device != checkpoint.device or not t.is_contiguous()
                    for t in (d, q)
                )
            ):
                raise ValueError("invalid MXFP8 column allocation")
    return columns, data, scales


# Column replay schedules per residual dtype: (VEC columns per thread, THREADS lanes across a
# 32-row tile, WARPS warps down it, MAXNREG registers, AHEAD transitions of update prefetch),
# from an 8K-token Nano/Super/Ultra sweep.
_MX_SCHEDULES = {torch.float32: (4, 8, 2, 128, 1), torch.bfloat16: (4, 8, 2, 128, 1)}


def _launch_columns(checkpoint, tapes, outputs, data, scales, requested, terminal):
    m, h = tapes[0].update.shape
    readers = tuple(tuple(r for r, need in enumerate(needs) if need) for needs in requested)
    first, flat = [], []  # per tape its first emitted operand; (tape, reader) per operand
    for tape, (index, emitted) in zip(tapes, enumerate(readers)):
        first.append(len(flat))
        flat.extend((index, r) for r in emitted)
    vector, threads, warps, maxnreg, ahead = _MX_SCHEDULES[checkpoint.dtype]
    block = vector * threads
    tiles = triton.cdiv(m, 128) * 4 * triton.cdiv(h, 128) * (128 // block)
    padded_h = triton.cdiv(h, 128) * 128
    compact = torch.empty(
        (len(flat), triton.cdiv(m, 128) * 4, padded_h), dtype=torch.uint8, device=checkpoint.device
    )
    _bwd_wr_recompute_mxfp8quant[(tiles,)](
        checkpoint,
        tuple(t.update for t in tapes),
        tuple(t.coefficients for t in tapes),
        tuple(tapes[t].rstd[r] for t, r in flat),
        tuple(tapes[t].gammas[r] for t, r in flat),
        outputs,
        tuple(data[t][r] for t, r in flat),
        compact,
        terminal,
        m,
        h,
        tapes[0].num_streams,
        tuple(t.has_retention for t in tapes),
        terminal is not None,
        readers,
        tuple(first),
        # Per transition, the coefficients it reads: write, retain and reads up to its last reader.
        tuple((3 + max(emitted, default=-1)) * tapes[0].num_streams for emitted in readers),
        all(t.coefficients.data_ptr() % 16 == 0 for t in tapes),
        vector,
        threads,
        warps,
        ahead,
        maxnreg=maxnreg,
        num_warps=warps,
    )
    _swizzle_scales[(triton.cdiv(m, 128) * triton.cdiv(h, 128), len(flat))](
        compact, tuple(scales[t][r] for t, r in flat), m, padded_h, num_warps=1
    )


def column_requests(tapes, prepare_columns):
    """Per tape, one bool per reader: ``prepare_columns`` is a bool for every tape or one entry
    per tape, each a bool (reader zero) or one bool per reader."""
    entries = (
        (prepare_columns,) * len(tapes)
        if isinstance(prepare_columns, bool)
        else tuple(prepare_columns)
    )
    if len(entries) != len(tapes):
        raise ValueError("one column request per transition is required")
    requested = []
    for tape, entry in zip(tapes, entries):
        readers = len(tape.rstd)
        needs = (entry,) + (False,) * (readers - 1) if isinstance(entry, bool) else tuple(entry)
        if len(needs) != readers or not all(isinstance(need, bool) for need in needs):
            raise ValueError("a transition's column request needs one bool per reader")
        requested.append(needs)
    return tuple(requested)


@torch.no_grad()
def replay_interval(
    checkpoint, tapes, outputs=None, columns=None, *, prepare_columns=False, terminal_output=None
):
    """Replay N tapes from ``checkpoint``: return N predecessors and, per tape, one columnwise
    MXFP8 operand per reader (None unless requested), all emitted by one launch.

    The predecessors are the checkpoint itself plus N - 1 scratch buffers (``outputs``), and
    ``terminal_output`` optionally receives the last tape's result. ``prepare_columns`` is
    ``column_requests``' argument; supplied ``columns`` follow the result's nesting. Scratch may
    be reused only after all readers, including deferred cross-stream weight gradients, have
    completed.
    """
    tapes = tuple(tapes)
    # Forward validated each saved write; check only how the interval fits together.
    m, h = tapes[0].update.shape
    streams = tapes[0].num_streams
    if (
        checkpoint.shape != (m, streams * h)
        or not checkpoint.is_contiguous()
        or any(t.update.shape != (m, h) or t.num_streams != streams for t in tapes)
    ):
        raise ValueError("inconsistent replay interval geometry")
    requested = column_requests(tapes, prepare_columns)
    if outputs is None:
        outputs = tuple(torch.empty_like(checkpoint) for _ in tapes[:-1])
    outputs = tuple(outputs)
    if len(outputs) != len(tapes) - 1:
        raise ValueError("need N-1 predecessor buffers")
    addresses = {checkpoint.untyped_storage().data_ptr()}
    for tape in tapes:
        for tensor in (tape.update, tape.coefficients, *tape.gammas, *tape.rstd):
            addresses.add(tensor.untyped_storage().data_ptr())

    def claim(tensor):
        address = tensor.untyped_storage().data_ptr()
        if address in addresses:
            raise ValueError("replay, column and input buffers must not alias")
        addresses.add(address)

    targets = outputs + (() if terminal_output is None else (terminal_output,))
    for out in targets:
        if (
            out.shape != checkpoint.shape
            or out.dtype != checkpoint.dtype
            or out.device != checkpoint.device
            or not out.is_contiguous()
            or out.storage_offset() != 0
            or out.untyped_storage().nbytes() != out.numel() * out.element_size()
        ):
            raise ValueError(
                "predecessor buffers require full contiguous storage of the checkpoint dtype"
            )
        claim(out)
    supplied = columns is not None
    columns, data, scales = _column_inputs(checkpoint, tapes, requested, columns)
    if supplied:  # fresh allocations cannot alias
        for tensor in (*sum(data, ()), *sum(scales, ())):
            if tensor is not None:
                claim(tensor)
    if any(any(needs) for needs in requested):
        _launch_columns(checkpoint, tapes, outputs, data, scales, requested, terminal_output)
    elif transitions := (tapes if terminal_output is not None else tapes[:-1]):
        _bwd_wr_recompute[(m * triton.cdiv(h, 512),)](
            checkpoint,
            tuple(t.update for t in transitions),
            tuple(t.coefficients for t in transitions),
            targets,
            h,
            streams,
            tuple(t.has_retention for t in transitions),
            512,
            enable_fp_fusion=False,
            num_warps=4,
        )
    return (checkpoint, *outputs), columns
