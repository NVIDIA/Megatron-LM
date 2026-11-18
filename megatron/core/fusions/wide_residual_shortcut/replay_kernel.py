# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Checkpoint forward replay and reader-column materialization.

Replayed residuals reproduce the forward's row/stream RN/FMA choices in FP32, rounded to the
residual's dtype as the forward stores them. MXFP8 columns encode the forward's folded FP32
reader value before output rounding. Callers own interval scheduling and buffer reuse.
"""

import torch
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout

from .kernel_utils import _emit_mxfp8, _forward_value, _mul_rn, allocate_operand, operand_buffers


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
            value = _forward_value(values[s], update, write, retain, row, s)
            output = outputs[step]
            # The next transition reads the stored (rounded) residual, as the forward does.
            value = value.to(output.dtype.element_ty)
            gl.store(output + row * STREAMS * H + s * H + col, value, mask, cache_modifier=".cs")
            next_values += (value.to(gl.float32),)
        values = next_values


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
    EMIT: gl.constexpr,
):
    """Replay transitions on 32x32 tiles, emitting the columns of every reader EMIT[tape] names
    (``rstd``/``gamma``/``data``/``scales`` are per tape, per reader)."""
    BLOCK_COLS: gl.constexpr = 32
    L: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    rowtile = gl.program_id(0) // (gl.cdiv(H, 128) * (128 // BLOCK_COLS))
    coltile = gl.program_id(0) % (gl.cdiv(H, 128) * (128 // BLOCK_COLS))
    rr = rowtile * 32 + gl.arange(0, 32, layout=gl.constexpr(gl.SliceLayout(1, L)))
    cc = coltile * BLOCK_COLS + gl.arange(0, BLOCK_COLS, layout=gl.constexpr(gl.SliceLayout(0, L)))
    mask = (rr[:, None] < M) & (cc[None, :] < H)
    values = ()
    for s in gl.static_range(S):
        values += (
            gl.load(checkpoint + rr[:, None] * (S * H) + s * H + cc[None, :], mask, 0).to(
                gl.float32
            ),
        )
    for j in gl.static_range(len(updates)):
        u = gl.load(updates[j] + rr[:, None] * H + cc[None, :], mask, 0).to(gl.float32)
        for reader in gl.static_range(len(EMIT[j])):
            if EMIT[j][reader]:
                x = gl.full((32, BLOCK_COLS), 0, gl.float32, L)
                q = 0.0
                for s in gl.static_range(S):
                    w = gl.load(cs[j] + s)
                    t = gl.load(cs[j] + S + s) if RETENTION[j] else 1.0
                    beta = gl.load(cs[j] + (2 + reader) * S + s)
                    # Explicit rn multiply prevents contraction into neighboring recurrence.
                    alpha = _mul_rn(beta, t)
                    x = gl.fma(alpha, values[s], x)
                    q = gl.fma(beta, w, q)
                x = gl.fma(q, u, x)
                r = gl.load(rstd[j][reader] + rr, rr < M, 0)
                weight = gl.load(gamma[j][reader] + cc, cc < H, 0).to(gl.float32)
                z = _mul_rn(x, r[:, None])
                y = gl.where(mask, _mul_rn(z, weight[None, :]), 0.0)
                # Columnwise blocks run down the rows; padded scales are overwritten too.
                _emit_mxfp8(
                    y,
                    (data[j][reader] + rr[:, None] * H + cc[None, :], mask),
                    (scales[j][reader], None),
                    (cc, rowtile, M),
                    AXIS=0,
                )
        if j < len(updates) - 1 or TERMINAL:
            next_values = ()
            for s in gl.static_range(S):
                w = gl.load(cs[j] + s)
                t = gl.load(cs[j] + S + s) if RETENTION[j] else 1.0
                value = _forward_value(values[s], u, w, t, rr[:, None], s)
                target = outputs[j] if j < len(updates) - 1 else terminal
                value = value.to(target.dtype.element_ty)
                gl.store(target + rr[:, None] * (S * H) + s * H + cc[None, :], value, mask)
                next_values += (value.to(gl.float32),)
            values = next_values


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


def _launch_columns(checkpoint, tapes, outputs, data, scales, requested, terminal):
    m, h = tapes[0].update.shape
    _bwd_wr_recompute_mxfp8quant[(triton.cdiv(m, 128) * 4 * triton.cdiv(h, 128) * 4,)](
        checkpoint,
        tuple(t.update for t in tapes),
        tuple(t.coefficients for t in tapes),
        tuple(
            tuple(t.rstd[r] if need else None for r, need in enumerate(needs))
            for t, needs in zip(tapes, requested)
        ),
        tuple(
            tuple(t.gammas[r] if need else None for r, need in enumerate(needs))
            for t, needs in zip(tapes, requested)
        ),
        outputs,
        data,
        scales,
        terminal,
        m,
        h,
        tapes[0].num_streams,
        tuple(t.has_retention for t in tapes),
        terminal is not None,
        requested,
        maxnreg=64,
        num_warps=4,
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
