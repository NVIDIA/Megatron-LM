# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Checkpoint forward replay and reader-column materialization.

Residual updates preserve the forward's row/stream RN/FMA choices. MXFP8
columns encode the matched folded FP32 reader value, before output rounding.
The caller owns interval scheduling and waits for all consumers before reuse.
"""

import torch
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout

from .kernel_utils import _e8m0, _forward_value, _mul_rn, allocate_operand, operand_buffers


@g.jit
def _replay_transitions(
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
            gl.load(checkpoint + row * STREAMS * H + s * H + col, mask, 0, cache_modifier=".cg"),
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
            gl.store(output + row * STREAMS * H + s * H + col, value, mask, cache_modifier=".cs")
            next_values += (value,)
        values = next_values


@g.jit
def _replay_mx(
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
    READER: gl.constexpr,
):
    BLOCK_COLS: gl.constexpr = 32
    L: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    rowtile = gl.program_id(0) // (gl.cdiv(H, 128) * (128 // BLOCK_COLS))
    coltile = gl.program_id(0) % (gl.cdiv(H, 128) * (128 // BLOCK_COLS))
    rr = rowtile * 32 + gl.arange(0, 32, layout=gl.constexpr(gl.SliceLayout(1, L)))
    cc = coltile * BLOCK_COLS + gl.arange(0, BLOCK_COLS, layout=gl.constexpr(gl.SliceLayout(0, L)))
    mask = (rr[:, None] < M) & (cc[None, :] < H)
    values = ()
    for s in gl.static_range(S):
        values += (gl.load(checkpoint + rr[:, None] * (S * H) + s * H + cc[None, :], mask, 0),)
    for j in gl.static_range(len(updates)):
        u = gl.load(updates[j] + rr[:, None] * H + cc[None, :], mask, 0).to(gl.float32)
        if EMIT[j]:
            x = gl.full((32, BLOCK_COLS), 0, gl.float32, L)
            q = 0.0
            for s in gl.static_range(S):
                w = gl.load(cs[j] + s)
                t = gl.load(cs[j] + S + s) if RETENTION[j] else 1.0
                beta = gl.load(cs[j] + (2 + READER) * S + s)
                # Explicit rn multiply prevents contraction into neighboring recurrence.
                alpha = _mul_rn(beta, t)
                x = gl.fma(alpha, values[s], x)
                q = gl.fma(beta, w, q)
            x = gl.fma(q, u, x)
            r = gl.load(rstd[j] + rr, rr < M, 0)
            weight = gl.load(gamma[j] + cc, cc < H, 0).to(gl.float32)
            z = _mul_rn(x, r[:, None])
            y = _mul_rn(z, weight[None, :])
            y = gl.where(mask, y, 0.0)
            exp, bits = _e8m0(gl.max(gl.abs(y), 0))
            quant = (
                (y * bits.to(gl.float32, bitcast=True)[None, :])
                .to(gl.float8e4nv)
                .to(gl.uint8, bitcast=True)
            )
            gl.store(data[j] + rr[:, None] * H + cc[None, :], quant, mask)
            index = (
                ((cc // 128) * gl.cdiv(M, 128) + rowtile // 4) * 512
                + (cc % 32) * 16
                + (cc % 128 // 32) * 4
                + rowtile % 4
            )
            gl.store(scales[j] + index, exp.to(gl.uint8))
        if j < len(updates) - 1 or TERMINAL:
            next_values = ()
            for s in gl.static_range(S):
                w = gl.load(cs[j] + s)
                t = gl.load(cs[j] + S + s) if RETENTION[j] else 1.0
                value = _forward_value(values[s], u, w, t, rr[:, None], s)
                target = outputs[j] if j < len(updates) - 1 else terminal
                gl.store(target + rr[:, None] * (S * H) + s * H + cc[None, :], value, mask)
                next_values += (value,)
            values = next_values


def _validate(checkpoint, tapes):
    # Forward validated each saved write; check only how the interval fits together.
    m, h = tapes[0].update.shape
    streams = tapes[0].num_streams
    if (
        checkpoint.shape != (m, streams * h)
        or checkpoint.dtype != torch.float32
        or not checkpoint.is_contiguous()
    ):
        raise ValueError("invalid FP32 predecessor checkpoint")
    for tape in tapes:
        if (
            tape.update.shape != (m, h)
            or tape.num_streams != streams
            or tape.update.device != checkpoint.device
        ):
            raise ValueError("inconsistent replay update geometry")
    return m, h, streams


def _column_inputs(checkpoint, tapes, requested, columns):
    m, h = tapes[0].update.shape
    if columns is None:
        columns = tuple(
            allocate_operand((m, h), checkpoint.device, columnwise=True) if need else None
            for need in requested
        )
        return (columns, *operand_buffers(columns, columnwise=True))
    columns = tuple(columns)
    if len(columns) != len(tapes):
        raise ValueError("one column entry per transition is required")
    if any((column is not None) != need for need, column in zip(requested, columns)):
        raise ValueError("column entries must match requested transitions")
    data, scales = operand_buffers(columns, columnwise=True)
    for d, q, need in zip(data, scales, requested):
        if not need:
            continue
        if (
            d is None
            or q is None
            or d.shape != (m, h)
            or d.dtype != torch.uint8
            or d.device != checkpoint.device
            or not d.is_contiguous()
        ):
            raise ValueError("invalid MXFP8 data allocation")
        if (
            q.dtype != torch.uint8
            or q.device != checkpoint.device
            or not q.is_contiguous()
            or q.numel() != triton.cdiv(m, 128) * triton.cdiv(h, 128) * 512
        ):
            raise ValueError("invalid MXFP8 scale allocation")
    return columns, data, scales


def _launch_columns(checkpoint, tapes, outputs, data, scales, requested, terminal, reader):
    m, h = tapes[0].update.shape
    streams = tapes[0].num_streams
    _replay_mx[(triton.cdiv(m, 128) * 4 * triton.cdiv(h, 128) * 4,)](
        checkpoint,
        tuple(t.update for t in tapes),
        tuple(t.coefficients for t in tapes),
        tuple(t.rstd[reader] if need else None for t, need in zip(tapes, requested)),
        tuple(t.gammas[reader] if need else None for t, need in zip(tapes, requested)),
        outputs,
        data,
        scales,
        terminal,
        m,
        h,
        streams,
        tuple(t.has_retention for t in tapes),
        terminal is not None,
        requested,
        reader,
        maxnreg=64,
        num_warps=4,
    )


@torch.no_grad()
def replay_interval(
    checkpoint, tapes, outputs=None, columns=None, *, prepare_columns=False, terminal_output=None
):
    """Return N predecessors (checkpoint identity plus N-1 scratch), and N columns.

    `prepare_columns` is a bool or one bool per transition. Column entries are
    reader-zero TE operands or None. Scratch may be reused only after all readers,
    including deferred cross-stream weight gradients, have completed.
    """
    tapes = tuple(tapes)
    if not tapes:
        if outputs or columns or terminal_output is not None:
            raise ValueError("empty interval has no outputs")
        return (), ()
    m, h, streams = _validate(checkpoint, tapes)
    requested = (
        (prepare_columns,) * len(tapes)
        if isinstance(prepare_columns, bool)
        else tuple(prepare_columns)
    )
    if len(requested) != len(tapes):
        raise ValueError("one column request per transition is required")
    outputs = (
        tuple(torch.empty_like(checkpoint) for _ in tapes[:-1])
        if outputs is None
        else tuple(outputs)
    )
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
            or out.dtype != torch.float32
            or out.device != checkpoint.device
            or not out.is_contiguous()
            or out.storage_offset() != 0
            or out.untyped_storage().nbytes() != out.numel() * out.element_size()
        ):
            raise ValueError("predecessor buffers require full contiguous FP32 storage")
        claim(out)
    supplied = columns is not None
    columns, data, scales = _column_inputs(checkpoint, tapes, requested, columns)
    if supplied:
        # Freshly allocated columns cannot alias; check only caller-supplied buffers.
        for tensor in (*data, *scales):
            if tensor is not None:
                claim(tensor)
    if any(requested):
        _launch_columns(checkpoint, tapes, outputs, data, scales, requested, terminal_output, 0)
    else:
        transitions = tapes if terminal_output is not None else tapes[:-1]
        if transitions:
            _replay_transitions[(m * triton.cdiv(h, 512),)](
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


@torch.no_grad()
def replay_columns(previous, saved, readers=None):
    """Materialize requested readers without a residual write or inverse operation."""
    _validate(previous, (saved,))
    readers = (True,) + (False,) * (saved.reader_count - 1) if readers is None else tuple(readers)
    if len(readers) != saved.reader_count:
        raise ValueError("one request per reader is required")
    columns = ()
    for reader, need in enumerate(readers):
        operands, data, scales = _column_inputs(previous, (saved,), (need,), None)
        if need:
            _launch_columns(previous, (saved,), (), data, scales, (True,), None, reader)
        columns += operands
    return columns
