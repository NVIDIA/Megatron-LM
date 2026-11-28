# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Forward seam kernel: write the current residual and the folded RMSNorm reader operands.

``configs.choose_forward_config`` selects a GB300-tuned schedule; inference skips backward state.
"""

from functools import lru_cache
from itertools import product
from typing import NamedTuple

import torch
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SwizzledSharedLayout
from triton.language.extra.cuda import gdc_wait

from .configs import _BATCH_INVARIANT_ROWS, choose_forward_config
from .kernel_utils import (
    OUTPUT_FORMATS,
    SavedWrite,
    _coefficient,
    _e8m0_scale,
    _swizzled_scale_index,
    _tensor,
    allocate_operand,
    named_jit,
    validate_num_streams,
)


class _WriteArgs(NamedTuple):
    """Residual write; ``other_update``/``summed_update`` belong to a shortcut add (else None)."""

    residual: object
    update: object
    other_update: object
    output: object
    summed_update: object


class _ReadArgs(NamedTuple):
    """Per reader: gamma, saved rstd (None in inference), BF16 output, rowwise MXFP8 data and
    scales (each None when that format is not emitted)."""

    gammas: tuple
    rstd: tuple
    outputs: tuple
    quantized: tuple
    scales: tuple


@lru_cache
def _programmatic_launch(device):
    """PDL launches and CTA clusters need sm_90+."""
    return torch.cuda.get_device_capability(device)[0] >= 9


@torch.no_grad()
def forward(
    residual,
    updates,
    write,
    retain,
    reads,
    gammas,
    *,
    outputs=None,
    compact_scales_for_tp=None,
    num_streams=3,
    epsilon=1.0e-5,
    inference=False,
    batch_invariant=False,
):
    """Write the current residual and one or two RMSNorm readers of it.

    ``updates`` is one BF16 ``[M, H]`` update or a shortcut add of two (its BF16 sum is rounded
    before the write); the ``[M, S * H]`` residual keeps its dtype. All arithmetic is FP32 and
    readers fold unrounded values. ``retain`` None is a static identity retention (not zero
    logits). ``outputs`` gives each reader's format (``OUTPUT_FORMATS``, default BF16), whatever
    the residual dtype: ``"mxfp8"`` is rowwise MXFP8 and ``"both"`` emits BF16 and MXFP8 from the
    same normalized values. A true entry in compact_scales_for_tp selects the scale layout for a TE
    sequence-parallel all-gather and records it for backward replay.

    ``batch_invariant`` (inference) schedules every batch size like the smallest one.

    Returns ``(current residual, SavedWrite or None in inference, reader operands)``: per reader a
    BF16 tensor, an MXFP8 tensor, or with ``"both"`` an ``(MXFP8, BF16)`` pair.
    """
    validate_num_streams(num_streams)
    n = len(reads)
    if n not in (1, 2) or len(gammas) != n or len(updates) not in (1, 2):
        raise ValueError("one or two readers write one update or a shortcut add of two")
    outputs = ("bf16",) * n if outputs is None else tuple(outputs)
    if len(outputs) != n or any(f not in OUTPUT_FORMATS for f in outputs):
        raise ValueError(f"each reader's output format must be one of {OUTPUT_FORMATS}")
    mxfp8 = any(f != "bf16" for f in outputs)
    shortcut_add = len(updates) == 2
    if not residual.is_cuda or updates[0].ndim != 2:
        raise ValueError("expected CUDA [M,H] updates")
    m, h = updates[0].shape
    device = residual.device
    _tensor(residual, (m, num_streams * h), residual.dtype, device, "residual")
    for u in updates:
        _tensor(u, (m, h), torch.bfloat16, device, "update")
    for controller in (write, *((retain,) if retain is not None else ()), *reads):
        logits = controller.dtype in (torch.float32, torch.bfloat16)
        if not logits or controller.ndim != 1 or controller.numel() < num_streams:
            raise ValueError(f"controllers require at least {num_streams} FP32 or BF16 logits")
        _tensor(controller, controller.shape, controller.dtype, device, "controller")
    for gamma in gammas:
        _tensor(gamma, (h,), torch.bfloat16, device, "gamma")
    rows = _BATCH_INVARIANT_ROWS if inference and batch_invariant else m
    config = choose_forward_config(
        rows,
        h,
        num_streams,
        n,
        outputs=outputs,
        inference=inference,
        updates=len(updates),
        bf16_residual=residual.dtype == torch.bfloat16,
    )
    c = None if inference else torch.empty((8 * num_streams,), device=device, dtype=torch.float32)

    if type(residual).__name__ not in ("FakeTensor", "FunctionalTensor"):
        # FP32 residuals use 16-byte packet loads; the backward reads the saved residual and
        # updates through TMA (rows and H divisible by 32).
        packets = h % 4 == 0 and residual.dtype == torch.float32
        tma = not inference and m % 32 == 0 and h % 32 == 0
        aligned = ((residual,) if packets or tma else ()) + (tuple(updates) if tma else ())
        if any(t.data_ptr() % 16 for t in aligned):
            raise ValueError("FP32 packet or training TMA operands must be 16-byte aligned")
    if mxfp8 and (m % 32 or h % 32):
        raise ValueError("MXFP8 outputs require rows and hidden width divisible by 32")

    current = torch.empty_like(residual)
    summed = torch.empty_like(updates[0]) if shortcut_add and not inference else None
    stats = tuple(
        None if inference else torch.empty(m, dtype=torch.float32, device=device) for _ in range(n)
    )
    buffers, operands = [], []
    compact_scales_for_tp = (
        (False,) * n if compact_scales_for_tp is None else tuple(compact_scales_for_tp)
    )
    for fmt, compact in zip(outputs, compact_scales_for_tp, strict=True):
        bf16 = None if fmt == "mxfp8" else torch.empty_like(updates[0])
        mx = (
            None
            if fmt == "bf16"
            else allocate_operand((m, h), device, compact_scales_for_tp=compact)
        )
        # Inference pads rows in-kernel; partial scale columns still need clearing.
        if mx is not None and (h % 128 or (m % 128 and not inference)):
            mx._rowwise_scale_inv.zero_()
        buffers.append(
            (bf16, getattr(mx, "_rowwise_data", None), getattr(mx, "_rowwise_scale_inv", None))
        )
        operands.append(bf16 if mx is None else mx if bf16 is None else (mx, bf16))

    saved = None
    if not inference:
        saved = SavedWrite(
            summed if summed is not None else updates[0].detach(),
            c,
            tuple(gamma.detach() for gamma in gammas),
            stats,
            tuple(0 if v is None else v.numel() for v in (write, retain, *reads)),
            num_streams,
            retain is not None,
            len(updates),
            compact_scales_for_tp,
        )
    # Inference is a programmatic dependent launch: it reads only parameters and prefetches
    # into L2 before waiting on the previous kernel (griddepcontrol.wait).
    dependent = inference and _programmatic_launch(device)
    grid = (m, 2 if config.SPLIT else 1, config.PARTS)
    write_args = _WriteArgs(
        residual, updates[0], updates[1] if shortcut_add else None, current, summed
    )
    kwargs = dict(
        read_tensors=_ReadArgs(gammas, stats, *zip(*buffers)),
        H=h,
        STREAMS=num_streams,
        # A runtime unit retention coefficient preserves Ultra register occupancy.
        HAS_RETENTION=retain is not None or h == 10240,
        EPSILON=epsilon,
        # Logits (write, retain, read_a, read_b); read_b repeats read_a with one reader.
        controllers=(write, retain, reads[0], reads[-1]),
        saved_coefficients=c,
        COMPACT_SCALES_FOR_TP=compact_scales_for_tp,
        PAD_ROWS=inference and mxfp8 and bool(m % 128),
        DEPENDENT=dependent,
        EXCHANGE=None,
        enable_fp_fusion=False,
        launch_pdl=dependent,
        CONFIG=config,
        num_warps=config.NUM_WARPS,
        maxnreg=config.MAXNREG,
    )
    kernel = FORWARD_KERNELS[outputs]
    if config.CLUSTER > 1:
        kwargs["EXCHANGE"] = _exchange_asm(config.CLUSTER, 1 if config.SPLIT else n)
        # Warmup and launch share the ordered kwargs so Triton reuses the patched metadata.
        if not (
            _programmatic_launch(device)
            and _as_cluster(kernel, write_args, grid, config.CLUSTER, kwargs)
        ):
            kwargs.update(CONFIG=config._replace(CLUSTER=1), EXCHANGE=None)
    kernel[grid](write_args, **kwargs)
    return current, saved, tuple(operands)


def _as_cluster(kernel, write_args, grid, ctas, kwargs):
    """Launch the forward ``kernel`` as ``ctas``-CTA clusters via Triton 3.7's private packed
    metadata.

    An exchange without its peers would hang, so an unexpected form returns False (unclustered).
    """
    kernel = kernel.warmup(write_args, grid=grid, **kwargs)
    meta = kernel.packed_metadata
    if not (isinstance(meta, tuple) and len(meta) == 3 and meta[1] in (1, ctas)):
        return False
    kernel.packed_metadata = (meta[0], ctas, meta[2])
    return True


@lru_cache
def _exchange_asm(ctas, partials):
    """PTX ``(init, exchange, constraints)``: peers ``st.async`` reader k's partial into
    ``xbuf[k * ctas + rank]``; every CTA sums the partials in rank order, so all compute
    identical bits.

    Inline PTX: Gluon cannot address a peer CTA's shared memory (``mapa.shared::cluster``,
    ``st.async.shared::cluster``), which the cluster schedule's partial-norm exchange needs;
    exchanging through global memory instead would defeat the latency-bound decode schedule.
    """
    init = f"""
        .shared .align 8 .b64 xbar;
        .shared .align 16 .f32 xbuf[{ctas * partials}];
        {{
            .reg .pred lead;
            .reg .u32 tid, mbar;
            mov.u32 tid, %tid.x;
            setp.eq.u32 lead, tid, 0;
            mov.u32 mbar, xbar;
            @lead mbarrier.init.shared::cta.b64 [mbar], 1;
            @lead mbarrier.arrive.expect_tx.shared::cta.b64 _, [mbar], {(ctas - 1) * partials * 4};
            fence.mbarrier_init.release.cluster;
            barrier.cluster.arrive.relaxed.aligned;
            mov.u32 $0, 0;
        }}
    """
    lines = ["""
        {
            .reg .pred lead, send, own, done;
            .reg .u32 tid, rank, buf, mbar, slot, peer_mbar, peer_slot;
            .reg .f32 v;
            mov.u32 tid, %tid.x;
            mov.u32 rank, %cluster_ctarank;
            setp.eq.u32 lead, tid, 0;
            mov.u32 buf, xbuf;
            mov.u32 mbar, xbar;
            mad.lo.u32 slot, rank, 4, buf;
            barrier.cluster.wait.aligned;
    """]
    for peer in range(ctas):
        lines += [
            f"setp.ne.and.u32 send, rank, {peer}, lead;",
            f"@send mapa.shared::cluster.u32 peer_mbar, mbar, {peer};",
            f"@send mapa.shared::cluster.u32 peer_slot, slot, {peer};",
        ]
        for k in range(partials):
            if k:
                lines.append(f"@send add.u32 peer_slot, peer_slot, {ctas * 4};")
            lines.append(
                f"@send st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [peer_slot], ${partials + k}, [peer_mbar];"
            )
    lines += [
        "WAIT_${:uid}:",
        "mbarrier.try_wait.parity.shared::cta.b64 done, [mbar], 0;",
        "@!done bra WAIT_${:uid};",
    ]
    for k in range(partials):
        for peer in range(ctas):
            lines += [
                f"setp.eq.u32 own, rank, {peer};",
                f"@own mov.f32 v, ${partials + k};",
                f"@!own ld.shared.f32 v, [buf + {(k * ctas + peer) * 4}];",
            ]
            if peer:
                lines.append(f"add.rn.f32 ${k}, ${k}, v;")
            else:
                lines.append(f"mov.f32 ${k}, v;")
    lines.append("}")
    constraints = ",".join(["=f"] * partials + ["f"] * partials)
    return init, "\n".join(lines), constraints


@g.jit
def _load_residual(
    write_tensors, row, stream, cols, H: gl.constexpr, STREAMS: gl.constexpr, CONFIG: gl.constexpr
):
    """One stream's residual fragment: FP32 4-wide packets when H allows, else plain loads.

    Inline PTX: each thread issues one predicated 16-byte read-only load
    (``ld.global.nc.v4.f32``) per packet. ``gl.load`` emits a plain ``ld.global.v4`` (no
    ``.nc``), which measured up to 22% slower in the training forward (hidden 3072, 16384 rows:
    197 -> 241 us).
    """
    residual = write_tensors.residual + row * (STREAMS * H) + stream * H + cols
    if write_tensors.residual.dtype.element_ty != gl.float32 or H % 4:
        old = gl.load(residual, cols < H, 0)  # BF16 stays packed until _mix_fragment widens it
    else:
        # Packets are wholly valid or wholly masked; whole tiles pass a constant-true predicate.
        if H % CONFIG.WIDTH:
            valid = (cols < H).to(gl.int32)
        else:
            valid = gl.full(cols.shape, 1, gl.int32, cols.type.layout)
        old = gl.inline_asm_elementwise(
            """
            {
                .reg .pred p;
                setp.ne.u32 p, $8, 0;
                mov.b32 $0, 0;
                mov.b32 $1, 0;
                mov.b32 $2, 0;
                mov.b32 $3, 0;
                @p ld.global.nc.v4.f32 {$0,$1,$2,$3},[$4];
            }
            """,
            "=f,=f,=f,=f,l,l,l,l,r,r,r,r",
            (residual.to(gl.uint64), valid),
            dtype=gl.float32,
            is_pure=False,
            pack=4,
        )
    return old


@g.jit
def _load_fragment(
    write_tensors, row, cols, H: gl.constexpr, STREAMS: gl.constexpr, CONFIG: gl.constexpr
):
    """``(update, other update or update again, per-stream residuals with LOADS_FIRST)``."""
    mask = cols < H
    update = gl.load(write_tensors.update + (row * H + cols), mask, 0)
    other = update  # unused without a shortcut add
    if write_tensors.other_update is not None:
        other = gl.load(write_tensors.other_update + (row * H + cols), mask, 0)
    olds = ()
    if CONFIG.LOADS_FIRST:
        for stream in gl.static_range(STREAMS):
            olds += (_load_residual(write_tensors, row, stream, cols, H, STREAMS, CONFIG),)
    return update, other, olds


@g.jit
def _mix_fragment(
    write_tensors,
    coefficients,
    row,
    cols,
    H: gl.constexpr,
    STREAMS: gl.constexpr,
    HAS_RETENTION: gl.constexpr,
    TWO_READERS: gl.constexpr,
    CONFIG: gl.constexpr,
    OWNER,
    LOADED=None,
):
    """Store one fragment's residual streams (columns in OWNER) and return both readers'
    pre-norm inputs (zero outside the row; the second is zero with one reader)."""
    mask = cols < H
    if LOADED is None:
        LOADED = _load_fragment(write_tensors, row, cols, H, STREAMS, CONFIG)
    update, other, olds = LOADED
    update = update.to(gl.float32)
    if write_tensors.other_update is not None:  # shortcut add
        other = other.to(gl.float32)
        update = (update + other).to(gl.bfloat16).to(gl.float32)
        if write_tensors.summed_update is not None:
            gl.store(write_tensors.summed_update + row * H + cols, update, mask)
    x0 = gl.full(mask.shape, 0, gl.float32, mask.type.layout)
    x1 = gl.full(mask.shape, 0, gl.float32, mask.type.layout)
    update_mix0 = 0.0
    update_mix1 = 0.0
    for stream in gl.static_range(STREAMS):
        owner = OWNER & (not CONFIG.SPLIT or gl.program_id(1) == (stream == STREAMS - 1))
        if CONFIG.LOADS_FIRST:
            old = olds[stream]
        else:
            old = _load_residual(write_tensors, row, stream, cols, H, STREAMS, CONFIG)
        old = old.to(gl.float32)  # exact for BF16

        write = _coefficient(coefficients, stream)
        retain = _coefficient(coefficients, STREAMS + stream) if HAS_RETENTION else 1.0

        # One rounded retain product, then a fused write: the rounding replay reproduces.
        value = gl.fma(write, update, retain * old)

        stored = value.to(write_tensors.output.dtype.element_ty)  # RN for a BF16 residual
        if not CONFIG.STORE_LATE:
            gl.store(
                write_tensors.output + row * (STREAMS * H) + stream * H + cols, stored, mask & owner
            )
        beta0 = _coefficient(coefficients, 2 * STREAMS + stream)
        x0 = gl.fma(beta0 * retain, old, x0)
        update_mix0 = gl.fma(beta0, write, update_mix0)
        if TWO_READERS:
            beta1 = _coefficient(coefficients, 3 * STREAMS + stream)
            x1 = gl.fma(beta1 * retain, old, x1)
            update_mix1 = gl.fma(beta1, write, update_mix1)
        if CONFIG.STORE_LATE:
            gl.store(
                write_tensors.output + row * (STREAMS * H) + stream * H + cols, stored, mask & owner
            )
    x0 = gl.fma(update_mix0, update, x0)
    if TWO_READERS:
        x1 = gl.fma(update_mix1, update, x1)
    return gl.where(mask, x0, 0.0), gl.where(mask, x1, 0.0)


@g.jit
def _owns_columns(cols, H: gl.constexpr, CONFIG: gl.constexpr):
    """Columns (or 32-column groups) this CTA stores of its row's PARTS replicas.

    A CLUSTER CTA's columns already lie in its own slice (WIDTH divides H / CLUSTER).
    """
    if CONFIG.SPLIT and CONFIG.PARTS == 2:
        owned = (cols < H // 2) == (gl.program_id(2) == 0)
    else:
        owned = True if CONFIG.PARTS == 1 else cols // (H // CONFIG.PARTS) == gl.program_id(2)
    return owned


@g.jit
def _emit_mxfp8(y, data, scale, coordinates, COMPACT_SCALES_FOR_TP: gl.constexpr, rows=None):
    """Store MXFP8 data and compact or GEMM-swizzled E8M0 scales of 32-value blocks per row.

    ``rows`` zeroes padded-row scales.
    """
    exponent, bits = _e8m0_scale(gl.max(gl.abs(y), 1))
    outer, inner, width = coordinates
    stride = gl.cdiv(width, 128) * 4
    index = (
        outer * stride + inner
        if COMPACT_SCALES_FOR_TP
        else _swizzled_scale_index(outer, inner, width)
    )
    scale_pointer, scale_mask = scale
    data_pointer, data_mask = data
    if rows is not None:
        for offset in gl.static_range(32, 128, 32):
            padding = (
                (outer // 128 == (rows - 1) // 128) & (outer % 128 < 32) & (outer + offset >= rows)
            )
            gl.store(
                scale_pointer
                + index
                + (offset * stride if COMPACT_SCALES_FOR_TP else offset // 32 * 4),
                0,
                scale_mask & padding,
            )
    q = y * gl.expand_dims(bits.to(gl.float32, bitcast=True), 1)
    q = q.to(gl.float8e4nv).to(gl.uint8, bitcast=True)
    gl.store(data_pointer, gl.reshape(q, data_pointer.shape), data_mask)
    gl.store(scale_pointer + index, exponent.to(gl.uint8), scale_mask)


@g.jit
def _emit(
    fragment,
    reads,
    target: gl.constexpr,
    row,
    cols,
    H: gl.constexpr,
    CONFIG: gl.constexpr,
    COMPACT_SCALES_FOR_TP: gl.constexpr,
    rows,
):
    """Normalize a queued fragment and emit its reader's requested formats.
    When supplied, rows gives the true row count for padding the MXFP8 scales.
    """
    x, weight, rstd = fragment
    output, quantized, scales = reads.outputs[target], reads.quantized[target], reads.scales[target]
    mask = (cols < H) & _owns_columns(cols, H, CONFIG)
    y = (x * rstd) * weight
    if output is not None:
        gl.store(output + row * H + cols, y.to(gl.bfloat16), mask)
    if quantized is not None:
        grouped = gl.reshape(y, (x.numel // 32, 32))
        group = gl.min(gl.reshape(cols, (x.numel // 32, 32)), 1) // 32
        _emit_mxfp8(
            grouped,
            (quantized + row * H + cols, mask),
            (scales, (group < H // 32) & _owns_columns(group, H // 32, CONFIG)),
            (row, group, H),
            COMPACT_SCALES_FOR_TP,
            rows,
        )


@g.jit
def _prefetch_l2(pointers, cols, H: gl.constexpr):
    """Prefetch the 128-byte lines under ``pointers`` (columns < H) into L2.

    Inline PTX: Gluon has no L2 prefetch (``prefetch.global.L2``). Dependent (PDL) decode
    launches warm their row while the previous kernel finishes; removing it measured up to 15%
    slower decode (hidden 3072, 256 rows, two readers).
    """
    LINE: gl.constexpr = 1024 // pointers.dtype.element_ty.primitive_bitwidth  # elements
    first = (cols < H) & (cols % LINE == 0)  # one prefetch per line
    gl.inline_asm_elementwise(
        """
        {
            .reg .pred p;
            setp.ne.u32 p, $2, 0;
            @p prefetch.global.L2 [$1];
            mov.u32 $0, 0;
        }
        """,
        "=r,l,r",
        [pointers.to(gl.uint64), first.to(gl.int32)],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@g.jit
def _prepare_coefficients(
    controllers, C, row, TWO_READERS: gl.constexpr, STREAMS: gl.constexpr, RETENTION: gl.constexpr
):
    """Return [write, retain, read_a, read_b] x STREAMS coefficients from the logits.

    Row zero also saves them and their sigmoid derivative terms to ``C`` (None in inference).
    """
    L: gl.constexpr = BlockedLayout([1], [32], [gl.num_warps()], [0])
    i = gl.arange(0, triton.next_power_of_2(4 * STREAMS), layout=L)
    controller, stream = i // STREAMS, i % STREAMS
    # One masked load per controller: each may be FP32 or BF16 and is widened exactly.
    logit = gl.zeros(i.shape, gl.float32, layout=L)
    for index in gl.static_range(4):
        if index != 1 or RETENTION:
            selected = (controller == index) & (i < 4 * STREAMS)
            logit += gl.load(controllers[index] + stream, selected, 0).to(gl.float32)
    value = 1.0 / (1.0 + gl.exp(gl.where(controller == 1, logit, -logit)))
    value = gl.where(
        ((controller != 1) | RETENTION) & ((controller != 3) | TWO_READERS), value, 0.0
    )
    factor = gl.where(controller == 0, 2.0, gl.where(controller == 1, 0.1, 1.0))
    coefficient = gl.where(controller == 1, gl.fma(-0.1, value, 1.0), factor * value)
    if C is not None:
        gl.store(C + i, coefficient, (row == 0) & (i < 4 * STREAMS))
        gl.store(
            C + i + 4 * STREAMS, (factor * value) * (1 - value), (row == 0) & (i < 4 * STREAMS)
        )
    return coefficient


def _forward_body(
    write_tensors,
    saved_coefficients,
    read_tensors,
    controllers,
    H: gl.constexpr,
    STREAMS: gl.constexpr,
    HAS_RETENTION: gl.constexpr,
    CONFIG: gl.constexpr,
    EPSILON: gl.constexpr = 1.0e-5,
    COMPACT_SCALES_FOR_TP: gl.constexpr = (False, False),
    PAD_ROWS: gl.constexpr = False,
    DEPENDENT: gl.constexpr = False,
    EXCHANGE: gl.constexpr = None,
):
    """One row (or slice) per CTA. DEPENDENT (inference PDL) prefetches the row into L2 from
    parameters only before ``gdc_wait``; EXCHANGE is ``_exchange_asm``'s PTX, or None."""
    gammas = read_tensors.gammas
    TWO_READERS: gl.constexpr = len(gammas) == 2  # Split CTAs still mix both updates.
    K: gl.constexpr = CONFIG.CLUSTER
    if CONFIG.SPLIT:
        reader = gl.program_id(1)
        gammas = (gl.where(reader == 0, gammas[0], gammas[1]),)
        read = gl.where(reader == 0, controllers[2], controllers[3])
        controllers = (controllers[0], controllers[1], read, controllers[3])
    row = gl.program_id(0) // K
    lead = gl.program_id(0) % K == 0  # one owner of per-row statistics
    # Only row zero saves coefficients (in a cluster, only its lead CTA).
    coefficients = _prepare_coefficients(
        controllers,
        saved_coefficients,
        gl.where(lead, row, 1),
        TWO_READERS,
        STREAMS,
        controllers[1] is not None,
    )
    width: gl.constexpr = CONFIG.WIDTH
    warps: gl.constexpr = gl.num_warps()
    lane_layout: gl.constexpr = BlockedLayout([1], [32], [warps], [0])
    if CONFIG.REDUCE_VECTOR and K == 1:
        vector_layout: gl.constexpr = BlockedLayout([CONFIG.REDUCE_VECTOR], [32], [warps], [0])
    paired: gl.constexpr = CONFIG.PAIRED_REDUCE and not CONFIG.SPLIT and K == 1
    NUM_SEGMENTS: gl.constexpr = triton.cdiv(H // K, width)

    cols = gl.arange(
        0, width, layout=gl.constexpr(BlockedLayout([CONFIG.VECTOR], [32], [gl.num_warps()], [0]))
    )
    cols += (gl.program_id(0) % K) * (H // K)  # this CTA's column slice
    if DEPENDENT:
        # Decode rows are latency bound: warm L2 before waiting. A prefetch is only a hint and
        # L2 is coherent, so it never observes a stale value; the real loads follow the wait.
        for segment in gl.static_range(NUM_SEGMENTS):
            cc = cols + segment * width
            for stream in gl.static_range(STREAMS):
                _prefetch_l2(write_tensors.residual + row * (STREAMS * H) + stream * H + cc, cc, H)
            for reader in gl.static_range(len(gammas)):
                _prefetch_l2(gammas[reader] + cc, cc, H)
        gdc_wait()
    mixed = ()
    totals = (0.0,) * len(gammas)
    loaded = ()
    if CONFIG.LOADS_FIRST:
        for segment in gl.static_range(NUM_SEGMENTS):
            loaded += (
                _load_fragment(write_tensors, row, cols + segment * width, H, STREAMS, CONFIG),
            )
    if EXCHANGE is not None:
        # With the loads in flight, arm this CTA's exchange barrier and signal the cluster
        # (``_exchange_asm``'s PTX: Gluon cannot address peer shared memory).
        zero = gl.full([1], 0, gl.int32, layout=lane_layout)
        gl.inline_asm_elementwise(
            EXCHANGE[0], "=r,r", [zero], dtype=gl.int32, is_pure=False, pack=1
        )

    for segment in gl.static_range(NUM_SEGMENTS):
        cc = cols + segment * width
        owner = _owns_columns(cc, H, CONFIG)
        fragments = _mix_fragment(
            write_tensors,
            coefficients,
            row,
            cc,
            H,
            STREAMS,
            HAS_RETENTION,
            TWO_READERS,
            CONFIG,
            OWNER=owner,
            LOADED=loaded[segment] if CONFIG.LOADS_FIRST else None,
        )
        next_totals = ()
        for reader in gl.static_range(len(gammas)):
            x = fragments[reader]
            fragment = x
            if segment < CONFIG.OFFLOAD_SEGMENTS:
                fragment = gl.allocate_shared_memory(
                    gl.float32, [width], gl.constexpr(SwizzledSharedLayout(1, 1, 1, [0])), x
                )
            # Group square sums to balance reduction work and register pressure.
            squares = gl.reshape(x * x, (width // 16, 16))
            next_totals += (totals[reader] + gl.sum(squares, 1),)
            mixed += (fragment,)
        totals = next_totals

    if EXCHANGE is not None:
        # Exchange the partial sums of squares across the cluster (``_exchange_asm``'s PTX).
        partials = ()
        for reader in gl.static_range(len(gammas)):
            partials += (gl.zeros([1], gl.float32, layout=lane_layout) + gl.sum(totals[reader], 0),)
        totals = gl.inline_asm_elementwise(
            EXCHANGE[1],
            EXCHANGE[2],
            list(partials),
            dtype=(gl.float32,) * len(partials),
            is_pure=False,
            pack=1,
        )
    queue = ()
    # The final reader also drains the queued stages.
    for reader in gl.static_range(len(gammas)):
        for segment in gl.static_range(
            NUM_SEGMENTS + (CONFIG.PIPELINE_DEPTH if reader == len(gammas) - 1 else 0)
        ):
            if segment < NUM_SEGMENTS:
                cc = cols + segment * width
                # Keep the segment as a load immediate after forming the base pointer.
                weight = gl.load(
                    gammas[reader] + cols + segment * width,
                    (cc < H) & _owns_columns(cc, H, CONFIG),
                    0,
                ).to(gl.float32)

                if segment == 0 and (not paired or reader == 0):
                    if paired:
                        if read_tensors.scales[0] is not None:
                            totals = (
                                gl.convert_layout(totals[0], lane_layout),
                                gl.convert_layout(totals[1], lane_layout),
                            )
                        pair_layout: gl.constexpr = BlockedLayout(
                            [1, 2], [32, 1], [warps, 1], [0, 1]
                        )
                        norm_total = gl.convert_layout(gl.join(totals[0], totals[1]), pair_layout)
                    else:
                        norm_total = totals[reader]
                        if CONFIG.REDUCE_VECTOR and K == 1:
                            norm_total = gl.convert_layout(norm_total, vector_layout)
                    rstd = gl.rsqrt(gl.sum(norm_total, 0) / H + EPSILON)
                    norms = gl.split(rstd) if paired else (rstd,)
                    if read_tensors.rstd[0] is not None:
                        for i in gl.static_range(len(norms)):
                            gl.store(
                                read_tensors.rstd[i if paired else reader] + row,
                                norms[i],
                                mask=lead,
                            )
                prefetch_rstd = norms[reader if paired else 0]
                x = mixed[segment * len(gammas) + reader]
                if segment < CONFIG.OFFLOAD_SEGMENTS:
                    x = x.load(cols.type.layout)
                queue += ((x, weight, prefetch_rstd),)

            if reader * NUM_SEGMENTS + segment >= CONFIG.PIPELINE_DEPTH:
                stage = reader * NUM_SEGMENTS + segment - CONFIG.PIPELINE_DEPTH
                for target in gl.static_range(len(read_tensors.gammas)):
                    if target == (gl.program_id(1) if CONFIG.SPLIT else stage // NUM_SEGMENTS):
                        output_cols = cols + (stage % NUM_SEGMENTS) * width
                        rows = gl.num_programs(0) // K if PAD_ROWS else None
                        _emit(
                            queue[0],
                            read_tensors,
                            target,
                            row,
                            output_cols,
                            H,
                            CONFIG,
                            COMPACT_SCALES_FOR_TP[target],
                            rows,
                        )
                queue = queue[1:]


def _forward_kernel_name(outputs):
    """Include the reader count and each reader's output formats in the profiler name."""
    tags = "_".join("bf16mxfp8" if fmt == "both" else fmt for fmt in outputs)
    return f"fwd_wr_writeread_{len(outputs)}reader_{tags}"


FORWARD_KERNELS = {
    formats: named_jit(_forward_body, _forward_kernel_name(formats))
    for readers in (1, 2)
    for formats in product(OUTPUT_FORMATS, repeat=readers)
}
