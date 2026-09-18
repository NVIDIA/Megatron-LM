# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Triton kernels for GDN convolution, L2 normalization, and gate preparation."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def sequence_bounds(
    cu, t, TOTAL: tl.constexpr, LENGTH: tl.constexpr, NSEQ: tl.constexpr, SEARCH: tl.constexpr
):
    """Locate the packed sequence containing each token."""
    if NSEQ:
        query = tl.minimum(t, TOTAL - 1)
        lo = tl.full(t.shape, 0, tl.int32)
        hi = tl.full(t.shape, NSEQ, tl.int32)
        for _ in range(SEARCH):
            mid = (lo + hi + 1) // 2
            boundary = tl.load(cu + mid).to(tl.int32)
            take = boundary <= query
            lo = tl.where(take, mid, lo)
            hi = tl.where(take, hi, mid - 1)
        begin = tl.load(cu + lo).to(tl.int32)
        end = tl.load(cu + lo + 1).to(tl.int32)
    else:
        begin = (t // LENGTH) * LENGTH
        end = begin + LENGTH
    return begin, end


@triton.jit
def conv_norm_repeat_fwd(
    x,
    weight,
    bias,
    cu,
    q,
    k,
    v,
    rstd,
    TOTAL: tl.constexpr,
    LENGTH: tl.constexpr,
    SX: tl.constexpr,
    HK: tl.constexpr,
    HV: tl.constexpr,
    D: tl.constexpr,
    WIDTH: tl.constexpr,
    EPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    NSEQ: tl.constexpr,
    SEARCH: tl.constexpr,
    BT: tl.constexpr,
    gate=None,
    g=None,
    beta=None,
    alog=None,
    dtbias=None,
    FULL_PROJECTION: tl.constexpr = False,
    COPY_GATE: tl.constexpr = True,
):
    """Convolve and normalize Q/K/V while preparing the projection gates."""
    tile, head = tl.program_id(0), tl.program_id(1)
    t = tile * BT + tl.arange(0, BT)
    d = tl.arange(0, D)
    c = head * D + d
    begin, _ = sequence_bounds(cu, t, TOTAL, LENGTH, NSEQ, SEARCH)
    acc = tl.full((BT, D), 0, tl.float32)
    for w in tl.static_range(WIDTH):
        tx = t + w - WIDTH + 1
        xv = tl.load(
            x + tx[:, None] * SX + c[None, :],
            (tx[:, None] >= begin[:, None]) & (t[:, None] < TOTAL),
            0,
        ).to(tl.float32)
        wv = tl.load(weight + c * WIDTH + w).to(tl.float32)
        acc += xv * wv[None, :]
    if HAS_BIAS:
        acc += tl.load(bias + c).to(tl.float32)[None, :]
    # The unfused convolution materializes BF16 before normalization.
    act = (acc * tl.sigmoid(acc)).to(x.dtype.element_ty).to(tl.float32)
    if head < 2 * HK:
        inv = tl.rsqrt(tl.sum(act * act, 1) + EPS)
        norm = act * inv[:, None]
        h = head % HK
        for r in tl.static_range(HV // HK):
            offset = (t[:, None] * HV + h * (HV // HK) + r) * D + d[None, :]
            out = tl.where(head < HK, q + offset, k + offset)
            tl.store(out, norm, t[:, None] < TOTAL)
        tl.store(rstd + t * (2 * HK) + head, inv, t < TOTAL)
    else:
        offset = (t[:, None] * HV + head - 2 * HK) * D + d[None, :]
        tl.store(v + offset, act, t[:, None] < TOTAL)
        if FULL_PROJECTION:
            hv = head - 2 * HK
            channels: tl.constexpr = (2 * HK + HV) * D
            if COPY_GATE:
                gate_val = tl.load(
                    x + t[:, None] * SX + channels + hv * D + d[None, :], t[:, None] < TOTAL, 0
                )
                tl.store(gate + offset, gate_val, t[:, None] < TOTAL)
            raw_beta = tl.load(x + t * SX + channels + HV * D + hv, t < TOTAL, 0).to(tl.float32)
            raw_alpha = tl.load(x + t * SX + channels + HV * D + HV + hv, t < TOTAL, 0).to(
                tl.float32
            )
            a = tl.load(alog + hv).to(tl.float32)
            dt = tl.load(dtbias + hv).to(tl.float32)
            u = raw_alpha + dt
            softplus = tl.where(u > 20, u, libdevice.log1p(tl.exp(u)))
            exp_a = tl.exp(a)
            tl.store(g + t * HV + hv, -exp_a * softplus, t < TOTAL)
            tl.store(beta + t * HV + hv, tl.sigmoid(raw_beta), t < TOTAL)


@triton.jit
def fold_norm_silu_bwd(
    x,
    weight,
    bias,
    cu,
    dq,
    dk,
    dv,
    q,
    k,
    rstd,
    dz,
    TOTAL: tl.constexpr,
    LENGTH: tl.constexpr,
    SX: tl.constexpr,
    HK: tl.constexpr,
    HV: tl.constexpr,
    D: tl.constexpr,
    WIDTH: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    NSEQ: tl.constexpr,
    SEARCH: tl.constexpr,
    BT: tl.constexpr,
):
    """Fold repeated heads and differentiate L2 normalization and SiLU."""
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    head = tl.program_id(1)
    d = tl.arange(0, D)
    c = head * D + d
    if head < 2 * HK:
        h = head % HK
        grad = tl.full((BT, D), 0, tl.float32)
        for r in tl.static_range(HV // HK):
            offset = (t[:, None] * HV + h * (HV // HK) + r) * D + d[None, :]
            src = tl.where(head < HK, dq + offset, dk + offset)
            grad += tl.load(src, t[:, None] < TOTAL, 0).to(tl.float32)
        grad = grad.to(dq.dtype.element_ty).to(tl.float32)
        offset = (t[:, None] * HV + h * (HV // HK)) * D + d[None, :]
        src = tl.where(head < HK, q + offset, k + offset)
        y = tl.load(src, t[:, None] < TOTAL, 0).to(tl.float32)
        inv = tl.load(rstd + t * (2 * HK) + head, t < TOTAL, 0)
        dot = tl.sum(grad * y, 1)
        dy = grad * inv[:, None] - dot[:, None] * y * inv[:, None]
        dy = dy.to(dq.dtype.element_ty).to(tl.float32)
    else:
        offset = (t[:, None] * HV + head - 2 * HK) * D + d[None, :]
        dy = tl.load(dv + offset, t[:, None] < TOTAL, 0).to(tl.float32)

    begin, _ = sequence_bounds(cu, t, TOTAL, LENGTH, NSEQ, SEARCH)
    acc = tl.full((BT, D), 0, tl.float32)
    for w in tl.static_range(WIDTH):
        tx = t + w - WIDTH + 1
        xv = tl.load(
            x + tx[:, None] * SX + c[None, :],
            (tx[:, None] >= begin[:, None]) & (t[:, None] < TOTAL),
            0,
        ).to(tl.float32)
        wv = tl.load(weight + c * WIDTH + w).to(tl.float32)
        acc += xv * wv[None, :]
    if HAS_BIAS:
        acc += tl.load(bias + c).to(tl.float32)[None, :]
    # FLA recomputes and stores the preactivation in the input dtype.
    acc = acc.to(x.dtype.element_ty).to(tl.float32)
    sig = tl.sigmoid(acc)
    out = dy * sig * (1 + acc * (1 - sig))
    offset = t[:, None] * ((2 * HK + HV) * D) + c[None, :]
    tl.store(dz + offset, out, t[:, None] < TOTAL)


@triton.jit
def projection_gates_bwd(
    x,
    alog,
    dtbias,
    dgate,
    dg,
    dbeta,
    dx,
    da_partial,
    ddt_partial,
    TOTAL: tl.constexpr,
    SX: tl.constexpr,
    HK: tl.constexpr,
    HV: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
):
    """Pack gate gradients and partial parameter reductions."""
    tile, h = tl.program_id(0), tl.program_id(1)
    t = tile * BT + tl.arange(0, BT)
    d = tl.arange(0, D)
    channels: tl.constexpr = (2 * HK + HV) * D
    full_channels: tl.constexpr = channels + HV * D + 2 * HV
    gate_offset = (t[:, None] * HV + h) * D + d[None, :]
    gate_grad = tl.load(dgate + gate_offset, t[:, None] < TOTAL, 0)
    tl.store(
        dx + t[:, None] * full_channels + channels + h * D + d[None, :],
        gate_grad,
        t[:, None] < TOTAL,
    )
    raw_beta = tl.load(x + t * SX + channels + HV * D + h, t < TOTAL, 0).to(tl.float32)
    raw_alpha = tl.load(x + t * SX + channels + HV * D + HV + h, t < TOTAL, 0).to(tl.float32)
    a = tl.load(alog + h).to(tl.float32)
    dt = tl.load(dtbias + h).to(tl.float32)
    grad_g = tl.load(dg + t * HV + h, t < TOTAL, 0).to(tl.float32)
    grad_beta = tl.load(dbeta + t * HV + h, t < TOTAL, 0).to(tl.float32)
    # sigmoid's autograd formula consumes its rounded BF16 output.
    b = tl.sigmoid(raw_beta).to(x.dtype.element_ty).to(tl.float32)
    db = grad_beta * b * (1 - b)
    u = raw_alpha + dt
    softplus = tl.where(u > 20, u, libdevice.log1p(tl.exp(u)))
    exp_a = tl.exp(a)
    scaled_grad = grad_g * (-exp_a)
    du = scaled_grad * tl.where(u > 20, 1.0, tl.sigmoid(u))
    da = -grad_g * softplus
    tl.store(dx + t * full_channels + channels + HV * D + h, db, t < TOTAL)
    tl.store(dx + t * full_channels + channels + HV * D + HV + h, du, t < TOTAL)
    tl.store(da_partial + tile * HV + h, tl.sum(tl.where(t < TOTAL, da, 0), 0))
    tl.store(ddt_partial + tile * HV + h, tl.sum(tl.where(t < TOTAL, du, 0), 0))
