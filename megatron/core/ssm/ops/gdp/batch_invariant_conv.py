# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared chronological convolution arithmetic for GDP training and inference."""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def _sequence_bounds(row, T, Cu, VARLEN: tl.constexpr, SEQUENCES: tl.constexpr):
    if VARLEN:
        lo = 0
        hi = SEQUENCES
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            start = tl.load(Cu + mid)
            if start <= row:
                lo = mid
            else:
                hi = mid
        seq = lo
        start = tl.load(Cu + seq)
        end = tl.load(Cu + seq + 1)
    else:
        seq = row // T
        start = seq * T
        end = start + T
    return seq, start, end


@triton.jit
def _conv_forward(
    X,
    W,
    Bias,
    Initial,
    Slots,
    Cu,
    Y,
    P,
    T,
    D: tl.constexpr,
    C: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    VARLEN: tl.constexpr,
    SEQUENCES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    channel = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.program_id(1)
    valid = channel < D
    seq, start, end = _sequence_bounds(row, T, Cu, VARLEN, SEQUENCES)
    slot = tl.load(Slots + seq)
    if slot < 0:
        tl.store(Y + row * D + channel, 0.0, valid)
        tl.store(P + row * D + channel, 0.0, valid)
        return
    acc = tl.full((BLOCK,), 0.0, tl.float32)
    if HAS_BIAS:
        acc = tl.load(Bias + channel, valid, other=0).to(tl.float32)
    for tap in tl.static_range(C):
        source = row + tap - C + 1
        x = tl.load(X + source * D + channel, valid & (source >= start), other=0).to(tl.float32)
        if HAS_INITIAL:
            previous = tl.load(
                Initial + (slot * D + channel) * C + source - start + C,
                valid & (source < start),
                other=0,
            ).to(tl.float32)
            x = tl.where(source < start, previous, x)
        w = tl.load(W + channel * C + tap, valid, other=0).to(tl.float32)
        acc = acc + x * w
    y = acc / (1.0 + tl.exp(-acc))
    tl.store(P + row * D + channel, acc, valid)
    tl.store(Y + row * D + channel, y, valid)


@triton.jit
def _conv_update(
    X,
    State,
    Slots,
    Cu,
    T,
    D: tl.constexpr,
    C: tl.constexpr,
    VARLEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    seq = tl.program_id(0)
    slot = tl.load(Slots + seq)
    if slot < 0:
        return
    if VARLEN:
        start = tl.load(Cu + seq)
        length = tl.load(Cu + seq + 1) - start
    else:
        start = seq * T
        length = T
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    channel, column = offset // C, offset % C
    valid = channel < D
    source = length + column - C
    x = tl.load(X + (start + source) * D + channel, valid & (source >= 0), other=0)
    old = tl.load(State + (slot * D + channel) * C + length + column, valid & (source < 0), other=0)
    tl.store(State + slot * D * C + offset, tl.where(source >= 0, x, old), valid)


@triton.jit
def _conv_dx(
    DP,
    W,
    Cu,
    DX,
    T,
    D: tl.constexpr,
    C: tl.constexpr,
    VARLEN: tl.constexpr,
    SEQUENCES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    channel = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.program_id(1)
    valid = channel < D
    seq, start, end = _sequence_bounds(row, T, Cu, VARLEN, SEQUENCES)
    dx = tl.full((BLOCK,), 0.0, tl.float32)
    for tap in tl.static_range(C):
        output = row + C - 1 - tap
        dp = tl.load(DP + output * D + channel, valid & (output < end), other=0)
        w = tl.load(W + channel * C + tap, valid, other=0).to(tl.float32)
        dx = dx + dp * w
    tl.store(DX + row * D + channel, dx, valid)


def _forward(x, weight, bias, initial, slots, cu):
    b, t, d = x.shape
    c = weight.shape[-1]
    if x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("Canonical convolution requires contiguous CUDA BF16 input")
    if c != 4 or weight.numel() != d * c or not weight.is_contiguous() or weight.device != x.device:
        raise ValueError("Canonical GDP convolution requires contiguous width-four weights")
    sequences = b if cu is None else cu.numel() - 1
    if cu is not None and (
        b != 1 or cu.dtype != torch.int32 or not cu.is_contiguous() or cu.device != x.device
    ):
        raise ValueError("Packed convolution requires B=1 and int32 CUDA offsets")
    if (
        slots.shape != (sequences,)
        or slots.dtype != torch.int32
        or slots.device != x.device
        or not slots.is_contiguous()
    ):
        raise ValueError("Convolution slots must be contiguous int32 on the input device")
    if initial is not None and (
        initial.shape[1:] != (d, c) or initial.dtype != x.dtype or not initial.is_contiguous()
    ):
        raise ValueError("Convolution cache must be contiguous BF16 [slots,channels,4]")
    y = torch.empty_like(x)
    p = torch.empty_like(x, dtype=torch.float32)
    _conv_forward[(triton.cdiv(d, 128), b * t)](
        x,
        weight,
        weight if bias is None else bias,
        x if initial is None else initial,
        slots,
        x if cu is None else cu,
        y,
        p,
        t,
        d,
        c,
        bias is not None,
        initial is not None,
        cu is not None,
        sequences,
        128,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return y, p


class _Conv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, cu):
        sequences = x.shape[0] if cu is None else cu.numel() - 1
        slots = torch.arange(sequences, device=x.device, dtype=torch.int32)
        y, p = _forward(x, weight, bias, None, slots, cu)
        ctx.save_for_backward(x, weight, bias, cu, p)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, weight, bias, cu, p = ctx.saved_tensors
        b, t, d = x.shape
        c = weight.shape[-1]
        sigmoid = torch.sigmoid(p)
        dp = (dy.float() * sigmoid * (1.0 + p * (1.0 - sigmoid))).contiguous()
        dx = torch.empty_like(x)
        sequences = b if cu is None else cu.numel() - 1
        _conv_dx[(triton.cdiv(d, 128), b * t)](
            dp,
            weight,
            x if cu is None else cu,
            dx,
            t,
            d,
            c,
            cu is not None,
            sequences,
            128,
            num_warps=4,
            enable_fp_fusion=False,
        )
        rows = torch.arange(b * t, device=x.device)
        if cu is None:
            starts = rows.div(t, rounding_mode="floor") * t
        else:
            ids = torch.bucketize(rows, cu[1:], right=True)
            starts = cu[ids]
        dw = []
        flat = x.reshape(b * t, d)
        for tap in range(c):
            source = rows + tap - c + 1
            selected = flat[source.clamp_min(0)].float()
            selected = selected.masked_fill((source < starts)[:, None], 0.0)
            dw.append((selected * dp.reshape(b * t, d)).sum(0))
        dw = torch.stack(dw, -1).reshape_as(weight).to(weight.dtype)
        db = None if bias is None else dp.sum((0, 1)).to(bias.dtype)
        return dx, dw, db, None


def causal_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    initial_state: torch.Tensor | None = None,
    slots: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    update_state: bool = False,
) -> torch.Tensor:
    """Apply width-four causal convolution and SiLU with shared rounded forward.

    Packed offsets and unique active slots are scheduler-owned metadata; -1 slots
    are inactive. Training starts from zero history and supports first-order
    gradients. Inference can carry and update slot-indexed history without CPU sync.
    """
    if torch.is_grad_enabled() and any(z.requires_grad for z in (x, weight, bias) if z is not None):
        if initial_state is not None or update_state:
            raise ValueError("Cache mutation is inference-only")
        return _Conv.apply(x, weight, bias, cu_seqlens)
    sequences = x.shape[0] if cu_seqlens is None else cu_seqlens.numel() - 1
    if slots is None:
        slots = torch.arange(sequences, device=x.device, dtype=torch.int32)
    y, _ = _forward(x, weight, bias, initial_state, slots, cu_seqlens)
    if update_state:
        if initial_state is None:
            raise ValueError("Updating convolution history requires an explicit cache")
        _conv_update[(sequences, triton.cdiv(x.shape[2] * 4, 128))](
            x,
            initial_state,
            slots,
            x if cu_seqlens is None else cu_seqlens,
            x.shape[1],
            x.shape[2],
            4,
            cu_seqlens is not None,
            128,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return y
