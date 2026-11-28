# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GDN/KDA output RMSNorm fusion preserving intermediate activation rounding."""

import torch
import triton
import triton.language as tl


@triton.jit
def _row_offset(
    row,
    SEQUENCE: tl.constexpr,
    HEADS: tl.constexpr,
    STRIDE_B: tl.constexpr,
    STRIDE_S: tl.constexpr,
    STRIDE_H: tl.constexpr,
    FLAT_TOKENS: tl.constexpr,
):
    """Address logical [batch, sequence, head] rows without copying tensor views."""
    token = row // HEADS
    if FLAT_TOKENS:
        offset = token * STRIDE_S
    else:
        offset = (token // SEQUENCE) * STRIDE_B + (token % SEQUENCE) * STRIDE_S
    return offset + (row % HEADS) * STRIDE_H


@triton.jit
def gated_norm_fwd(
    x,
    gate,
    weight,
    out,
    rstd,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    SEQUENCE: tl.constexpr,
    X_STRIDES: tl.constexpr,
    GATE_STRIDES: tl.constexpr,
    X_FLAT_TOKENS: tl.constexpr,
    GATE_FLAT_TOKENS: tl.constexpr,
    EPS: tl.constexpr,
    ZERO_CENTERED: tl.constexpr,
    BT: tl.constexpr,
    GATE_ACTIVATION: tl.constexpr,
):
    """Apply output RMSNorm and SiLU or sigmoid gating."""
    row = tl.program_id(0) * BT + tl.arange(0, BT)
    d = tl.arange(0, D)
    off = row[:, None] * D + d[None, :]
    if X_FLAT_TOKENS and X_STRIDES[1] == HEADS * D and X_STRIDES[2] == D and X_STRIDES[3] == 1:
        xoff = off
    else:
        xrow = _row_offset(
            row, SEQUENCE, HEADS, X_STRIDES[0], X_STRIDES[1], X_STRIDES[2], X_FLAT_TOKENS
        )
        xoff = xrow[:, None] + d[None, :] * X_STRIDES[3]
    xv = tl.load(x + xoff, row[:, None] < ROWS, 0).to(tl.float32)
    w = tl.load(weight + d).to(tl.float32)
    if ZERO_CENTERED:
        w += 1
    inv = tl.rsqrt(tl.sum(xv * xv, 1) / D + EPS)
    # TE materializes RMSNorm in the activation dtype before FP32 gating.
    norm = (xv * inv[:, None] * w[None, :]).to(x.dtype.element_ty).to(tl.float32)
    grow = _row_offset(
        row, SEQUENCE, HEADS, GATE_STRIDES[0], GATE_STRIDES[1], GATE_STRIDES[2], GATE_FLAT_TOKENS
    )
    goff = grow[:, None] + d[None, :] * GATE_STRIDES[3]
    gv = tl.load(gate + goff, row[:, None] < ROWS, 0).to(tl.float32)
    if GATE_ACTIVATION == "sigmoid":
        result = norm * tl.sigmoid(gv)
    else:
        result = norm * (gv * tl.sigmoid(gv))
    tl.store(out + off, result, row[:, None] < ROWS)
    tl.store(rstd + row, inv, row < ROWS)


@triton.jit
def gated_norm_bwd(
    x,
    gate,
    weight,
    rstd,
    dy,
    dx,
    dgate,
    dw_partial,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    SEQUENCE: tl.constexpr,
    X_STRIDES: tl.constexpr,
    GATE_STRIDES: tl.constexpr,
    X_FLAT_TOKENS: tl.constexpr,
    GATE_FLAT_TOKENS: tl.constexpr,
    ZERO_CENTERED: tl.constexpr,
    BT: tl.constexpr,
    GATE_ACTIVATION: tl.constexpr,
):
    """Differentiate output RMSNorm and SiLU or sigmoid gating."""
    tile = tl.program_id(0)
    row = tile * BT + tl.arange(0, BT)
    d = tl.arange(0, D)
    off = row[:, None] * D + d[None, :]
    if X_FLAT_TOKENS and X_STRIDES[1] == HEADS * D and X_STRIDES[2] == D and X_STRIDES[3] == 1:
        xoff = off
    else:
        xrow = _row_offset(
            row, SEQUENCE, HEADS, X_STRIDES[0], X_STRIDES[1], X_STRIDES[2], X_FLAT_TOKENS
        )
        xoff = xrow[:, None] + d[None, :] * X_STRIDES[3]
    xv = tl.load(x + xoff, row[:, None] < ROWS, 0).to(tl.float32)
    w = tl.load(weight + d).to(tl.float32)
    if ZERO_CENTERED:
        w += 1
    inv = tl.load(rstd + row, row < ROWS, 0)
    xn = xv * inv[:, None]
    norm = (xn * w[None, :]).to(x.dtype.element_ty).to(tl.float32)
    grad = tl.load(dy + off, row[:, None] < ROWS, 0).to(tl.float32)
    grow = _row_offset(
        row, SEQUENCE, HEADS, GATE_STRIDES[0], GATE_STRIDES[1], GATE_STRIDES[2], GATE_FLAT_TOKENS
    )
    goff = grow[:, None] + d[None, :] * GATE_STRIDES[3]
    gv = tl.load(gate + goff, row[:, None] < ROWS, 0).to(tl.float32)
    sig = tl.sigmoid(gv)
    if GATE_ACTIVATION == "sigmoid":
        grad_norm = (grad * sig).to(x.dtype.element_ty).to(tl.float32)
        grad_gate = (grad * norm) * (sig * (1 - sig))
    else:
        grad_norm = (grad * (gv * sig)).to(x.dtype.element_ty).to(tl.float32)
        grad_gate = (grad * norm) * (sig * (1 + gv * (1 - sig)))
    weighted_grad = grad_norm * w[None, :]
    dot = tl.sum(weighted_grad * xn, 1) / D
    grad_x = (weighted_grad - xn * dot[:, None]) * inv[:, None]
    tl.store(dx + off, grad_x, row[:, None] < ROWS)
    tl.store(dgate + off, grad_gate, row[:, None] < ROWS)
    dw = tl.sum(tl.where(row[:, None] < ROWS, grad_norm * xn, 0), 0)
    tl.store(dw_partial + tile * D + d, dw)


def forward_impl(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    zero_centered: bool = False,
    bt: int = 4,
    warps: int = 2,
    gate_activation: str = "silu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize and gate [batch, sequence, head, dimension] tensor views."""
    batch, sequence, heads, d = x.shape
    rows = batch * sequence * heads
    x_strides, gate_strides = x.stride(), gate.stride()
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    rstd = torch.empty((rows,), device=x.device, dtype=torch.float32)
    gated_norm_fwd[(triton.cdiv(rows, bt),)](
        x,
        gate,
        weight,
        out,
        rstd,
        ROWS=rows,
        HEADS=heads,
        D=d,
        SEQUENCE=sequence,
        X_STRIDES=x_strides,
        GATE_STRIDES=gate_strides,
        X_FLAT_TOKENS=batch == 1 or x_strides[0] == sequence * x_strides[1],
        GATE_FLAT_TOKENS=batch == 1 or gate_strides[0] == sequence * gate_strides[1],
        EPS=eps,
        ZERO_CENTERED=zero_centered,
        BT=bt,
        GATE_ACTIVATION=gate_activation,
        num_warps=warps,
        enable_fp_fusion=True,
    )
    return out, rstd


def backward_impl(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    dy: torch.Tensor,
    zero_centered: bool = False,
    bt: int = 16,
    warps: int = 2,
    gate_activation: str = "silu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute activation, gate, and RMSNorm-weight gradients."""
    dy = dy.contiguous()
    batch, sequence, heads, d = x.shape
    rows = batch * sequence * heads
    x_strides, gate_strides = x.stride(), gate.stride()
    dx = torch.empty_like(x, memory_format=torch.contiguous_format)
    dgate = torch.empty_like(gate, memory_format=torch.contiguous_format)
    dw = torch.empty((triton.cdiv(rows, bt), d), device=x.device, dtype=torch.float32)
    gated_norm_bwd[(triton.cdiv(rows, bt),)](
        x,
        gate,
        weight,
        rstd,
        dy,
        dx,
        dgate,
        dw,
        ROWS=rows,
        HEADS=heads,
        D=d,
        SEQUENCE=sequence,
        X_STRIDES=x_strides,
        GATE_STRIDES=gate_strides,
        X_FLAT_TOKENS=batch == 1 or x_strides[0] == sequence * x_strides[1],
        GATE_FLAT_TOKENS=batch == 1 or gate_strides[0] == sequence * gate_strides[1],
        ZERO_CENTERED=zero_centered,
        BT=bt,
        GATE_ACTIVATION=gate_activation,
        num_warps=warps,
        enable_fp_fusion=True,
    )
    return dx, dgate, dw.sum(0).to(weight.dtype)


class _FusedGatedNorm(torch.autograd.Function):
    """First-order autograd for fused output gating."""

    @staticmethod
    def forward(ctx, x, gate, weight, eps, zero_centered, gate_activation):
        """Save activations and inverse norms for backward."""
        out, rstd = forward_impl(
            x, gate, weight, eps, zero_centered, gate_activation=gate_activation
        )
        ctx.save_for_backward(x, gate, weight, rstd)
        ctx.zero_centered = zero_centered
        ctx.gate_activation = gate_activation
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        """Return activation, gate, and norm-weight gradients."""
        gradients = backward_impl(
            *ctx.saved_tensors, dy, ctx.zero_centered, gate_activation=ctx.gate_activation
        )
        return (*gradients, None, None, None)


@torch.compiler.disable
def fused_gated_norm(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    zero_centered: bool = False,
    gate_activation: str = "silu",
) -> torch.Tensor:
    """Run fused RMSNorm with SiLU (GDN) or sigmoid (KDA) gating.

    The RMSNorm result and its incoming gradient round to the activation dtype,
    matching the unfused materialization boundaries. Supports first-order autograd.
    """
    if gate_activation not in ("silu", "sigmoid"):
        raise ValueError("gate_activation must be 'silu' or 'sigmoid'.")
    return _FusedGatedNorm.apply(x, gate, weight, eps, zero_centered, gate_activation)


def validate_gated_norm(
    module: torch.nn.Module, x: torch.Tensor, gate: torch.Tensor, gate_activation: str = "silu"
) -> None:
    """Reject unsupported GDN-family output fusion configurations and layouts.

    Called on every fused module forward, including output-norm recomputation.
    These checks inspect host-side metadata only and do not synchronize CUDA.
    """
    prefix = "gdn_gated_output_norm_fusion requires "
    if module.config.deterministic_mode:
        raise ValueError(prefix + "deterministic_mode=False.")
    if gate_activation == "silu" and module.activation not in ("silu", "swish"):
        raise ValueError(prefix + "SiLU/Swish activation.")
    if type(module.out_norm).__name__ != "RMSNorm":
        raise ValueError(prefix + "an RMSNorm output normalization module.")
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(prefix + "CUDA BF16 or FP16 core attention output.")
    if x.ndim != 4 or gate.shape != x.shape:
        raise ValueError(prefix + "matching output and gate shapes [batch, sequence, heads, dim].")
    if x.numel() == 0:
        raise ValueError(prefix + "nonempty core attention output and gate.")
    d = x.shape[-1]
    if d & (d - 1):
        raise ValueError(prefix + "a power-of-two head dimension.")
    if gate.dtype not in (x.dtype, torch.float32):
        raise ValueError(prefix + "a gate in the activation dtype or FP32.")
    if gate.device != x.device:
        raise ValueError(prefix + "core attention output and gate on the same CUDA device.")
    weight = module.out_norm.weight
    if weight.device != x.device or weight.shape != (d,) or not weight.is_contiguous():
        raise ValueError(prefix + "a contiguous RMSNorm weight of size dim on the input device.")
    if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(prefix + "a BF16, FP16 or FP32 RMSNorm weight.")
