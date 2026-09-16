# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GDN output RMSNorm/SiLU fusion preserving intermediate BF16 rounding."""

import logging
import os

import torch
import triton
import triton.language as tl


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
    GATE_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    ZERO_CENTERED: tl.constexpr,
    BT: tl.constexpr,
):
    """Apply output RMSNorm and SiLU gating."""
    row = tl.program_id(0) * BT + tl.arange(0, BT)
    d = tl.arange(0, D)
    off = row[:, None] * D + d[None, :]
    xv = tl.load(x + off, row[:, None] < ROWS, 0).to(tl.float32)
    w = tl.load(weight + d).to(tl.float32)
    if ZERO_CENTERED:
        w += 1
    inv = tl.rsqrt(tl.sum(xv * xv, 1) / D + EPS)
    # TE produces a BF16 RMSNorm output before the FP32 gating multiply.
    norm = (xv * inv[:, None] * w[None, :]).to(x.dtype.element_ty).to(tl.float32)
    goff = (row // HEADS)[:, None] * GATE_STRIDE + (row % HEADS)[:, None] * D + d[None, :]
    gv = tl.load(gate + goff, row[:, None] < ROWS, 0).to(tl.float32)
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
    GATE_STRIDE: tl.constexpr,
    ZERO_CENTERED: tl.constexpr,
    BT: tl.constexpr,
):
    """Differentiate output RMSNorm and SiLU gating."""
    tile = tl.program_id(0)
    row = tile * BT + tl.arange(0, BT)
    d = tl.arange(0, D)
    off = row[:, None] * D + d[None, :]
    xv = tl.load(x + off, row[:, None] < ROWS, 0).to(tl.float32)
    w = tl.load(weight + d).to(tl.float32)
    if ZERO_CENTERED:
        w += 1
    inv = tl.load(rstd + row, row < ROWS, 0)
    xn = xv * inv[:, None]
    norm = (xn * w[None, :]).to(x.dtype.element_ty).to(tl.float32)
    grad = tl.load(dy + off, row[:, None] < ROWS, 0).to(tl.float32)
    goff = (row // HEADS)[:, None] * GATE_STRIDE + (row % HEADS)[:, None] * D + d[None, :]
    gv = tl.load(gate + goff, row[:, None] < ROWS, 0).to(tl.float32)
    sig = tl.sigmoid(gv)
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize and gate contiguous activations with an optionally strided gate."""
    rows, d = x.numel() // x.shape[-1], x.shape[-1]
    heads = gate.shape[-2]
    out = torch.empty_like(x)
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
        GATE_STRIDE=gate.stride(-3),
        EPS=eps,
        ZERO_CENTERED=zero_centered,
        BT=bt,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute activation, gate, and RMSNorm-weight gradients."""
    dy = dy.contiguous()
    rows, d = x.numel() // x.shape[-1], x.shape[-1]
    dx = torch.empty_like(x)
    dgate = torch.empty(gate.shape, device=gate.device, dtype=gate.dtype)
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
        HEADS=gate.shape[-2],
        D=d,
        GATE_STRIDE=gate.stride(-3),
        ZERO_CENTERED=zero_centered,
        BT=bt,
        num_warps=warps,
        enable_fp_fusion=True,
    )
    return dx, dgate, dw.sum(0).to(weight.dtype)


class _FusedGatedNorm(torch.autograd.Function):
    """First-order autograd for fused output gating."""

    @staticmethod
    def forward(ctx, x, gate, weight, eps, zero_centered):
        """Save activations and inverse norms for backward."""
        out, rstd = forward_impl(x, gate, weight, eps, zero_centered)
        ctx.save_for_backward(x, gate, weight, rstd)
        ctx.zero_centered = zero_centered
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        """Return activation, gate, and norm-weight gradients."""
        return (*backward_impl(*ctx.saved_tensors, dy, ctx.zero_centered), None, None)


@torch.compiler.disable
def fused_gated_norm(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    zero_centered: bool = False,
) -> torch.Tensor:
    """Run fused RMSNorm/SiLU gating with first-order autograd support."""
    return _FusedGatedNorm.apply(x, gate, weight, eps, zero_centered)


_LOGGED = False


def enabled(module: torch.nn.Module, x: torch.Tensor, gate: torch.Tensor) -> bool:
    """Return whether the opt-in output fusion supports these activation layouts."""
    global _LOGGED
    use_fusion = (
        os.getenv("MCORE_GDN_FUSION", "0") == "1"
        and not module.config.deterministic_mode
        and module.cp_size == 1
        and module.activation in ("silu", "swish")
        and type(module.out_norm).__name__ == "RMSNorm"
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.shape[-1] == 128
        and gate.ndim == 4
        and gate.shape[0] == 1
        and gate.shape[-2:] == (16, 128)
        and x.numel() == gate.numel()
        and gate.stride(-1) == 1
        and gate.stride(-2) == 128
    )
    if use_fusion and not _LOGGED:
        logging.getLogger(__name__).warning(
            "GDN_OUTPUT_FUSION_ACTIVE shape=%s gate_strides=%s", tuple(x.shape), gate.stride()
        )
        _LOGGED = True
    return use_fusion
