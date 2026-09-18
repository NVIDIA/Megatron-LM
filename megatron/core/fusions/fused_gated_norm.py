# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GDN output RMSNorm/SiLU fusion preserving intermediate BF16 rounding."""

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


def validate_gated_norm(module: torch.nn.Module, x: torch.Tensor, gate: torch.Tensor) -> None:
    """Reject unsupported GDN output fusion configurations and layouts.

    Called on every fused module forward, including output-norm recomputation.
    These checks inspect host-side metadata only and do not synchronize CUDA.
    """
    prefix = "gdn_gated_output_norm_fusion requires "
    if module.config.deterministic_mode:
        raise ValueError(prefix + "deterministic_mode=False.")
    if module.cp_size != 1:
        raise ValueError(prefix + "context_parallel_size=1.")
    if module.activation not in ("silu", "swish"):
        raise ValueError(prefix + "SiLU/Swish activation.")
    if type(module.out_norm).__name__ != "RMSNorm":
        raise ValueError(prefix + "an RMSNorm output normalization module.")
    if not x.is_cuda or x.dtype != torch.bfloat16:
        raise ValueError(prefix + "CUDA BF16 core attention output.")
    if not x.is_contiguous() or x.ndim == 0 or x.shape[-1] != 128:
        raise ValueError(prefix + "contiguous core attention output with head dimension 128.")
    if gate.ndim != 4 or gate.shape[0] != 1 or gate.shape[-2:] != (16, 128):
        raise ValueError(prefix + "gate shape [1, sequence_length, 16, 128].")
    if x.numel() == 0 or x.numel() != gate.numel():
        raise ValueError(
            prefix + "nonempty core attention output and gate with equal element counts."
        )
    if gate.stride(-1) != 1 or gate.stride(-2) != 128:
        raise ValueError(prefix + "contiguous elements within each gate token (strides 1 and 128).")
    if gate.device != x.device:
        raise ValueError(prefix + "core attention output and gate on the same CUDA device.")
    weight = module.out_norm.weight
    if weight.device != x.device or weight.shape != (128,) or not weight.is_contiguous():
        raise ValueError(prefix + "a contiguous 128-element RMSNorm weight on the input device.")
