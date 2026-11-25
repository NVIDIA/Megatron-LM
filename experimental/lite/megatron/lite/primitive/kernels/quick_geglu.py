# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Clamped quick-GEGLU ("swigluoai") autograd helpers.

``y = gate * sigmoid(alpha * gate) * (up + offset)`` on ``[gate | up]`` halves of
the last dimension, with ``gate`` clamped from above and ``up`` clamped to
``[-limit, limit]`` first (gradients are zero where a value was clamped). The
token-weighted variant multiplies by a per-row weight (MoE routing probability)
before the output cast.

The elementwise bodies are compiled with ``jit_fuser`` (``torch.compile``), so
each runs as a single fp32 kernel with one final rounding; their expression
order is fixed and matches Megatron-Core's quick-GEGLU helpers.
"""

from __future__ import annotations

import torch

from megatron.lite.primitive.kernels.jit import jit_fuser

QUICK_GELU_ALPHA = 1.702


@jit_fuser
def quick_gelu(y: torch.Tensor) -> torch.Tensor:
    return y * torch.sigmoid(1.702 * y)


@jit_fuser
def quick_geglu(y: torch.Tensor, linear_offset: float = 0.0) -> torch.Tensor:
    y_1, y_2 = torch.chunk(y, 2, dim=-1)
    return quick_gelu(y_1) * (y_2 + linear_offset)


@jit_fuser
def weighted_quick_geglu(
    y: torch.Tensor, weights: torch.Tensor, linear_offset: float = 0.0
) -> torch.Tensor:
    dtype = y.dtype
    res = quick_geglu(y, linear_offset) * weights
    return res.to(dtype)


@jit_fuser
def quick_geglu_back(g, y, linear_offset: float = 0.0) -> torch.Tensor:
    y_1, y_2 = torch.chunk(y, 2, -1)
    sigmoid_out = torch.sigmoid(1.702 * y_1)
    dy_1 = g * sigmoid_out * (1 + 1.702 * y_1 * (1 - sigmoid_out)) * (y_2 + linear_offset)
    dy_2 = g * y_1 * sigmoid_out
    return torch.cat((dy_1, dy_2), -1)


@jit_fuser
def weighted_quick_geglu_back(g, y, weights, linear_offset: float = 0.0):
    input_dtype = y.dtype
    w_dtype = weights.dtype
    input_grad = quick_geglu_back(g * weights, y, linear_offset)
    weights_grad = quick_geglu(y, linear_offset) * g.to(w_dtype)
    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)
    return input_grad.to(input_dtype), weights_grad.to(w_dtype)


def clamp_glu_input(y: torch.Tensor, limit: float) -> torch.Tensor:
    """Clamp the gate half from above and the linear half to ``[-limit, limit]``."""
    gate, linear = y.chunk(2, -1)
    return torch.cat((gate.clamp(max=limit), linear.clamp(min=-limit, max=limit)), -1)


class _QuickGeGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, linear_offset):
        ctx.save_for_backward(y)
        ctx.linear_offset = linear_offset
        return quick_geglu(y, linear_offset)

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        return quick_geglu_back(grad_output, y, ctx.linear_offset), None


class _WeightedQuickGeGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, weights, linear_offset):
        ctx.save_for_backward(y, weights)
        ctx.linear_offset = linear_offset
        return weighted_quick_geglu(y, weights, linear_offset)

    @staticmethod
    def backward(ctx, grad_output):
        y, weights = ctx.saved_tensors
        input_grad, weights_grad = weighted_quick_geglu_back(
            grad_output, y, weights, ctx.linear_offset
        )
        return input_grad, weights_grad, None


def clamped_quick_geglu(
    y: torch.Tensor, *, limit: float, linear_offset: float, weights: torch.Tensor | None = None
) -> torch.Tensor:
    """``[..., 2H] -> [..., H]``; ``weights`` (fp32 ``[N, 1]``) scales each of the ``N`` rows."""
    shape = y.shape
    y = clamp_glu_input(y, limit).view(-1, shape[-1])
    if weights is None:
        out = _QuickGeGLU.apply(y, linear_offset)
    else:
        out = _WeightedQuickGeGLU.apply(y, weights, linear_offset)
    return out.view(*shape[:-1], shape[-1] // 2)


__all__ = [
    "QUICK_GELU_ALPHA",
    "clamp_glu_input",
    "clamped_quick_geglu",
    "quick_geglu",
    "quick_geglu_back",
    "weighted_quick_geglu",
    "weighted_quick_geglu_back",
]
