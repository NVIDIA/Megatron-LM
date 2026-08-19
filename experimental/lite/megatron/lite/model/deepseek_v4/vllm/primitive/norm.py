"""Analytic backward bridges for vLLM-visible RMSNorm kernels."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from ._contract import (
    check_parameter_versions,
    own_visible_tensor,
    parameter_versions,
)


def _rms_norm_vjp(
    grad_output: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = value.float()
    dy = grad_output.float()
    w = weight.float()
    rstd = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    weighted_dy = dy * w
    projection = (weighted_dy * x).mean(dim=-1, keepdim=True)
    dx = rstd * weighted_dy - x * rstd.pow(3) * projection
    reduce_dims = tuple(range(value.ndim - 1))
    dw = (dy * x * rstd).sum(dim=reduce_dims)
    return dx.to(value.dtype), dw.to(weight.dtype)


class _VLLMRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        value: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        visible_op: Callable[..., torch.Tensor],
    ) -> torch.Tensor:
        output = own_visible_tensor(visible_op(value, weight, eps))
        ctx.save_for_backward(value)
        ctx.weights = (weight,)
        ctx.eps = eps
        ctx.versions = parameter_versions((weight,))
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        (weight,) = ctx.weights
        check_parameter_versions((weight,), ctx.versions)
        (value,) = ctx.saved_tensors
        grad_value, grad_weight = _rms_norm_vjp(
            grad_output, value, weight, ctx.eps
        )
        return grad_value, grad_weight, None, None


class _VLLMFusedQKVRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        kv: torch.Tensor,
        q_weight: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
        visible_op: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_out, kv_out = visible_op(q, kv, q_weight, kv_weight, eps)
        ctx.save_for_backward(q, kv)
        ctx.weights = (q_weight, kv_weight)
        ctx.eps = eps
        ctx.versions = parameter_versions((q_weight, kv_weight))
        return own_visible_tensor(q_out), own_visible_tensor(kv_out)

    @staticmethod
    def backward(
        ctx: Any,
        grad_q: torch.Tensor,
        grad_kv: torch.Tensor,
    ):
        q_weight, kv_weight = ctx.weights
        check_parameter_versions((q_weight, kv_weight), ctx.versions)
        q, kv = ctx.saved_tensors
        dq, dq_weight = _rms_norm_vjp(grad_q, q, q_weight, ctx.eps)
        dkv, dkv_weight = _rms_norm_vjp(grad_kv, kv, kv_weight, ctx.eps)
        return dq, dkv, dq_weight, dkv_weight, None, None


def rms_norm(
    visible_op: Callable[..., torch.Tensor],
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return _VLLMRMSNormFunction.apply(value, weight, eps, visible_op)


def fused_qkv_rms_norm(
    visible_op: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    q: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _VLLMFusedQKVRMSNormFunction.apply(
        q, kv, q_weight, kv_weight, eps, visible_op
    )


__all__ = [
    "_VLLMFusedQKVRMSNormFunction",
    "_VLLMRMSNormFunction",
    "fused_qkv_rms_norm",
    "rms_norm",
]
