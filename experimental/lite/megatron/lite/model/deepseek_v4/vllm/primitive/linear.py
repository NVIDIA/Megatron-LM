from __future__ import annotations

from collections.abc import Callable

import torch

from .recompute import (
    check_parameter_versions,
    fp32_linear_vjp,
    parameter_versions,
)


class _VisibleLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, visible_op: Callable, value: torch.Tensor, *weights):
        output = visible_op(value, *weights)
        if isinstance(output, (tuple, list)):
            output = output[0]
        ctx.save_for_backward(value)
        ctx.weights = weights
        ctx.versions = parameter_versions(weights)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        check_parameter_versions(ctx.weights, ctx.versions)
        (value,) = ctx.saved_tensors
        fused = torch.cat(ctx.weights, dim=0)
        grad_value, grad_weight = fp32_linear_vjp(grad_output, value, fused)
        return (
            None,
            grad_value,
            *grad_weight.split([weight.shape[0] for weight in ctx.weights], dim=0),
        )


def visible_linear(visible_op, value, master_weight):
    if not torch.is_grad_enabled():
        output = visible_op(value, master_weight)
        return output[0] if isinstance(output, (tuple, list)) else output
    return _VisibleLinear.apply(visible_op, value, master_weight)


block_fp8_linear = visible_linear


def fused_block_fp8_linear(visible_op, value, *master_weights):
    if not torch.is_grad_enabled():
        output = visible_op(value, *master_weights)
        return output[0] if isinstance(output, (tuple, list)) else output
    return _VisibleLinear.apply(visible_op, value, *master_weights)


def gate_linear(visible_op, value, master_weight):
    if not torch.is_grad_enabled():
        output = visible_op(value)
        return output[0] if isinstance(output, (tuple, list)) else output
    return _VisibleLinear.apply(
        lambda input_value, _weight: visible_op(input_value),
        value,
        master_weight,
    )
