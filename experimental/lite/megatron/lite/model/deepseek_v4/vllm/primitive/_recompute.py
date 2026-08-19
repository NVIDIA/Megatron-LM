from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from ._contract import check_parameter_versions, own_visible_tensor, parameter_versions


def _own(value):
    if isinstance(value, torch.Tensor):
        return own_visible_tensor(value)
    if isinstance(value, tuple):
        return tuple(_own(item) for item in value)
    if isinstance(value, list):
        return tuple(_own(item) for item in value)
    raise TypeError("training bridge outputs must be tensors")


class _VisibleFunctionalVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, visible_op, functional_op, version_indices, *inputs):
        output = _own(visible_op(*inputs))
        ctx.functional_op = functional_op
        ctx.inputs = inputs
        ctx.version_indices = tuple(version_indices)
        versioned = tuple(inputs[index] for index in ctx.version_indices)
        ctx.versions = parameter_versions(versioned)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs):
        versioned = tuple(ctx.inputs[index] for index in ctx.version_indices)
        check_parameter_versions(versioned, ctx.versions)
        with torch.enable_grad():
            recompute_inputs = tuple(
                value.detach().requires_grad_(value.is_floating_point())
                for value in ctx.inputs
            )
            outputs = ctx.functional_op(*recompute_inputs)
            outputs = outputs if isinstance(outputs, tuple) else (outputs,)
            differentiable_inputs = tuple(
                value for value in recompute_inputs if value.requires_grad
            )
            grads = torch.autograd.grad(
                outputs,
                differentiable_inputs,
                grad_outputs=grad_outputs,
                allow_unused=True,
            )
        grad_iter = iter(grads)
        input_grads = tuple(
            next(grad_iter) if value.requires_grad else None
            for value in recompute_inputs
        )
        return None, None, None, *input_grads


def visible_functional_vjp(
    visible_op: Callable,
    functional_op: Callable,
    inputs: tuple[torch.Tensor, ...],
    *,
    version_indices: tuple[int, ...] = (),
):
    return _VisibleFunctionalVJP.apply(
        visible_op, functional_op, version_indices, *inputs
    )
