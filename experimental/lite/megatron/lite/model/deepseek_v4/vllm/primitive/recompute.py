from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch

def parameter_versions(parameters: Iterable[torch.Tensor]) -> tuple[int, ...]:
    return tuple(parameter._version for parameter in parameters)


def check_parameter_versions(
    parameters: Iterable[torch.Tensor], expected: tuple[int, ...]
) -> None:
    actual = parameter_versions(parameters)
    if actual != expected:
        raise RuntimeError(
            "DS4 vLLM master parameter changed between forward and backward; "
            f"versions={expected}->{actual}"
        )


def fp32_linear_vjp(
    grad_output: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x2d = value.reshape(-1, value.shape[-1]).float()
    dy2d = grad_output.reshape(-1, grad_output.shape[-1]).float()
    if grad_output.shape[:-1] != value.shape[:-1]:
        raise RuntimeError("linear bridge received incompatible grad_output shape")
    return (
        torch.mm(dy2d, weight.float()).to(value.dtype).reshape(value.shape),
        torch.mm(dy2d.T, x2d).to(weight.dtype),
    )


def _own(value):
    if isinstance(value, torch.Tensor):
        return value
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
    # Forward-only callers (including VERL old-logprob evaluation) should use
    # the visible implementation exactly as ordinary MCore modules do.  The
    # custom autograd owner exists only to attach the functional backward.
    if not torch.is_grad_enabled():
        return visible_op(*inputs)
    return _VisibleFunctionalVJP.apply(
        visible_op, functional_op, version_indices, *inputs
    )
