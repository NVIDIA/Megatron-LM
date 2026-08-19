"""Shared contracts for DS4 vLLM-visible training bridges."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def own_visible_tensor(value: torch.Tensor) -> torch.Tensor:
    """Make a deployment result eligible for an outer autograd owner.

    Cloning an inference tensor after the deployment call preserves its value
    while avoiding inference-mode tensor restrictions. Ordinary tensors are
    returned as-is so an exact/path gate observes no extra numerical work.
    """

    if not isinstance(value, torch.Tensor):
        raise TypeError("a vLLM training bridge must own a tensor output")
    return value.clone() if value.is_inference() else value


def parameter_versions(parameters: Iterable[torch.Tensor]) -> tuple[int, ...]:
    # vLLM audit/eager inference constructs inference tensors, which
    # intentionally have no version counter. Training tensors still retain
    # the mutation guard used by custom backward.
    return tuple(
        -1 if parameter.is_inference() else parameter._version
        for parameter in parameters
    )


def check_parameter_versions(
    parameters: Iterable[torch.Tensor], expected: tuple[int, ...]
) -> None:
    parameters = tuple(parameters)
    actual = parameter_versions(parameters)
    if actual != expected:
        changed = [
            index
            for index, (before, after) in enumerate(zip(expected, actual, strict=True))
            if before != after
        ]
        raise RuntimeError(
            "DS4 vLLM master parameter changed between forward and backward; "
            f"changed dependency indices={changed}, versions={expected}->{actual}"
        )


def fp32_linear_vjp(
    grad_output: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16-master linear VJP with FP32 accumulation."""

    input_shape = value.shape
    output_shape = grad_output.shape
    x2d = value.reshape(-1, value.shape[-1]).float()
    dy2d = grad_output.reshape(-1, grad_output.shape[-1]).float()
    dx = torch.mm(dy2d, weight.float()).to(value.dtype).reshape(input_shape)
    dw = torch.mm(dy2d.T, x2d).to(weight.dtype)
    if output_shape[:-1] != input_shape[:-1]:
        raise RuntimeError("linear bridge received incompatible grad_output shape")
    return dx, dw


__all__ = [
    "check_parameter_versions",
    "fp32_linear_vjp",
    "own_visible_tensor",
    "parameter_versions",
]
