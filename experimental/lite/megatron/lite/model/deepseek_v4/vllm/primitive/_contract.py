from __future__ import annotations

from collections.abc import Iterable

import torch


def own_visible_tensor(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("a vLLM training bridge must own a tensor output")
    if value.is_inference():
        raise RuntimeError("the vLLM training model cannot run under inference_mode")
    return value


def parameter_versions(parameters: Iterable[torch.Tensor]) -> tuple[int, ...]:
    return tuple(parameter._version for parameter in parameters)


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
    input_shape = value.shape
    output_shape = grad_output.shape
    x2d = value.reshape(-1, value.shape[-1]).float()
    dy2d = grad_output.reshape(-1, grad_output.shape[-1]).float()
    dx = torch.mm(dy2d, weight.float()).to(value.dtype).reshape(input_shape)
    dw = torch.mm(dy2d.T, x2d).to(weight.dtype)
    if output_shape[:-1] != input_shape[:-1]:
        raise RuntimeError("linear bridge received incompatible grad_output shape")
    return dx, dw
