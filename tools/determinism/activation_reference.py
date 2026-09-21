# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Independent eager FP64 references for the three local activation pilots."""

from __future__ import annotations

import torch

from tools.determinism.reduction_reference import ReductionReference
from tools.determinism.reference import TensorPair


def activation_reference(
    case: str, inputs: tuple
) -> tuple[TensorPair, dict[str, ReductionReference], TensorPair]:
    """Return the interface reference, reduction contract and mathematical values.

    No production activation, custom backward or compiled helper is reused.
    This adapter is scoped to 2D pilot inputs and all-ones upstream gradients.
    """
    if case not in ("bias_swiglu", "weighted_swiglu", "weighted_squared_relu"):
        raise ValueError(f"Unsupported activation reference: {case}")
    values = tuple(
        (
            value.detach().double().requires_grad_(value.requires_grad)
            if isinstance(value, torch.Tensor)
            else value
        )
        for value in inputs
    )
    x = values[0]
    if x.ndim != 2 or any(
        not value.requires_grad for value in values if isinstance(value, torch.Tensor)
    ):
        raise ValueError("Pilot reference requires 2D input and every tensor gradient")
    if case == "weighted_squared_relu":
        activation = torch.relu(x).square()
        output = activation * values[1]
    else:
        if case == "bias_swiglu":
            x = x + values[1]
        gate, linear = x.chunk(2, dim=-1)
        activation = torch.nn.functional.silu(gate) * linear
        output = activation * values[2] if case == "weighted_swiglu" else activation
    leaves = {f"in[{index}]": value for index, value in enumerate(values) if value is not None}
    gradients = torch.autograd.grad(output, tuple(leaves.values()), torch.ones_like(output))
    mathematical = ({"out": output.detach()}, dict(zip(leaves, gradients)))
    expected = (
        {"out": output.detach().to(inputs[0].dtype)},
        {
            f"in[{index}]": mathematical[1][f"in[{index}]"].to(value.dtype)
            for index, value in enumerate(inputs)
            if value is not None
        },
    )
    if case == "bias_swiglu":
        name = "in[1]"
        reduction = ReductionReference.from_terms(
            mathematical[1]["in[0]"].to(inputs[0].dtype),
            dim=0,
            keepdim=False,
            rounding="per_token_gradient_to_input_dtype_before_bias_sum",
            term_dtype=inputs[0].dtype,
        )
    else:
        name = "in[2]" if case == "weighted_swiglu" else "in[1]"
        reduction = ReductionReference.from_terms(
            activation, dim=-1, keepdim=True, rounding="fp32_fused_terms_before_weight_sum"
        )
    expected[1][name] = reduction.total.to(expected[1][name].dtype)
    return expected, {"gradient:" + name: reduction}, mathematical
