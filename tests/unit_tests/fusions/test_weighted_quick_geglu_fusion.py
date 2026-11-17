# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.fusions.fused_bias_geglu import weighted_bias_quick_geglu_impl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LINEAR_OFFSET = 1.0


def _reference(x, bias, weights, clamp_value, linear_offset=LINEAR_OFFSET):
    """Unfused GPT-OSS ordering: add the FC1 bias, clamp both halves, then Quick-GEGLU."""
    y = x if bias is None else x + bias
    gate, up = y.chunk(2, dim=-1)
    if clamp_value is not None:
        gate = gate.clamp(max=clamp_value)
        up = up.clamp(min=-clamp_value, max=clamp_value)
    out = gate * torch.sigmoid(1.702 * gate) * (up + linear_offset) * weights
    return out.to(x.dtype)


@pytest.mark.internal
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bias_is_added_before_clamp(dtype):
    """A bias that moves the gate back under the clamp must not be applied after clamping."""
    clamp_value = 7.0
    x = torch.tensor([[10.0, 1.0]], dtype=dtype, device=DEVICE, requires_grad=True)
    bias = torch.tensor([[-10.0, 0.0]], dtype=dtype, device=DEVICE)
    weights = torch.ones((1, 1), dtype=torch.float32, device=DEVICE)

    actual = weighted_bias_quick_geglu_impl(
        x, bias, weights, False, LINEAR_OFFSET, clamp_value=clamp_value
    )
    # The biased gate is 10 - 10 = 0, which quick_geglu maps to exactly 0.
    assert actual.item() == 0.0
    # The biased gate is inside the clamp range, so the gradient reaches the gate input.
    (dx,) = torch.autograd.grad(actual.sum(), x)
    assert dx.tolist() == [[1.0, 0.0]]


@pytest.mark.internal
@pytest.mark.parametrize("bias_shape", ["feature", "full"])
@pytest.mark.parametrize("clamp_value", [None, 3.0])
def test_bias_quick_geglu_matches_unfused_reference(bias_shape, clamp_value):
    """Outputs and gradients match ``quick_geglu(clamp(x + bias))``, with and without clamps."""
    torch.manual_seed(0)
    tokens, hidden = 16, 32
    tols = dict(rtol=1.0e-5, atol=1.0e-5)

    # Scaled so both the bias and the input push values across the clamp boundaries.
    x = torch.randn(tokens, 2 * hidden, device=DEVICE) * 3.0
    bias_dims = (2 * hidden,) if bias_shape == "feature" else (tokens, 2 * hidden)
    bias = torch.randn(*bias_dims, device=DEVICE) * 3.0
    weights = torch.rand(tokens, 1, device=DEVICE, dtype=torch.float32)
    grad_output = torch.randn(tokens, hidden, device=DEVICE)

    ref_x, ref_bias, ref_weights = (
        t.detach().clone().requires_grad_(True) for t in (x, bias, weights)
    )
    expected = _reference(ref_x, ref_bias, ref_weights, clamp_value)
    expected.backward(grad_output)

    act_x, act_bias, act_weights = (
        t.detach().clone().requires_grad_(True) for t in (x, bias, weights)
    )
    actual = weighted_bias_quick_geglu_impl(
        act_x, act_bias, act_weights, False, LINEAR_OFFSET, clamp_value=clamp_value
    )
    actual.backward(grad_output)

    torch.testing.assert_close(actual, expected, **tols)
    torch.testing.assert_close(act_x.grad, ref_x.grad, **tols)
    torch.testing.assert_close(act_bias.grad, ref_bias.grad, **tols)
    torch.testing.assert_close(act_weights.grad, ref_weights.grad, **tols)
