# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression tests for the output-layer gradient-clipping hook."""

import pytest
import torch

from megatron.core.optimizer.utils import clip_output_grad_hook


def test_output_grad_clipping_preserves_other_gradients():
    gradient = torch.tensor([-100.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 100.0])
    weight_gradient = torch.ones(2, 2)
    original = gradient.clone()

    clipped = clip_output_grad_hook(None, (gradient, weight_gradient, None), ())

    torch.testing.assert_close(
        clipped[0], torch.tensor([-5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0])
    )
    assert clipped[1] is weight_gradient
    assert clipped[2] is None
    torch.testing.assert_close(gradient, original)


def test_output_grad_clipping_without_input_gradient():
    gradients = (None, torch.ones(2, 2))
    assert clip_output_grad_hook(None, gradients, ()) is gradients


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_output_grad_clipping_zero_spread(dtype):
    gradient = torch.tensor([0.0, 0.0, 0.0, 0.0, 100.0], dtype=dtype)

    clipped = clip_output_grad_hook(None, (gradient,), ())[0]

    expected = torch.tensor([0.0, 0.0, 0.0, 0.0, 3e-6], dtype=dtype)
    torch.testing.assert_close(clipped, expected)
    assert clipped.dtype == dtype


def test_output_grad_clipping_all_zero():
    gradient = torch.zeros(2, 3, 4)
    clipped = clip_output_grad_hook(None, (gradient,), ())[0]
    torch.testing.assert_close(clipped, gradient)
    assert torch.isfinite(clipped).all()


def test_output_grad_clipping_in_backward():
    class OutputLayer(torch.nn.Module):
        def forward(self, hidden):
            return hidden * 2, None

    layer = OutputLayer()
    layer.register_backward_hook(clip_output_grad_hook)
    hidden = torch.zeros(1, 1, 9, requires_grad=True)
    upstream_gradient = torch.tensor([-50.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 50.0])

    with pytest.warns(FutureWarning, match="non-full backward hook"):
        output, _ = layer(hidden)
    output.backward(upstream_gradient.view_as(output))

    expected = torch.tensor([-5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0])
    torch.testing.assert_close(hidden.grad.flatten(), expected)
