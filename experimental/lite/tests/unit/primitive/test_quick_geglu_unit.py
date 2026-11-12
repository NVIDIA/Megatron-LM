# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Clamped quick-GEGLU ("swigluoai") against the same formula under plain autograd (CPU)."""

import pytest
import torch

from megatron.lite.primitive.kernels.quick_geglu import clamped_quick_geglu

LIMIT = 7.0


def _autograd_quick_geglu(y, weights=None):
    gate, up = y.chunk(2, -1)
    gate = gate.clamp(max=LIMIT)
    up = up.clamp(min=-LIMIT, max=LIMIT)
    out = gate * torch.sigmoid(1.702 * gate) * (up + 1.0)
    return out if weights is None else out * weights


@pytest.mark.parametrize("weighted", [False, True])
def test_clamped_quick_geglu_matches_autograd(weighted):
    gen = torch.Generator().manual_seed(0)
    y = (torch.randn(16, 2 * 8, generator=gen) * 6).requires_grad_(True)
    y_autograd = y.detach().clone().requires_grad_(True)
    weights = weights_autograd = None
    if weighted:
        weights = torch.rand(16, 1, generator=gen).requires_grad_(True)
        weights_autograd = weights.detach().clone().requires_grad_(True)

    out = clamped_quick_geglu(y, limit=LIMIT, linear_offset=1.0, weights=weights)
    expected = _autograd_quick_geglu(y_autograd, weights_autograd)
    torch.testing.assert_close(out, expected)

    grad = torch.randn(out.shape, generator=gen)
    out.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(y.grad, y_autograd.grad)
    if weighted:
        torch.testing.assert_close(weights.grad, weights_autograd.grad)

    # Clamped inputs get no gradient.
    gate, up = y.detach().chunk(2, -1)
    clamped = torch.cat((gate > LIMIT, up.abs() > LIMIT), -1)
    assert clamped.any()
    assert torch.all(y.grad[clamped] == 0)
