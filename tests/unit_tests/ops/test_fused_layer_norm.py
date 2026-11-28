# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the Apex-free LayerNorm path."""

from types import SimpleNamespace

import pytest
import torch

import megatron.core.ops.norm.fused_layer_norm as fused_layer_norm


@pytest.mark.parametrize("zero_centered_gamma", [False, True])
def test_fused_layer_norm_falls_back_to_torch(monkeypatch, zero_centered_gamma):
    monkeypatch.setattr(fused_layer_norm, "HAVE_PERSIST_LAYER_NORM", False)
    monkeypatch.setattr(fused_layer_norm, "HAVE_FUSED_LAYER_NORM", False)
    config = SimpleNamespace(
        normalization="LayerNorm",
        layernorm_zero_centered_gamma=zero_centered_gamma,
        persist_layer_norm=False,
        sequence_parallel=True,
        memory_efficient_layer_norm=False,
    )

    norm = fused_layer_norm.FusedLayerNorm(config, hidden_size=4, eps=1e-5)

    assert norm.use_torch_fallback
    assert not norm.persist_layer_norm
    assert norm.weight.sequence_parallel
    assert norm.bias.sequence_parallel
    assert list(norm.state_dict()) == ["weight", "bias"]
    assert norm.weight.shape == norm.bias.shape == (4,)

    with torch.no_grad():
        norm.weight.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
        norm.bias.copy_(torch.tensor([-1.0, -0.5, 0.5, 1.0]))
    restored = fused_layer_norm.FusedLayerNorm(config, hidden_size=4, eps=1e-5)
    restored.load_state_dict(norm.state_dict())
    torch.testing.assert_close(restored.weight, norm.weight)
    torch.testing.assert_close(restored.bias, norm.bias)

    value = torch.tensor([[1.0, -2.0, 3.0, 4.0]], requires_grad=True)
    reference_value = value.detach().clone().requires_grad_(True)
    reference_weight = norm.weight.detach().clone().requires_grad_(True)
    reference_bias = norm.bias.detach().clone().requires_grad_(True)

    actual = norm(value)
    expected = torch.nn.functional.layer_norm(
        reference_value,
        norm.hidden_size,
        reference_weight + int(zero_centered_gamma),
        reference_bias,
        norm.eps,
    )
    torch.testing.assert_close(actual, expected)

    actual.square().sum().backward()
    expected.square().sum().backward()

    torch.testing.assert_close(value.grad, reference_value.grad)
    torch.testing.assert_close(norm.weight.grad, reference_weight.grad)
    torch.testing.assert_close(norm.bias.grad, reference_bias.grad)
    torch.testing.assert_close(restored(value.detach()), norm(value.detach()))


def test_fused_layer_norm_keeps_apex_path_when_available(monkeypatch):
    monkeypatch.setattr(fused_layer_norm, "HAVE_PERSIST_LAYER_NORM", False)
    monkeypatch.setattr(fused_layer_norm, "HAVE_FUSED_LAYER_NORM", True)
    calls = []

    class FakeFusedLayerNormAffineFunction:
        @staticmethod
        def forward(input, weight, bias, hidden_size, eps):
            return None

        @staticmethod
        def apply(input, weight, bias, hidden_size, eps):
            calls.append((hidden_size, eps))
            return torch.nn.functional.layer_norm(input, hidden_size, weight, bias, eps)

    monkeypatch.setattr(
        fused_layer_norm,
        "FusedLayerNormAffineFunction",
        FakeFusedLayerNormAffineFunction,
        raising=False,
    )
    config = SimpleNamespace(
        normalization="LayerNorm",
        layernorm_zero_centered_gamma=False,
        persist_layer_norm=False,
        sequence_parallel=False,
        memory_efficient_layer_norm=False,
    )
    norm = fused_layer_norm.FusedLayerNorm(config, hidden_size=4, eps=1e-5)
    value = torch.tensor([[1.0, -2.0, 3.0, 4.0]])

    actual = norm(value)

    assert not norm.use_torch_fallback
    assert calls == [(torch.Size([4]), 1e-5)]
    expected = torch.nn.functional.layer_norm(value, norm.hidden_size, norm.weight, norm.bias, norm.eps)
    torch.testing.assert_close(actual, expected)
