# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Liger RMSNorm selection, configuration checks and tensor contracts."""

import sys
from types import SimpleNamespace

import pytest
import torch

from megatron.core.extensions.liger_kernel_spec_provider import LigerSpecProvider
from megatron.core.models.backends import LocalSpecProvider, get_backend_from_config
from megatron.core.ops.norm.liger_norm import LigerRMSNorm
from megatron.core.ops.norm.torch_norm import WrappedTorchNorm
from megatron.core.transformer.heterogeneous.heterogeneous_config import (
    HeterogeneousTransformerConfig,
)
from megatron.core.transformer.transformer_config import TransformerConfig


def test_unselected_paths_do_not_import_liger(monkeypatch):
    for module in ("liger_kernel", "liger_kernel.megatron"):
        monkeypatch.setitem(sys.modules, module, None)  # any import attempt now fails
    provider = get_backend_from_config(SimpleNamespace(transformer_impl="local", use_liger=True))
    assert isinstance(provider, LigerSpecProvider)
    assert provider.layer_norm() is LocalSpecProvider().layer_norm()
    assert LocalSpecProvider().layer_norm(rms_norm=True) is WrappedTorchNorm
    with pytest.raises(ImportError, match="Liger RMSNorm requires liger_kernel.megatron"):
        provider.layer_norm(rms_norm=True)


def test_config_rejects_unsupported_setups():
    options = dict(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        normalization="RMSNorm",
        transformer_impl="local",
        use_liger=True,
    )
    TransformerConfig(**options)
    for invalid in (
        {"normalization": "LayerNorm"},
        {"transformer_impl": "transformer_engine"},
        {"transformer_impl": "inference_optimized"},
        {"use_kitchen": True},
        {"experimental_attention_variant": "dsa"},
    ):
        with pytest.raises(ValueError, match="use_liger requires"):
            TransformerConfig(**(options | invalid))
    with pytest.raises(ValueError, match="use_liger does not support heterogeneous"):
        HeterogeneousTransformerConfig(**options)


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_norm_interface_and_checkpoint(sequence_parallel):
    pytest.importorskip("liger_kernel.megatron")
    config = SimpleNamespace(sequence_parallel=sequence_parallel)
    norm = LigerRMSNorm(config=config, hidden_size=128, eps=1e-6)
    assert norm.returns_residual is False
    assert norm.weight.sequence_parallel is sequence_parallel
    assert norm.weight.is_rmsnorm_weight
    assert list(norm.state_dict()) == ["weight"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.parametrize("hidden_size", [123, 512])
def test_forward_backward_parity(dtype, zero_centered, hidden_size):
    pytest.importorskip("liger_kernel.megatron")
    norm = LigerRMSNorm(
        config=SimpleNamespace(layernorm_zero_centered_gamma=zero_centered),
        hidden_size=hidden_size,
        eps=1e-6,
    ).to(device="cuda", dtype=dtype)
    torch.manual_seed(123)
    with torch.no_grad():
        norm.weight.add_(0.1 * torch.randn_like(norm.weight))
    x = torch.randn(7, 3, hidden_size, device="cuda", dtype=dtype).transpose(0, 1)
    x.requires_grad_(True)
    original = x.detach().clone()
    ref_x = x.detach().clone().requires_grad_(True)
    ref_weight = norm.weight.detach().clone().requires_grad_(True)
    normalized = (
        ref_x.float() * torch.rsqrt(ref_x.float().square().mean(-1, keepdim=True) + 1e-6)
    ).to(dtype)
    expected = normalized * (ref_weight + int(zero_centered))
    actual = norm(x)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    tolerance = dict(atol=2e-5, rtol=2e-5) if dtype == torch.float32 else dict(atol=0.1, rtol=0.03)
    torch.testing.assert_close(actual, expected, **tolerance)
    torch.testing.assert_close(x.grad, ref_x.grad, **tolerance)
    torch.testing.assert_close(norm.weight.grad, ref_weight.grad, **tolerance)
    torch.testing.assert_close(x, original, atol=0, rtol=0)
