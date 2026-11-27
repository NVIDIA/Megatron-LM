# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Liger RMSNorm as a norm backend, configuration checks and tensor contracts."""

import sys
from types import SimpleNamespace

import pytest
import torch

from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.ops.norm.backends import get_norm_module_spec
from megatron.core.ops.norm.liger_norm import LigerRMSNorm, _normalize_in_weight_dtype
from megatron.core.transformer.transformer_config import TransformerConfig

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Liger kernels need CUDA")


def _config(**overrides):
    options = dict(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        normalization="RMSNorm",
        rmsnorm_impl="liger",
        use_cpu_initialization=True,
    )
    return TransformerConfig(**(options | overrides))


def test_unselected_paths_do_not_import_liger(monkeypatch):
    for module in ("liger_kernel", "liger_kernel.megatron"):
        monkeypatch.setitem(sys.modules, module, None)  # any import attempt now fails
    assert get_norm_module_spec("te") is TENorm
    assert get_norm_module_spec("liger") is LigerRMSNorm  # naming it in a spec imports nothing
    with pytest.raises(ImportError, match="Liger RMSNorm requires liger_kernel.megatron"):
        LigerRMSNorm(config=_config(), hidden_size=8)
    with pytest.raises(ValueError, match="Unknown norm backend 'apex'"):
        get_norm_module_spec("apex")


def test_liger_backend_builds_a_megatron_norm():
    pytest.importorskip("liger_kernel.megatron")
    norm = get_norm_module_spec("liger")(
        config=_config(params_dtype=torch.bfloat16), hidden_size=8, eps=1e-6
    )
    assert type(norm).__name__ == "LigerMegatronRMSNorm"
    assert norm.weight.dtype == torch.bfloat16 and norm.weight.device.type == "cpu"
    assert norm.weight.is_rmsnorm_weight and norm.returns_residual is False
    fp32_residual = torch.ones(2, 8)
    assert _normalize_in_weight_dtype(norm, (fp32_residual,))[0].dtype == torch.bfloat16


def test_liger_needs_a_minimum_version(monkeypatch):
    pytest.importorskip("liger_kernel.megatron")
    monkeypatch.setattr(sys.modules["liger_kernel"], "__version__", "0.8.4", raising=False)
    with pytest.raises(ImportError, match=r"Liger RMSNorm requires liger_kernel.megatron>=0.8.5"):
        LigerRMSNorm(config=_config(), hidden_size=8)


def test_config_rejects_unsupported_setups():
    _config()
    for invalid, requirement in (
        ({"normalization": "LayerNorm"}, "normalization='RMSNorm'"),
        ({"transformer_impl": "local"}, "transformer_impl='transformer_engine'"),
        ({"fused_residual_rmsnorm": True}, "fused_residual_rmsnorm=False"),
        ({"use_kitchen": True}, "use_kitchen=False"),
        ({"experimental_attention_variant": "dsa"}, "no experimental_attention_variant"),
    ):
        with pytest.raises(ValueError, match=f"rmsnorm_impl='liger' requires {requirement}"):
            _config(**invalid)
    with pytest.raises(ValueError, match="rmsnorm_impl must be 'te' or 'liger'"):
        _config(rmsnorm_impl="apex")
    with pytest.raises(ValueError, match="supported only by HybridModel"):
        GPTModel(config=_config(), transformer_layer_spec=None, vocab_size=8, max_sequence_length=8)


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_norm_interface_and_checkpoint(sequence_parallel):
    pytest.importorskip("liger_kernel.megatron")
    config = SimpleNamespace(sequence_parallel=sequence_parallel, use_cpu_initialization=True)
    norm = LigerRMSNorm(config=config, hidden_size=128, eps=1e-6)
    assert norm.weight.sequence_parallel is sequence_parallel
    assert list(norm.state_dict()) == ["weight"]


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.parametrize("hidden_size", [123, 512])
def test_forward_backward_parity(dtype, zero_centered, hidden_size):
    pytest.importorskip("liger_kernel.megatron")
    config = SimpleNamespace(layernorm_zero_centered_gamma=zero_centered, params_dtype=dtype)
    norm = LigerRMSNorm(config=config, hidden_size=hidden_size, eps=1e-6)
    torch.manual_seed(123)
    with torch.no_grad():
        norm.weight.add_(0.1 * torch.randn_like(norm.weight))
    x = torch.randn(7, 3, hidden_size, device="cuda", dtype=dtype).transpose(0, 1)
    x.requires_grad_(True)
    original = x.detach().clone()
    ref_x = x.detach().clone().requires_grad_(True)
    ref_weight = norm.weight.detach().clone()
    # Liger's "llama" casting: rstd in FP32, round to the input dtype, then scale in FP32.
    rstd = torch.rsqrt(ref_x.float().square().mean(-1, keepdim=True) + 1e-6)
    normalized = (ref_x.float() * rstd).to(dtype)
    expected = (normalized.float() * (ref_weight.float() + int(zero_centered))).to(dtype)
    actual = norm(x)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    # Its weight gradient rounds each dY * x_hat product to the input dtype, then sums in FP32.
    expected_dw = (grad * normalized.detach()).float().sum(dim=(0, 1)).to(dtype)
    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        grad_tolerance = dict(atol=2e-5, rtol=2e-5)
    else:
        torch.testing.assert_close(actual, expected)  # BF16 defaults: one rounding step
        # A different rstd summation order can move x_hat by one BF16 step in a few elements.
        grad_tolerance = dict(atol=2e-2, rtol=1.6e-2)
    torch.testing.assert_close(x.grad, ref_x.grad, **grad_tolerance)
    torch.testing.assert_close(norm.weight.grad, expected_dw, **grad_tolerance)
    torch.testing.assert_close(x, original, atol=0, rtol=0)


@requires_cuda
def test_fp32_input_normalizes_in_weight_dtype():
    pytest.importorskip("liger_kernel.megatron")
    norm = LigerRMSNorm(config=SimpleNamespace(params_dtype=torch.bfloat16), hidden_size=64)
    x = torch.randn(4, 64, device="cuda")
    torch.testing.assert_close(norm(x), norm(x.bfloat16()), atol=0, rtol=0)
    assert norm(x).dtype == torch.bfloat16
