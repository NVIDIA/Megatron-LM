# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU proofs for owned derivatives; deployment kernel forward is a GPU gate."""
from types import SimpleNamespace

import pytest
import torch

from megatron.lite.model.deepseek_v41.vllm.linear import _BF16Linear, _MXFP8Linear, linear
from megatron.lite.model.deepseek_v41.vllm.norm import _RMSNorm


@pytest.mark.parametrize("shape", [(3, 32), (2, 3, 32), (0, 32)])
def test_bf16_linear_owned_vjp_uses_rounded_weight(shape):
    x = torch.randn(shape, requires_grad=True)
    weight = torch.randn(64, 32, requires_grad=True)
    compute = weight.bfloat16()
    surrogate = weight + (compute.float() - weight).detach()
    out = x @ surrogate.T
    grad = torch.randn_like(out)
    expected = torch.autograd.grad(out, (x, weight), grad)
    ctx = SimpleNamespace(saved_tensors=(x.detach(), compute.detach()), weight_dtype=weight.dtype)
    actual = _BF16Linear.backward(ctx, grad)
    for value, reference in zip(actual, expected):
        if not reference.numel() or not reference.count_nonzero():
            torch.testing.assert_close(value, reference)
            continue
        a, b = value.double().flatten(), reference.double().flatten()
        assert torch.nn.functional.cosine_similarity(a, b, dim=0) >= 0.9999
        assert (a - b).norm() / b.norm() <= 1e-2


def test_mxfp8_owned_vjp_uses_actual_decoded_operands():
    x, weight = torch.randn(7, 32).bfloat16(), torch.randn(64, 32).bfloat16()
    grad = torch.randn(7, 64).bfloat16()
    ctx = SimpleNamespace(
        saved_tensors=(x, weight), input_shape=x.shape, weight_dtype=torch.float32
    )
    dx, dw, ignored = _MXFP8Linear.backward(ctx, grad)
    assert ignored is None
    torch.testing.assert_close(dx, (grad.float() @ weight.float()).bfloat16())
    torch.testing.assert_close(dw, grad.float().T @ x.float())


def test_rms_owned_vjp_matches_fp32_reference_with_rounded_weight():
    x, weight = torch.randn(2, 3, 32, requires_grad=True), torch.randn(32, requires_grad=True)
    rounded = weight + (weight.bfloat16().float() - weight).detach()
    out = torch.nn.functional.rms_norm(x, (32,), rounded, 1e-6)
    grad = torch.randn_like(out)
    expected = torch.autograd.grad(out, (x, weight), grad)
    ctx = SimpleNamespace(saved_tensors=(x.detach(), weight.detach()), eps=1e-6)
    dx, dw, ignored = _RMSNorm.backward(ctx, grad)
    assert ignored is None
    for value, reference in zip((dx, dw), expected):
        torch.testing.assert_close(value, reference, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("fp8", [False, True])
def test_no_deployment_cpu_fallback(fp8):
    with pytest.raises(RuntimeError, match="CUDA"):
        linear(torch.zeros(2, 32), torch.zeros(64, 32), fp8=fp8)


def test_mhc_adapter_reads_actual_native_parameter_contract():
    from megatron.core.transformer.hyper_connection import HyperConnectionModule
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.lite.model.deepseek_v41.vllm.mhc import joint

    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=1,
        num_residual_streams=4,
        use_fused_mhc=False,
    )
    mixes = HyperConnectionModule(config, 1)
    with pytest.raises(RuntimeError, match="CUDA"):
        joint(torch.zeros(2, 1, 4, 32), torch.zeros(2, 1, 4), torch.ones(32), mixes)
