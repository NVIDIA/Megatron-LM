# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl, weighted_bias_swiglu_impl
from megatron.core.transformer.transformer_config import TransformerConfig


def _clamped_swiglu_config(**kwargs):
    defaults = dict(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=4,
        gated_linear_unit=True,
        activation_func=F.silu,
        activation_func_clamp_value=10.0,
    )
    return TransformerConfig(**(defaults | kwargs))


def test_clamped_swiglu_config_accepts_positive_moe_clamp():
    assert _clamped_swiglu_config().activation_func_clamp_value == 10.0


@pytest.mark.parametrize("clamp_value", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_clamped_swiglu_config_requires_positive_clamp(clamp_value):
    with pytest.raises(ValueError, match="greater than zero"):
        _clamped_swiglu_config(activation_func_clamp_value=clamp_value)


def test_clamped_swiglu_config_rejects_linear_offset():
    with pytest.raises(ValueError, match="glu_linear_offset must be zero"):
        _clamped_swiglu_config(glu_linear_offset=1.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_moe_experts": None}, "only supported with MoE"),
        ({"use_te_activation_func": True}, "use_te_activation_func must be False"),
    ],
)
def test_clamped_swiglu_config_rejects_unsupported_paths(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _clamped_swiglu_config(**kwargs)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
def test_weighted_bias_swiglu(input_dtype):
    if input_dtype == torch.float32:
        tols = dict(rtol=1.0e-6, atol=1.0e-6)
    elif input_dtype == torch.bfloat16:
        tols = dict(rtol=2.0e-2, atol=1.0e-3)
    else:
        raise ValueError(f"Invalid input dtype: {input_dtype}")

    x = torch.randn(16, 64, dtype=input_dtype, device="cuda")
    x.requires_grad = True
    weights = torch.randn(16, 1, dtype=torch.float32, device="cuda")
    weights.requires_grad = True
    bwd_input = torch.randn(16, 32, dtype=input_dtype, device="cuda")

    y = bias_swiglu_impl(x, None) * weights
    y = y.to(input_dtype)
    y.backward(bwd_input)

    x_2 = x.detach()
    x_2.requires_grad = True
    weights_2 = weights.detach()
    weights_2.requires_grad = True
    bwd_input_2 = bwd_input.detach()

    y_2 = weighted_bias_swiglu_impl(x_2, None, weights_2)
    y_2.backward(bwd_input_2)

    assert y_2.dtype == y.dtype
    assert torch.allclose(y, y_2, **tols)
    assert x_2.grad.dtype == x.grad.dtype
    assert torch.allclose(x.grad, x_2.grad, **tols)
    assert weights_2.grad.dtype == weights.grad.dtype
    if input_dtype == torch.float32:
        assert torch.allclose(weights.grad, weights_2.grad, **tols)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
def test_clamped_weighted_bias_swiglu(input_dtype):
    clamp_value = 10.0

    if input_dtype == torch.float32:
        tols = dict(rtol=1.0e-6, atol=1.0e-6)
    elif input_dtype == torch.bfloat16:
        tols = dict(rtol=2.0e-2, atol=1.0e-3)
    else:
        raise ValueError(f"Invalid input dtype: {input_dtype}")

    x = (torch.randn(16, 64, dtype=input_dtype, device="cuda") * 5.0).requires_grad_(True)
    weights = torch.randn(16, 1, dtype=torch.float32, device="cuda", requires_grad=True)
    bwd_input = torch.randn(16, 32, dtype=input_dtype, device="cuda")

    # Reference: clamp and activate in FP32, then restore the input dtype.
    y_1, y_2 = torch.chunk(x.to(torch.float32), 2, -1)
    y = (
        F.silu(y_1.clamp(min=None, max=clamp_value))
        * y_2.clamp(min=-clamp_value, max=clamp_value)
        * weights
    ).to(input_dtype)
    y.backward(bwd_input)

    x_fused = x.detach().clone().requires_grad_(True)
    weights_fused = weights.detach().clone().requires_grad_(True)
    y_fused = weighted_bias_swiglu_impl(x_fused, None, weights_fused, clamp_value=clamp_value)
    y_fused.backward(bwd_input.detach().clone())

    assert y_fused.dtype == y.dtype
    assert torch.allclose(y, y_fused, **tols)
    assert x_fused.grad.dtype == x.grad.dtype
    assert torch.allclose(x.grad, x_fused.grad, **tols)
    assert weights_fused.grad.dtype == weights.grad.dtype
    if input_dtype == torch.float32:
        assert torch.allclose(weights.grad, weights_fused.grad, **tols)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_bias", [False, True])
def test_clamped_bias_swiglu_impl(input_dtype, with_bias):
    clamp_value = 10.0

    if input_dtype == torch.float32:
        tols = dict(rtol=1.0e-6, atol=1.0e-6)
    elif input_dtype == torch.bfloat16:
        tols = dict(rtol=2.0e-2, atol=1.0e-3)
    else:
        raise ValueError(f"Invalid input dtype: {input_dtype}")

    x = (torch.randn(16, 64, dtype=input_dtype, device="cuda") * 5.0).requires_grad_(True)
    bias = (
        torch.randn(64, dtype=input_dtype, device="cuda").requires_grad_(True)
        if with_bias
        else None
    )
    bwd_input = torch.randn(16, 32, dtype=input_dtype, device="cuda")

    x_fp32 = x.to(torch.float32)
    x_effective = x_fp32 + bias.to(torch.float32) if with_bias else x_fp32
    y_1, y_2 = torch.chunk(x_effective, 2, -1)
    y = (
        F.silu(y_1.clamp(min=None, max=clamp_value)) * y_2.clamp(min=-clamp_value, max=clamp_value)
    ).to(input_dtype)
    y.backward(bwd_input)

    x_fused = x.detach().clone().requires_grad_(True)
    bias_fused = bias.detach().clone().requires_grad_(True) if with_bias else None
    y_fused = bias_swiglu_impl(x_fused, bias_fused, clamp_value=clamp_value)
    y_fused.backward(bwd_input.detach().clone())

    assert y_fused.dtype == y.dtype
    assert torch.allclose(y, y_fused, **tols)
    assert x_fused.grad.dtype == x.grad.dtype
    assert torch.allclose(x.grad, x_fused.grad, **tols)
    if with_bias:
        assert bias_fused.grad.dtype == bias.grad.dtype
        bias_grad_cos = F.cosine_similarity(
            bias.grad.flatten().float().unsqueeze(0), bias_fused.grad.flatten().float().unsqueeze(0)
        ).item()
        assert bias_grad_cos > 0.999, f"bias.grad cosine similarity = {bias_grad_cos:.6f}"


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_bias", [False, True])
def test_bias_swiglu_impl_clamp_none_matches_unclamped(input_dtype, with_bias):
    if input_dtype == torch.float32:
        tols = dict(rtol=1.0e-6, atol=1.0e-6)
    else:
        tols = dict(rtol=2.0e-2, atol=1.0e-3)

    x = torch.randn(16, 64, dtype=input_dtype, device="cuda").requires_grad_(True)
    bias = (
        torch.randn(64, dtype=input_dtype, device="cuda").requires_grad_(True)
        if with_bias
        else None
    )
    bwd_input = torch.randn(16, 32, dtype=input_dtype, device="cuda")

    y = bias_swiglu_impl(x, bias)
    y.backward(bwd_input)

    x_explicit = x.detach().clone().requires_grad_(True)
    bias_explicit = bias.detach().clone().requires_grad_(True) if with_bias else None
    y_explicit = bias_swiglu_impl(x_explicit, bias_explicit, clamp_value=None)
    y_explicit.backward(bwd_input.detach().clone())

    assert torch.allclose(y, y_explicit, **tols)
    assert torch.allclose(x.grad, x_explicit.grad, **tols)
    if with_bias:
        bias_grad_cos = F.cosine_similarity(
            bias.grad.flatten().float().unsqueeze(0),
            bias_explicit.grad.flatten().float().unsqueeze(0),
        ).item()
        assert bias_grad_cos > 0.999, f"bias.grad cosine similarity = {bias_grad_cos:.6f}"
