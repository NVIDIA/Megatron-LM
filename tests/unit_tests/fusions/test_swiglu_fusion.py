# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl, weighted_bias_swiglu_impl


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

    x = torch.randn(16, 64, dtype=input_dtype, device="cuda")
    x.requires_grad = True
    weights = torch.randn(16, 1, dtype=torch.float32, device="cuda")
    weights.requires_grad = True
    bwd_input = torch.randn(16, 32, dtype=input_dtype, device="cuda")

    # Reference: manual clamp + silu + weight
    y_1, y_2 = torch.chunk(x, 2, -1)
    y_1c = y_1.clamp(min=None, max=clamp_value)
    y_2c = y_2.clamp(min=-clamp_value, max=clamp_value)
    y = (F.silu(y_1c) * y_2c * weights).to(input_dtype)
    y.backward(bwd_input)

    x_2 = x.detach().clone()
    x_2.requires_grad = True
    weights_2 = weights.detach().clone()
    weights_2.requires_grad = True
    bwd_input_2 = bwd_input.detach().clone()

    # Fused implementation
    y_2_out = weighted_bias_swiglu_impl(x_2, None, weights_2, clamp_value=clamp_value)
    y_2_out.backward(bwd_input_2)

    assert y_2_out.dtype == y.dtype
    assert torch.allclose(y, y_2_out, **tols)
    assert x_2.grad.dtype == x.grad.dtype
    assert torch.allclose(x.grad, x_2.grad, **tols)
    assert weights_2.grad.dtype == weights.grad.dtype
    if input_dtype == torch.float32:
        assert torch.allclose(weights.grad, weights_2.grad, **tols)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_bias", [False, True])
def test_clamped_bias_swiglu_impl(input_dtype, with_bias):
    """``bias_swiglu_impl`` (with and without bias) must respect clamp_value."""
    clamp_value = 10.0

    if input_dtype == torch.float32:
        tols = dict(rtol=1.0e-6, atol=1.0e-6)
    elif input_dtype == torch.bfloat16:
        tols = dict(rtol=2.0e-2, atol=1.0e-3)
    else:
        raise ValueError(f"Invalid input dtype: {input_dtype}")

    # Use a large input range so the clamp actually triggers in many positions.
    x = (torch.randn(16, 64, dtype=input_dtype, device="cuda") * 5.0).requires_grad_(True)
    bias = (
        torch.randn(64, dtype=input_dtype, device="cuda").requires_grad_(True)
        if with_bias
        else None
    )
    bwd_input = torch.randn(16, 32, dtype=input_dtype, device="cuda")

    # Reference: manual clamp + silu in fp32 then cast back, mirroring ``clamped_swiglu``.
    # Cast BEFORE the bias-add so the addition happens in fp32 (matches the fused
    # kernel's internal accumulation); a bf16-precision bias-add can flip clamp
    # saturation near the boundary and yield 0 grad where fp32 sees a finite slope.
    x_fp32 = x.to(torch.float32)
    x_eff = x_fp32 + bias.to(torch.float32) if with_bias else x_fp32
    y_1, y_2 = torch.chunk(x_eff, 2, -1)
    y_1c = y_1.clamp(min=None, max=clamp_value)
    y_2c = y_2.clamp(min=-clamp_value, max=clamp_value)
    y_ref = (F.silu(y_1c) * y_2c).to(input_dtype)
    y_ref.backward(bwd_input)

    x_2 = x.detach().clone().requires_grad_(True)
    bias_2 = bias.detach().clone().requires_grad_(True) if with_bias else None
    bwd_input_2 = bwd_input.detach().clone()

    y_fused = bias_swiglu_impl(x_2, bias_2, clamp_value=clamp_value)
    y_fused.backward(bwd_input_2)

    assert y_fused.dtype == y_ref.dtype
    assert torch.allclose(y_ref, y_fused, **tols)
    assert x_2.grad.dtype == x.grad.dtype
    assert torch.allclose(x.grad, x_2.grad, **tols)
    if with_bias:
        assert bias_2.grad.dtype == bias.grad.dtype
        bias_grad_cos = torch.nn.functional.cosine_similarity(
            bias.grad.flatten().float().unsqueeze(0), bias_2.grad.flatten().float().unsqueeze(0)
        ).item()
        assert bias_grad_cos > 0.999, f"bias.grad cosine similarity = {bias_grad_cos:.6f}"


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_bias", [False, True])
def test_bias_swiglu_impl_clamp_none_matches_unclamped(input_dtype, with_bias):
    """``clamp_value=None`` (default) must match the legacy unclamped behavior."""
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

    y_unclamped = bias_swiglu_impl(x, bias)
    y_unclamped.backward(bwd_input)

    x_2 = x.detach().clone().requires_grad_(True)
    bias_2 = bias.detach().clone().requires_grad_(True) if with_bias else None

    y_default_clamp = bias_swiglu_impl(x_2, bias_2, clamp_value=None)
    y_default_clamp.backward(bwd_input.detach().clone())

    assert torch.allclose(y_unclamped, y_default_clamp, **tols)
    assert torch.allclose(x.grad, x_2.grad, **tols)
    if with_bias:
        bias_grad_cos = torch.nn.functional.cosine_similarity(
            bias.grad.flatten().float().unsqueeze(0), bias_2.grad.flatten().float().unsqueeze(0)
        ).item()
        assert bias_grad_cos > 0.999, f"bias.grad cosine similarity = {bias_grad_cos:.6f}"
