# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.activations import squared_relu, tanh_soft_clamp
from megatron.core.fusions.fused_weighted_squared_relu import weighted_squared_relu_impl


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
def test_weighted_squared_relu_fusion(input_dtype):
    torch.manual_seed(0)

    # Tolerances depend on dtype precision
    if input_dtype == torch.float32:
        tols = dict(rtol=1.0e-6, atol=1.0e-6)
    elif input_dtype == torch.bfloat16:
        tols = dict(rtol=2.0e-2, atol=1.0e-3)
    else:
        raise ValueError(f"Unsupported dtype {input_dtype}")

    # Inputs
    x = torch.randn(16, 64, dtype=input_dtype, device="cuda", requires_grad=True)
    weights = torch.randn(16, 1, dtype=torch.float32, device="cuda", requires_grad=True)
    grad_output = torch.randn(16, 64, dtype=input_dtype, device="cuda")

    # Baseline: legacy squared_relu followed by weighting.
    y_baseline = squared_relu(x) * weights
    y_baseline = y_baseline.to(input_dtype)
    y_baseline.backward(grad_output)

    # Fused implementation.
    x_fused = x.detach().clone().requires_grad_(True)
    weights_fused = weights.detach().clone().requires_grad_(True)
    grad_output_fused = grad_output.detach().clone()

    y_fused = weighted_squared_relu_impl(x_fused, weights_fused)
    y_fused.backward(grad_output_fused)

    # Forward accuracy
    assert y_fused.dtype == y_baseline.dtype
    assert torch.allclose(y_fused, y_baseline, **tols)

    # Grad accuracy w.r.t input
    assert x_fused.grad.dtype == x.grad.dtype
    assert torch.allclose(x_fused.grad, x.grad, **tols)

    # Grad accuracy w.r.t weights
    assert weights_fused.grad.dtype == weights.grad.dtype
    if input_dtype == torch.float32:
        # For bf16 baseline weight grad computed in fp32 then cast may lose precision.
        assert torch.allclose(weights_fused.grad, weights.grad, **tols)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
def test_clamped_weighted_squared_relu_fusion_matches_unfused(input_dtype):
    """Fused clamped Squared-ReLU must match an eager tanh soft clamp plus the activation."""
    torch.manual_seed(0)
    clamp_scale = 5.0

    if input_dtype == torch.float32:
        tols = dict(rtol=1.0e-6, atol=1.0e-6)
    elif input_dtype == torch.bfloat16:
        tols = dict(rtol=2.0e-2, atol=1.0e-3)
    else:
        raise ValueError(f"Unsupported dtype {input_dtype}")

    # Scaled up so a good share of the inputs land in the saturating region of the clamp.
    x = (torch.randn(16, 64, dtype=input_dtype, device="cuda") * 5.0).requires_grad_(True)
    weights = torch.randn(16, 1, dtype=torch.float32, device="cuda", requires_grad=True)
    grad_output = torch.randn(16, 64, dtype=input_dtype, device="cuda")

    # Baseline: the unfused ordering used by MLP/experts when bias_activation_fusion is False.
    y_baseline = (squared_relu(tanh_soft_clamp(x, clamp_scale)) * weights).to(input_dtype)
    y_baseline.backward(grad_output)

    x_fused = x.detach().clone().requires_grad_(True)
    weights_fused = weights.detach().clone().requires_grad_(True)
    y_fused = weighted_squared_relu_impl(x_fused, weights_fused, clamp_scale=clamp_scale)
    y_fused.backward(grad_output.detach().clone())

    assert y_fused.dtype == y_baseline.dtype
    assert torch.allclose(y_fused, y_baseline, **tols)

    assert x_fused.grad.dtype == x.grad.dtype
    assert torch.allclose(x_fused.grad, x.grad, **tols)

    assert weights_fused.grad.dtype == weights.grad.dtype
    if input_dtype == torch.float32:
        assert torch.allclose(weights_fused.grad, weights.grad, rtol=1.0e-5, atol=1.0e-5)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(16, 64), (4, 8, 64)], ids=["2d", "3d"])
def test_unweighted_clamped_squared_relu_fusion_matches_unfused(input_dtype, shape):
    """Fused unweighted clamped Squared-ReLU must match the eager clamp + activation."""
    torch.manual_seed(0)
    clamp_scale = 5.0

    if input_dtype == torch.float32:
        tols = dict(rtol=1.0e-6, atol=1.0e-6)
    elif input_dtype == torch.bfloat16:
        tols = dict(rtol=2.0e-2, atol=1.0e-3)
    else:
        raise ValueError(f"Unsupported dtype {input_dtype}")

    # Scaled up so a good share of the inputs land in the saturating region of the clamp.
    x = (torch.randn(*shape, dtype=input_dtype, device="cuda") * 5.0).requires_grad_(True)
    grad_output = torch.randn(*shape, dtype=input_dtype, device="cuda")

    # Baseline: the unfused ordering used by MLP for non-squared-relu activations.
    y_baseline = squared_relu(tanh_soft_clamp(x, clamp_scale))
    y_baseline.backward(grad_output)

    x_fused = x.detach().clone().requires_grad_(True)
    y_fused = weighted_squared_relu_impl(x_fused, None, clamp_scale)
    y_fused.backward(grad_output.detach().clone())

    assert y_fused.shape == y_baseline.shape
    assert y_fused.dtype == y_baseline.dtype
    assert torch.allclose(y_fused, y_baseline, **tols)

    assert x_fused.grad.dtype == x.grad.dtype
    assert torch.allclose(x_fused.grad, x.grad, **tols)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_unweighted_squared_relu_fusion_saves_only_input():
    """Without weights, the fused op must keep a single input-sized tensor alive."""
    saved = []

    def pack(t):
        saved.append(t)
        return t

    x = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = weighted_squared_relu_impl(x, None, 5.0)

    assert len(saved) == 1
    assert saved[0].data_ptr() == x.data_ptr()
    y.sum().backward()
    assert x.grad is not None


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_squared_relu_fusion_rejects_no_weights_and_no_clamp():
    """With neither weights nor clamp there is nothing to fuse; plain squared_relu should be used."""
    x = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    with pytest.raises(AssertionError):
        weighted_squared_relu_impl(x, None, None)
