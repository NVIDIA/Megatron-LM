# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.activations import squared_relu
from megatron.core.fusions.fused_weighted_squared_relu import weighted_squared_relu_impl


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
def test_weighted_squared_relu_fusion(input_dtype):
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
