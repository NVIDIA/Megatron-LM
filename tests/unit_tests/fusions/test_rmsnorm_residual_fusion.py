# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch import RMSNorm

from megatron.core.extensions.transformer_engine import TEFusedResidualRMSNorm


def baseline_rmsnorm_residual(x, rmsnorm: RMSNorm):
    return rmsnorm(x), x


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("normalized_shape", [256, 256 * 2, 256 * 4])
def test_rmsnorm_residual_fusion(input_dtype, normalized_shape):
    x_baseline = torch.randn(16, 32, normalized_shape, dtype=input_dtype, device="cuda")
    x_baseline.requires_grad = True
    x_fused = x_baseline.detach()
    x_fused.requires_grad = True
    baseline_rmsnorm = RMSNorm(normalized_shape=normalized_shape, dtype=input_dtype).cuda()
    fused_rmsnorm = TEFusedResidualRMSNorm(
        normalized_shape=normalized_shape, dtype=input_dtype
    ).cuda()

    # baseline
    baseline_y, baseline_residual = baseline_rmsnorm_residual(x_baseline, baseline_rmsnorm)
    baseline_loss = baseline_y.sum() + baseline_residual.sum()
    baseline_loss.backward()

    # fused
    fused_y, fused_residual = fused_rmsnorm(x_fused)
    fused_loss = fused_y.sum() + fused_residual.sum()
    fused_loss.backward()

    # Use tolerances appropriate for dtype (pattern from other tests)
    tols = (
        dict(rtol=1e-6, atol=1e-6) if input_dtype is torch.float32 else dict(rtol=2e-2, atol=1e-2)
    )

    assert fused_y.dtype == baseline_y.dtype
    assert torch.allclose(fused_y, baseline_y, **tols)
    assert fused_residual.dtype == baseline_residual.dtype
    assert torch.allclose(fused_residual, baseline_residual, **tols)
    assert x_fused.grad.dtype == x_baseline.grad.dtype
    assert torch.allclose(x_baseline.grad, x_fused.grad, **tols)


class _GradSpy(torch.autograd.Function):
    """Identity that records the gradient it passes on."""

    @staticmethod
    def forward(ctx, x, grads):
        ctx.grads = grads
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        ctx.grads.append(grad)
        return grad, None


@pytest.mark.skipif(not te.is_mxfp8_available(), reason="MXFP8 required")
def test_rmsnorm_residual_fusion_quantize_grad(monkeypatch):
    """With quantize_grad_input the input's gradient leaves the fused norm as an MXFP8 tensor
    carrying its high-precision values, so it accumulates exactly with another gradient of the
    same tensor."""
    monkeypatch.setenv("NVTE_RMSNORM_BWD_USE_TRITON", "1")
    recipe = te.fp8.MXFP8BlockScaling()
    x = torch.randn(16, 32, 256, dtype=torch.bfloat16, device="cuda")
    other = torch.randn_like(x)
    grads = {}
    seen = []
    for quantize in (False, True):
        torch.manual_seed(0)
        norm = TEFusedResidualRMSNorm(normalized_shape=256, dtype=torch.bfloat16).cuda()
        norm.quantize_grad_input = quantize
        x_test = x.clone().requires_grad_()
        x_norm = _GradSpy.apply(x_test, seen) if quantize else x_test
        with te.autocast(enabled=True, recipe=recipe):
            y, residual = norm(x_norm)
        loss = y.sum() + residual.sum() + (x_test * other).sum()
        loss.backward()
        grads[quantize] = x_test.grad
    assert isinstance(seen[0], te.MXFP8Tensor)
    assert torch.equal(seen[0].dequantize(), seen[0]._high_precision_data)
    assert torch.equal(grads[True], grads[False])
