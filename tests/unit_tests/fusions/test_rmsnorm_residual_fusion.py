import pytest
import torch
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
        normalized_shape=normalized_shape, dtype=input_dtype, quantize=False
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
