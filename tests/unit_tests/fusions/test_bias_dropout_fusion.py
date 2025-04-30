import pytest
import torch

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("training", [True, False])
def test_bias_dropout_add(dtype, training):
    torch.manual_seed(42)
    device = "cuda"
    B, H = 16, 64

    # Initialize inputs
    x = torch.randn(B, H, dtype=dtype, device=device, requires_grad=training)
    residual = torch.randn(B, H, dtype=dtype, device=device, requires_grad=training)
    bias = torch.randn(H, dtype=dtype, device=device)

    # Run un-fused as reference
    torch.manual_seed(42)
    ref_fn = get_bias_dropout_add(training=training, fused=False)
    x_ref = x.clone().detach().requires_grad_(training)
    residual_ref = residual.clone().detach().requires_grad_(training)
    out_ref = ref_fn((x_ref, bias), residual_ref, prob=0.0)

    # Run fused
    torch.manual_seed(42)
    fused_fn = get_bias_dropout_add(training=training, fused=True)
    x_fused = x.clone().detach().requires_grad_(training)
    residual_fused = residual.clone().detach().requires_grad_(training)
    out_fused = fused_fn((x_fused, bias), residual_fused, prob=0.0)

    tols = dict(rtol=1e-6, atol=1e-6) if dtype is torch.float32 else dict(rtol=2e-2, atol=1e-2)

    assert out_fused.dtype == out_ref.dtype
    assert torch.allclose(out_fused, out_ref, **tols)

    if training:
        grad = torch.randn_like(out_ref)
        out_ref.backward(grad)
        out_fused.backward(grad)

        assert torch.allclose(x_ref.grad, x_fused.grad, **tols)
        assert torch.allclose(residual_ref.grad, residual_fused.grad, **tols)
    else:
        # In‑place check for inference
        assert out_fused.data_ptr() == x_fused.data_ptr()
        assert torch.allclose(out_fused, x_fused, **tols)
