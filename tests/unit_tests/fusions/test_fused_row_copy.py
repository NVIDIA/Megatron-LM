# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.fusions.fused_row_copy import cat_last_dim, contiguous_rows

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_contiguous_input_is_returned_unchanged():
    x = torch.randn(64, 2, 128, device="cuda")
    assert contiguous_rows(x) is x


@pytest.mark.parametrize(
    "shape, sl", [((256, 2, 2112), slice(0, 1536)), ((256, 1, 576), slice(0, 512))]
)
def test_row_strided_view_matches_contiguous(shape, sl):
    base = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    x = base[..., sl].requires_grad_(True)
    ref = base[..., sl].requires_grad_(True)
    y = contiguous_rows(x)
    y_ref = ref.contiguous()
    assert y.is_contiguous()
    assert torch.equal(y, y_ref)
    grad = torch.randn_like(y_ref)
    y.backward(grad)
    y_ref.backward(grad)
    assert torch.equal(x.grad, ref.grad)


@pytest.mark.parametrize(
    "make",
    [lambda: torch.randn(64, 32, device="cuda").t(), lambda: torch.randn(64, 2, 32)[:, :, :16]],
)
def test_other_layouts_fall_back_to_contiguous(make):
    """Views that are not rows at a uniform stride, and CPU tensors, use Tensor.contiguous."""
    x = make()
    y = contiguous_rows(x)
    assert y.is_contiguous()
    assert torch.equal(y, x.contiguous())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("widths", [(7168, 7168), (512, 1536), (300, 77)])
@pytest.mark.parametrize("needs_grad", [(True, True), (True, False), (False, True)])
def test_cat_last_dim_matches_torch_cat(dtype, widths, needs_grad):
    """Same output and same input gradients as torch.cat, and the gradients are contiguous."""
    a0 = torch.randn(256, 2, widths[0], device="cuda", dtype=dtype)
    b0 = torch.randn(256, 2, widths[1], device="cuda", dtype=dtype)
    grad = torch.randn(256, 2, sum(widths), device="cuda", dtype=dtype)
    a, a_ref = (a0.clone().requires_grad_(needs_grad[0]) for _ in range(2))
    b, b_ref = (b0.clone().requires_grad_(needs_grad[1]) for _ in range(2))
    y = cat_last_dim(a, b)
    y_ref = torch.cat((a_ref, b_ref), -1)
    assert torch.equal(y, y_ref)
    y.backward(grad)
    y_ref.backward(grad)
    for x, x_ref in ((a, a_ref), (b, b_ref)):
        if x.requires_grad:
            assert x.grad.is_contiguous()
            assert torch.equal(x.grad, x_ref.grad)
        else:
            assert x.grad is None


def test_cat_last_dim_without_grad_is_torch_cat():
    a = torch.randn(16, 1, 64, device="cuda")
    b = torch.randn(16, 1, 32, device="cuda")
    with torch.no_grad():
        assert torch.equal(cat_last_dim(a, b), torch.cat((a, b), -1))
    assert torch.equal(cat_last_dim(a, b), torch.cat((a, b), -1))
