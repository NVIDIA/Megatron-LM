# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.fusions.fused_row_copy import contiguous_rows

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
