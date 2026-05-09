# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the fused residual-add + RMSNorm Triton kernel."""

import pytest
import torch

from megatron.core.fusions.fused_add_rmsnorm import HAVE_TRITON, fused_add_rmsnorm

pytestmark = pytest.mark.skipif(
    not HAVE_TRITON or not torch.cuda.is_available(),
    reason="fused_add_rmsnorm requires Triton and CUDA",
)


def _reference(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
    """PyTorch reference: residual += x; return rmsnorm(residual)."""
    new_residual = residual + x
    # Match the kernel's fp32 accumulation convention for variance.
    r32 = new_residual.to(torch.float32)
    var = r32.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    normed = (r32 * rstd * weight.to(torch.float32)).to(x.dtype)
    return normed, new_residual


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("hidden_size", [256, 1024, 4096])
@pytest.mark.parametrize("n_rows", [1, 16, 128])
def test_matches_reference(dtype, hidden_size, n_rows):
    torch.manual_seed(0)
    eps = 1e-5
    device = "cuda"

    x = torch.randn(n_rows, hidden_size, dtype=dtype, device=device)
    residual = torch.randn(n_rows, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    ref_normed, ref_residual = _reference(x, residual.clone(), weight, eps)

    fused_residual = residual.clone()
    fused_normed, fused_residual_out = fused_add_rmsnorm(x, fused_residual, weight, eps=eps)

    if dtype is torch.float32:
        tols = dict(rtol=1e-5, atol=1e-5)
    else:
        tols = dict(rtol=2e-2, atol=1e-2)

    assert fused_normed.shape == ref_normed.shape
    assert fused_normed.dtype == ref_normed.dtype
    torch.testing.assert_close(fused_normed, ref_normed, **tols)

    assert fused_residual_out.shape == ref_residual.shape
    assert fused_residual_out.dtype == ref_residual.dtype
    torch.testing.assert_close(fused_residual_out, ref_residual, **tols)


def test_residual_in_place_when_contiguous():
    """Contiguous residual: the returned buffer aliases the caller's tensor,
    so the input ``residual`` is mutated in place with ``residual + x``."""
    torch.manual_seed(0)
    device = "cuda"
    hidden = 1024
    x = torch.randn(4, hidden, dtype=torch.bfloat16, device=device)
    residual = torch.randn(4, hidden, dtype=torch.bfloat16, device=device)
    weight = torch.randn(hidden, dtype=torch.bfloat16, device=device)

    expected = residual + x
    _, residual_out = fused_add_rmsnorm(x, residual, weight, eps=1e-5)
    # Same storage -- caller's tensor sees the update.
    assert residual_out.data_ptr() == residual.data_ptr()
    torch.testing.assert_close(residual, expected, rtol=2e-2, atol=1e-2)


def test_residual_out_of_place_when_noncontiguous():
    """Non-contiguous residual: the kernel allocates a contiguous copy and
    returns *that* buffer. Callers must use the returned residual."""
    torch.manual_seed(0)
    device = "cuda"
    hidden = 1024
    x = torch.randn(4, hidden, dtype=torch.bfloat16, device=device)
    base = torch.randn(4, 2, hidden, dtype=torch.bfloat16, device=device)
    residual = base[:, 0]  # shape (4, hidden), stride (2*hidden, 1) -- non-contig
    assert not residual.is_contiguous()
    weight = torch.randn(hidden, dtype=torch.bfloat16, device=device)

    expected = residual + x
    residual_before = residual.clone()
    _, residual_out = fused_add_rmsnorm(x, residual, weight, eps=1e-5)

    # Caller's tensor unchanged; returned buffer holds the sum.
    torch.testing.assert_close(residual, residual_before)
    torch.testing.assert_close(residual_out, expected, rtol=2e-2, atol=1e-2)


def test_higher_rank_input_flattens_correctly():
    """Inputs of shape (s, b, h) should be treated the same as (s*b, h)."""
    torch.manual_seed(0)
    device = "cuda"
    s, b, h = 4, 8, 2048
    x_3d = torch.randn(s, b, h, dtype=torch.bfloat16, device=device)
    r_3d = torch.randn(s, b, h, dtype=torch.bfloat16, device=device)
    weight = torch.randn(h, dtype=torch.bfloat16, device=device)

    r_3d_ref = r_3d.clone()
    out_3d, r_3d_out = fused_add_rmsnorm(x_3d, r_3d, weight, eps=1e-5)

    r_2d = r_3d_ref.reshape(s * b, h)
    x_2d = x_3d.reshape(s * b, h)
    out_2d, r_2d_out = fused_add_rmsnorm(x_2d, r_2d, weight, eps=1e-5)

    torch.testing.assert_close(out_3d.reshape(s * b, h), out_2d, rtol=2e-2, atol=1e-2)
    torch.testing.assert_close(r_3d_out.reshape(s * b, h), r_2d_out, rtol=2e-2, atol=1e-2)
