# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for ``HyperConnection.post`` delegating to the Core mHC kernels.

``post`` used to spell the residual mixing as a batched ``comb @ residual``. With
``hc_mult`` residual streams the contracted dimension is single-digit, so cuBLAS
had no aligned path to dispatch to and fell back to an ``sm80`` WMMA kernel with
``align2`` on Hopper -- measured at ~7-8% of all GPU kernel time on DeepSeek-V4.
Core already ships fused mHC kernels for exactly this expression, so ``post`` now
calls ``fused_h_post_bda``.

The conventions differ by a transpose: Core computes ``h_res.T @ residual`` while
this module carries ``comb`` the other way round. That is invisible to a reader
matching on names -- both operands are square and equally plausible -- and a run
with the wrong orientation still produces correctly shaped, finite activations.
The reference test below therefore pins the orientation against an explicit
recomputation, and ``test_orientation_is_not_symmetric`` asserts that the check
can actually fail.
"""

from __future__ import annotations

import pytest
import torch

from megatron.lite.primitive.modules.attention.hca import HyperConnection, split_sinkhorn

pytestmark = [pytest.mark.mlite]

S, B, N, C = 3, 2, 4, 16


def _make_inputs(dtype: torch.dtype, device: str):
    """Build (x, residual, post, comb) with a real Sinkhorn-normalised ``comb``."""
    generator = torch.Generator(device="cpu").manual_seed(0)
    mixes = torch.randn(S, B, (2 + N) * N, generator=generator).to(device=device, dtype=dtype)
    scale = torch.ones(3, device=device, dtype=torch.float32)
    base = torch.zeros((2 + N) * N, device=device, dtype=torch.float32)
    _, post, comb = split_sinkhorn(mixes, scale, base, N, 3, 1e-6)
    x = torch.randn(S, B, C, generator=generator).to(device=device, dtype=dtype)
    residual = torch.randn(S, B, N, C, generator=generator).to(device=device, dtype=dtype)
    return x, residual, post, comb


def _reference_post(x, residual, post, comb):
    """The pre-fusion expression, kept verbatim as the numerical contract."""
    dtype = x.dtype
    placed = post.to(dtype).unsqueeze(-1) * x.unsqueeze(-2)
    mixed = torch.matmul(comb.to(dtype), residual.to(dtype))
    return placed + mixed


def test_post_matches_pre_fusion_reference() -> None:
    """The fused path reproduces the expression it replaced, in fp32."""
    x, residual, post, comb = _make_inputs(torch.float32, "cpu")
    expected = _reference_post(x, residual, post, comb)
    actual = HyperConnection.post(x, residual, post, comb)
    assert actual.shape == (S, B, N, C)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_orientation_is_not_symmetric() -> None:
    """Guard the guard: passing ``comb`` un-transposed must disagree.

    Without this, a regression that drops the transpose would still satisfy
    ``test_post_matches_pre_fusion_reference`` whenever ``comb`` happened to be
    symmetric, and the suite would pass while every layer mixed residual streams
    the wrong way.
    """
    _, residual, _, comb = _make_inputs(torch.float32, "cpu")
    assert not torch.allclose(comb, comb.transpose(-1, -2), rtol=1e-3, atol=1e-3)
    correct = torch.matmul(comb, residual)
    swapped = torch.matmul(comb.transpose(-1, -2), residual)
    assert not torch.allclose(correct, swapped, rtol=1e-3, atol=1e-3)


def test_post_is_differentiable_through_both_terms() -> None:
    """Both the mixing and the placement term must carry gradient."""
    x, residual, post, comb = _make_inputs(torch.float32, "cpu")
    x = x.detach().requires_grad_(True)
    residual = residual.detach().requires_grad_(True)
    HyperConnection.post(x, residual, post, comb).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert residual.grad is not None and torch.isfinite(residual.grad).all()
    assert x.grad.abs().sum() > 0
    assert residual.grad.abs().sum() > 0


@pytest.mark.gpus(1)
def test_post_matches_pre_fusion_reference_bf16_gpu() -> None:
    """Same contract in the dtype and on the device production actually uses."""
    x, residual, post, comb = _make_inputs(torch.bfloat16, "cuda")
    expected = _reference_post(x, residual, post, comb)
    actual = HyperConnection.post(x, residual, post, comb)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
