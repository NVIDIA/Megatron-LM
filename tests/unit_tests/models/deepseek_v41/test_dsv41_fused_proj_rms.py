# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fused V4.1 mHC projection + RMS factor (Triton) vs the eager fp32 reference (GPU)."""

import pytest
import torch

from megatron.core.models.deepseek_v41.fused_proj_rms import (
    fused_v41_proj_rms_available,
    fused_v41_projection_and_rms,
)
from megatron.core.models.deepseek_v41.hyper_connection import v41_projection_and_rms


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize(
    "rows,n,hidden,dtype,contig",
    [
        (512, 4, 640, torch.bfloat16, True),  # aligned shapes
        (1000, 4, 625, torch.bfloat16, True),  # M and K not multiples of the tile sizes
        (300, 4, 640, torch.float32, True),  # fp32 streams
        (256, 4, 640, torch.bfloat16, False),  # non-contiguous x (transposed view)
        (16384, 4, 640, torch.bfloat16, True),  # long M reduction for grad_W (3xTF32 accumulation)
    ],
)
def test_fused_proj_rms_matches_eager(rows, n, hidden, dtype, contig):
    if not fused_v41_proj_rms_available():
        pytest.skip("Triton not available")
    torch.manual_seed(0)
    dev = "cuda"
    K, N, eps = n * hidden, n * n + 2 * n, 1e-6
    if contig:
        x = (torch.randn(rows, K, device=dev) * 0.7).to(dtype).requires_grad_(True)
    else:
        x = (torch.randn(K, rows, device=dev) * 0.7).to(dtype).t().requires_grad_(True)
        assert not x.is_contiguous()
    w = torch.empty(N, K, device=dev, dtype=torch.float32)
    torch.nn.init.xavier_uniform_(w)
    w.requires_grad_(True)
    # downstream weights so that both proj and r receive non-trivial gradients
    gp = torch.randn(rows, N, device=dev)
    gr = torch.randn(rows, 1, device=dev)

    def run(fn):
        xx = x.detach().clone().requires_grad_(True)
        ww = w.detach().clone().requires_grad_(True)
        proj, r = fn(xx, ww, eps)
        (proj * gp).sum().add_((r * gr).sum()).backward()
        return proj.detach(), r.detach(), xx.grad.detach(), ww.grad.detach()

    proj_f, r_f, gx_f, gw_f = run(fused_v41_projection_and_rms)
    # full-precision fp32 reference (no tf32 in the eager matmuls)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        proj_e, r_e, gx_e, gw_e = run(
            lambda xx, ww, e: v41_projection_and_rms(xx.to(torch.float32), ww, e)
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    assert proj_f.dtype == torch.float32 and r_f.shape == (rows, 1)
    torch.testing.assert_close(proj_f, proj_e, rtol=1e-4, atol=1e-4 * proj_e.abs().max().item())
    torch.testing.assert_close(r_f, r_e, rtol=1e-5, atol=1e-6)
    # grad_x is cast once to the stream dtype in both paths; compare in fp32
    gscale = gx_e.float().abs().max().item()
    torch.testing.assert_close(gx_f.float(), gx_e.float(), rtol=1e-2, atol=1e-2 * gscale)
    torch.testing.assert_close(gw_f, gw_e, rtol=1e-4, atol=1e-4 * gw_e.abs().max().item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_fused_proj_rms_deterministic_grad_w():
    if not fused_v41_proj_rms_available():
        pytest.skip("Triton not available")
    torch.manual_seed(1)
    dev = "cuda"
    x = torch.randn(2048, 4 * 640, device=dev).to(torch.bfloat16)
    w = torch.randn(24, 4 * 640, device=dev) * 0.02
    grads = []
    for _ in range(2):
        ww = w.clone().requires_grad_(True)
        proj, r = fused_v41_projection_and_rms(x, ww, 1e-6)
        (proj.sum() + r.sum()).backward()
        grads.append(ww.grad.clone())
    assert torch.equal(grads[0], grads[1])
