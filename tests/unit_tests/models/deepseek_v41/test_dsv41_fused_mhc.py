# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fused (merged Triton/cuTile) vs fp32 mHC stream aggregation and residual update (GPU)."""

import pytest
import torch

from megatron.core.models.deepseek_v41.hyper_connection import (
    aggregate_streams_fp32,
    aggregate_streams_fused,
    residual_update_fp32,
    residual_update_fused,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("s,b,n,c", [(1024, 1, 4, 512), (333, 2, 4, 256)])
def test_fused_mhc_ops_match_fp32(s, b, n, c):
    torch.manual_seed(0)
    dev = "cuda"
    streams = (torch.randn(s, b, n * c, device=dev) * 0.5).to(torch.bfloat16).requires_grad_(True)
    branch = (torch.randn(s, b, c, device=dev) * 0.5).to(torch.bfloat16).requires_grad_(True)
    h_pre = torch.sigmoid(torch.randn(s, b, n, device=dev)) + 1e-6
    h_post = 2 * torch.sigmoid(torch.randn(s, b, n, device=dev))
    h_res = torch.softmax(torch.randn(s, b, n, n, device=dev), dim=-1)
    ulp = 2**-7  # bf16 relative spacing

    ref = aggregate_streams_fp32(streams, h_pre, n)
    out = aggregate_streams_fused(streams, h_pre, n)
    assert out.dtype == ref.dtype and out.shape == ref.shape
    tol = ulp * ref.float().abs().amax().item() + 1e-3
    assert (out.float() - ref.float()).abs().max().item() <= tol

    ref_r = residual_update_fp32(streams, branch, h_post, h_res, n)
    out_r = residual_update_fused(streams, branch, h_post, h_res, n)
    assert out_r.dtype == ref_r.dtype and out_r.shape == ref_r.shape
    tol = ulp * ref_r.float().abs().amax().item() + 1e-3
    assert (out_r.float() - ref_r.float()).abs().max().item() <= tol

    # gradients through the fused residual update
    g = torch.randn_like(ref_r)
    ref_r.backward(g, retain_graph=True)
    gs_ref, gb_ref = streams.grad.clone(), branch.grad.clone()
    streams.grad = None
    branch.grad = None
    out_r.backward(g)
    tol_s = ulp * gs_ref.float().abs().amax().item() + 1e-3
    tol_b = ulp * gb_ref.float().abs().amax().item() + 1e-3
    assert (streams.grad.float() - gs_ref.float()).abs().max().item() <= tol_s
    assert (branch.grad.float() - gb_ref.float()).abs().max().item() <= tol_b
