# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""The cuTile / Triton mHC kernels under a published live-row bound: live rows match the unbounded
kernels, padding rows get zero outputs and zero gradients."""

import pytest
import torch

from megatron.core.fusions import fused_mhc_kernels as mhc
from megatron.core.transformer import padding_aware_gemm as pag

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and pag.HAVE_TRITON), reason="needs a GPU and Triton"
)


def _run_proj_rms(x, weight, alphas, bias, n, bound, T):
    leaves = [t.clone().requires_grad_() for t in (x, weight, *alphas, bias)]
    with pag.padding_aware_live_rows(bound, T) if bound is not None else _nullcontext():
        outs = mhc.fused_proj_rms_compute_h(
            leaves[0], leaves[1], leaves[2], leaves[3], leaves[4], leaves[5], n, 1e-6
        )
    return leaves, outs


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


@pytest.mark.launch_on_gb200  # cuTile (SM100) kernels: only the GB200 CI can exercise them
@pytest.mark.parametrize(
    "C", [256, 2048]
)  # K = n*C below / above the small-K backward threshold (8192)
def test_cutile_proj_rms_compute_h_skips_the_padding_tail(C):
    if not mhc.is_cutile_available():
        pytest.skip("cuTile unavailable")
    torch.manual_seed(0)
    T, n, m = 2048, 4, 1280
    K, N = n * C, 2 * n + n * n
    x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(N, K, device="cuda") * 0.02).float()
    alphas = [
        torch.full((1,), 1.0, device="cuda"),
        torch.full((1,), 1.0, device="cuda"),
        torch.full((1,), 1.0, device="cuda"),
    ]
    bias = torch.zeros(N, device="cuda")
    bound = torch.tensor([m], device="cuda", dtype=torch.int64)

    leaves_b, outs_b = _run_proj_rms(x, weight, alphas, bias, n, bound, T)
    leaves_r, outs_r = _run_proj_rms(x, weight, alphas, bias, n, None, T)
    # outputs: h_pre, h_post, h_res (zero tail) and r (tail = 1, a finite value the backward can
    # divide by; zero would make grad_x's grad_r_total * x / (r K) term NaN).
    for i, (ob, orf) in enumerate(zip(outs_b, outs_r)):
        torch.testing.assert_close(ob[:m], orf[:m], rtol=1e-3, atol=1e-3)
        if i == 3:
            assert torch.all(ob[m:] == 1), "padding rows of r must be 1"
        else:
            assert not ob[m:].any(), "padding rows of the forward outputs must be zero"
    grads = [torch.randn_like(o) for o in outs_r]
    for g in grads:
        g[m:] = 0  # padding rows carry no loss gradient
    torch.autograd.backward(outs_b, grads)
    torch.autograd.backward(outs_r, grads)
    gx_b, gx_r = leaves_b[0].grad, leaves_r[0].grad
    torch.testing.assert_close(gx_b[:m].float(), gx_r[:m].float(), rtol=2e-2, atol=2e-2)
    assert not gx_b[m:].any(), "padding rows of grad_x must be zero"
    for lb, lr in zip(leaves_b[1:], leaves_r[1:]):  # weight, alphas, bias: sums over live rows only
        torch.testing.assert_close(lb.grad, lr.grad, rtol=2e-2, atol=2e-2)


def test_sinkhorn_skips_the_padding_tail():
    if mhc._get_triton_sinkhorn() is None:
        pytest.skip("Triton sinkhorn unavailable")
    torch.manual_seed(0)
    T, n, m = 4096, 4, 2688
    logits = torch.randn(T, 1, n, n, device="cuda", dtype=torch.bfloat16)
    bound = torch.tensor([m], device="cuda", dtype=torch.int64)
    a = logits.clone().requires_grad_()
    b = logits.clone().requires_grad_()
    with pag.padding_aware_live_rows(bound, T):
        out_b = mhc.fused_sinkhorn(a, 20)
    out_r = mhc.fused_sinkhorn(b, 20)
    torch.testing.assert_close(out_b[:m], out_r[:m])
    assert not out_b[m:].any()
    g = torch.randn_like(out_r)
    g[m:] = 0
    out_b.backward(g)
    out_r.backward(g)
    torch.testing.assert_close(a.grad[:m], b.grad[:m])
    assert not a.grad[m:].any()
