# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for fused mHC kernels (cuTile / torch.compile fallback).

Each test compares the fused kernel's forward output AND backward gradients
against a pure-PyTorch differentiable reference to catch numerical drift
introduced by kernel fusion.
"""

import math
from typing import Optional

import pytest
import torch
from torch import Tensor

from megatron.core.fusions.fused_mhc_kernels import (
    fused_h_aggregate,
    fused_h_post_bda,
    fused_proj_rms,
    fused_sinkhorn,
    is_cutile_available,
)


@pytest.fixture(autouse=True)
def _skip_without_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


DTYPE = torch.bfloat16
DEVICE = "cuda"
FWD_ATOL, FWD_RTOL = 2e-2, 2e-2
BWD_ATOL, BWD_RTOL = 5e-2, 5e-2
RAND_LO, RAND_HI = -0.1, 0.1


def _rand(*shape, **kwargs):
    """Uniform in [RAND_LO, RAND_HI] to keep magnitudes small for bf16 stability."""
    return torch.empty(*shape, dtype=DTYPE, device=DEVICE, **kwargs).uniform_(RAND_LO, RAND_HI)


def _info():
    backend = "cuTile" if is_cutile_available() else "torch.compile"
    print(f"\n  [backend: {backend}]")


# ============================================================================
# Pure-PyTorch differentiable references (used by both fwd AND bwd tests)
# ============================================================================


@torch.compile
def _ref_sinkhorn(logits: Tensor, num_iters: int, eps: float = 1e-8) -> Tensor:
    M = torch.exp(logits)
    for _ in range(num_iters):
        M = M / M.sum(dim=-1, keepdim=True).clamp(min=eps)
        M = M / M.sum(dim=-2, keepdim=True).clamp(min=eps)
    return M


@torch.compile
def _ref_h_aggregate(x: Tensor, h_pre: Tensor) -> Tensor:
    return (x * h_pre.unsqueeze(-1)).sum(dim=2)


@torch.compile
def _ref_h_post_bda(
    h_res: Tensor, orig_res: Tensor, h_post: Tensor, x: Tensor, bias: Optional[Tensor]
) -> Tensor:
    s, b, n, C = orig_res.shape
    mixed = torch.bmm(h_res.view(s * b, n, n), orig_res.view(s * b, n, C)).view(s, b, n, C)
    x_exp = h_post.unsqueeze(-1) * x.unsqueeze(2)
    out = x_exp + mixed
    if bias is not None:
        out = out + h_post.unsqueeze(-1) * bias.view(1, 1, 1, C)
    return out


@torch.compile
def _ref_proj_rms(x: Tensor, weight: Tensor, eps: float = 1e-8):
    proj = torch.matmul(x, weight.t())
    norm = x.norm(dim=-1, keepdim=True)
    K = x.shape[-1]
    r = 1.0 / (norm / math.sqrt(K) + eps)
    return proj, r


# ============================================================================
# Sinkhorn
# ============================================================================


class TestFusedSinkhorn:
    @pytest.mark.parametrize("s,b,n,iters", [(2, 4, 4, 5), (1, 1, 2, 10)])
    def test_fwd_bwd_vs_reference(self, s, b, n, iters):
        """E2E: fwd output and bwd grad must match the PyTorch reference."""
        _info()
        eps = 1e-8
        data = _rand(s, b, n, n)
        grad_out = _rand(s, b, n, n)

        # -- fused path --
        inp_f = data.clone().requires_grad_(True)
        out_f = fused_sinkhorn(inp_f, iters, eps)
        out_f.backward(grad_out)
        grad_f = inp_f.grad.clone()

        # -- reference path (fully differentiable) --
        inp_r = data.clone().requires_grad_(True)
        out_r = _ref_sinkhorn(inp_r, iters, eps)
        out_r.backward(grad_out)
        grad_r = inp_r.grad.clone()

        torch.testing.assert_close(out_f, out_r, atol=FWD_ATOL, rtol=FWD_RTOL)
        torch.testing.assert_close(grad_f, grad_r, atol=BWD_ATOL, rtol=BWD_RTOL)


# ============================================================================
# H_aggregate
# ============================================================================


class TestFusedHAggregate:
    @pytest.mark.parametrize("s,b,n,C", [(2, 4, 4, 1024), (1, 1, 2, 256)])
    def test_fwd_bwd_vs_reference(self, s, b, n, C):
        """E2E: fwd output and bwd grads must match the PyTorch reference."""
        _info()
        x_data = _rand(s, b, n, C)
        h_data = _rand(s, b, n)
        grad_out = _rand(s, b, C)

        # -- fused path --
        xf = x_data.clone().requires_grad_(True)
        hf = h_data.clone().requires_grad_(True)
        of = fused_h_aggregate(xf, hf)
        of.backward(grad_out)

        # -- reference path --
        xr = x_data.clone().requires_grad_(True)
        hr = h_data.clone().requires_grad_(True)
        oref = _ref_h_aggregate(xr, hr)
        oref.backward(grad_out)

        torch.testing.assert_close(of, oref, atol=FWD_ATOL, rtol=FWD_RTOL)
        torch.testing.assert_close(xf.grad, xr.grad, atol=BWD_ATOL, rtol=BWD_RTOL)
        torch.testing.assert_close(hf.grad, hr.grad, atol=BWD_ATOL, rtol=BWD_RTOL)


# ============================================================================
# H_post BDA
# ============================================================================


class TestFusedHPostBDA:
    @pytest.mark.parametrize("with_bias", [True, False])
    @pytest.mark.parametrize("s,b,n,C", [(2, 4, 4, 1024), (1, 2, 2, 256)])
    def test_fwd_bwd_vs_reference(self, s, b, n, C, with_bias):
        """E2E: fwd output and bwd grads must match the PyTorch reference."""
        _info()
        hr_data = _rand(s, b, n, n)
        orig_data = _rand(s, b, n, C)
        hp_data = _rand(s, b, n)
        x_data = _rand(s, b, C)
        bias_data = _rand(C) if with_bias else None
        grad_out = _rand(s, b, n, C)

        def _make_inputs():
            hr = hr_data.clone().requires_grad_(True)
            orig = orig_data.clone().requires_grad_(True)
            hp = hp_data.clone().requires_grad_(True)
            x = x_data.clone().requires_grad_(True)
            bi = bias_data.clone().requires_grad_(True) if with_bias else None
            return hr, orig, hp, x, bi

        # -- fused path --
        hr_f, orig_f, hp_f, x_f, bi_f = _make_inputs()
        out_f = fused_h_post_bda(hr_f, orig_f, hp_f, x_f, bi_f)
        out_f.backward(grad_out)

        # -- reference path --
        hr_r, orig_r, hp_r, x_r, bi_r = _make_inputs()
        out_r = _ref_h_post_bda(hr_r, orig_r, hp_r, x_r, bi_r)
        out_r.backward(grad_out)

        torch.testing.assert_close(out_f, out_r, atol=FWD_ATOL, rtol=FWD_RTOL)
        for name, gf, gr in [
            ("h_res", hr_f.grad, hr_r.grad),
            ("orig_res", orig_f.grad, orig_r.grad),
            ("h_post", hp_f.grad, hp_r.grad),
            ("x", x_f.grad, x_r.grad),
        ]:
            torch.testing.assert_close(
                gf, gr, atol=BWD_ATOL, rtol=BWD_RTOL, msg=f"backward mismatch on {name}"
            )
        if with_bias:
            torch.testing.assert_close(
                bi_f.grad, bi_r.grad, atol=BWD_ATOL, rtol=BWD_RTOL, msg="backward mismatch on bias"
            )


# ============================================================================
# Proj RMS
# ============================================================================


class TestFusedProjRms:
    @pytest.mark.parametrize("M,N,K", [(256, 20, 4096), (64, 8, 512)])
    def test_fwd_bwd_vs_reference(self, M, N, K):
        """E2E: fwd output and bwd grads must match the PyTorch reference."""
        _info()
        eps = 1e-8
        x_data = _rand(M, K)
        w_data = _rand(N, K)
        grad_proj = _rand(M, N)
        grad_r = _rand(M, 1)

        # -- fused path --
        xf = x_data.clone().requires_grad_(True)
        wf = w_data.clone().requires_grad_(True)
        proj_f, r_f = fused_proj_rms(xf, wf, eps)
        (proj_f * grad_proj + r_f * grad_r).sum().backward()

        # -- reference path --
        xr = x_data.clone().requires_grad_(True)
        wr = w_data.clone().requires_grad_(True)
        proj_r, r_r = _ref_proj_rms(xr, wr, eps)
        (proj_r * grad_proj + r_r * grad_r).sum().backward()

        torch.testing.assert_close(proj_f, proj_r, atol=FWD_ATOL, rtol=FWD_RTOL)
        torch.testing.assert_close(r_f, r_r, atol=FWD_ATOL, rtol=FWD_RTOL)
        torch.testing.assert_close(
            xf.grad, xr.grad, atol=BWD_ATOL, rtol=BWD_RTOL, msg="backward mismatch on x"
        )
        torch.testing.assert_close(
            wf.grad, wr.grad, atol=BWD_ATOL, rtol=BWD_RTOL, msg="backward mismatch on weight"
        )


# ============================================================================
# End-to-end pipeline (all four kernels chained)
# ============================================================================


class TestEndToEnd:
    """Full mHC pipeline: proj_rms -> compute_h -> sinkhorn -> aggregate -> h_post_bda.

    Runs the fused path and a pure-PyTorch reference path in lock-step, then
    compares both the final forward output and the gradients back to the input
    hidden_states.
    """

    def test_full_pipeline_fwd_bwd(self):
        _info()
        s, b, n, C = 2, 4, 4, 1024
        eps = 1e-6
        sinkhorn_iters = 5

        hs_data = _rand(s, b, n * C)
        w_data = _rand(n * n + 2 * n, n * C)
        layer_out_data = _rand(s, b, C)
        layer_bias_data = _rand(C)

        def _run_fused():
            hs = hs_data.clone().requires_grad_(True)
            w = w_data.clone().requires_grad_(True)

            x_2d = hs.reshape(s * b, n * C)
            proj, r = fused_proj_rms(x_2d, w, eps)
            proj = proj.view(s, b, -1)
            r = r.view(s, b, 1)

            h = r * proj
            h_pre = h[..., :n].sigmoid()
            h_post = h[..., n : 2 * n].sigmoid() * 2
            h_res_logits = h[..., 2 * n :]
            h_res = fused_sinkhorn(h_res_logits.view(s, b, n, n), sinkhorn_iters)

            aggregated = fused_h_aggregate(hs.view(s, b, n, C), h_pre)

            output = fused_h_post_bda(
                h_res, hs.view(s, b, n, C), h_post, layer_out_data, layer_bias_data
            )

            loss = output.sum() + aggregated.sum()
            loss.backward()
            return output.detach(), aggregated.detach(), hs.grad.clone()

        def _run_ref():
            hs = hs_data.clone().requires_grad_(True)
            w = w_data.clone().requires_grad_(True)

            x_2d = hs.reshape(s * b, n * C)
            proj, r = _ref_proj_rms(x_2d, w, eps)
            proj = proj.view(s, b, -1)
            r = r.view(s, b, 1)

            h = r * proj
            h_pre = h[..., :n].sigmoid()
            h_post = h[..., n : 2 * n].sigmoid() * 2
            h_res_logits = h[..., 2 * n :]
            h_res = _ref_sinkhorn(h_res_logits.view(s, b, n, n), sinkhorn_iters)

            aggregated = _ref_h_aggregate(hs.view(s, b, n, C), h_pre)

            output = _ref_h_post_bda(
                h_res, hs.view(s, b, n, C), h_post, layer_out_data, layer_bias_data
            )

            loss = output.sum() + aggregated.sum()
            loss.backward()
            return output.detach(), aggregated.detach(), hs.grad.clone()

        out_f, agg_f, grad_f = _run_fused()
        out_r, agg_r, grad_r = _run_ref()

        torch.testing.assert_close(
            agg_f, agg_r, atol=FWD_ATOL, rtol=FWD_RTOL, msg="aggregated output mismatch"
        )
        torch.testing.assert_close(
            out_f, out_r, atol=FWD_ATOL, rtol=FWD_RTOL, msg="h_post_bda output mismatch"
        )
        torch.testing.assert_close(
            grad_f,
            grad_r,
            atol=BWD_ATOL,
            rtol=BWD_RTOL,
            msg=f"hidden_states grad mismatch (E2E backward), max diff: {grad_f.max() - grad_r.max()}",
        )
