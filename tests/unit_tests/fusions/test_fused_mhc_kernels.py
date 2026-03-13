# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for fused mHC kernels (cuTile) and native implementations.

Each test compares the fused kernel's forward output AND backward gradients
against a pure-PyTorch differentiable reference to catch numerical drift
introduced by kernel fusion.
"""

import math
from typing import Optional

import pytest
import torch
from torch import Tensor

from megatron.core.fusions.fused_mhc_kernels import is_cutile_available
from megatron.core.transformer.hyper_connection import (
    NativeHAggregate,
    NativeHPostBDA,
    NativeProjRms,
    native_sinkhorn,
)

_require_cutile = pytest.mark.skipif(not is_cutile_available(), reason="cuTile not installed")


@pytest.fixture(autouse=True)
def _skip_without_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


DTYPE = torch.bfloat16
DEVICE = "cuda"
FWD_ATOL, FWD_RTOL = 2e-2, 2e-2
BWD_ATOL, BWD_RTOL = 5e-2, 5e-2
RAND_LO, RAND_HI = -0.1, 0.1
COSINE_SIM_THRESH = 0.999


def _assert_cosine_similar(a: Tensor, b: Tensor, threshold: float, msg: str = ""):
    """Assert that flattened tensors have cosine similarity >= threshold."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    sim = torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()
    assert sim >= threshold, (
        f"{msg}: cosine similarity {sim:.6f} < {threshold} "
        f"(max_abs_diff={torch.max(torch.abs(a_flat - b_flat)):.6e})"
    )


def _rand(*shape, **kwargs):
    """Uniform in [RAND_LO, RAND_HI] to keep magnitudes small for bf16 stability."""
    return torch.empty(*shape, dtype=DTYPE, device=DEVICE, **kwargs).uniform_(RAND_LO, RAND_HI)


def _info():
    backend = "cuTile" if is_cutile_available() else "native"
    print(f"\n  [backend: {backend}]")


# ============================================================================
# Pure-PyTorch differentiable references (used by both fwd AND bwd tests)
# ============================================================================


def _ref_sinkhorn(logits: Tensor, num_iters: int, eps: float = 1e-6) -> Tensor:
    row_max = logits.max(dim=-1, keepdim=True).values
    M = torch.exp(logits - row_max)
    for _ in range(num_iters):
        M = M / M.sum(dim=-1, keepdim=True).clamp(min=eps)
        M = M / M.sum(dim=-2, keepdim=True).clamp(min=eps)
    return M


def _ref_h_aggregate(x: Tensor, h_pre: Tensor) -> Tensor:
    return (x * h_pre.unsqueeze(-1)).sum(dim=2)


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


def _ref_proj_rms(x: Tensor, weight: Tensor, eps: float = 1e-6):
    proj = torch.matmul(x, weight.t())
    norm = x.norm(dim=-1, keepdim=True)
    K = x.shape[-1]
    r = 1.0 / (norm / math.sqrt(K) + eps)
    return proj, r


# ============================================================================
# Sinkhorn
# ============================================================================


class TestNativeSinkhorn:
    """Tests for the native SinkhornKnopp implementation."""

    @pytest.mark.parametrize("s,b,n,iters", [(2, 4, 4, 5), (1, 1, 2, 10)])
    def test_fwd_bwd_vs_torch_reference(self, s, b, n, iters):
        """native_sinkhorn fwd output and bwd grad must match the inline PyTorch reference."""
        _info()
        eps = 1e-6
        data = _rand(s, b, n, n)
        grad_out = _rand(s, b, n, n)

        # -- native_sinkhorn path (autograd.Function) --
        inp_f = data.clone().requires_grad_(True)
        out_f = native_sinkhorn(inp_f, iters, eps)
        out_f.backward(grad_out)
        grad_f = inp_f.grad.clone()

        # -- inline torch reference (fully differentiable) --
        inp_r = data.clone().requires_grad_(True)
        out_r = _ref_sinkhorn(inp_r, iters, eps)
        out_r.backward(grad_out)
        grad_r = inp_r.grad.clone()

        torch.testing.assert_close(out_f, out_r, atol=FWD_ATOL, rtol=FWD_RTOL)
        torch.testing.assert_close(grad_f, grad_r, atol=BWD_ATOL, rtol=BWD_RTOL)


class TestFusedSinkhorn:
    @_require_cutile
    @pytest.mark.parametrize("s,b,n,iters", [(2, 4, 4, 5), (1, 1, 2, 10)])
    def test_fwd_bwd_vs_reference(self, s, b, n, iters):
        """E2E: fused cuTile fwd output and bwd grad must match the PyTorch reference."""
        from megatron.core.fusions.fused_mhc_kernels import fused_sinkhorn

        _info()
        eps = 1e-6
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


class TestNativeHAggregate:
    """Tests for the NativeHAggregate module."""

    @pytest.mark.parametrize("s,b,n,C", [(2, 4, 4, 1024), (1, 1, 2, 256)])
    def test_fwd_bwd_vs_torch_reference(self, s, b, n, C):
        _info()
        native_mod = NativeHAggregate().to(DEVICE)
        x_data = _rand(s, b, n, C)
        h_data = _rand(s, b, n)
        grad_out = _rand(s, b, C)

        xf = x_data.clone().requires_grad_(True)
        hf = h_data.clone().requires_grad_(True)
        of = native_mod(xf, hf)
        of.backward(grad_out)

        xr = x_data.clone().requires_grad_(True)
        hr = h_data.clone().requires_grad_(True)
        oref = _ref_h_aggregate(xr, hr)
        oref.backward(grad_out)

        torch.testing.assert_close(of, oref, atol=FWD_ATOL, rtol=FWD_RTOL)
        torch.testing.assert_close(xf.grad, xr.grad, atol=BWD_ATOL, rtol=BWD_RTOL)
        torch.testing.assert_close(hf.grad, hr.grad, atol=BWD_ATOL, rtol=BWD_RTOL)


class TestFusedHAggregate:
    @_require_cutile
    @pytest.mark.parametrize("s,b,n,C", [(2, 4, 4, 1024), (1, 1, 2, 256)])
    def test_fwd_bwd_vs_reference(self, s, b, n, C):
        """E2E: fused cuTile fwd output and bwd grads must match the PyTorch reference."""
        from megatron.core.fusions.fused_mhc_kernels import fused_h_aggregate

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


class TestNativeHPostBDA:
    """Tests for the NativeHPostBDA module."""

    @pytest.mark.parametrize("with_bias", [True, False])
    @pytest.mark.parametrize("s,b,n,C", [(2, 4, 4, 1024), (1, 2, 2, 256)])
    def test_fwd_bwd_vs_torch_reference(self, s, b, n, C, with_bias):
        _info()
        native_mod = NativeHPostBDA().to(DEVICE)
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

        hr_f, orig_f, hp_f, x_f, bi_f = _make_inputs()
        out_f = native_mod(hr_f, orig_f, hp_f, x_f, bi_f)
        out_f.backward(grad_out)

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


class TestFusedHPostBDA:
    @_require_cutile
    @pytest.mark.parametrize("with_bias", [True, False])
    @pytest.mark.parametrize("s,b,n,C", [(2, 4, 4, 1024), (1, 2, 2, 256)])
    def test_fwd_bwd_vs_reference(self, s, b, n, C, with_bias):
        """E2E: fused cuTile fwd output and bwd grads must match the PyTorch reference."""
        from megatron.core.fusions.fused_mhc_kernels import fused_h_post_bda

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


class TestNativeProjRms:
    """Tests for the NativeProjRms module."""

    @pytest.mark.parametrize("M,N,K", [(256, 20, 4096), (64, 8, 512)])
    def test_fwd_bwd_vs_torch_reference(self, M, N, K):
        _info()
        eps = 1e-6
        native_mod = NativeProjRms().to(DEVICE)
        x_data = _rand(M, K)
        w_data = _rand(N, K)
        grad_proj = _rand(M, N)
        grad_r = _rand(M, 1)

        xf = x_data.clone().requires_grad_(True)
        wf = w_data.clone().requires_grad_(True)
        proj_f, r_f = native_mod(xf, wf, eps)
        (proj_f * grad_proj + r_f * grad_r).sum().backward()

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


class TestFusedProjRms:
    @_require_cutile
    @pytest.mark.parametrize("M,N,K", [(256, 20, 4096), (64, 8, 512)])
    def test_fwd_bwd_vs_reference(self, M, N, K):
        """E2E: fused cuTile fwd output and bwd grads must match the PyTorch reference."""
        from megatron.core.fusions.fused_mhc_kernels import fused_proj_rms

        _info()
        eps = 1e-6
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


class TestEndToEndNative:
    """Full mHC pipeline using native modules.

    proj_rms -> compute_h -> sinkhorn -> aggregate -> h_post_bda.
    Compares the native modules against inline PyTorch reference.
    """

    def test_full_pipeline_fwd_bwd(self):
        _info()
        s, b, n, C = 2, 4, 4, 1024
        eps = 1e-6
        sinkhorn_iters = 5

        native_proj_rms_mod = NativeProjRms().to(DEVICE)
        native_h_agg_mod = NativeHAggregate().to(DEVICE)
        native_hpb_mod = NativeHPostBDA().to(DEVICE)

        hs_data = _rand(s, b, n * C)
        w_data = _rand(n * n + 2 * n, n * C)
        layer_out_data = _rand(s, b, C)
        layer_bias_data = _rand(C)

        def _run_native_modules():
            hs = hs_data.clone().requires_grad_(True)
            w = w_data.clone().requires_grad_(True)

            x_2d = hs.reshape(s * b, n * C)
            proj, r = native_proj_rms_mod(x_2d, w, eps)
            proj = proj.view(s, b, -1)
            r = r.view(s, b, 1)

            h = r * proj
            h_pre = h[..., :n].sigmoid()
            h_post = h[..., n : 2 * n].sigmoid() * 2
            h_res_logits = h[..., 2 * n :]
            h_res = native_sinkhorn(h_res_logits.view(s, b, n, n), sinkhorn_iters, eps)

            aggregated = native_h_agg_mod(hs.view(s, b, n, C), h_pre)

            output = native_hpb_mod(
                h_res, hs.view(s, b, n, C), h_post, layer_out_data, layer_bias_data
            )

            loss = output.sum() + aggregated.sum()
            loss.backward()
            return output.detach(), aggregated.detach(), hs.grad.clone()

        def _run_inline_ref():
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
            h_res = _ref_sinkhorn(h_res_logits.view(s, b, n, n), sinkhorn_iters, eps)

            aggregated = _ref_h_aggregate(hs.view(s, b, n, C), h_pre)

            output = _ref_h_post_bda(
                h_res, hs.view(s, b, n, C), h_post, layer_out_data, layer_bias_data
            )

            loss = output.sum() + aggregated.sum()
            loss.backward()
            return output.detach(), aggregated.detach(), hs.grad.clone()

        out_m, agg_m, grad_m = _run_native_modules()
        out_r, agg_r, grad_r = _run_inline_ref()

        torch.testing.assert_close(
            agg_m, agg_r, atol=FWD_ATOL, rtol=FWD_RTOL, msg="aggregated output mismatch"
        )
        torch.testing.assert_close(
            out_m, out_r, atol=FWD_ATOL, rtol=FWD_RTOL, msg="h_post_bda output mismatch"
        )
        _assert_cosine_similar(
            grad_m, grad_r, COSINE_SIM_THRESH, msg="hidden_states grad (E2E backward)"
        )


class TestEndToEndFused:
    """Full mHC pipeline using fused cuTile kernels (requires cuTile)."""

    @_require_cutile
    def test_full_pipeline_fwd_bwd(self):
        from megatron.core.fusions.fused_mhc_kernels import (
            fused_h_aggregate,
            fused_h_post_bda,
            fused_proj_rms,
            fused_sinkhorn,
        )

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
            h_res = fused_sinkhorn(h_res_logits.view(s, b, n, n), sinkhorn_iters, eps)

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
            h_res = _ref_sinkhorn(h_res_logits.view(s, b, n, n), sinkhorn_iters, eps)

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
        _assert_cosine_similar(
            grad_f, grad_r, COSINE_SIM_THRESH, msg="hidden_states grad (E2E backward)"
        )
