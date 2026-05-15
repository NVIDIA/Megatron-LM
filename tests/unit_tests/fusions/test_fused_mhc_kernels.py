# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for unified fused mHC kernels and native implementations.

Each test compares the fused kernel's forward output AND backward gradients
against a pure-PyTorch differentiable reference to catch numerical drift
introduced by kernel fusion.
"""

import math
from typing import Optional

import pytest
import torch
from torch import Tensor

from megatron.core.fusions.fused_mhc_kernels import is_cutile_available, is_triton_available
from megatron.core.transformer.hyper_connection import (
    native_h_aggregate,
    native_h_post_bda,
    native_proj_rms,
    native_sinkhorn,
)

_require_cutile = pytest.mark.skipif(not is_cutile_available(), reason="cuTile not installed")
_require_triton = pytest.mark.skipif(not is_triton_available(), reason="Triton not installed")


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
    if is_triton_available() and is_cutile_available():
        backend = "triton+cuTile"
    elif is_triton_available():
        backend = "triton+native"
    elif is_cutile_available():
        backend = "cuTile"
    else:
        backend = "native"
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


def _ref_proj_rms_compute_h(
    x: Tensor,
    weight: Tensor,
    alpha_pre: Tensor,
    alpha_post: Tensor,
    alpha_res: Tensor,
    bias: Tensor,
    n: int,
    eps: float = 1e-6,
):
    """Reference: fused proj_rms + compute_h."""
    proj = torch.matmul(x, weight.t())
    norm = x.norm(dim=-1, keepdim=True)
    K = x.shape[-1]
    r = norm / math.sqrt(K)  # [M, 1]
    N = proj.shape[-1]
    alpha = torch.cat([alpha_pre.expand(n), alpha_post.expand(n), alpha_res.expand(N - 2 * n)])
    h = proj * alpha.unsqueeze(0) / (r + eps) + bias.unsqueeze(0)
    h_pre = h[..., :n].sigmoid()
    h_post = h[..., n : 2 * n].sigmoid() * 2
    h_res = h[..., 2 * n :]
    return h_pre, h_post, h_res, r


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
    @pytest.mark.parametrize("s,b,n,iters", [(2, 4, 4, 5), (1, 1, 2, 10)])
    def test_fwd_bwd_vs_reference(self, s, b, n, iters):
        """E2E: public fused fwd output and bwd grad must match the PyTorch reference."""
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
    """Tests for native_h_aggregate."""

    @pytest.mark.parametrize("s,b,n,C", [(2, 4, 4, 1024), (1, 1, 2, 256)])
    def test_fwd_bwd_vs_torch_reference(self, s, b, n, C):
        _info()
        x_data = _rand(s, b, n, C)
        h_data = _rand(s, b, n)
        grad_out = _rand(s, b, C)

        xf = x_data.clone().requires_grad_(True)
        hf = h_data.clone().requires_grad_(True)
        of = native_h_aggregate(xf, hf)
        of.backward(grad_out)

        xr = x_data.clone().requires_grad_(True)
        hr = h_data.clone().requires_grad_(True)
        oref = _ref_h_aggregate(xr, hr)
        oref.backward(grad_out)

        torch.testing.assert_close(of, oref, atol=FWD_ATOL, rtol=FWD_RTOL)
        torch.testing.assert_close(xf.grad, xr.grad, atol=BWD_ATOL, rtol=BWD_RTOL)
        torch.testing.assert_close(hf.grad, hr.grad, atol=BWD_ATOL, rtol=BWD_RTOL)


class TestFusedHAggregate:
    @pytest.mark.parametrize("s,b,n,C", [(2, 4, 4, 1024), (1, 1, 2, 256)])
    def test_fwd_bwd_vs_reference(self, s, b, n, C):
        """E2E: public fused fwd output and bwd grads must match the PyTorch reference."""
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


class TestTritonHAggregate:
    """Tests for Triton h_aggregate forward against PyTorch reference."""

    @_require_triton
    @pytest.mark.parametrize("s,b,n,C", [(2, 4, 4, 1024), (1, 1, 2, 256), (64, 8, 4, 4096)])
    def test_fwd_vs_reference(self, s, b, n, C):
        from megatron.core.fusions.fused_mhc_kernels import _triton_h_aggregate_fwd

        _info()
        x_data = _rand(s, b, n, C)
        h_data = _rand(s, b, n)

        out_t = _triton_h_aggregate_fwd(x_data, h_data)
        out_r = _ref_h_aggregate(x_data, h_data)

        torch.testing.assert_close(out_t, out_r, atol=FWD_ATOL, rtol=FWD_RTOL)


# ============================================================================
# H_post BDA
# ============================================================================


class TestNativeHPostBDA:
    """Tests for native_h_post_bda."""

    @pytest.mark.parametrize("with_bias", [True, False])
    @pytest.mark.parametrize("s,b,n,C", [(2, 4, 4, 1024), (1, 2, 2, 256)])
    def test_fwd_bwd_vs_torch_reference(self, s, b, n, C, with_bias):
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

        hr_f, orig_f, hp_f, x_f, bi_f = _make_inputs()
        out_f = native_h_post_bda(hr_f, orig_f, hp_f, x_f, bi_f)
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
    @pytest.mark.parametrize("with_bias", [True, False])
    @pytest.mark.parametrize("s,b,n,C", [(2, 4, 4, 1024), (1, 2, 2, 256)])
    def test_fwd_bwd_vs_reference(self, s, b, n, C, with_bias):
        """E2E: public fused fwd output and bwd grads must match the PyTorch reference."""
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


class TestTritonHPostBDA:
    """Tests for Triton h_post_bda kernels against PyTorch reference."""

    @_require_triton
    @pytest.mark.parametrize("with_bias", [True, False])
    @pytest.mark.parametrize("s,b,n,C", [(2, 4, 4, 1024), (1, 2, 2, 256), (64, 8, 4, 4096)])
    def test_fwd_vs_reference(self, s, b, n, C, with_bias):
        """Triton hpb forward output must match the PyTorch reference."""
        from megatron.core.fusions.fused_mhc_kernels import _triton_h_post_bda_fwd

        _info()
        hr_data = _rand(s, b, n, n)
        orig_data = _rand(s, b, n, C)
        hp_data = _rand(s, b, n)
        x_data = _rand(s, b, C)
        bias_data = _rand(C) if with_bias else None

        out_t = _triton_h_post_bda_fwd(hr_data, orig_data, hp_data, x_data, bias_data)
        out_r = _ref_h_post_bda(hr_data, orig_data, hp_data, x_data, bias_data)

        torch.testing.assert_close(out_t, out_r, atol=FWD_ATOL, rtol=FWD_RTOL)

    @_require_triton
    @pytest.mark.parametrize("with_bias", [True, False])
    @pytest.mark.parametrize(
        "s,b,n,C", [(2, 4, 4, 1024), (1, 2, 2, 256), (64, 8, 4, 4096), (128, 1, 8, 7168)]
    )
    def test_bwd_vs_reference(self, s, b, n, C, with_bias):
        """Triton hpb backward grads must match the PyTorch reference."""
        from megatron.core.fusions.fused_mhc_kernels import _triton_h_post_bda_bwd

        _info()
        hr_data = _rand(s, b, n, n)
        orig_data = _rand(s, b, n, C)
        hp_data = _rand(s, b, n)
        x_data = _rand(s, b, C)
        bias_data = _rand(C) if with_bias else None
        grad_out = _rand(s, b, n, C)

        # -- Triton backward --
        g_hr_t, g_res_t, g_hp_t, g_x_t, g_bias_t = _triton_h_post_bda_bwd(
            grad_out, hr_data, orig_data, hp_data, x_data, bias_data
        )

        # -- Reference backward via autograd --
        hr_r = hr_data.clone().requires_grad_(True)
        orig_r = orig_data.clone().requires_grad_(True)
        hp_r = hp_data.clone().requires_grad_(True)
        x_r = x_data.clone().requires_grad_(True)
        bi_r = bias_data.clone().requires_grad_(True) if with_bias else None
        out_r = _ref_h_post_bda(hr_r, orig_r, hp_r, x_r, bi_r)
        out_r.backward(grad_out)

        for name, gt, gr in [
            ("h_res", g_hr_t, hr_r.grad),
            ("orig_res", g_res_t, orig_r.grad),
            ("h_post", g_hp_t, hp_r.grad),
            ("x", g_x_t, x_r.grad),
        ]:
            torch.testing.assert_close(
                gt, gr, atol=BWD_ATOL, rtol=BWD_RTOL, msg=f"Triton backward mismatch on {name}"
            )
        if with_bias:
            torch.testing.assert_close(
                g_bias_t,
                bi_r.grad,
                atol=BWD_ATOL,
                rtol=BWD_RTOL,
                msg="Triton backward mismatch on bias",
            )

    @_require_triton
    @_require_cutile
    @pytest.mark.parametrize("with_bias", [True, False])
    @pytest.mark.parametrize("s,b,n,C", [(2, 4, 4, 1024), (1, 2, 2, 256), (64, 8, 4, 4096)])
    def test_triton_vs_cutile(self, s, b, n, C, with_bias):
        """Triton and cuTile backward must produce identical results."""
        from megatron.core.fusions.fused_mhc_kernels import (
            _cutile_h_post_bda_bwd,
            _triton_h_post_bda_bwd,
        )

        _info()
        hr_data = _rand(s, b, n, n)
        orig_data = _rand(s, b, n, C)
        hp_data = _rand(s, b, n)
        x_data = _rand(s, b, C)
        bias_data = _rand(C) if with_bias else None
        grad_out = _rand(s, b, n, C)

        triton_out = _triton_h_post_bda_bwd(
            grad_out, hr_data, orig_data, hp_data, x_data, bias_data
        )
        cutile_out = _cutile_h_post_bda_bwd(
            grad_out, hr_data, orig_data, hp_data, x_data, bias_data
        )

        for i, name in enumerate(["h_res", "orig_res", "h_post", "x", "bias"]):
            if triton_out[i] is None:
                continue
            torch.testing.assert_close(
                triton_out[i],
                cutile_out[i],
                atol=BWD_ATOL,
                rtol=BWD_RTOL,
                msg=f"Triton vs cuTile mismatch on {name}",
            )


class TestTritonHPostBDABwdE2EDebug:
    """Debug: run E2E forward, then compare cuTile vs Triton backward per-output."""

    @_require_triton
    @_require_cutile
    def test_e2e_inputs_no_nan(self):
        """Feed actual E2E backward inputs to Triton kernel and check for NaN."""
        from megatron.core.fusions.fused_mhc_kernels import (
            _cutile_h_post_bda_bwd,
            _triton_h_post_bda_bwd,
            fused_h_aggregate,
            fused_h_post_bda,
            fused_proj_rms,
            fused_sinkhorn,
        )

        s, b, n, C = 8, 4, 4, 1024
        eps = 1e-6
        sinkhorn_iters = 5

        hs_data = _rand(s, b, n * C)
        w_data = _rand(n * n + 2 * n, n * C)
        layer_out_data = _rand(s, b, C)
        layer_bias_data = _rand(C)

        # Run E2E forward to produce realistic backward inputs
        hs = hs_data.clone().requires_grad_(True)
        w = w_data.clone().requires_grad_(True)
        x_2d = hs.reshape(s * b, n * C)
        proj, r = fused_proj_rms(x_2d, w, eps)
        proj = proj.view(s, b, -1)
        r = r.view(s, b, 1)
        h = r * proj
        h_pre = h[..., :n].sigmoid()
        h_post_val = h[..., n : 2 * n].sigmoid() * 2
        h_res = fused_sinkhorn(h[..., 2 * n :].view(s, b, n, n), sinkhorn_iters, eps)
        _ = fused_h_aggregate(hs.view(s, b, n, C), h_pre)
        output = fused_h_post_bda(
            h_res, hs.view(s, b, n, C), h_post_val, layer_out_data, layer_bias_data
        )
        go = torch.ones_like(output)

        # Capture inputs (detach from graph)
        hr = h_res.detach()
        orig = hs.view(s, b, n, C).detach()
        hp = h_post_val.detach()
        x = layer_out_data.detach()
        bias = layer_bias_data.detach()

        # Compare cuTile vs Triton per-output
        ct_out = _cutile_h_post_bda_bwd(go, hr, orig, hp, x, bias)
        tr_out = _triton_h_post_bda_bwd(go, hr, orig, hp, x, bias)

        names = ["g_hr", "g_res", "g_hp", "g_x", "g_bias"]
        for name, ct_t, tr_t in zip(names, ct_out, tr_out):
            if tr_t is None:
                continue
            assert not tr_t.isnan().any(), f"Triton {name} has NaN"
            assert not tr_t.isinf().any(), f"Triton {name} has Inf"
            torch.testing.assert_close(
                tr_t,
                ct_t,
                atol=BWD_ATOL,
                rtol=BWD_RTOL,
                msg=f"Triton vs cuTile mismatch on {name} (E2E inputs)",
            )


# ============================================================================
# Triton: Sinkhorn
# ============================================================================


class TestTritonSinkhorn:
    @_require_triton
    @pytest.mark.parametrize("s,b,n,iters", [(2, 4, 4, 5), (1, 1, 2, 10), (8, 4, 4, 20)])
    def test_fwd_bwd_vs_reference(self, s, b, n, iters):
        from megatron.core.fusions.fused_mhc_kernels import triton_fused_sinkhorn

        eps = 1e-6
        data = _rand(s, b, n, n)
        grad_out = _rand(s, b, n, n)

        inp_f = data.clone().requires_grad_(True)
        out_f = triton_fused_sinkhorn(inp_f, iters, eps)
        out_f.backward(grad_out)

        inp_r = data.clone().requires_grad_(True)
        out_r = _ref_sinkhorn(inp_r, iters, eps)
        out_r.backward(grad_out)

        torch.testing.assert_close(out_f, out_r, atol=FWD_ATOL, rtol=FWD_RTOL)
        torch.testing.assert_close(inp_f.grad, inp_r.grad, atol=BWD_ATOL, rtol=BWD_RTOL)

    @_require_triton
    @_require_cutile
    @pytest.mark.parametrize("s,b,n,iters", [(2, 4, 4, 5)])
    def test_triton_vs_cutile(self, s, b, n, iters):
        from megatron.core.fusions.fused_mhc_kernels import (
            _cutile_sinkhorn_bwd,
            _cutile_sinkhorn_fwd,
            triton_fused_sinkhorn,
        )

        eps = 1e-6
        data = _rand(s, b, n, n)
        grad_out = _rand(s, b, n, n)

        inp_t = data.clone().requires_grad_(True)
        out_t = triton_fused_sinkhorn(inp_t, iters, eps)
        out_t.backward(grad_out)

        out_c, M_init = _cutile_sinkhorn_fwd(data.clone(), iters, eps)
        grad_c = _cutile_sinkhorn_bwd(grad_out, M_init, iters, eps)

        torch.testing.assert_close(out_t, out_c, atol=FWD_ATOL, rtol=FWD_RTOL)
        torch.testing.assert_close(inp_t.grad, grad_c, atol=BWD_ATOL, rtol=BWD_RTOL)


# ============================================================================
# Proj RMS
# ============================================================================


class TestNativeProjRms:
    """Tests for native_proj_rms."""

    @pytest.mark.parametrize("M,N,K", [(256, 20, 4096), (64, 8, 512)])
    def test_fwd_bwd_vs_torch_reference(self, M, N, K):
        _info()
        eps = 1e-6
        x_data = _rand(M, K)
        w_data = _rand(N, K)
        grad_proj = _rand(M, N)
        grad_r = _rand(M, 1)

        xf = x_data.clone().requires_grad_(True)
        wf = w_data.clone().requires_grad_(True)
        proj_f, r_f = native_proj_rms(xf, wf, eps)
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
    @pytest.mark.parametrize("M,N,K", [(256, 20, 4096), (64, 8, 512)])
    def test_fwd_bwd_vs_reference(self, M, N, K):
        """E2E: public fused fwd output and bwd grads must match the PyTorch reference."""
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
# Proj RMS + Compute H (fused)
# ============================================================================


class TestFusedProjRmsComputeH:
    @pytest.mark.parametrize("M,n,K", [(256, 4, 4096), (64, 2, 512), (128, 4, 2048)])
    def test_fwd_bwd_vs_reference(self, M, n, K):
        """E2E: public fused fwd output and bwd grads must match the PyTorch reference."""
        from megatron.core.fusions.fused_mhc_kernels import fused_proj_rms_compute_h

        _info()
        N = n * n + 2 * n
        eps = 1e-6
        x_data = _rand(M, K)
        w_data = _rand(N, K)
        ap_data = _rand(1)
        apo_data = _rand(1)
        ar_data = _rand(1)
        bias_data = _rand(N)
        grad_y = _rand(M, N)
        grad_h_pre = grad_y[:, :n]
        grad_h_post = grad_y[:, n : 2 * n]
        grad_h_res = grad_y[:, 2 * n :]
        grad_r = _rand(M, 1)

        def _make_inputs():
            return (
                x_data.clone().requires_grad_(True),
                w_data.clone().requires_grad_(True),
                ap_data.clone().requires_grad_(True),
                apo_data.clone().requires_grad_(True),
                ar_data.clone().requires_grad_(True),
                bias_data.clone().requires_grad_(True),
            )

        # -- fused path --
        xf, wf, apf, apof, arf, bf = _make_inputs()
        h_pre_f, h_post_f, h_res_f, r_f = fused_proj_rms_compute_h(
            xf, wf, apf, apof, arf, bf, n, eps
        )
        loss_f = (
            (h_pre_f * grad_h_pre).sum()
            + (h_post_f * grad_h_post).sum()
            + (h_res_f * grad_h_res).sum()
            + (r_f * grad_r).sum()
        )
        loss_f.backward()

        # -- reference path --
        xr, wr, apr, apor, arr, br = _make_inputs()
        h_pre_r, h_post_r, h_res_r, r_r = _ref_proj_rms_compute_h(
            xr, wr, apr, apor, arr, br, n, eps
        )
        loss_r = (
            (h_pre_r * grad_h_pre).sum()
            + (h_post_r * grad_h_post).sum()
            + (h_res_r * grad_h_res).sum()
            + (r_r * grad_r).sum()
        )
        loss_r.backward()

        torch.testing.assert_close(
            h_pre_f, h_pre_r, atol=FWD_ATOL, rtol=FWD_RTOL, msg="h_pre mismatch"
        )
        torch.testing.assert_close(
            h_post_f, h_post_r, atol=FWD_ATOL, rtol=FWD_RTOL, msg="h_post mismatch"
        )
        torch.testing.assert_close(
            h_res_f, h_res_r, atol=FWD_ATOL, rtol=FWD_RTOL, msg="h_res mismatch"
        )
        torch.testing.assert_close(r_f, r_r, atol=FWD_ATOL, rtol=FWD_RTOL, msg="r mismatch")
        torch.testing.assert_close(
            xf.grad, xr.grad, atol=BWD_ATOL, rtol=BWD_RTOL, msg="backward mismatch on x"
        )
        torch.testing.assert_close(
            wf.grad, wr.grad, atol=BWD_ATOL, rtol=BWD_RTOL, msg="backward mismatch on weight"
        )
        _assert_cosine_similar(
            apf.grad, apr.grad, COSINE_SIM_THRESH, msg="backward mismatch on alpha_pre"
        )
        _assert_cosine_similar(
            apof.grad, apor.grad, COSINE_SIM_THRESH, msg="backward mismatch on alpha_post"
        )
        _assert_cosine_similar(
            arf.grad, arr.grad, COSINE_SIM_THRESH, msg="backward mismatch on alpha_res"
        )
        _assert_cosine_similar(bf.grad, br.grad, COSINE_SIM_THRESH, msg="backward mismatch on bias")


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

        hs_data = _rand(s, b, n * C)
        w_data = _rand(n * n + 2 * n, n * C)
        layer_out_data = _rand(s, b, C)
        layer_bias_data = _rand(C)

        def _run_native_modules():
            hs = hs_data.clone().requires_grad_(True)
            w = w_data.clone().requires_grad_(True)

            x_2d = hs.reshape(s * b, n * C)
            proj, r = native_proj_rms(x_2d, w, eps)
            proj = proj.view(s, b, -1)
            r = r.view(s, b, 1)

            h = r * proj
            h_pre = h[..., :n].sigmoid()
            h_post = h[..., n : 2 * n].sigmoid() * 2
            h_res_logits = h[..., 2 * n :]
            h_res = native_sinkhorn(h_res_logits.view(s, b, n, n), sinkhorn_iters, eps)

            aggregated = native_h_aggregate(hs.view(s, b, n, C), h_pre)

            output = native_h_post_bda(
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
    """Full mHC pipeline using the public fused API."""

    def test_full_pipeline_fwd_bwd(self):
        from megatron.core.fusions.fused_mhc_kernels import (
            fused_h_aggregate,
            fused_h_post_bda,
            fused_proj_rms,
            fused_sinkhorn,
        )

        _info()
        s, b, n, C = 8, 4, 4, 1024
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
            return proj.detach(), r.detach(), output.detach(), aggregated.detach(), hs.grad.clone()

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
            return proj.detach(), r.detach(), output.detach(), aggregated.detach(), hs.grad.clone()

        proj_f, r_f, out_f, agg_f, grad_f = _run_fused()
        proj_r, r_r, out_r, agg_r, grad_r = _run_ref()

        torch.testing.assert_close(proj_f, proj_r, atol=FWD_ATOL, rtol=FWD_RTOL)
        torch.testing.assert_close(r_f, r_r, atol=FWD_ATOL, rtol=FWD_RTOL)
        torch.testing.assert_close(agg_f, agg_r, atol=FWD_ATOL, rtol=FWD_RTOL)
        torch.testing.assert_close(
            out_f, out_r, atol=FWD_ATOL, rtol=FWD_RTOL, msg="h_post_bda output mismatch"
        )
        _assert_cosine_similar(
            grad_f, grad_r, COSINE_SIM_THRESH, msg="hidden_states grad (E2E backward)"
        )

    def test_full_pipeline_fused_compute_h(self):
        """E2E: fused proj_rms_compute_h replaces separate proj_rms + compute_h."""
        from megatron.core.fusions.fused_mhc_kernels import (
            fused_h_aggregate,
            fused_h_post_bda,
            fused_proj_rms_compute_h,
            fused_sinkhorn,
        )

        _info()
        s, b, n, C = 8, 4, 4, 1024
        N = n * n + 2 * n
        eps = 1e-6
        sinkhorn_iters = 5

        hs_data = _rand(s, b, n * C)
        w_data = _rand(N, n * C)
        ap_data = _rand(1)
        apo_data = _rand(1)
        ar_data = _rand(1)
        bias_data = _rand(N)
        layer_out_data = _rand(s, b, C)
        layer_bias_data = _rand(C)

        def _run_fused_compute_h():
            hs = hs_data.clone().requires_grad_(True)
            w = w_data.clone().requires_grad_(True)
            ap = ap_data.clone().requires_grad_(True)
            apo = apo_data.clone().requires_grad_(True)
            ar = ar_data.clone().requires_grad_(True)
            bias_p = bias_data.clone().requires_grad_(True)

            x_2d = hs.reshape(s * b, n * C)
            h_pre, h_post, h_res_logits, _ = fused_proj_rms_compute_h(
                x_2d, w, ap, apo, ar, bias_p, n, eps
            )

            h_pre = h_pre.view(s, b, n)
            h_post = h_post.view(s, b, n)
            h_res_logits = h_res_logits.view(s, b, n, n)
            h_res = fused_sinkhorn(h_res_logits, sinkhorn_iters, eps)

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
            ap = ap_data.clone().requires_grad_(True)
            apo = apo_data.clone().requires_grad_(True)
            ar = ar_data.clone().requires_grad_(True)
            bias_p = bias_data.clone().requires_grad_(True)

            x_2d = hs.reshape(s * b, n * C)
            h_pre, h_post, h_res_logits, _ = _ref_proj_rms_compute_h(
                x_2d, w, ap, apo, ar, bias_p, n, eps
            )

            h_pre = h_pre.view(s, b, n)
            h_post = h_post.view(s, b, n)
            h_res_logits = h_res_logits.view(s, b, n, n)
            h_res = _ref_sinkhorn(h_res_logits, sinkhorn_iters, eps)

            aggregated = _ref_h_aggregate(hs.view(s, b, n, C), h_pre)

            output = _ref_h_post_bda(
                h_res, hs.view(s, b, n, C), h_post, layer_out_data, layer_bias_data
            )

            loss = output.sum() + aggregated.sum()
            loss.backward()
            return output.detach(), aggregated.detach(), hs.grad.clone()

        out_f, agg_f, grad_f = _run_fused_compute_h()
        out_r, agg_r, grad_r = _run_ref()

        torch.testing.assert_close(
            agg_f, agg_r, atol=FWD_ATOL, rtol=FWD_RTOL, msg="aggregated mismatch"
        )
        torch.testing.assert_close(
            out_f, out_r, atol=FWD_ATOL, rtol=FWD_RTOL, msg="h_post_bda output mismatch"
        )
        _assert_cosine_similar(
            grad_f,
            grad_r,
            COSINE_SIM_THRESH,
            msg="hidden_states grad (E2E backward, fused compute_h)",
        )


# ============================================================================
# fused_add_3 kernel tests
# ============================================================================


class TestFusedAdd3:
    """Tests for fused_add_3 (torch.compile backend, no cuTile dependency)."""

    def test_fused_add_3_forward_bf16(self):
        """fused_add_3 matches a + b + c for bf16 tensors."""
        from megatron.core.fusions.fused_mhc_kernels import fused_add_3

        a = torch.randn(128, 256, dtype=DTYPE, device=DEVICE)
        b = torch.randn(128, 256, dtype=DTYPE, device=DEVICE)
        c = torch.randn(128, 256, dtype=DTYPE, device=DEVICE)
        result = fused_add_3(a, b, c)
        expected = (a.float() + b.float() + c.float()).to(DTYPE)
        torch.testing.assert_close(result, expected, atol=FWD_ATOL, rtol=FWD_RTOL)

    def test_fused_add_3_forward_fp32(self):
        """fused_add_3 matches a + b + c for fp32 tensors."""
        from megatron.core.fusions.fused_mhc_kernels import fused_add_3

        a = torch.randn(128, 256, dtype=torch.float32, device=DEVICE)
        b = torch.randn(128, 256, dtype=torch.float32, device=DEVICE)
        c = torch.randn(128, 256, dtype=torch.float32, device=DEVICE)
        result = fused_add_3(a, b, c)
        expected = a + b + c
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_fused_add_3_large_tensor(self):
        """fused_add_3 handles large tensors."""
        from megatron.core.fusions.fused_mhc_kernels import fused_add_3

        a = torch.randn(8192, 4096, dtype=DTYPE, device=DEVICE)
        b = torch.randn(8192, 4096, dtype=DTYPE, device=DEVICE)
        c = torch.randn(8192, 4096, dtype=DTYPE, device=DEVICE)
        result = fused_add_3(a, b, c)
        expected = (a.float() + b.float() + c.float()).to(DTYPE)
        torch.testing.assert_close(result, expected, atol=FWD_ATOL, rtol=FWD_RTOL)

    def test_fused_add_3_gradient(self):
        """fused_add_3 produces correct gradients."""
        from megatron.core.fusions.fused_mhc_kernels import fused_add_3

        a = torch.randn(64, 128, dtype=torch.float32, device=DEVICE, requires_grad=True)
        b = torch.randn(64, 128, dtype=torch.float32, device=DEVICE, requires_grad=True)
        c = torch.randn(64, 128, dtype=torch.float32, device=DEVICE, requires_grad=True)
        result = fused_add_3(a, b, c)
        result.sum().backward()
        torch.testing.assert_close(a.grad, torch.ones_like(a))
        torch.testing.assert_close(b.grad, torch.ones_like(b))
        torch.testing.assert_close(c.grad, torch.ones_like(c))


# ============================================================================
# End-to-end pipeline with BroadcastTensorFused
# ============================================================================


class TestEndToEndNativeBroadcast:
    """Full mHC pipeline using native modules + BroadcastTensorFused.

    Same pipeline as TestEndToEndNative but hidden_states is split via
    BroadcastTensorFused so each consumer (proj_rms/compute_h, aggregate,
    h_post_bda) gets a distinct autograd graph node. Verifies gradient
    correctness versus the inline reference that uses the tensor directly.
    """

    def test_full_pipeline_fwd_bwd(self):
        from megatron.core.transformer.hyper_connection import (
            BroadcastTensorFused,
            native_fused_add_3,
        )

        _info()
        s, b, n, C = 2, 4, 4, 1024
        eps = 1e-6
        sinkhorn_iters = 5

        hs_data = _rand(s, b, n * C)
        w_data = _rand(n * n + 2 * n, n * C)
        layer_out_data = _rand(s, b, C)
        layer_bias_data = _rand(C)

        def _run_broadcast():
            hs = hs_data.clone().requires_grad_(True)
            w = w_data.clone().requires_grad_(True)

            # Split via BroadcastTensorFused
            hs_map, hs_agg, hs_res = BroadcastTensorFused.apply(hs, native_fused_add_3)

            x_2d = hs_map.reshape(s * b, n * C)
            proj, r = native_proj_rms(x_2d, w, eps)
            proj = proj.view(s, b, -1)
            r = r.view(s, b, 1)

            h = r * proj
            h_pre = h[..., :n].sigmoid()
            h_post = h[..., n : 2 * n].sigmoid() * 2
            h_res_logits = h[..., 2 * n :]
            h_res = native_sinkhorn(h_res_logits.view(s, b, n, n), sinkhorn_iters, eps)

            aggregated = native_h_aggregate(hs_agg.view(s, b, n, C), h_pre)

            output = native_h_post_bda(
                h_res, hs_res.view(s, b, n, C), h_post, layer_out_data, layer_bias_data
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

        out_b, agg_b, grad_b = _run_broadcast()
        out_r, agg_r, grad_r = _run_inline_ref()

        torch.testing.assert_close(
            agg_b, agg_r, atol=FWD_ATOL, rtol=FWD_RTOL, msg="aggregated mismatch (broadcast)"
        )
        torch.testing.assert_close(
            out_b, out_r, atol=FWD_ATOL, rtol=FWD_RTOL, msg="h_post_bda output mismatch (broadcast)"
        )
        _assert_cosine_similar(
            grad_b, grad_r, COSINE_SIM_THRESH, msg="hidden_states grad (E2E backward, broadcast)"
        )


class TestEndToEndFusedBroadcast:
    """Full mHC pipeline using the public fused API + BroadcastTensorFused."""

    def test_full_pipeline_fwd_bwd(self):
        from megatron.core.fusions.fused_mhc_kernels import (
            fused_add_3,
            fused_h_aggregate,
            fused_h_post_bda,
            fused_proj_rms,
            fused_sinkhorn,
        )
        from megatron.core.transformer.hyper_connection import BroadcastTensorFused

        _info()
        s, b, n, C = 8, 4, 4, 1024
        eps = 1e-6
        sinkhorn_iters = 5

        hs_data = _rand(s, b, n * C)
        w_data = _rand(n * n + 2 * n, n * C)
        layer_out_data = _rand(s, b, C)
        layer_bias_data = _rand(C)

        def _run_fused_broadcast():
            hs = hs_data.clone().requires_grad_(True)
            w = w_data.clone().requires_grad_(True)

            hs_map, hs_agg, hs_res = BroadcastTensorFused.apply(hs, fused_add_3)

            x_2d = hs_map.reshape(s * b, n * C)
            proj, r = fused_proj_rms(x_2d, w, eps)
            proj = proj.view(s, b, -1)
            r = r.view(s, b, 1)

            h = r * proj
            h_pre = h[..., :n].sigmoid()
            h_post = h[..., n : 2 * n].sigmoid() * 2
            h_res_logits = h[..., 2 * n :]
            h_res = fused_sinkhorn(h_res_logits.view(s, b, n, n), sinkhorn_iters, eps)

            aggregated = fused_h_aggregate(hs_agg.view(s, b, n, C), h_pre)

            output = fused_h_post_bda(
                h_res, hs_res.view(s, b, n, C), h_post, layer_out_data, layer_bias_data
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

        out_f, agg_f, grad_f = _run_fused_broadcast()
        out_r, agg_r, grad_r = _run_ref()

        torch.testing.assert_close(
            agg_f, agg_r, atol=FWD_ATOL, rtol=FWD_RTOL, msg="aggregated mismatch (fused broadcast)"
        )
        torch.testing.assert_close(
            out_f,
            out_r,
            atol=FWD_ATOL,
            rtol=FWD_RTOL,
            msg="h_post_bda output mismatch (fused broadcast)",
        )
        _assert_cosine_similar(
            grad_f,
            grad_r,
            COSINE_SIM_THRESH,
            msg="hidden_states grad (E2E backward, fused broadcast)",
        )

    def test_full_pipeline_fused_compute_h_broadcast(self):
        """E2E: fused proj_rms_compute_h + BroadcastTensorFused."""
        from megatron.core.fusions.fused_mhc_kernels import (
            fused_add_3,
            fused_h_aggregate,
            fused_h_post_bda,
            fused_proj_rms_compute_h,
            fused_sinkhorn,
        )
        from megatron.core.transformer.hyper_connection import BroadcastTensorFused

        _info()
        s, b, n, C = 8, 4, 4, 1024
        N = n * n + 2 * n
        eps = 1e-6
        sinkhorn_iters = 5

        hs_data = _rand(s, b, n * C)
        w_data = _rand(N, n * C)
        ap_data = _rand(1)
        apo_data = _rand(1)
        ar_data = _rand(1)
        bias_data = _rand(N)
        layer_out_data = _rand(s, b, C)
        layer_bias_data = _rand(C)

        def _run_fused_compute_h_broadcast():
            hs = hs_data.clone().requires_grad_(True)
            w = w_data.clone().requires_grad_(True)
            ap = ap_data.clone().requires_grad_(True)
            apo = apo_data.clone().requires_grad_(True)
            ar = ar_data.clone().requires_grad_(True)
            bias_p = bias_data.clone().requires_grad_(True)

            hs_map, hs_agg, hs_res = BroadcastTensorFused.apply(hs, fused_add_3)

            x_2d = hs_map.reshape(s * b, n * C)
            h_pre, h_post, h_res_logits, _ = fused_proj_rms_compute_h(
                x_2d, w, ap, apo, ar, bias_p, n, eps
            )

            h_pre = h_pre.view(s, b, n)
            h_post = h_post.view(s, b, n)
            h_res_logits = h_res_logits.view(s, b, n, n)
            h_res = fused_sinkhorn(h_res_logits, sinkhorn_iters, eps)

            aggregated = fused_h_aggregate(hs_agg.view(s, b, n, C), h_pre)

            output = fused_h_post_bda(
                h_res, hs_res.view(s, b, n, C), h_post, layer_out_data, layer_bias_data
            )

            loss = output.sum() + aggregated.sum()
            loss.backward()
            return output.detach(), aggregated.detach(), hs.grad.clone()

        def _run_ref():
            hs = hs_data.clone().requires_grad_(True)
            w = w_data.clone().requires_grad_(True)
            ap = ap_data.clone().requires_grad_(True)
            apo = apo_data.clone().requires_grad_(True)
            ar = ar_data.clone().requires_grad_(True)
            bias_p = bias_data.clone().requires_grad_(True)

            x_2d = hs.reshape(s * b, n * C)
            h_pre, h_post, h_res_logits, _ = _ref_proj_rms_compute_h(
                x_2d, w, ap, apo, ar, bias_p, n, eps
            )

            h_pre = h_pre.view(s, b, n)
            h_post = h_post.view(s, b, n)
            h_res_logits = h_res_logits.view(s, b, n, n)
            h_res = _ref_sinkhorn(h_res_logits, sinkhorn_iters, eps)

            aggregated = _ref_h_aggregate(hs.view(s, b, n, C), h_pre)

            output = _ref_h_post_bda(
                h_res, hs.view(s, b, n, C), h_post, layer_out_data, layer_bias_data
            )

            loss = output.sum() + aggregated.sum()
            loss.backward()
            return output.detach(), aggregated.detach(), hs.grad.clone()

        out_f, agg_f, grad_f = _run_fused_compute_h_broadcast()
        out_r, agg_r, grad_r = _run_ref()

        torch.testing.assert_close(
            agg_f,
            agg_r,
            atol=FWD_ATOL,
            rtol=FWD_RTOL,
            msg="aggregated mismatch (fused compute_h broadcast)",
        )
        torch.testing.assert_close(
            out_f,
            out_r,
            atol=FWD_ATOL,
            rtol=FWD_RTOL,
            msg="h_post_bda output mismatch (fused compute_h broadcast)",
        )
        _assert_cosine_similar(
            grad_f,
            grad_r,
            COSINE_SIM_THRESH,
            msg="hidden_states grad (E2E backward, fused compute_h broadcast)",
        )
