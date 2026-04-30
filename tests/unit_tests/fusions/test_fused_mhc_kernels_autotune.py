# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for cuTile mHC kernel autotuning correctness.

Each test clears the autotune cache, runs the fused kernel (triggering a fresh
autotune search), and verifies the output matches a pure-PyTorch reference.
This ensures every config in the search space that the autotuner might select
produces numerically correct results.

Usage:
    pytest tests/unit_tests/fusions/test_fused_mhc_kernels_autotune.py -s
"""

import math

import pytest
import torch
from torch import Tensor

from megatron.core.fusions.fused_mhc_kernels import (
    _CUTILE_EXPERIMENTAL_AVAILABLE,
    is_cutile_available,
)

_require_cutile_experimental = pytest.mark.skipif(
    not (is_cutile_available() and _CUTILE_EXPERIMENTAL_AVAILABLE),
    reason="cuTile + cuda.tile_experimental required for autotune tests",
)


@pytest.fixture(autouse=True)
def _skip_without_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


DTYPE = torch.bfloat16
DEVICE = "cuda"
FWD_ATOL, FWD_RTOL = 2e-2, 2e-2
BWD_ATOL, BWD_RTOL = 5e-2, 5e-2


def _rand(*shape, requires_grad=False):
    return (
        torch.empty(*shape, dtype=DTYPE, device=DEVICE)
        .uniform_(-0.1, 0.1)
        .requires_grad_(requires_grad)
    )


# ---------------------------------------------------------------------------
# Pure-PyTorch references
# ---------------------------------------------------------------------------


def _ref_sinkhorn(logits: Tensor, num_iters: int, eps: float = 1e-6) -> Tensor:
    row_max = logits.max(dim=-1, keepdim=True).values
    M = torch.exp(logits - row_max)
    for _ in range(num_iters):
        M = M / M.sum(dim=-1, keepdim=True).clamp(min=eps)
        M = M / M.sum(dim=-2, keepdim=True).clamp(min=eps)
    return M


def _ref_proj_rms(x: Tensor, weight: Tensor, eps: float = 1e-6):
    proj = torch.matmul(x, weight.t())
    norm = x.norm(dim=-1, keepdim=True)
    K = x.shape[-1]
    r = 1.0 / (norm / math.sqrt(K) + eps)
    return proj, r


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clear_autotune_caches():
    """Clear all cuTile mHC autotune caches to force a fresh search."""
    from megatron.core.fusions import fused_mhc_kernels as mod

    mod._sinkhorn_fwd_best_cfg.clear()
    mod._sinkhorn_bwd_best_cfg.clear()
    mod._proj_rms_fwd_best_cfg.clear()
    mod._proj_rms_bwd_best_cfg.clear()


# ============================================================================
# Sinkhorn autotune
# ============================================================================


class TestAutotunesSinkhorn:
    @_require_cutile_experimental
    @pytest.mark.parametrize("s,b,n,iters", [(2, 4, 4, 5), (4, 8, 4, 20)])
    def test_sinkhorn_fwd_autotune(self, s, b, n, iters):
        """Autotuned sinkhorn fwd must match PyTorch reference."""
        from megatron.core.fusions.fused_mhc_kernels import _cutile_sinkhorn_fwd

        _clear_autotune_caches()

        eps = 1e-6
        data = _rand(s, b, n, n)

        out_fused, _ = _cutile_sinkhorn_fwd(data.clone(), iters, eps)

        out_ref = _ref_sinkhorn(data.clone(), iters, eps)

        torch.testing.assert_close(out_fused, out_ref, atol=FWD_ATOL, rtol=FWD_RTOL)

    @_require_cutile_experimental
    @pytest.mark.parametrize("s,b,n,iters", [(2, 4, 4, 5), (4, 8, 4, 20)])
    def test_sinkhorn_bwd_autotune(self, s, b, n, iters):
        """Autotuned sinkhorn bwd grads must match PyTorch reference."""
        from megatron.core.fusions.fused_mhc_kernels import (
            _cutile_sinkhorn_bwd,
            _cutile_sinkhorn_fwd,
        )

        _clear_autotune_caches()

        eps = 1e-6
        data = _rand(s, b, n, n)
        grad_out = _rand(s, b, n, n)

        out_f, M_init = _cutile_sinkhorn_fwd(data.clone(), iters, eps)
        grad_f = _cutile_sinkhorn_bwd(grad_out, M_init, iters, eps)

        inp_r = data.clone().requires_grad_(True)
        out_r = _ref_sinkhorn(inp_r, iters, eps)
        out_r.backward(grad_out)
        grad_r = inp_r.grad.clone()

        torch.testing.assert_close(out_f, out_r, atol=FWD_ATOL, rtol=FWD_RTOL)
        torch.testing.assert_close(grad_f, grad_r, atol=BWD_ATOL, rtol=BWD_RTOL)

    @_require_cutile_experimental
    def test_sinkhorn_autotune_cache_populated(self):
        """After first call, autotune cache should contain the best config."""
        from megatron.core.fusions import fused_mhc_kernels as mod
        from megatron.core.fusions.fused_mhc_kernels import _cutile_sinkhorn_fwd

        _clear_autotune_caches()

        s, b, n, iters = 2, 4, 4, 5
        data = _rand(s, b, n, n)

        # First call triggers autotune
        _cutile_sinkhorn_fwd(data, iters)
        N_batch = s * b
        assert (N_batch, n, iters) in mod._sinkhorn_fwd_best_cfg

        # Second call should use cache (no error)
        _cutile_sinkhorn_fwd(data, iters)


# ============================================================================
# Proj RMS autotune
# ============================================================================


class TestAutotunesProjRms:
    @_require_cutile_experimental
    @pytest.mark.parametrize("M,N,K", [(256, 24, 4096), (64, 8, 512)])
    def test_proj_rms_fwd_autotune(self, M, N, K):
        """Autotuned proj_rms fwd must match PyTorch reference."""
        from megatron.core.fusions.fused_mhc_kernels import fused_proj_rms

        _clear_autotune_caches()

        eps = 1e-6
        x_data = _rand(M, K)
        w_data = _rand(N, K)

        xf = x_data.clone()
        wf = w_data.clone()
        proj_f, r_f = fused_proj_rms(xf, wf, eps)

        xr = x_data.clone()
        wr = w_data.clone()
        proj_r, r_r = _ref_proj_rms(xr, wr, eps)

        torch.testing.assert_close(proj_f, proj_r, atol=FWD_ATOL, rtol=FWD_RTOL)
        torch.testing.assert_close(r_f, r_r, atol=FWD_ATOL, rtol=FWD_RTOL)

    @_require_cutile_experimental
    @pytest.mark.parametrize("M,N,K", [(256, 24, 16384), (64, 8, 512)])
    def test_proj_rms_bwd_autotune(self, M, N, K):
        """Autotuned proj_rms bwd grads must match PyTorch reference."""
        from megatron.core.fusions.fused_mhc_kernels import fused_proj_rms

        _clear_autotune_caches()

        eps = 1e-6
        x_data = _rand(M, K)
        w_data = _rand(N, K)
        grad_proj = _rand(M, N)
        grad_r = _rand(M, 1)

        xf = x_data.clone().requires_grad_(True)
        wf = w_data.clone().requires_grad_(True)
        proj_f, r_f = fused_proj_rms(xf, wf, eps)
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

    @_require_cutile_experimental
    def test_proj_rms_autotune_cache_populated(self):
        """After first call, autotune caches should contain the best configs."""
        from megatron.core.fusions import fused_mhc_kernels as mod
        from megatron.core.fusions.fused_mhc_kernels import fused_proj_rms

        _clear_autotune_caches()

        M, N, K = 256, 24, 16384
        eps = 1e-6
        x = _rand(M, K, requires_grad=True)
        w = _rand(N, K, requires_grad=True)

        proj, r = fused_proj_rms(x, w, eps)
        assert (M, N, K) in mod._proj_rms_fwd_best_cfg, "fwd cache not populated"

        (proj.sum() + r.sum()).backward()
        # bwd autotune only triggers for K >= 8192
        assert (M, N, K) in mod._proj_rms_bwd_best_cfg, "bwd cache not populated"
