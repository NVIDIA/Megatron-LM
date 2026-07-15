# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the fused gated-RMSNorm path in GatedDeltaNet.

Two independent concerns are covered:

1. ``_gated_norm_rmsnorm_compatible`` gating logic -- must enable the fused
   kernel only for a plain RMSNorm output norm and must reject the default
   ``IdentityOp`` and any LayerNorm-based norm. Pure Python (the rejection
   cases run without fla; the acceptance case is skipped without fla).
2. Numerical parity of ``fla.rms_norm_gated`` against the reference two-step
   ``rmsnorm(x, weight) * silu(gate)`` for both forward and backward, across
   dtypes and with/without zero-centered gamma. Requires fla + CUDA.
"""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.ssm.gated_delta_net import _gated_norm_rmsnorm_compatible

try:
    from fla.modules.fused_norm_gate import rms_norm_gated

    HAVE_FUSED_GATED_NORM = True
except ImportError:
    rms_norm_gated = None
    HAVE_FUSED_GATED_NORM = False


class _Cfg:
    def __init__(self, normalization="RMSNorm"):
        self.normalization = normalization


class _RMSNormLike(torch.nn.Module):
    """RMSNorm exposes ``weight`` and no ``bias``."""

    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = 1e-5


class _LayerNormLike(torch.nn.Module):
    """LayerNorm exposes both ``weight`` and ``bias``."""

    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.bias = torch.nn.Parameter(torch.zeros(dim))
        self.eps = 1e-5


class _IdentityLike(torch.nn.Module):
    """The default out_norm (IdentityOp) has neither weight nor bias."""


# --------------------------------------------------------------------------- #
# 1. Gating predicate
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not HAVE_FUSED_GATED_NORM, reason="predicate returns False without fla")
def test_predicate_accepts_rmsnorm():
    assert _gated_norm_rmsnorm_compatible(_Cfg("RMSNorm"), _RMSNormLike(8)) is True


def test_predicate_rejects_identity():
    # default output norm -> no weight -> must never fuse (would AttributeError).
    assert _gated_norm_rmsnorm_compatible(_Cfg("RMSNorm"), _IdentityLike()) is False


def test_predicate_rejects_layernorm():
    # LayerNorm carries a bias / mean-subtraction the fused RMSNorm would drop.
    assert _gated_norm_rmsnorm_compatible(_Cfg("RMSNorm"), _LayerNormLike(8)) is False


def test_predicate_rejects_non_rmsnorm_config():
    # even with a weight-bearing norm, a non-RMSNorm normalization must not fuse.
    assert _gated_norm_rmsnorm_compatible(_Cfg("LayerNorm"), _RMSNormLike(8)) is False


def test_predicate_false_without_fla(monkeypatch):
    import megatron.core.ssm.gated_delta_net as gdn

    monkeypatch.setattr(gdn, "rms_norm_gated", None)
    assert gdn._gated_norm_rmsnorm_compatible(_Cfg("RMSNorm"), _RMSNormLike(8)) is False


# --------------------------------------------------------------------------- #
# 2. Numerical parity (forward + backward)
# --------------------------------------------------------------------------- #


def _ref_gated_norm(x, gate, weight, eps, zero_centered):
    """Reference: rmsnorm(x, gamma) * silu(gate), matching _apply_gated_norm_ref."""
    x_dtype = x.dtype
    xf = x.float()
    gamma = (weight.float() + 1.0) if zero_centered else weight.float()
    rms = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    y = rms * gamma
    y = y * F.silu(gate.float())
    return y.to(x_dtype)


@pytest.mark.skipif(
    not (HAVE_FUSED_GATED_NORM and torch.cuda.is_available()),
    reason="fused parity requires fla + CUDA",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("zero_centered", [False, True])
def test_fused_matches_reference_fwd_bwd(dtype, zero_centered):
    torch.manual_seed(0)
    T, D, eps = 512, 128, 1e-5
    # tolerances: fp32 tight; bf16/fp16 within the ~2.5e-4 rel-dev budget observed.
    ftol = 1e-5 if dtype == torch.float32 else 3e-3
    gtol = 1e-4 if dtype == torch.float32 else 1e-2

    x = torch.randn(T, D, device="cuda", dtype=dtype)
    gate = torch.randn(T, D, device="cuda", dtype=dtype)
    weight = torch.randn(D, device="cuda", dtype=dtype) * 0.1

    def run(fused):
        xr = x.clone().requires_grad_(True)
        gr = gate.clone().requires_grad_(True)
        wr = weight.clone().requires_grad_(True)
        if fused:
            w = wr + 1.0 if zero_centered else wr
            y = rms_norm_gated(xr, gr, w, None, activation="swish", eps=eps)
        else:
            y = _ref_gated_norm(xr, gr, wr, eps, zero_centered)
        y.float().pow(2).sum().backward()
        return y.float(), xr.grad.float(), gr.grad.float(), wr.grad.float()

    yf, xgf, ggf, wgf = run(True)
    yr, xgr, ggr, wgr = run(False)

    def rel(a, b):
        return (a - b).abs().max().item() / (b.abs().max().item() + 1e-12)

    assert rel(yf, yr) < ftol, f"forward rel dev {rel(yf, yr):.2e}"
    assert rel(xgf, xgr) < gtol, f"grad_x rel dev {rel(xgf, xgr):.2e}"
    assert rel(ggf, ggr) < gtol, f"grad_gate rel dev {rel(ggf, ggr):.2e}"
    assert rel(wgf, wgr) < gtol, f"grad_weight rel dev {rel(wgf, wgr):.2e}"
