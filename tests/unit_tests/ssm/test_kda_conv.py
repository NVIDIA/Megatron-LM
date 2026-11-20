# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The row-walking short-conv kernels match FLA's ``causal_conv1d_fwd`` / ``causal_conv1d_bwd``.

Covers packed sequences with per-sequence initial states (what FLA's context parallelism passes),
one sequence with and without an initial state, SiLU and no activation, with and without bias,
and channel / row counts that are not multiples of the tiles. FLA rounds its recomputed
pre-activation to BF16 before SiLU's derivative and this kernel keeps it in FP32, so the
comparison allows BF16-level differences.
"""

import pytest
import torch

fla_ops = pytest.importorskip("fla.modules.conv.triton.ops")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

from megatron.core.ssm.gated_delta_net import kda_conv  # noqa: E402

_CASES = [
    # (cu_seqlens or None, D, use_h0, activation, bias)
    ([0, 300], 200, True, "silu", False),
    ([0, 300], 200, False, "silu", True),
    ([0, 64, 65, 190, 512], 384, True, "silu", True),
    ([0, 64, 65, 190, 512], 384, True, None, False),
    (None, 136, False, "silu", False),
    ([0, 1000, 1003, 4096], 1024, True, "silu", False),
]


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


@pytest.mark.parametrize(("cu", "D", "use_h0", "activation", "bias"), _CASES)
def test_one_pass_bwd_matches_fla(cu, D, use_h0, activation, bias):
    torch.manual_seed(0)
    T = cu[-1] if cu is not None else 777
    x = torch.randn(1, T, D, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(1, T, D, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(D, 4, device="cuda", dtype=torch.float32) * 0.5
    b = torch.randn(D, device="cuda", dtype=torch.float32) * 0.1 if bias else None
    cu_t = torch.tensor(cu, device="cuda", dtype=torch.int32) if cu is not None else None
    n = len(cu) - 1 if cu is not None else 1
    h0 = torch.randn(n, D, 4, device="cuda", dtype=torch.bfloat16) if use_h0 else None
    kwargs = dict(
        x=x,
        dy=dy,
        dht=None,
        weight=w,
        bias=b,
        residual=None,
        initial_state=h0,
        activation=activation,
        cu_seqlens=cu_t,
    )
    ref = kda_conv._FLA_CAUSAL_CONV1D_BWD(**kwargs)
    got = kda_conv.causal_conv1d_bwd(**kwargs)
    names = ("dx", "dw", "db", "dr", "dh0")
    for name, r, g in zip(names, ref, got):
        if r is None:
            assert g is None, name
            continue
        assert g.shape == r.shape and g.dtype == r.dtype, name
        assert _rel(g, r) < 2e-2, f"{name}: relative error {_rel(g, r):.3e}"


def test_unsupported_inputs_keep_fla():
    x = torch.randn(2, 64, 32, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(32, 4, device="cuda", dtype=torch.float32)
    assert not kda_conv._supported(x, x, None, w, None, "silu", None)
    w3 = torch.randn(32, 3, device="cuda", dtype=torch.float32)
    x1 = x[:1]
    assert not kda_conv._supported(x1, x1, None, w3, None, "silu", None)
    assert kda_conv._supported(x1, x1, None, w, None, "silu", None)


@pytest.mark.parametrize(("cu", "D", "use_h0", "activation", "bias"), _CASES)
def test_row_walking_fwd_matches_fla(cu, D, use_h0, activation, bias):
    torch.manual_seed(1)
    T = cu[-1] if cu is not None else 777
    x = torch.randn(1, T, D, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(D, 4, device="cuda", dtype=torch.float32) * 0.5
    b = torch.randn(D, device="cuda", dtype=torch.float32) * 0.1 if bias else None
    cu_t = torch.tensor(cu, device="cuda", dtype=torch.int32) if cu is not None else None
    n = len(cu) - 1 if cu is not None else 1
    h0 = torch.randn(n, D, 4, device="cuda", dtype=torch.bfloat16) if use_h0 else None
    kwargs = dict(
        x=x,
        weight=w,
        bias=b,
        residual=None,
        initial_state=h0,
        activation=activation,
        cu_seqlens=cu_t,
    )
    ref, _ = kda_conv._FLA_CAUSAL_CONV1D_FWD(**kwargs)
    got, final = kda_conv.causal_conv1d_fwd(**kwargs)
    assert final is None and got.shape == ref.shape and got.dtype == ref.dtype
    # Same FP32 products in the same order; at most a BF16 rounding step apart.
    assert _rel(got, ref) < 1e-2


def test_installed_kernels_match_fla_through_autograd(monkeypatch):
    from fla.modules.conv import causal_conv1d

    torch.manual_seed(2)
    cu = torch.tensor([0, 100, 333, 1024], device="cuda", dtype=torch.int32)
    x = torch.randn(1, 1024, 512, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(512, 4, device="cuda", dtype=torch.float32, requires_grad=True)
    grad = torch.randn_like(x)

    def run():
        xi = x.detach().clone().requires_grad_(True)
        wi = w.detach().clone().requires_grad_(True)
        y, _ = causal_conv1d(x=xi, weight=wi, bias=None, activation="silu", cu_seqlens=cu)
        y.backward(grad)
        return y.detach(), xi.grad, wi.grad

    ref = run()
    monkeypatch.setattr(fla_ops, "causal_conv1d_fwd", kda_conv.causal_conv1d_fwd)
    monkeypatch.setattr(fla_ops, "causal_conv1d_bwd", kda_conv.causal_conv1d_bwd)
    got = run()
    for name, r, g in zip(("y", "dx", "dw"), ref, got):
        assert _rel(g, r) < 2e-2, f"{name}: relative error {_rel(g, r):.3e}"
