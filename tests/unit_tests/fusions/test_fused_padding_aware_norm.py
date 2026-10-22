# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
import pytest
import torch

from megatron.core.fusions import fused_padding_aware_norm as pan

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and pan.HAVE_TRITON), reason="needs a GPU and Triton"
)


def _ref_rmsnorm(x, w, eps, zero_centered):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if w is not None:
        wf = w.float() + (1.0 if zero_centered else 0.0)
        y = y * wf
    return y.to(x.dtype)


@pytest.mark.parametrize("n_cols", [512, 4096, 5120])  # pow2 and non-pow2 widths
@pytest.mark.parametrize("weighted,zero_centered", [(True, False), (True, True), (False, False)])
@pytest.mark.parametrize("m_valid", [128, 2688, 4096])
def test_matches_reference_on_live_rows_and_zeros_the_tail(
    n_cols, weighted, zero_centered, m_valid
):
    torch.manual_seed(0)
    rows = 4096
    x = (torch.randn(rows, 1, n_cols, device="cuda") * 2).to(torch.bfloat16)
    w = (
        (torch.randn(n_cols, device="cuda") * 0.1 + (0.0 if zero_centered else 1.0))
        if weighted
        else None
    )
    bound = torch.tensor([m_valid], device="cuda", dtype=torch.int64)
    xa = x.clone().requires_grad_()
    wa = w.clone().requires_grad_() if weighted else None
    xr = x.clone().requires_grad_()
    wr = w.clone().requires_grad_() if weighted else None
    ya = pan.padding_aware_rms_norm(xa, wa, 1e-6, bound, zero_centered)
    yr = _ref_rmsnorm(xr, wr, 1e-6, zero_centered)
    torch.testing.assert_close(ya[:m_valid].float(), yr[:m_valid].float(), rtol=1e-2, atol=1e-2)
    assert not ya[m_valid:].any()
    g = torch.randn_like(yr)
    g[m_valid:] = 0
    ya.backward(g)
    yr.backward(g)
    torch.testing.assert_close(
        xa.grad[:m_valid].float(), xr.grad[:m_valid].float(), rtol=2e-2, atol=2e-2
    )
    assert not xa.grad[m_valid:].any()
    if weighted:
        torch.testing.assert_close(wa.grad, wr.grad, rtol=2e-2, atol=2e-2 * rows**0.5)


def test_cuda_graph_replay_follows_the_device_bound():
    torch.manual_seed(0)
    rows, n_cols = 1024, 1024
    x = torch.randn(rows, 1, n_cols, device="cuda").to(torch.bfloat16)
    w = torch.randn(n_cols, device="cuda")
    bound = torch.tensor([512], device="cuda", dtype=torch.int64)
    xs = x.clone().requires_grad_()
    ws = w.clone().requires_grad_()
    g = torch.ones(rows, 1, n_cols, device="cuda", dtype=torch.bfloat16)

    def step():
        y = pan.padding_aware_rms_norm(xs, ws, 1e-6, bound)
        dx, dw = torch.autograd.grad(y, (xs, ws), g)
        return y, dx, dw

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y, dx, dw = step()
    for m in (512, 256, 1024, 128):
        bound.fill_(m)
        graph.replay()
        torch.cuda.synchronize()
        yr = _ref_rmsnorm(x, w, 1e-6, False)
        torch.testing.assert_close(y[:m].float(), yr[:m].float(), rtol=1e-2, atol=1e-2)
        assert not y[m:].any() and not dx[m:].any()


def test_te_rmsnorm_wrapper_skips_the_tail_under_a_bound(monkeypatch):
    monkeypatch.setenv("MCORE_PADDING_AWARE_SKIP", "")  # the row-bounded norm is off by default
    te = pytest.importorskip("transformer_engine.pytorch")
    from megatron.core.extensions.transformer_engine import PaddingAwareTERMSNorm
    from megatron.core.transformer import padding_aware_gemm as pag

    torch.manual_seed(0)
    rows, n_cols, m = 2048, 1024, 1280
    ref = te.RMSNorm(n_cols, eps=1e-6).cuda()
    pa = PaddingAwareTERMSNorm(n_cols, eps=1e-6).cuda()
    with torch.no_grad():
        ref.weight.normal_(1.0, 0.1)
        pa.weight.copy_(ref.weight)
    x = torch.randn(rows, 1, n_cols, device="cuda").to(torch.bfloat16)
    bound = torch.tensor([m], device="cuda", dtype=torch.int64)
    xa, xr = x.clone().requires_grad_(), x.clone().requires_grad_()
    with pag.padding_aware_live_rows(bound, rows):
        ya = pa(xa)
    yr = ref(xr)
    torch.testing.assert_close(ya[:m].float(), yr[:m].float(), rtol=1e-2, atol=1e-2)
    assert not ya[m:].any()
    g = torch.randn_like(yr)
    g[m:] = 0
    ya.backward(g)
    yr.backward(g)
    torch.testing.assert_close(xa.grad[:m].float(), xr.grad[:m].float(), rtol=2e-2, atol=2e-2)
    assert not xa.grad[m:].any()
    torch.testing.assert_close(pa.weight.grad, ref.weight.grad, rtol=2e-2, atol=2e-2 * rows**0.5)
    # Without a bound the TE kernels run and every row is normalised.
    yb = pa(x.clone())
    assert yb[m:].any()


def test_padding_aware_add3_zeroes_the_tail():
    from megatron.core.fusions import fused_padding_aware_elementwise as pae

    torch.manual_seed(0)
    rows, n_cols, m = 4096, 4 * 256, 2688
    a, b, c = (torch.randn(rows, 1, n_cols, device="cuda").to(torch.bfloat16) for _ in range(3))
    bound = torch.tensor([m], device="cuda", dtype=torch.int64)
    out = pae.padding_aware_add3(
        a.view(rows, -1), b.view(rows, -1), c.view(rows, -1), bound
    ).view_as(a)
    ref = (a.float() + b.float() + c.float()).to(torch.bfloat16)
    torch.testing.assert_close(out[:m], ref[:m])
    assert not out[m:].any()


def test_padding_aware_row_argmax_matches_torch_on_live_rows():
    from megatron.core.fusions import fused_padding_aware_elementwise as pae

    torch.manual_seed(0)
    rows, vocab, m = 1024, 30016, 640
    logits = torch.randn(rows, 1, vocab, device="cuda").to(torch.bfloat16)
    bound = torch.tensor([m], device="cuda", dtype=torch.int64)
    out = pae.padding_aware_row_argmax(logits, bound)
    ref = torch.argmax(logits, dim=-1)
    assert torch.equal(out[:m], ref[:m])
    assert torch.all(out[m:] == -1)
