# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused Triton gated-residual kernels vs an fp64 reference and vs the torch path.

* kernel level: the forward kernels' intermediates (rstd, per-stream partials,
  down / u / write gate) and the write-back Function against fp64 math;
* module level: ``GatedResidualModule`` forward + write-back + backward with
  ``deterministic_mode=False`` (fused) — every output and gradient must be at
  least as close to fp64 as the torch path's (within 1.5x);
* path selection on ``deterministic_mode``, determinism of the deterministic
  path, bitwise-reproducible fused forward, CUDA-graph capture/replay.
"""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import gated_residual_triton as grt
from megatron.core.transformer.gated_residual import GatedResidualModule
from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

STREAMS = 4
EPS = 1e-6


def _cfg(hidden, lowrank, deterministic=False, streams=STREAMS):
    return TransformerConfig(
        num_layers=1,
        hidden_size=hidden,
        num_attention_heads=8,
        use_cpu_initialization=True,
        is_hybrid_model=True,
        enable_mhc_connections=True,
        mhc_num_residual_streams=streams,
        mhc_connection_variant="gated_residual",
        hc_lowrank=lowrank,
        layernorm_zero_centered_gamma=True,
        layernorm_epsilon=EPS,
        deterministic_mode=deterministic,
    )


def _module(
    hidden,
    lowrank,
    use_combine=True,
    params_dtype=torch.bfloat16,
    deterministic=False,
    streams=STREAMS,
):
    torch.manual_seed(1234)
    mod = GatedResidualModule(
        _cfg(hidden, lowrank, deterministic, streams), layer_number=1, use_combine=use_combine
    )
    with torch.no_grad():
        mod.hc_norm.weight.normal_(0.0, 0.1)  # non-trivial gamma
    mod = mod.cuda()
    if params_dtype != torch.float32:
        convert_module_to_dtype_except_fp32_marked(mod, params_dtype)
    return mod


def _ref64(h, gamma, wd, wu, wi, n):
    """fp64 gated-residual math with no intermediate rounding."""
    h, gamma, wd = h.double(), gamma.double(), wd.double()
    M, nC = h.shape
    C = nC // n
    x = h.view(M, n, C)
    rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    xn = (x * rstd).reshape(M, nC) * (1.0 + gamma)
    down = xn @ wd.t()
    u = F.silu(down / n)
    mixed = None
    if wu is not None:
        G = torch.sigmoid(u @ wu.double().t())
        mixed = (G.view(M, n, C) * xn.view(M, n, C)).mean(1)
    gate = None
    if wi is not None:
        gate = 2.0 * torch.sigmoid(xn @ wi.double().t() / n)
    return dict(rstd=rstd.view(M, n), xn=xn, down=down, u=u, mixed=mixed, gate=gate)


def _rel(a, ref):
    a, ref = a.double(), ref.double()
    return ((a - ref).norm() / ref.norm().clamp_min(1e-30)).item()


def _run_module(mod, h0, y, dmixed, dout, fused):
    """One sublayer call (gates + write-back) and its backward; returns outputs and grads."""
    mod.use_fused_kernels = fused
    for p in mod.parameters():
        p.grad = None
    h = h0.detach().clone().requires_grad_(True)
    yy = y.detach().clone().requires_grad_(True)
    res = {}
    if mod.use_combine:
        mixed, h_res, g_write, residual = mod(h)
        assert h_res is None and residual is h
        out = mod.fused_h_res_h_post_bda(None, residual, g_write, (yy, None), 0.0, True, True)
        torch.autograd.backward([mixed, out], [dmixed, dout])
        res.update(g_write=g_write.detach(), out=out.detach(), dy=yy.grad)
    else:
        mixed = mod(h)
        mixed.backward(dmixed)
    res.update(mixed=mixed.detach(), dh=h.grad)
    for name, p in mod.named_parameters():
        res["d_" + name] = p.grad.detach().clone()
    return res


def _ref_module64(mod, h0, y, dmixed, dout):
    """fp64 autograd of the same sublayer call using the module's parameters."""
    params = {name: p.detach().double().requires_grad_(True) for name, p in mod.named_parameters()}
    h = h0.detach().double().requires_grad_(True)
    yy = y.detach().double().requires_grad_(True)
    n = mod.n
    shape = h.shape
    h2 = h.reshape(-1, shape[-1])
    r = _ref64(
        h2,
        params["hc_norm.weight"],
        params["input_mix_weight_down.weight"],
        params["input_mix_weight_up.weight"],
        params.get("block_inject_weight.weight"),
        n,
    )
    mixed = r["mixed"].view(*shape[:-1], -1)
    out_t, grads_t = [mixed], [dmixed.double()]
    res = {}
    if mod.use_combine:
        gate = r["gate"].view(*shape[:-1], n)
        C = shape[-1] // n
        out = h + (gate.unsqueeze(-1) * yy.unsqueeze(-2)).reshape(*shape[:-1], n * C)
        out_t.append(out)
        grads_t.append(dout.double())
        res.update(g_write=gate.detach(), out=out.detach())
    torch.autograd.backward(out_t, grads_t)
    res.update(mixed=mixed.detach(), dh=h.grad)
    if mod.use_combine:
        res["dy"] = yy.grad
    for name, p in params.items():
        res["d_" + name] = p.grad
    return res


class TestGatedResidualTritonKernels:
    """Kernel level: forward intermediates and the write-back Function vs fp64."""

    @pytest.mark.parametrize("M", [1, 37, 4096])
    @pytest.mark.parametrize("C,R", [(2560, 320), (1024, 64)])
    @pytest.mark.parametrize("h_dtype", [torch.bfloat16, torch.float32])
    def test_forward_kernels_vs_fp64(self, M, C, R, h_dtype):
        torch.manual_seed(0)
        n = STREAMS
        h = torch.randn(M, n * C, device="cuda").to(h_dtype)
        gamma = 0.1 * torch.randn(n * C, device="cuda")
        wd = (torch.randn(R, n * C, device="cuda") * 0.02).bfloat16()
        wi = torch.randn(n, n * C, device="cuda") * 0.02
        rstd, xnb, p, z, down, u, gate = grt._launch_gates_fwd(h, gamma, wd, wi, EPS, n, C, True)
        ref = _ref64(h, gamma, wd, None, wi, n)
        assert _rel(rstd, ref["rstd"]) < 1e-6
        # x̃ may round to a neighbouring bf16 value (fp32 vs fp64 rounding boundary).
        assert _rel(xnb, ref["xn"]) < 4e-3
        # P is the bf16-input projection with fp32 accumulation: compare with fp64 on the
        # kernel's own bf16 x̃.
        p_ref = torch.einsum("mjc,rjc->jmr", xnb.double().view(M, n, C), wd.double().view(R, n, C))
        assert _rel(p, p_ref) < 1e-5
        assert _rel(down, ref["down"]) < 1e-2
        u_ref = F.silu(down.double() / n)
        assert _rel(u, u_ref) < 4e-3
        # The write-gate partials follow torch's TF32 policy for fp32 x fp32 products.
        z_tol = 2e-3 if torch.backends.cuda.matmul.allow_tf32 else 1e-5
        z_ref = torch.einsum("mjc,ijc->mji", ref["xn"].view(M, n, C), wi.double().view(n, n, C))
        assert _rel(z[:, :, :n], z_ref) < z_tol
        assert _rel(gate, ref["gate"]) < 4e-3

    @pytest.mark.parametrize("M", [1, 37, 4096])
    @pytest.mark.parametrize("with_bias", [False, True])
    @pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float32])
    def test_write_back_vs_fp64(self, M, with_bias, res_dtype):
        torch.manual_seed(1)
        n, C = STREAMS, 1024
        res = torch.randn(M, n * C, device="cuda").to(res_dtype).requires_grad_(True)
        gate = (torch.rand(M, n, device="cuda") * 2).bfloat16().requires_grad_(True)
        x = torch.randn(M, C, device="cuda").bfloat16().requires_grad_(True)
        bias = torch.randn(C, device="cuda").bfloat16().requires_grad_(True) if with_bias else None
        out = grt.fused_gated_residual_write_back(res, gate, x, bias)
        go = torch.randn_like(out)
        out.backward(go)

        r64, g64, x64 = (t.detach().double().requires_grad_(True) for t in (res, gate, x))
        b64 = bias.detach().double().requires_grad_(True) if with_bias else None
        xe = x64 + b64 if with_bias else x64
        ref = r64 + (g64.unsqueeze(-1) * xe.unsqueeze(1)).reshape(M, n * C)
        ref.backward(go.double())
        assert out.dtype == res_dtype
        assert _rel(out, ref) < 4e-3
        assert torch.equal(res.grad, go.to(res_dtype))
        assert _rel(gate.grad, g64.grad) < 4e-3
        assert _rel(x.grad, x64.grad) < 4e-3
        if with_bias:
            assert _rel(bias.grad, b64.grad) < 4e-3


class TestGatedResidualTritonModule:
    """Module level: fused vs torch path, both measured against fp64."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _check(mod, M, h_dtype, seed=0):
        torch.manual_seed(seed)
        n, C = mod.n, mod.hidden_size
        h0 = torch.randn(M, 1, n * C, device="cuda").to(h_dtype)
        y = torch.randn(M, 1, C, device="cuda").bfloat16()
        dmixed = torch.randn(M, 1, C, device="cuda").bfloat16()
        dout = torch.randn(M, 1, n * C, device="cuda").to(h_dtype)
        fused = _run_module(mod, h0, y, dmixed, dout, fused=True)
        torch_path = _run_module(mod, h0, y, dmixed, dout, fused=False)
        ref = _ref_module64(mod, h0, y, dmixed, dout)
        report = {}
        for key, r in ref.items():
            e_f, e_t = _rel(fused[key], r), _rel(torch_path[key], r)
            report[key] = (e_f, e_t)
            assert fused[key].dtype == torch_path[key].dtype, key
            assert e_f <= 1.5 * e_t + 1e-6, f"{key}: fused {e_f:.3e} vs torch {e_t:.3e}"
        return report

    @pytest.mark.parametrize("M", [1, 37, 4096])
    @pytest.mark.parametrize("C,R", [(2560, 320), (1024, 64)])
    @pytest.mark.parametrize("use_combine", [True, False])
    def test_fused_vs_torch_vs_fp64(self, M, C, R, use_combine):
        mod = _module(C, R, use_combine=use_combine)
        self._check(mod, M, torch.bfloat16)

    def test_proxy_shape_32k_rows(self):
        """Rank-local size of the 64K CP2 proxy run."""
        mod = _module(2560, 320)
        report = self._check(mod, 32768, torch.bfloat16)
        print({k: f"{a:.2e}/{b:.2e}" for k, (a, b) in report.items()})

    @pytest.mark.parametrize("M", [37, 4096])
    def test_fp32_residual_stream(self, M):
        mod = _module(1024, 64)
        self._check(mod, M, torch.float32)

    @pytest.mark.parametrize("streams", [1, 2, 3])
    def test_fewer_streams(self, streams):
        mod = _module(1024, 64, streams=streams)
        assert mod.use_fused_kernels
        self._check(mod, 37, torch.bfloat16)

    def test_more_streams_use_torch_path(self):
        mod = _module(512, 64, streams=grt.MAX_FUSED_STREAMS * 2)
        assert not mod.use_fused_kernels

    def test_fp32_params_use_torch_gates(self, monkeypatch):
        """fp32 gate weights keep the torch gates (identical to the reference)."""
        calls = []
        orig = grt.fused_gated_residual_gates
        monkeypatch.setattr(
            grt, "fused_gated_residual_gates", lambda *a, **k: calls.append(1) or orig(*a, **k)
        )
        mod = _module(1024, 64, params_dtype=torch.float32)
        h = torch.randn(37, 1, STREAMS * 1024, device="cuda")
        mod(h)
        assert calls == []

    def test_path_selection(self, monkeypatch):
        calls = []
        orig_g, orig_w = grt.fused_gated_residual_gates, grt.fused_gated_residual_write_back
        monkeypatch.setattr(
            grt, "fused_gated_residual_gates", lambda *a, **k: calls.append("g") or orig_g(*a, **k)
        )
        monkeypatch.setattr(
            grt,
            "fused_gated_residual_write_back",
            lambda *a, **k: calls.append("w") or orig_w(*a, **k),
        )
        for deterministic, expected in ((False, ["g", "w"]), (True, [])):
            mod = _module(1024, 64, deterministic=deterministic)
            assert mod.use_fused_kernels is (not deterministic)
            calls.clear()
            h = torch.randn(37, 1, STREAMS * 1024, device="cuda").bfloat16()
            y = torch.randn(37, 1, 1024, device="cuda").bfloat16()
            _, _, g, res = mod(h)
            mod.fused_h_res_h_post_bda(None, res, g, (y, None), 0.0, True, True)
            assert calls == expected

    @pytest.mark.parametrize("deterministic", [True, False])
    def test_bitwise_reproducible(self, deterministic):
        """deterministic_mode=True (torch path) is required to be bitwise reproducible.

        The fused path has no atomics either; its forward must be reproducible because the
        mHC recompute path re-runs it, and the backward is checked too to document it.
        """
        mod = _module(2560, 320, deterministic=deterministic)
        torch.manual_seed(3)
        M = 4096
        h0 = torch.randn(M, 1, STREAMS * 2560, device="cuda").bfloat16()
        y = torch.randn(M, 1, 2560, device="cuda").bfloat16()
        dm = torch.randn(M, 1, 2560, device="cuda").bfloat16()
        do = torch.randn(M, 1, STREAMS * 2560, device="cuda").bfloat16()
        a = _run_module(mod, h0, y, dm, do, fused=not deterministic)
        b = _run_module(mod, h0, y, dm, do, fused=not deterministic)
        for key in a:
            assert torch.equal(a[key], b[key]), key

    def test_cuda_graph_capture_replay(self):
        mod = _module(2560, 320)
        M, n, C = 4096, STREAMS, 2560

        class Call(torch.nn.Module):
            def __init__(self, gr):
                super().__init__()
                self.gr = gr

            def forward(self, h, y):
                mixed, _, g, res = self.gr(h)
                out = self.gr.fused_h_res_h_post_bda(None, res, g, (y, None), 0.0, True, True)
                return mixed, out

        call = Call(mod)
        torch.manual_seed(5)
        h = torch.randn(M, 1, n * C, device="cuda").bfloat16().requires_grad_(True)
        y = torch.randn(M, 1, C, device="cuda").bfloat16().requires_grad_(True)
        dm = torch.randn(M, 1, C, device="cuda").bfloat16()
        do = torch.randn(M, 1, n * C, device="cuda").bfloat16()

        def eager():
            for p in mod.parameters():
                p.grad = None
            hh, yy = h.detach().clone().requires_grad_(True), y.detach().clone().requires_grad_(
                True
            )
            outs = call(hh, yy)
            torch.autograd.backward(outs, [dm, do])
            return (
                [o.detach().clone() for o in outs]
                + [hh.grad, yy.grad]
                + [p.grad.clone() for p in mod.parameters()]
            )

        expected = eager()
        graphed = torch.cuda.make_graphed_callables(call, (h, y), num_warmup_iters=3)
        for _ in range(2):
            for p in mod.parameters():
                p.grad = None
            hh, yy = h.detach().clone().requires_grad_(True), y.detach().clone().requires_grad_(
                True
            )
            outs = graphed(hh, yy)
            torch.autograd.backward(outs, [dm, do])
            got = (
                [o.detach().clone() for o in outs]
                + [hh.grad, yy.grad]
                + [p.grad.clone() for p in mod.parameters()]
            )
            for e, g in zip(expected, got):
                assert torch.equal(e, g)
