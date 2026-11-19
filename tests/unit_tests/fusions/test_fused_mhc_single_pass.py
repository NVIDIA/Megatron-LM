# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests of the opt-in fast single-pass mHC path (MCORE_MHC_SP_FAST).

The aggregate, the BDA and the gradient-sum chain must match the default Triton kernels bit for
bit; the mapping projection and its gradients must be at least as accurate as the TF32 paths
against an FP64 reference; and a chain of sublayers must give the same bits with and without
selective mHC recompute.
"""

import pytest
import torch

from megatron.core.fusions import fused_mhc_single_pass as SP
from megatron.core.fusions.fused_mhc_kernels import is_triton_available
from megatron.core.tensor_parallel import random as tensor_parallel_random
from megatron.core.transformer.hyper_connection import (
    HyperConnectionExecutionContext,
    HyperConnectionModule,
    HyperConnectionResidualConnection,
    SinglePassMHCState,
    build_mhc_recompute_layer_plan,
    finalize_mhc_recompute_layer,
    get_mhc_aggregate,
)
from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked
from megatron.core.transformer.residual_recompute import ResidualStreamRecomputeContext
from megatron.core.transformer.transformer_config import TransformerConfig

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_triton_available()), reason="needs CUDA and Triton"
)

T, C, N = 256, 512, 4
NOUT = N * N + 2 * N
EPS, H_EPS, SK_EPS, ITERS = 1e-6, 1e-6, 1e-6, 20


def _bits_equal(a, b):
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    )


def _rel(a, ref):
    a, ref = a.double(), ref.double()
    return ((a - ref).norm() / ref.norm().clamp_min(1e-300)).item()


@pytest.fixture
def data():
    g = torch.Generator(device="cuda").manual_seed(11)
    d = dict(
        x=(torch.randn(T, N * C, device="cuda", generator=g) * 0.7).to(torch.bfloat16),
        pre_mix=torch.rand(T, N, device="cuda", generator=g) + 0.05,
        w=torch.randn(NOUT, N * C, device="cuda", generator=g) * 0.05,
        alphas=[torch.tensor([v], device="cuda") for v in (0.3, -0.2, 0.5)],
        bias=torch.randn(NOUT, device="cuda", generator=g) * 0.5,
        y=torch.randn(T, C, device="cuda", generator=g).to(torch.bfloat16),
        go=torch.randn(T, N * C, device="cuda", generator=g).to(torch.bfloat16),
        g_agg=torch.randn(T, C, device="cuda", generator=g).to(torch.bfloat16),
        g_res=torch.randn(T, N * C, device="cuda", generator=g).to(torch.bfloat16) * 0.1,
        g_pre=torch.randn(T, N, device="cuda", generator=g) * 1e-2,
        g_post=torch.randn(T, N, device="cuda", generator=g) * 1e-2,
        g_hres=torch.randn(T, N, N, device="cuda", generator=g) * 1e-2,
    )
    return d


def _forward(d, pre_mix):
    return SP.sp_pre_forward(
        d["x"], pre_mix, d["w"], *d["alphas"], d["bias"], N, EPS, H_EPS, SK_EPS, ITERS
    )


def test_supports():
    w = torch.zeros(NOUT, N * C, device="cuda")
    x = torch.zeros(T, 1, N * C, device="cuda", dtype=torch.bfloat16)
    assert SP.single_pass_fast_supports(x, N, w)
    assert not SP.single_pass_fast_supports(x[:0], N, w)
    assert not SP.single_pass_fast_supports(x.float(), N, w)
    assert not SP.single_pass_fast_supports(x[..., : N * 200].contiguous(), N, w[:, : N * 200])
    strided = torch.zeros(T, 2, N * C, device="cuda", dtype=torch.bfloat16)[:, :1]
    assert not SP.single_pass_fast_supports(strided, N, w)
    assert not SP.single_pass_fast_supports(x, N, w.to(torch.bfloat16))


def test_aggregate_bitwise(data):
    from megatron.core.fusions.fused_mhc_kernels import _triton_h_aggregate_fwd

    x, pre_mix = data["x"], data["pre_mix"]
    agg = _forward(data, pre_mix)[0]
    ref = _triton_h_aggregate_fwd(x.view(T, 1, N, C), pre_mix.view(T, 1, N)).view(T, C)
    assert _bits_equal(agg, ref)
    assert _bits_equal(SP.sp_aggregate(x, pre_mix, N), agg)
    entry = _forward(data, None)[0]
    assert _bits_equal(entry, x.view(T, N, C)[:, 0].contiguous())
    assert _bits_equal(SP.sp_aggregate(x, None, N), entry)


def test_bda_bitwise(data):
    """BDA backward equals the default Triton kernels at the same launch shape; the forward is an
    explicit FMA chain (the default kernel's contraction depends on its launch shape), so it is
    checked against FP64 instead.

    The default kernels' bits depend on their (autotuned) launch shape: BLOCK_C 64 changes
    FP32 contraction in a few elements. The fast path pins the shapes autotuned for the 64K
    proxy, so the comparison launches the default kernels with those shapes.
    """
    import triton

    from megatron.core.fusions import fused_mhc_kernels as fk

    x, y, go = data["x"], data["y"], data["go"]
    hr = torch.rand(T, N, N, device="cuda")
    hp = torch.rand(T, N, device="cuda")
    K = N * C
    out = SP.sp_bda_forward(hr, x, hp, y, N)
    ref = (
        torch.einsum("tji,tjc->tic", hr.double(), x.view(T, N, C).double())
        + hp.double().unsqueeze(-1) * y.double().unsqueeze(1)
    ).reshape(T, K)
    # Within half a BF16 ulp of the exact value, up to the FP32 accumulation.
    assert ((out.double() - ref).abs() <= ref.abs() * 2.0**-8 + 1e-6).all()
    g_res, g_y, g_hp, g_hr = SP.sp_bda_backward(go, hr, x, hp, y, N)
    a, b = SP.BDA_BWD_A_CFG, SP.BDA_BWD_B_CFG
    ref_res, ref_y = torch.empty_like(g_res), torch.empty_like(g_y)
    ref_hp, ref_hr = torch.empty_like(g_hp), torch.empty_like(g_hr)
    fk._triton_hpb_bwd_g_x_orig_kernel.fn[(triton.cdiv(T, a["block_s"]), C // a["block_c"])](
        go,
        hr,
        hp,
        ref_res,
        ref_y,
        T,
        C,
        N,
        K,
        C,
        1,
        N * N,
        N,
        1,
        K,
        C,
        1,
        BLOCK_C=a["block_c"],
        BLOCK_S=a["block_s"],
        num_warps=a["num_warps"],
        num_stages=a["num_stages"],
    )
    fk._triton_hpb_bwd_g_hp_hr_kernel.fn[(triton.cdiv(T, b["block_s"]),)](
        go,
        x,
        y,
        y,
        ref_hr,
        ref_hp,
        T,
        C,
        N,
        K,
        C,
        1,
        K,
        C,
        1,
        N * N,
        N,
        1,
        HAS_BIAS=False,
        BLOCK_C=b["block_c"],
        BLOCK_S=b["block_s"],
        num_warps=b["num_warps"],
        num_stages=b["num_stages"],
    )
    for got, exp in ((g_res, ref_res), (g_y, ref_y), (g_hp, ref_hp), (g_hr, ref_hr)):
        assert _bits_equal(got, exp)


def test_bda_next_aggregate_bitwise(data):
    """The BDA's next-sublayer aggregate equals the aggregate kernel's on the BDA output."""
    x, y, pre = data["x"], data["y"], data["pre_mix"]
    hr = torch.rand(T, N, N, device="cuda")
    hp = torch.rand(T, N, device="cuda")
    holder = SP.NextAggregate(pre)
    out = SP.sp_bda_forward(hr, x, hp, y, N, holder)
    assert holder.filled
    assert _bits_equal(out, SP.sp_bda_forward(hr, x, hp, y, N))
    assert _bits_equal(holder.agg, SP.sp_aggregate(out, pre, N))
    sq = torch.empty(C // SP.AGG_CFG["bc"], T, device="cuda")
    SP._launch_agg(out, pre, torch.empty_like(holder.agg), sq, N, True)
    torch.testing.assert_close(holder.sq.sum(0), sq.sum(0), rtol=1e-6, atol=0)
    # A filled holder is left alone (checkpoint replays of the BDA).
    agg_before = holder.agg.clone()
    SP.sp_bda_forward(hr, x * 2, hp, y, N, holder)
    assert _bits_equal(holder.agg, agg_before)


def _fp64_mappings(x, w, alphas, bias, pre_mix):
    proj = x @ w.t()
    inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    u = proj * inv
    pre = torch.sigmoid(u[:, :N] * alphas[0] + bias[:N]) + H_EPS
    post = 2 * torch.sigmoid(u[:, N : 2 * N] * alphas[1] + bias[N : 2 * N])
    m = (u[:, 2 * N :] * alphas[2] + bias[2 * N :]).view(-1, N, N).softmax(-1) + SK_EPS
    m = m / (m.sum(-2, keepdim=True) + SK_EPS)
    for _ in range(ITERS - 1):
        m = m / (m.sum(-1, keepdim=True) + SK_EPS)
        m = m / (m.sum(-2, keepdim=True) + SK_EPS)
    agg = (x.view(-1, N, C) * pre_mix.unsqueeze(-1)).sum(1)
    return agg, pre, post, m


def test_mappings_and_backward_vs_fp64(data):
    agg, pre, post, hres, proj, inv = _forward(data, data["pre_mix"])
    leaves = [
        t.double().requires_grad_()
        for t in (data["x"], data["w"], *data["alphas"], data["bias"], data["pre_mix"])
    ]
    xd, wd, a0, a1, a2, bd, pmd = leaves
    ref = _fp64_mappings(xd, wd, (a0, a1, a2), bd, pmd)
    # TF32 projections land near 1e-4 here; the BF16 hi + lo split is far below.
    for got, exp in zip((pre, post, hres), ref[1:]):
        assert _rel(got, exp) < 2e-5
    upstream = (data["g_agg"], data["g_pre"], data["g_post"], data["g_hres"])
    sum(((r * g.double()).sum() for r, g in zip(ref, upstream))).backward()
    g_x, g_pm, g_w, g_ab = SP.sp_pre_backward(
        data["x"],
        data["pre_mix"],
        data["w"],
        *data["alphas"],
        data["bias"],
        proj,
        inv,
        *upstream,
        data["g_res"],
        N,
        SK_EPS,
        ITERS,
        True,
        True,
    )
    assert _rel(g_x, xd.grad + data["g_res"].double()) < 8e-3  # BF16 rounding of the sum
    assert _rel(g_w, wd.grad) < 1e-4
    assert _rel(g_pm, pmd.grad) < 1e-5
    assert _rel(g_ab[3:], bd.grad) < 1e-4
    for i, leaf in enumerate((a0, a1, a2)):
        assert _rel(g_ab[i : i + 1], leaf.grad) < 2e-4


@pytest.mark.parametrize("entry", [False, True])
def test_gradient_chain_matches_ordered_sum(data, entry):
    """x's gradient is round(round(g_res + round(g_agg * pre_j)) + round(g_map)), bit for bit."""
    from megatron.core.fusions.fused_mhc_kernels import fused_add_3_sequential

    pre_mix = None if entry else data["pre_mix"]
    _, _, _, _, proj, inv = _forward(data, pre_mix)
    args = (data["x"], pre_mix, data["w"], *data["alphas"], data["bias"], proj, inv)
    rest = (data["g_pre"], data["g_post"], data["g_hres"])
    g_x = SP.sp_pre_backward(
        *args, data["g_agg"], *rest, data["g_res"], N, SK_EPS, ITERS, True, True
    )[0]
    g_map = SP.sp_pre_backward(*args, None, *rest, None, N, SK_EPS, ITERS, False, False)[0]
    if entry:
        term = torch.zeros(T, N, C, device="cuda", dtype=torch.bfloat16)
        term[:, 0] = data["g_agg"]
    else:
        term = (data["g_agg"].view(T, 1, C).float() * data["pre_mix"].view(T, N, 1)).to(
            torch.bfloat16
        )
    assert _bits_equal(g_x, fused_add_3_sequential(data["g_res"], term.view(T, N * C), g_map))


def _run_chain(monkeypatch, fast, recompute, layers=4):
    monkeypatch.setattr(SP, "SINGLE_PASS_FAST_NEXT_AGG", True)
    config = TransformerConfig(
        num_layers=layers,
        hidden_size=C,
        num_attention_heads=8,
        enable_mhc_connections=True,
        mhc_single_pass=True,
        use_fused_mhc=True,
        recompute_granularity="selective" if recompute else None,
        recompute_modules=["mhc"] if recompute else None,
        mhc_recompute_layer_num=2 if recompute else None,
    )
    torch.manual_seed(3)
    mhcs = []
    for i in range(layers):
        m = HyperConnectionModule(config, i + 1).cuda()
        if not fast:
            m._sp_fast = False
        convert_module_to_dtype_except_fp32_marked(m, torch.bfloat16)
        with torch.no_grad():
            m.bias.normal_(0, 0.5)
            m.alpha_res.fill_(0.5)
        mhcs.append(m)
    branches = [torch.nn.Linear(C, C, bias=False).cuda().to(torch.bfloat16) for _ in range(layers)]
    g = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn(T, 1, N * C, device="cuda", generator=g).to(torch.bfloat16).requires_grad_()
    probe = torch.randn(T, 1, C, device="cuda", generator=g).to(torch.bfloat16)
    state = SinglePassMHCState()
    managers, ends = build_mhc_recompute_layer_plan(
        layers, config.mhc_recompute_layer_num, recompute
    )
    read = HyperConnectionResidualConnection(config, operation="read")
    write = HyperConnectionResidualConnection(config, operation="write")
    out = x
    for i, (hc, branch) in enumerate(zip(mhcs, branches)):
        manager = managers[i]
        if manager is not None:
            manager.is_last_layer_in_recompute_block = ends[i]
        replay = ResidualStreamRecomputeContext(manager, ends[i]) if manager is not None else None
        execution = HyperConnectionExecutionContext(hc, mhc_state=state)
        agg, carry = read(out, recompute_context=replay, execution_context=execution)
        assert (hc._sp_fast_pending is not None) == fast
        if fast and i > 0:
            assert state.next_aggregate is None  # consumed by this sublayer
        y = torch.tanh(branch(agg))
        out = write(
            y,
            state=carry,
            dropout_probability=0.0,
            training=True,
            recompute_context=replay,
            execution_context=execution,
        )
        finalize_mhc_recompute_layer(manager, out, ends[i])
    final = state.contract(out, N, aggregate=get_mhc_aggregate(config))
    final.backward(probe)
    params = [p for m in (*mhcs, *branches) for p in m.parameters()]
    return [final.detach(), x.grad] + [p.grad for p in params]


def test_chain_fast_vs_default_and_recompute(monkeypatch):
    launches = []
    launch_agg = SP._launch_agg

    def counting(*args):
        launches.append(args[-1])
        return launch_agg(*args)

    monkeypatch.setattr(SP, "_launch_agg", counting)
    fast = _run_chain(monkeypatch, True, False)
    # Only the entry sublayer runs the aggregate kernel; the others get it from the previous BDA.
    assert launches.count(True) == 1
    fast_rc = _run_chain(monkeypatch, True, True)
    for a, b in zip(fast, fast_rc):
        assert _bits_equal(a, b), "recompute must replay the same kernels"
    default = _run_chain(monkeypatch, False, False)
    # The mapping projection differs (FP32-accurate vs TF32), everything else is shared.
    for a, b in zip(fast, default):
        cos = torch.nn.functional.cosine_similarity(
            a.double().flatten(), b.double().flatten(), dim=0
        )
        assert cos > 0.999


def test_three_tuple_callers_keep_default_path(monkeypatch):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=C,
        num_attention_heads=8,
        enable_mhc_connections=True,
        mhc_single_pass=True,
        use_fused_mhc=True,
    )
    hc = HyperConnectionModule(config, 1).cuda()
    convert_module_to_dtype_except_fp32_marked(hc, torch.bfloat16)
    x = torch.randn(T, 1, N * C, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    assert hc._sp_fast
    result = hc(x, return_residual=False, mhc_state=SinglePassMHCState())
    assert len(result) == 3 and hc._sp_fast_pending is None


def test_fast_chain_releases_memory(monkeypatch):
    """Nothing from a fast-path step outlives it (the NextAggregate holder, kept in checkpoint
    contexts, must not close a reference cycle through the autograd graph)."""
    _run_chain(monkeypatch, True, True)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    for _ in range(3):
        _run_chain(monkeypatch, True, True)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() <= base + (1 << 20)


@pytest.mark.parametrize("rows", [T, T - 3])
def test_bda_backward_tiled_matches_row_loop(data, monkeypatch, rows):
    """The tiled BDA backward (VR200 choice) gives the row-loop kernel's bits, also with a partial
    last row block."""
    x, y, go = data["x"][:rows], data["y"][:rows], data["go"][:rows]
    hr = torch.rand(rows, N, N, device="cuda")
    hp = torch.rand(rows, N, device="cuda")
    monkeypatch.setattr(SP, "BDA_BWD4T_CFG", None)
    ref = SP.sp_bda_backward(go, hr, x, hp, y, N)
    monkeypatch.setattr(SP, "BDA_BWD4T_CFG", dict(block_s=8, num_warps=2))
    got = SP.sp_bda_backward(go, hr, x, hp, y, N)
    for a, b in zip(got, ref):
        assert _bits_equal(a, b)


@pytest.mark.parametrize("entry", [False, True])
def test_pre_backward_prefetch_matches(data, monkeypatch, entry):
    """The prefetching x-gradient kernel (VR200 choice) gives the plain kernel's bits."""
    pre_mix = None if entry else data["pre_mix"]
    _, _, _, _, proj, inv = _forward(data, pre_mix)
    args = (data["x"], pre_mix, data["w"], *data["alphas"], data["bias"], proj, inv)
    upstream = (data["g_agg"], data["g_pre"], data["g_post"], data["g_hres"], data["g_res"])
    shape = dict(bt=16, bc=128, tch=64, num_warps=2, num_stages=1)
    monkeypatch.setattr(SP, "PRE_BWD_MAIN_CFG", dict(shape))
    ref = SP.sp_pre_backward(*args, *upstream, N, SK_EPS, ITERS, True, True)
    monkeypatch.setattr(SP, "PRE_BWD_MAIN_CFG", dict(shape, prefetch=True))
    got = SP.sp_pre_backward(*args, *upstream, N, SK_EPS, ITERS, True, True)
    for a, b in zip(got, ref):
        assert (a is None and b is None) or _bits_equal(a, b)
    # Without the aggregate and BDA gradients (only the mapping term).
    no_up = (None, data["g_pre"], data["g_post"], data["g_hres"], None)
    monkeypatch.setattr(SP, "PRE_BWD_MAIN_CFG", dict(shape))
    ref = SP.sp_pre_backward(*args, *no_up, N, SK_EPS, ITERS, False, False)[0]
    monkeypatch.setattr(SP, "PRE_BWD_MAIN_CFG", dict(shape, prefetch=True))
    assert _bits_equal(SP.sp_pre_backward(*args, *no_up, N, SK_EPS, ITERS, False, False)[0], ref)


@pytest.mark.skipif(
    SP._TensorDescriptor is None
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 9,
    reason="needs host TMA descriptors (Triton 3.4+, sm90+)",
)
@pytest.mark.parametrize("entry", [False, True])
def test_tma_kernels_match(data, monkeypatch, entry):
    """The TMA variants (VR200 choices) of the projection, x-gradient and W-gradient kernels give
    the plain kernels' results at the same tile shapes."""
    pre_mix = None if entry else data["pre_mix"]
    proj_shape = dict(bm=128, bk=64, split=8, num_warps=4, num_stages=3)
    monkeypatch.setattr(SP, "PROJ_CFG", dict(proj_shape))
    ref_fwd = _forward(data, pre_mix)
    monkeypatch.setattr(SP, "PROJ_CFG", dict(proj_shape, tma=True))
    got_fwd = _forward(data, pre_mix)
    torch.testing.assert_close(got_fwd[4], ref_fwd[4], rtol=1e-6, atol=1e-6)  # projection
    for a, b in zip(got_fwd[:4], ref_fwd[:4]):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    proj, inv = ref_fwd[4], ref_fwd[5]
    args = (data["x"], pre_mix, data["w"], *data["alphas"], data["bias"], proj, inv)
    upstream = (data["g_agg"], data["g_pre"], data["g_post"], data["g_hres"], data["g_res"])
    main_shape = dict(bt=32, bc=64, tch=64, num_warps=2, num_stages=3)
    gw_shape = dict(bk=128, bt=32, tch=128, num_warps=4, num_stages=3)
    monkeypatch.setattr(SP, "PRE_BWD_MAIN_CFG", dict(main_shape))
    monkeypatch.setattr(SP, "GW_CFG", dict(gw_shape))
    ref = SP.sp_pre_backward(*args, *upstream, N, SK_EPS, ITERS, True, True)
    monkeypatch.setattr(SP, "PRE_BWD_MAIN_CFG", dict(main_shape, tma=True))
    monkeypatch.setattr(SP, "GW_CFG", dict(gw_shape, tma=True))
    got = SP.sp_pre_backward(*args, *upstream, N, SK_EPS, ITERS, True, True)
    # Token-major tensor-core tiles: the mapping term can round differently in rare elements.
    differing = (got[0].view(torch.int16) != ref[0].view(torch.int16)).sum().item()
    assert differing <= max(1, got[0].numel() // 10000)
    torch.testing.assert_close(got[0], ref[0], rtol=2**-7, atol=1e-6)
    for a, b in zip(got[1:], ref[1:]):
        if a is not None or b is not None:
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
