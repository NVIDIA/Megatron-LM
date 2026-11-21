# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_rank_tree_unpermute_kernel`` (megatron/core/transformer/moe/batch_invariant.py),
through its two launchers: ``unpermute`` (``probs=None``, ``_RankTreeUnpermute``) and
``unpermute_with_bf16_rank_partials``, and through ``moe_utils.unpermute`` in batch-invariant
mode.

Inference reaches it when generation runs a non-inference-optimized MoE model in
batch-invariant mode: such a model must use the all-to-all dispatcher, whose combine calls
``moe_utils.unpermute`` without probabilities (packet O1 doc, medium-risk batch, with the
``path:line`` chain). Only the forward kernel is in scope; the backward kernel runs only in
training.

Contract (from ``unpermute``'s docstring and the reference-loop comments; ``combine_reference``):

* ``out[t]`` adds token ``t``'s routes by EP rank, then by top-k slot. A route ``(t, k)`` counts
  on rank ``r`` when ``inverse_rows[t, k] >= 0`` and ``inverse_experts[t, k]`` lies in rank
  ``r``'s expert range ``[r * E / ep, (r + 1) * E / ep)``; other routes (``-1`` rows or experts,
  experts outside ``[0, E)``) are not read.
* Rank partials: unit weights (``unpermute``, ``probs=None``): the fp64 sum of the rows in slot
  order, rounded to fp32. FlashInfer finalize (``unpermute_with_bf16_rank_partials``): fp32
  fused multiply-adds of the fp32 probability and the row in slot order, from zero, rounded to
  bf16.
* Cross-rank: ascending rank order; the ``ordered`` collective adds in fp32, ``multimem`` in
  fp64 with one rounding to fp32 (``get_batch_invariant_collective``). The fp32 result is
  rounded to the rows' dtype; ``out`` has shape ``restore_shape`` and every element is written.
* EP size from ``parallel_state`` (``unpermute``) or the argument; ``E % ep == 0`` asserted.
  Inputs are read-only. A token with no counted route reads ``+0``.
* Tolerance class: **exact** (bitwise): the contract fixes every rounding, because the result
  must equal generation's combine bit for bit. Where a NaN is expected, any NaN is accepted.

Contract-first caveat: written from the docstrings and the reference loop's comments before
reading the kernel body; the reference evaluates that definition with torch elementwise ops on
whole token columns (``combine_reference``) and shares no indexing with the kernel. The exact
fp32 FMA is emulated in fp64 with a TwoSum tie correction (checked against rational arithmetic
on 40k random triples, where a separate multiply and add differ on 6%).

Structure (read after the contract): one program per (token, hidden block); constexpr
``EP_SIZE`` and ``TOPK`` (unrolled), ``EXPERTS_PER_RANK``, ``CROSS_RANK_FP64``,
``BF16_RANK_PARTIALS`` and ``BLOCK_H = min(next_pow2(H), 512)``. Inputs therefore cover EP {1,
2, 4, 8, 16} x experts per rank {1, 2, 5, 16} x top-k {1, 2, 6, 8} x both partial modes x both
collectives x bf16 / fp32 rows x H in {1, 3, 100, 511, 512, 513, 1025, 2688} (one to six
blocks, partial last block) x tokens {1, 2, 37, 300} x values (scaled normal, a 2^+-40 range
with signed zeros where the fp32 / fp64 rounding points and the order matter, Inf and NaN) x
invalid routes (``-1`` rows, ``-1`` experts, experts past ``E``, each pointing at a NaN row),
pairwise; hand-placed values at each rounding point; zero tokens; production shapes (EP 1, 8
and 64); CUDA-graph replay; the ``moe_utils.unpermute`` dispatch; opt-in (``large``), row and
output offsets ``row * H`` at 2^31 (``H = 7168``).
Unreferenced rows hold NaN, and the output is allocated NaN-poisoned.

Hardware: any CUDA GPU; marked for H100 and GB200, where batch-invariant generation runs.
"""

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.moe_routing import combine_reference as cr
from tests.unit_tests.inference.oracles.moe_routing import routing_reference as rr

bi = pytest.importorskip("megatron.core.transformer.moe.batch_invariant")
pytest.importorskip("triton")

pytestmark = [pytest.mark.launch_on_gb200, ou.requires_sm(ou.HOPPER, ou.BLACKWELL)]

MODES = ("sum", "bf16-partials")
COLLECTIVES = ("ordered", "multimem")


@pytest.fixture
def ep_and_collective(monkeypatch):
    """``set(ep, collective)``: the EP world size ``unpermute`` reads and the collective."""

    def _set(ep, collective):
        monkeypatch.setattr(bi, "get_batch_invariant_collective", lambda: collective)
        bi.parallel_state.set_expert_model_parallel_world_size(ep)

    yield _set
    bi.parallel_state.set_expert_model_parallel_world_size(None)


def build_case(
    tokens, hidden, topk, num_experts, *, dtype, kind, invalid, dev, seed=0, extra_rows=5
):
    """Rows, inverse map and probabilities for ``tokens`` tokens routed to ``topk`` distinct
    experts each, in random slot order, over a row buffer with ``extra_rows`` unreferenced rows.

    Rows that no counted route names (unreferenced, or named only by an invalid route) hold NaN.
    ``invalid`` turns about a sixth of the routes into a ``-1`` row, a ``-1`` expert or an expert
    past ``E``.
    """
    g = ou.make_generator(ou.DEFAULT_SEED + seed)
    experts = torch.rand(tokens, num_experts, generator=g).argsort(dim=1)[:, :topk]
    routes = tokens * topk
    num_rows = routes + extra_rows
    inverse_rows = torch.randperm(num_rows, generator=g)[:routes].view(tokens, topk).clone()
    inverse_experts = experts.clone()
    if invalid and routes:
        pick = torch.rand(tokens, topk, generator=g) < 1 / 6
        how = torch.randint(0, 3, (tokens, topk), generator=g)
        inverse_rows[pick & (how == 0)] = -1
        inverse_experts[pick & (how == 1)] = -1
        inverse_experts[pick & (how == 2)] = num_experts + 3
    if kind == "randn":
        rows = torch.randn(num_rows, hidden, generator=g) * torch.exp2(
            torch.randint(-8, 8, (num_rows, 1), generator=g).float()
        )
    elif kind == "wide":
        rows = torch.randn(num_rows, hidden, generator=g) * torch.exp2(
            torch.randint(-40, 40, (num_rows, hidden), generator=g).float()
        )
        rows.view(-1)[::7] = -0.0
        rows.view(-1)[3::11] = 0.0
    elif kind == "special":
        rows = torch.randn(num_rows, hidden, generator=g)
        special = torch.rand(num_rows, hidden, generator=g)
        rows[special < 0.02] = float("inf")
        rows[(special >= 0.02) & (special < 0.04)] = float("-inf")
        rows[(special >= 0.04) & (special < 0.05)] = float("nan")
    else:
        raise ValueError(kind)
    counted = torch.zeros(num_rows, dtype=torch.bool)
    ok = (inverse_rows >= 0) & (inverse_experts >= 0) & (inverse_experts < num_experts)
    counted[inverse_rows[ok]] = True
    rows[~counted] = float("nan")
    probs = torch.rand(tokens, topk, generator=g)
    probs[:, 0] = 1.0  # exact weights next to random ones
    return (
        rows.to(dtype=dtype, device=dev),
        inverse_rows.to(dev),
        inverse_experts.to(dev),
        probs.to(dev),
    )


def assert_matches(out, ref, *, what):
    """Bitwise, except that any NaN matches any NaN (NaN payloads differ between devices)."""
    nan_out, nan_ref = torch.isnan(out), torch.isnan(ref)
    ou.assert_exact(nan_out, nan_ref, what=f"NaN positions ({what})")
    ou.assert_exact(
        torch.where(nan_out, torch.zeros_like(out), out),
        torch.where(nan_ref, torch.zeros_like(ref), ref),
        what=what,
    )


def run_and_check(
    ep_and_collective,
    tokens,
    hidden,
    topk,
    ep,
    per_rank,
    *,
    mode,
    collective,
    dtype,
    kind,
    invalid,
    seed=0,
):
    dev = rr.require_cuda()
    num_experts = ep * per_rank
    topk = min(topk, num_experts)
    ep_and_collective(ep, collective)
    rows, inv_rows, inv_experts, probs = build_case(
        tokens,
        hidden,
        topk,
        num_experts,
        dtype=dtype,
        kind=kind,
        invalid=invalid,
        dev=dev,
        seed=seed,
    )
    before = (rows.clone(), inv_rows.clone(), inv_experts.clone(), probs.clone())
    with torch.no_grad(), rr.poisoned_outputs():
        if mode == "sum":
            out = bi.unpermute(
                rows,
                torch.Size((tokens, hidden)),
                probs=None,
                num_experts=num_experts,
                inverse_map=torch.stack((inv_rows, inv_experts)),
            )
        else:
            out = bi.unpermute_with_bf16_rank_partials(
                rows, inv_rows, inv_experts, probs, num_experts, ep
            )
    ref = cr.rank_tree_combine(
        rows,
        inv_rows,
        inv_experts,
        num_experts,
        ep,
        cross_rank_fp64=collective == "multimem",
        probs=probs if mode == "bf16-partials" else None,
    )
    what = (
        f"T={tokens} H={hidden} topk={topk} ep={ep} E={num_experts} {mode} {collective}"
        f" {dtype} {kind} invalid={invalid}"
    )
    assert out.dtype == dtype and out.shape == (tokens, hidden), what
    assert_matches(out, ref, what=what)
    for name, now, then in zip(
        ("rows", "inverse rows", "inverse experts", "probs"),
        (rows, inv_rows, inv_experts, probs),
        before,
    ):
        ou.assert_exact(now, then, what=f"{name} are read-only")


_VARIANTS = ou.pairwise_cases(
    {
        "ep": [1, 2, 4, 8, 16],
        "per_rank": [1, 2, 5, 16],
        "topk": [1, 2, 6, 8],
        "mode": list(MODES),
        "collective": list(COLLECTIVES),
        "dtype": [torch.bfloat16, torch.float32],
        "hidden": [1, 3, 100, 511, 512, 513, 1025, 2688],
        "tokens": [1, 2, 37, 300],
        "kind": ["randn", "wide", "special"],
        "invalid": [False, True],
    }
)


def _variant_id(c):
    return (
        f"ep{c['ep']}-epr{c['per_rank']}-topk{c['topk']}-{c['mode']}-{c['collective']}"
        f"-{str(c['dtype'])[6:]}-H{c['hidden']}-T{c['tokens']}-{c['kind']}-inv{int(c['invalid'])}"
    )


@pytest.mark.parametrize("case", _VARIANTS, ids=[_variant_id(c) for c in _VARIANTS])
def test_variants_pairwise(ep_and_collective, case):
    """Every compiled variant and both launchers, pairwise (module docstring)."""
    run_and_check(
        ep_and_collective,
        case["tokens"],
        case["hidden"],
        case["topk"],
        case["ep"],
        case["per_rank"],
        mode=case["mode"],
        collective=case["collective"],
        dtype=case["dtype"],
        kind=case["kind"],
        invalid=case["invalid"],
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("collective", COLLECTIVES)
def test_rounding_points(ep_and_collective, mode, collective):
    """Hand-placed values whose result depends on each rounding point: within a rank
    ``1 + 2^-30 - 1`` (2^-30 in fp64, 0 in fp32), across ranks the same on three ranks (fp64
    keeps 2^-30, an fp32 chain loses it), a bf16 partial ``1 + 2^-9`` (rounds to 1), and
    ``2^40 - 2^40 + 1`` across ranks (order matters in fp32)."""
    dev = rr.require_cuda()
    hidden, ep, per_rank = 4, 4, 1
    ep_and_collective(ep, collective)
    values = [
        # (slots: (value, expert)) per token
        [(1.0, 0), (2.0**-30, 0), (-1.0, 0)],
        [(1.0, 0), (2.0**-30, 1), (-1.0, 2)],
        [(1.0, 0), (2.0**-9, 0), (0.0, 3)],
        [(2.0**40, 1), (-(2.0**40), 2), (1.0, 3)],
        [(1.0, 3), (2.0**-30, 2), (-1.0, 1)],
    ]
    tokens, topk = len(values), 3
    rows = torch.full((tokens * topk + 1, hidden), float("nan"))
    inv_rows = torch.zeros(tokens, topk, dtype=torch.int64)
    inv_experts = torch.zeros(tokens, topk, dtype=torch.int64)
    for t, slots in enumerate(values):
        for k, (v, e) in enumerate(slots):
            row = 1 + t * topk + k
            rows[row] = v
            inv_rows[t, k], inv_experts[t, k] = row, e
    probs = torch.ones(tokens, topk)
    rows, inv_rows, inv_experts, probs = (x.to(dev) for x in (rows, inv_rows, inv_experts, probs))
    rows = rows.float()
    with torch.no_grad(), rr.poisoned_outputs():
        if mode == "sum":
            out = bi.unpermute(
                rows,
                torch.Size((tokens, hidden)),
                probs=None,
                num_experts=ep * per_rank,
                inverse_map=torch.stack((inv_rows, inv_experts)),
            )
        else:
            out = bi.unpermute_with_bf16_rank_partials(
                rows, inv_rows, inv_experts, probs, ep * per_rank, ep
            )
    ref = cr.rank_tree_combine(
        rows,
        inv_rows,
        inv_experts,
        ep * per_rank,
        ep,
        cross_rank_fp64=collective == "multimem",
        probs=probs if mode == "bf16-partials" else None,
    )
    assert_matches(out, ref, what=f"rounding points ({mode}, {collective})")


@pytest.mark.parametrize("mode", MODES)
def test_no_tokens(ep_and_collective, mode):
    """Zero tokens: an empty ``[0, H]`` output, nothing launched."""
    dev = rr.require_cuda()
    ep_and_collective(2, "ordered")
    rows = torch.randn(3, 64, device=dev, dtype=torch.bfloat16)
    empty = torch.zeros(0, 2, dtype=torch.int64, device=dev)
    with torch.no_grad():
        if mode == "sum":
            out = bi.unpermute(
                rows,
                torch.Size((0, 64)),
                probs=None,
                num_experts=4,
                inverse_map=torch.stack((empty, empty)),
            )
        else:
            out = bi.unpermute_with_bf16_rank_partials(
                rows, empty, empty, torch.zeros(0, 2, device=dev), 4, 2
            )
    assert out.shape == (0, 64) and out.dtype == torch.bfloat16


@pytest.mark.parametrize("mode", MODES)
def test_rejects_uneven_expert_shards(ep_and_collective, mode):
    """``E`` must be a multiple of the EP size (asserted)."""
    dev = rr.require_cuda()
    ep_and_collective(4, "ordered")
    rows = torch.randn(4, 8, device=dev, dtype=torch.bfloat16)
    inv = torch.zeros(2, 2, dtype=torch.int64, device=dev)
    with pytest.raises(AssertionError):
        if mode == "sum":
            bi.unpermute(
                rows,
                torch.Size((2, 8)),
                probs=None,
                num_experts=6,
                inverse_map=torch.stack((inv, inv)),
            )
        else:
            bi.unpermute_with_bf16_rank_partials(rows, inv, inv, torch.ones(2, 2, device=dev), 6, 4)


@pytest.mark.parametrize(
    "tokens,hidden,topk,num_experts,ep",
    [
        pytest.param(8192, 2688, 6, 128, 1, id="nanov3-ep1"),
        pytest.param(8192, 2688, 6, 128, 8, id="nanov3-ep8"),
        pytest.param(4096, 7168, 8, 256, 8, id="dsv3-ep8"),
        pytest.param(1024, 7168, 8, 256, 64, id="dsv3-ep64"),
    ],
)
@pytest.mark.parametrize("mode", MODES)
def test_production_sizes(ep_and_collective, tokens, hidden, topk, num_experts, ep, mode):
    """Production shapes, bf16 rows, both partial modes (``ordered`` for the unit-weight sum,
    ``multimem`` for the FlashInfer finalize, as configured on GB200)."""
    collective = "ordered" if mode == "sum" else "multimem"
    run_and_check(
        ep_and_collective,
        tokens,
        hidden,
        topk,
        ep,
        num_experts // ep,
        mode=mode,
        collective=collective,
        dtype=torch.bfloat16,
        kind="randn",
        invalid=False,
    )


@pytest.mark.sanitizer
def test_sanitizer_small_case(ep_and_collective):
    """Small representative case for the compute-sanitizer memcheck pass."""
    run_and_check(
        ep_and_collective,
        13,
        520,
        4,
        4,
        2,
        mode="sum",
        collective="ordered",
        dtype=torch.bfloat16,
        kind="wide",
        invalid=True,
    )


def test_moe_utils_unpermute_dispatches_here(ep_and_collective, monkeypatch):
    """The all-to-all dispatcher's combine (``moe_utils.unpermute`` without probabilities) in
    batch-invariant mode returns the rank-tree combine."""
    from megatron.core.transformer.moe import moe_utils

    dev = rr.require_cuda()
    tokens, hidden, topk, ep, num_experts = 37, 300, 4, 4, 16
    ep_and_collective(ep, "ordered")
    monkeypatch.setattr(moe_utils, "is_batch_invariant_mode_enabled", lambda: True)
    rows, inv_rows, inv_experts, _ = build_case(
        tokens, hidden, topk, num_experts, dtype=torch.bfloat16, kind="wide", invalid=False, dev=dev
    )
    routing_map = torch.zeros(tokens, num_experts, dtype=torch.bool, device=dev)
    routing_map.scatter_(1, inv_experts, True)
    with torch.no_grad(), rr.poisoned_outputs():
        out = moe_utils.unpermute(
            rows,
            torch.zeros(rows.shape[0], dtype=torch.int64, device=dev),
            torch.Size((tokens, hidden)),
            routing_map=routing_map,
            batch_invariant_inverse_map=torch.stack((inv_rows, inv_experts)),
        )
    ref = cr.rank_tree_combine(rows, inv_rows, inv_experts, num_experts, ep, cross_rank_fp64=False)
    assert_matches(out, ref, what="moe_utils.unpermute, batch-invariant")


@pytest.mark.parametrize("collective", COLLECTIVES)
def test_cuda_graph_replay(ep_and_collective, collective):
    """``unpermute`` captured once (fixed shapes, as a decode CUDA graph at EP 1 captures it) and
    replayed after the rows and the inverse map are rewritten in place; the output is
    re-poisoned before every replay."""
    dev = rr.require_cuda()
    tokens, hidden, topk, ep, num_experts = 40, 700, 6, 2, 8
    ep_and_collective(ep, collective)
    rows, inv_rows, inv_experts, _ = build_case(
        tokens, hidden, topk, num_experts, dtype=torch.bfloat16, kind="wide", invalid=True, dev=dev
    )
    inverse_map = torch.stack((inv_rows, inv_experts))

    def call():
        return bi.unpermute(
            rows,
            torch.Size((tokens, hidden)),
            probs=None,
            num_experts=num_experts,
            inverse_map=inverse_map,
        )

    with torch.no_grad():
        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = call()
    for step in range(1, 5):
        r, ir, ie, _ = build_case(
            tokens,
            hidden,
            topk,
            num_experts,
            dtype=torch.bfloat16,
            kind="wide",
            invalid=step % 2 == 1,
            dev=dev,
            seed=step,
        )
        rows.copy_(r)
        inverse_map.copy_(torch.stack((ir, ie)))
        ou.poison_(out)
        graph.replay()
        torch.cuda.synchronize()
        ref = cr.rank_tree_combine(
            r, ir, ie, num_experts, ep, cross_rank_fp64=collective == "multimem"
        )
        assert_matches(out, ref, what=f"replay {step} ({collective})")


@pytest.mark.parametrize("tokens", ou.int32_overflow_params(7168))
def test_int32_row_offsets(ep_and_collective, tokens):
    """Rows read (``row * H``) and output rows written (``token * H``) whose element offsets
    reach 2^31 (``H = 7168``). Top-1, EP 1, token ``t`` reading row ``perm[t]``: the combine is
    then that row exactly (one fp64 add from +0, exact roundings), compared on the device.
    Opt-in ``large``; about 9 GiB.
    """
    dev = rr.require_cuda()
    hidden = 7168
    ep_and_collective(1, "ordered")
    ou.require_free_memory(tokens * hidden * 4 + (2 << 30))
    g = ou.make_generator(device=dev)
    rows = torch.randn(tokens, hidden, device=dev, generator=g, dtype=torch.bfloat16)
    # Token t reads row perm[t]; the last token reads the last row (both offsets highest).
    last = torch.tensor([tokens - 1], device=dev)
    perm = torch.cat([torch.randperm(tokens - 1, device=dev, generator=g), last]).view(tokens, 1)
    experts = torch.zeros(tokens, 1, dtype=torch.int64, device=dev)
    with torch.no_grad():
        out = bi.unpermute(
            rows,
            torch.Size((tokens, hidden)),
            probs=None,
            num_experts=1,
            inverse_map=torch.stack((perm, experts)),
        )
    for lo in range(0, tokens, 1 << 16):
        expected = rows[perm[lo : lo + (1 << 16), 0]] + 0.0  # -0.0 -> +0.0, as a sum from +0
        ou.assert_exact(out[lo : lo + (1 << 16)], expected, what=f"output rows from {lo}")
