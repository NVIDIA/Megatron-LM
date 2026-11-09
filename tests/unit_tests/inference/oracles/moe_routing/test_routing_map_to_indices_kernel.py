# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_routing_map_to_indices_kernel``
(megatron/core/transformer/moe/megakernel/mok/route_adapter.py), through its launcher
``routing_map_to_mok_inputs`` (``RoutingMapToIndices.forward``).

The MOK MoE backend calls it on every forward of a layer with ``moe_megakernel_backend='mok'``,
before anything inference-specific: generation on such a model (for example RL rollouts on the
training model) reaches it (packet O1 doc, medium-risk batch, with the ``path:line`` chain).
Only the forward kernel is in scope; the backward kernel runs only in training.

Contract (from ``RoutingMapToIndices``' and ``routing_map_to_mok_inputs``' docstrings):

* Inputs: ``probs`` ``[T, E]`` fp16, bf16 or fp32 and ``routing_map`` ``[T, E]`` (bool or
  integer, nonzero = routed), both CUDA, contiguous and on one device; ``1 <= topk <= E``
  (``int``). Violations raise ``ValueError`` / ``TypeError`` before any launch. Read-only.
* Outputs: ``probs_indices`` ``[T, topk]`` fp32 and ``indices`` ``[T, topk]`` int32. Row ``t``
  lists token ``t``'s routed experts in increasing expert index, at most ``topk`` of them (the
  first ``topk``; "maximum number of valid routes emitted per token"), with
  ``probs_indices = float32(probs[t, e])`` (exact for every input dtype, special values
  included); the remaining slots hold ``-1`` and ``+0.0``.
* ``T = 0``: empty outputs.
* Tolerance class: **exact** (integers and exact conversions; any NaN matches any NaN).

Contract-first caveat: written from the docstrings before reading the kernel body. The
reference is a per-row Python listing of ``nonzero`` experts.

Structure (read after the contract): one program per token; constexpr ``num_experts``,
``topk``, ``num_experts_block_size = next_pow2(E)`` and ``topk_block_size = next_pow2(topk)``;
``num_warps`` 4 up to a 512-expert block, else 8; the padding store, a block barrier, then the
compacted scatter. Inputs therefore cover E in {1, 2, 3, 7, 8, 9, 64, 127, 128, 129, 256, 511,
512, 513, 1024} (around the power-of-two blocks and the warp switch), top-k {1, 2, 3, 6, 8, E},
routes per row {0, fewer than, exactly, more than top-k, all}, bool / uint8 / int32 / int64 maps,
fp32 / bf16 / fp16 probabilities with signed zeros, subnormals, Inf and NaN, and T {1, 2, 37,
300}, pairwise; zero tokens; the input checks; production shapes; opt-in (``large``), row
offsets ``t * E`` past 2^31 (suspected finding O1-M1: the offset is int32).

Hardware: any CUDA GPU (H100 in the manifest).
"""

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.moe_routing import routing_reference as rr

ra = pytest.importorskip("megatron.core.transformer.moe.megakernel.mok.route_adapter")
pytest.importorskip("triton")

pytestmark = [ou.requires_sm(ou.HOPPER)]


def reference(probs, routing_map, topk):
    """Per row: the first ``topk`` routed experts in increasing order, their fp32 probabilities,
    then ``-1`` / ``+0.0``."""
    tokens = probs.shape[0]
    indices = torch.full((tokens, topk), -1, dtype=torch.int32)
    weights = torch.zeros(tokens, topk, dtype=torch.float32)
    rm, p = routing_map.cpu(), probs.cpu().float()
    for t in range(tokens):
        experts = (rm[t] != 0).nonzero().reshape(-1)[:topk]
        indices[t, : experts.numel()] = experts.to(torch.int32)
        weights[t, : experts.numel()] = p[t, experts]
    return weights, indices


def assert_weights(got, expected, *, what):
    """Bitwise, any NaN matching any NaN."""
    nan_got, nan_expected = torch.isnan(got.cpu()), torch.isnan(expected)
    ou.assert_exact(nan_got, nan_expected, what=f"NaN positions ({what})")
    ou.assert_exact(
        torch.where(nan_got, torch.zeros_like(expected), got.cpu()),
        torch.where(nan_expected, torch.zeros_like(expected), expected),
        what=what,
    )


def build_inputs(tokens, num_experts, *, routes, map_dtype, pdtype, special, dev, seed=0):
    """``routes`` routed experts per row (a row's count varies around it by -1..+1 when
    ``routes`` is ``"mixed"``), at random experts; probabilities random, or with special values
    (signed zeros, subnormals, Inf, NaN) at random entries."""
    g = ou.make_generator(ou.DEFAULT_SEED + seed)
    if routes == "mixed":
        counts = torch.randint(0, num_experts + 1, (tokens,), generator=g)
    else:
        counts = torch.full((tokens,), min(routes, num_experts), dtype=torch.int64)
    order = torch.rand(tokens, num_experts, generator=g).argsort(dim=1)
    routed = torch.arange(num_experts)[None, :] < counts[:, None]
    routing_map = torch.zeros(tokens, num_experts, dtype=torch.bool)
    routing_map.scatter_(1, order, routed)
    probs = torch.rand(tokens, num_experts, generator=g)
    if special:
        info = torch.finfo(pdtype)
        values = torch.tensor(
            [-0.0, 0.0, info.tiny / 4, -info.tiny / 2, float("inf"), -float("inf"), float("nan")]
        )
        pick = torch.rand(tokens, num_experts, generator=g) < 0.3
        which = torch.randint(0, values.numel(), (tokens, num_experts), generator=g)
        probs = torch.where(pick, values[which], probs)
    return probs.to(dtype=pdtype, device=dev), routing_map.to(dtype=map_dtype, device=dev)


def run_and_check(tokens, num_experts, topk, *, routes, map_dtype, pdtype, special, seed=0):
    dev = rr.require_cuda()
    topk = num_experts if topk == "E" else min(topk, num_experts)
    probs, routing_map = build_inputs(
        tokens,
        num_experts,
        routes=routes,
        map_dtype=map_dtype,
        pdtype=pdtype,
        special=special,
        dev=dev,
        seed=seed,
    )
    before = (probs.clone(), routing_map.clone())
    with torch.no_grad(), rr.poisoned_outputs():
        weights, indices = ra.routing_map_to_mok_inputs(probs, routing_map, topk)
    what = (
        f"T={tokens} E={num_experts} topk={topk} routes={routes} map={map_dtype} probs={pdtype}"
        f" special={special}"
    )
    assert weights.dtype == torch.float32 and weights.shape == (tokens, topk), what
    assert indices.dtype == torch.int32 and indices.shape == (tokens, topk), what
    ref_w, ref_i = reference(probs, routing_map, topk)
    ou.assert_exact(indices, ref_i, what=f"indices ({what})")
    assert_weights(weights, ref_w, what=f"probabilities ({what})")
    ou.assert_exact(probs, before[0], what="probs are read-only")
    ou.assert_exact(routing_map, before[1], what="routing_map is read-only")


_VARIANTS = ou.pairwise_cases(
    {
        "num_experts": [1, 2, 3, 7, 8, 9, 64, 127, 128, 129, 256, 511, 512, 513, 1024],
        "topk": [1, 2, 3, 6, 8, "E"],
        "routes": [0, 1, 6, 8, 9, "mixed", 1024],
        "map_dtype": [torch.bool, torch.uint8, torch.int32, torch.int64],
        "pdtype": [torch.float32, torch.bfloat16, torch.float16],
        "special": [False, True],
        "tokens": [1, 2, 37, 300],
    }
)


def _variant_id(c):
    return (
        f"E{c['num_experts']}-topk{c['topk']}-routes{c['routes']}-{str(c['map_dtype'])[6:]}"
        f"-{str(c['pdtype'])[6:]}-sp{int(c['special'])}-T{c['tokens']}"
    )


@pytest.mark.parametrize("case", _VARIANTS, ids=[_variant_id(c) for c in _VARIANTS])
def test_variants_pairwise(case):
    """Every compiled variant, route count and dtype, pairwise (module docstring)."""
    run_and_check(
        case["tokens"],
        case["num_experts"],
        case["topk"],
        routes=case["routes"],
        map_dtype=case["map_dtype"],
        pdtype=case["pdtype"],
        special=case["special"],
    )


def test_no_tokens():
    """``T = 0``: empty ``[0, topk]`` outputs."""
    dev = rr.require_cuda()
    probs = torch.zeros(0, 16, device=dev)
    weights, indices = ra.routing_map_to_mok_inputs(probs, probs != 0, 4)
    assert weights.shape == (0, 4) and indices.shape == (0, 4)
    assert weights.dtype == torch.float32 and indices.dtype == torch.int32


@pytest.mark.parametrize(
    "probs_kind,map_kind,topk,error",
    [
        pytest.param("ok", "ok", 0, ValueError, id="topk-0"),
        pytest.param("ok", "ok", 17, ValueError, id="topk-above-E"),
        pytest.param("ok", "ok", 2.0, ValueError, id="topk-float"),
        pytest.param("3d", "3d", 2, ValueError, id="three-dims"),
        pytest.param("ok", "short", 2, ValueError, id="shape-mismatch"),
        pytest.param("cpu", "cpu", 2, ValueError, id="cpu"),
        pytest.param("strided", "ok", 2, ValueError, id="non-contiguous"),
        pytest.param("fp64", "ok", 2, TypeError, id="fp64-probs"),
    ],
)
def test_rejects_invalid_inputs(probs_kind, map_kind, topk, error):
    """Violated input conditions raise before any launch."""
    dev = rr.require_cuda()
    probs = {
        "ok": lambda: torch.rand(8, 16, device=dev),
        "3d": lambda: torch.rand(2, 4, 16, device=dev),
        "cpu": lambda: torch.rand(8, 16),
        "strided": lambda: torch.rand(16, 8, device=dev).t(),
        "fp64": lambda: torch.rand(8, 16, device=dev, dtype=torch.float64),
    }[probs_kind]()
    routing_map = {
        "ok": lambda: torch.ones(8, 16, dtype=torch.bool, device=dev),
        "3d": lambda: torch.ones(2, 4, 16, dtype=torch.bool, device=dev),
        "short": lambda: torch.ones(7, 16, dtype=torch.bool, device=dev),
        "cpu": lambda: torch.ones(8, 16, dtype=torch.bool),
    }[map_kind]()
    with pytest.raises(error):
        ra.routing_map_to_mok_inputs(probs, routing_map, topk)


@pytest.mark.parametrize(
    "tokens,num_experts,topk",
    [
        pytest.param(8192, 256, 8, id="dsv3"),
        pytest.param(8192, 128, 6, id="nanov3"),
        pytest.param(16384, 512, 10, id="E512"),
    ],
)
def test_production_sizes(tokens, num_experts, topk):
    """Router-shaped maps (exactly ``topk`` routes per token) at production sizes."""
    run_and_check(
        tokens,
        num_experts,
        topk,
        routes=topk,
        map_dtype=torch.bool,
        pdtype=torch.float32,
        special=False,
    )


@pytest.mark.sanitizer
def test_sanitizer_small_case():
    """Small representative case for the compute-sanitizer memcheck pass."""
    run_and_check(
        37, 129, 6, routes="mixed", map_dtype=torch.bool, pdtype=torch.bfloat16, special=True
    )


@pytest.mark.large
def test_int32_row_offsets():
    """Suspected finding O1-M1: the row offset ``token * E`` of the probabilities and the map is
    int32. With ``E = 256`` and ``2^23 + 1`` tokens the last token's offset is ``2^31``, which
    wraps to ``-2^31``.

    Both inputs sit behind a ``2^31``-element front guard in one allocation (map guard all
    routed, probability guard NaN), so a wrapped read lands in the guard and changes the result
    instead of faulting. Token ``t`` routes to experts ``(t + 32 j) % 256``, ``j < 8``, with
    probability ``e + 1``, so the expected outputs are computed in closed form. Expected to fail
    on origin/main (the last token reads the guard). About 22 GiB; opt-in ``large``.
    """
    dev = rr.require_cuda()
    num_experts, topk, guard = 256, 8, 2**31
    tokens = guard // num_experts + 1
    ou.require_free_memory((guard + tokens * num_experts) * 5 + (4 << 30))
    map_buf = torch.ones(guard + tokens * num_experts, dtype=torch.bool, device=dev)
    probs_buf = torch.full((guard + tokens * num_experts,), float("nan"), device=dev)
    routing_map = map_buf[guard:].view(tokens, num_experts)
    probs = probs_buf[guard:].view(tokens, num_experts)
    probs.copy_(torch.arange(1, num_experts + 1, device=dev, dtype=torch.float32).expand_as(probs))
    token = torch.arange(tokens, device=dev)[:, None]
    experts = (token + 32 * torch.arange(topk, device=dev)[None, :]) % num_experts
    routing_map.zero_()
    routing_map.scatter_(1, experts, True)
    weights, indices = ra.routing_map_to_mok_inputs(probs, routing_map, topk)
    expected = experts.sort(dim=1).values
    for lo in range(0, tokens, 1 << 20):
        hi = min(tokens, lo + (1 << 20))
        ou.assert_exact(indices[lo:hi], expected[lo:hi], what=f"indices of tokens {lo}..{hi}")
        ou.assert_exact(
            weights[lo:hi], (expected[lo:hi] + 1).float(), what=f"weights of tokens {lo}..{hi}"
        )
