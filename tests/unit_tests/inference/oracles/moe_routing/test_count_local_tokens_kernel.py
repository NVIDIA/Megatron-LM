# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_count_local_tokens_kernel`` (megatron/core/inference/moe/permute.py).

Launcher: ``compute_local_tokens_per_expert(routing_map, local_expert_start, num_local_experts,
valid_tokens, persistent=False, pre_initialized_output_buffer=None)``; called by
``permute_tokens`` and ``permute_and_quantize_mxfp8`` on every torch-backend / MXFP8 MoE layer.

Contract (from the launcher's documented API and its callers, written before reading the body):

* Computes, for each local expert ``e in [0, E_l)``::

      counts[e] = |{(t, k) : t < min(valid, rows), routing_map[t, k] - local_expert_start == e}|

  over the ``[rows, topk]`` routing map. ``valid`` is the int32 device scalar ``valid_tokens``
  (read on device, so the launch is CUDA-graph safe and the grid depends only on ``rows``).
* Inputs: ``routing_map`` ``[rows, topk]`` int64 (the router's output or the AGV buffer; int32
  is accepted too), contiguous, read-only. ``local_expert_start`` host int, ``E_l`` compile-time.
* Output: ``[E_l]`` int32, freshly zeroed by the launcher, or a caller-zeroed contiguous buffer
  (the counts are added into it). Nothing outside ``[0, E_l)`` of it is written.
* Live rows: rows ``>= valid`` are never read; a ``valid`` above ``rows`` counts the ``rows``
  rows of the map and nothing past them. Ids ``-1`` (masked CUDA-graph padding rows) and ids of
  other ranks' experts are not counted. Duplicate ids within a token count once per pair.
* Degenerate: ``valid = 0`` or ``rows = 0`` gives all zeros.
* Tolerance class: **exact** (integer counts).

Oracle: ``torch.bincount`` of the live pairs on CPU (``routing_reference.local_counts``).

Structure (read after the contract): one compiled variant per ``E_l`` (constexpr) and per
routing-map dtype; ``BLOCK_SIZE = 1024`` pairs per program; grid ``cdiv(rows * topk, 1024)``;
no autotune. Inputs therefore cover ``rows * topk`` and ``valid * topk`` at 1024-pair block
boundaries (B-1/B/B+1 and 2B), ``valid`` in {0, 1, rows-1, rows}, ``topk`` in {1 (integer
specialisation), 2, 6, 8, 16}, ``E_l`` in {1, 2, 7, 16, 32, 128}, the first / second / last
local slice of a 4-rank model (``start`` 0, 1, 16 included), ids at the slice edges, all pairs
on one expert (atomic contention), production sizes and, opt-in (``large``), live pair counts
at the int32 limits and a map of more than 2^31 pairs behind a front guard (block offsets).

Memory: every routing map is the head of a canary buffer whose tail routes to a local expert
(a read past the map's rows changes a count); the caller-zeroed output sits between guard
bands; the routing map is checked unchanged. One small case is marked ``sanitizer``.

Known finding F1 (audit family 01): a ``valid`` above ``rows`` reaches this kernel in MTP
forwards on main, and the kernel reads past the routing map (up to the end of its last 1024-pair
block). ``test_valid_above_rows_counts_only_the_map_rows`` therefore **fails on origin/main**
and passes with #8563 (``fix/d3-s6-mtp-moe-metadata``), which bounds the count kernels by the
routing-map rows.
"""

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.moe_routing import routing_reference as rr

permute = pytest.importorskip("megatron.core.inference.moe.permute")
pytest.importorskip("triton")

PERSISTENT = False


def _count(routing_map, start, num_local, valid, out=None):
    return permute.compute_local_tokens_per_expert(
        routing_map,
        start,
        num_local,
        valid,
        persistent=PERSISTENT,
        pre_initialized_output_buffer=out,
    )


def _check(routing_map, start, num_local, valid_n, *, what, dev):
    """Both output paths: the launcher's own zeroed buffer (production) and a caller-zeroed
    buffer between guard bands (catches a count landing outside ``[0, E_l)``)."""
    before = routing_map.clone()
    expected = rr.local_counts(routing_map, valid_n, start, num_local)
    counts = _count(routing_map, start, num_local, rr.valid_tensor(valid_n, dev))
    ou.assert_exact(counts, expected, what=f"counts ({what})")
    assert counts.dtype == torch.int32 and counts.shape == (num_local,)
    out = ou.guarded_empty((num_local,), torch.int32, dev, poison=False)
    out.tensor.zero_()
    _count(routing_map, start, num_local, rr.valid_tensor(valid_n, dev), out=out.tensor)
    ou.check_guards(out=out)
    ou.assert_exact(out.tensor, expected, what=f"counts into a caller buffer ({what})")
    ou.assert_exact(routing_map, before, what="routing_map is read-only")


def _grid_cases():
    cases = []
    for topk in (1, 2, 6, 8):
        for rows in [0, 1] + rr.rows_around_pairs(rr.COUNT_BLOCK, topk):
            for valid in sorted({rows, max(rows - 1, 0)}):
                cases.append(pytest.param(rows, topk, valid, id=f"rows{rows}-topk{topk}-v{valid}"))
    return cases


@pytest.mark.parametrize("rows,topk,valid", _grid_cases())
def test_grid_boundaries(rows, topk, valid):
    """``rows * topk`` just below / at / above one and two 1024-pair blocks, and 0 / 1 rows."""
    dev = rr.device()
    g = ou.make_generator()
    num_local, start = 8, 8
    head = rr.boundary_expert_ids(rows, topk, 32, start, num_local, generator=g, dev=dev)
    routing_map, _ = rr.with_canary_tail(head, rr.COUNT_BLOCK, start)
    _check(routing_map, start, num_local, valid, what=f"rows={rows} topk={topk}", dev=dev)


def _valid_cases():
    return [
        pytest.param(topk, valid, id=f"topk{topk}-v{valid}[pairs{valid * topk}]")
        for topk in (1, 6, 8)
        for valid in [0, 1] + rr.rows_around_pairs(rr.COUNT_BLOCK, topk)
    ]


@pytest.mark.parametrize("topk,valid", _valid_cases())
def test_valid_at_block_boundaries(topk, valid):
    """``valid * topk`` just below / at / above one and two blocks inside a fixed 3-block map."""
    dev = rr.device()
    g = ou.make_generator()
    rows = 3 * rr.COUNT_BLOCK // topk
    num_local, start = 16, 16
    head = rr.boundary_expert_ids(rows, topk, 64, start, num_local, generator=g, dev=dev)
    routing_map, _ = rr.with_canary_tail(head, rr.COUNT_BLOCK, start)
    _check(routing_map, start, num_local, valid, what=f"valid={valid}", dev=dev)


_VARIANTS = ou.pairwise_cases(
    {
        "num_local": [1, 2, 7, 16, 32, 128],
        "slice": ["first", "second", "last"],
        "topk": [1, 2, 6, 8, 16],
        "dtype": [torch.int64, torch.int32],
        "valid": ["rows", "rows-1", "half", "0", "1"],
    }
)


@pytest.mark.parametrize(
    "case",
    _VARIANTS,
    ids=[
        f"E{c['num_local']}-{c['slice']}-topk{c['topk']}-{str(c['dtype'])[6:]}-v{c['valid']}"
        for c in _VARIANTS
    ],
)
def test_variants_pairwise(case):
    """Every ``E_l`` variant x local slice x topk x map dtype x live-row count, pairwise."""
    dev = rr.device()
    g = ou.make_generator()
    num_local, topk = case["num_local"], case["topk"]
    start = dict(rr.counting_slices(num_local))[case["slice"]]
    rows = 777
    valid = {"rows": rows, "rows-1": rows - 1, "half": rows // 2, "0": 0, "1": 1}[case["valid"]]
    head = rr.boundary_expert_ids(
        rows, topk, 4 * num_local, start, num_local, dtype=case["dtype"], generator=g, dev=dev
    )
    routing_map, _ = rr.with_canary_tail(head, rr.COUNT_BLOCK, start)
    _check(routing_map, start, num_local, valid, what=str(case), dev=dev)


@pytest.mark.parametrize(
    "kind", ["all-on-first", "all-on-last", "none-local", "all-masked", "duplicates"]
)
def test_degenerate_routing(kind):
    """Atomic contention on one expert, nothing local, all ``-1``, duplicate ids per token."""
    dev = rr.device()
    rows, topk, num_local, start = 2 * rr.COUNT_BLOCK + 3, 4, 8, 8
    value = {
        "all-on-first": start,
        "all-on-last": start + num_local - 1,
        "none-local": start + num_local,
        "all-masked": -1,
        "duplicates": start + 3,
    }[kind]
    head = torch.full((rows, topk), value, dtype=torch.int64, device=dev)
    if kind == "duplicates":
        head[:, ::2] = start + 5
    routing_map, _ = rr.with_canary_tail(head, rr.COUNT_BLOCK, start)
    _check(routing_map, start, num_local, rows, what=kind, dev=dev)


@pytest.mark.parametrize(
    "rows,topk,num_experts,ep",
    [
        pytest.param(8192, 6, 128, 1, id="nanov3-ep1-prefill"),
        pytest.param(8 * 8192, 8, 256, 8, id="dsv3-ep8-agv"),
        pytest.param(64 * 256, 8, 256, 64, id="dsv3-ep64-decode"),
    ],
)
@pytest.mark.parametrize("live", ["all", "partial"])
def test_production_sizes(rows, topk, num_experts, ep, live):
    """Production shapes: router-shaped maps with CUDA-graph padding rows masked to ``-1``."""
    dev = rr.device()
    g = ou.make_generator()
    num_local = num_experts // ep
    start = (ep // 2) * num_local
    masked = rows // 7
    head = rr.topk_distinct_ids(rows, topk, num_experts, masked_rows=masked, generator=g, dev=dev)
    routing_map, _ = rr.with_canary_tail(head, rr.COUNT_BLOCK, start)
    valid = rows if live == "all" else rows - masked - 5
    _check(routing_map, start, num_local, valid, what=f"{rows}x{topk}", dev=dev)


@pytest.mark.parametrize("rows", [5, 1023, 1025])
def test_caller_zeroed_buffer_guarded(rows):
    """The caller-zeroed output path: counts land in the buffer, nothing outside it changes."""
    dev = rr.device()
    g = ou.make_generator()
    topk, num_local, start = 2, 7, 7
    head = rr.boundary_expert_ids(rows, topk, 28, start, num_local, generator=g, dev=dev)
    routing_map, _ = rr.with_canary_tail(head, rr.COUNT_BLOCK, start)
    out = ou.guarded_empty((num_local,), torch.int32, dev, poison=False)
    out.tensor.zero_()
    result = _count(routing_map, start, num_local, rr.valid_tensor(rows - 1, dev), out=out.tensor)
    assert result.data_ptr() == out.tensor.data_ptr()
    ou.check_guards(out=out)
    ou.assert_exact(out.tensor, rr.local_counts(routing_map, rows - 1, start, num_local))


@pytest.mark.sanitizer
def test_sanitizer_small_case():
    """Small representative case for the compute-sanitizer memcheck pass."""
    dev = rr.device()
    g = ou.make_generator()
    rows, topk, num_local, start = 300, 6, 8, 8
    head = rr.boundary_expert_ids(rows, topk, 32, start, num_local, generator=g, dev=dev)
    out = ou.guarded_empty((num_local,), torch.int32, dev, poison=False)
    out.tensor.zero_()
    _count(head, start, num_local, rr.valid_tensor(rows - 7, dev), out=out.tensor)
    ou.check_guards(out=out)
    ou.assert_exact(out.tensor, rr.local_counts(head, rows - 7, start, num_local))


def test_cuda_graph_replay_follows_valid():
    """Captured once at a fixed buffer shape, replayed with ``valid`` changing every step.

    The production pattern (``permute_tokens`` inside a CUDA graph): the launcher allocates and
    zeroes the output inside the graph; only the device scalar changes between replays.
    """
    dev = rr.require_cuda()
    g = ou.make_generator()
    rows, topk, num_local, start = 300, 8, 8, 8
    head = rr.boundary_expert_ids(rows, topk, 32, start, num_local, generator=g, dev=dev)
    routing_map, _ = rr.with_canary_tail(head, rr.COUNT_BLOCK, start)
    valid = rr.valid_tensor(rows, dev)
    _count(routing_map, start, num_local, valid)  # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        counts = _count(routing_map, start, num_local, valid)
    for v in [rows, 0, 1, 127, 128, 129, 255, 256, 257, rows - 1, 17, rows]:
        valid.fill_(v)
        graph.replay()
        torch.cuda.synchronize()
        ou.assert_exact(
            counts, rr.local_counts(routing_map, v, start, num_local), what=f"replay valid={v}"
        )


@pytest.mark.parametrize(
    "rows,topk,extra",
    [
        pytest.param(6, 2, 4, id="mtp-rows6-topk2"),
        pytest.param(37, 6, 3, id="partial-last-block"),
        pytest.param(170, 6, 1000, id="valid-far-above-rows"),
    ],
)
def test_valid_above_rows_counts_only_the_map_rows(rows, topk, extra):
    """``valid > rows`` (finding F1): only the ``rows`` rows of the map are counted.

    Fails on origin/main: the kernel masks by ``valid * topk`` alone and reads the canary tail
    up to the end of the last 1024-pair block of its grid. Passes with #8563, which bounds both
    count kernels by the routing-map rows.
    """
    dev = rr.device()
    g = ou.make_generator()
    num_local, start = 8, 0
    head = rr.topk_distinct_ids(rows, topk, num_local - 1, generator=g, dev=dev)
    routing_map, _ = rr.with_canary_tail(head, rows + extra + rr.COUNT_BLOCK, num_local - 1)
    _check(routing_map, start, num_local, rows + extra, what="valid > rows", dev=dev)


@pytest.mark.parametrize("rows", rr.int32_pair_count_params(8))
def test_int32_pair_count(rows):
    """Live pair counts ``valid * topk`` at the int32 limits (opt-in ``large``; 16 GiB map).

    Finding O1-N1 (GPU-confirmed, dfw 21220533: 0 instead of 2^26 on every expert at 2^31
    pairs): ``valid * topk`` is int32 and wraps to ``-2^31``, so the mask drops every pair.
    Expected on origin/main and with #8563: ``count-fits`` and ``cdiv-wraps`` pass,
    ``count-wraps`` fails. Unreachable at production sizes (2^31 pairs is about 268M routing
    rows at top-8).
    """
    rr.check_int32_pair_count(_count, rows, dev=rr.require_cuda())


@pytest.mark.large
def test_int32_block_offsets():
    """Finding O1-N2: ``pid * 1024`` is int32, so with a map of more than 2^31 pairs the last
    program's offsets wrap to ``-2^31 .. -2^31 + 1023``, pass the ``< valid * topk`` mask and
    read 16 GiB before the map, whatever ``valid`` is.

    The map (2^31 + 8 pairs, 1000 live rows) sits behind a 2^31-entry front guard routed to
    expert 31, so the stray read is counted instead of faulting. Expected to fail on
    origin/main and with #8563 (``counts[31] = 1024``). On dfw 21220533 the unguarded
    ``2^31 + 8``-pair case faulted with an illegal memory access. 32 GiB, opt-in ``large``.
    """
    rr.check_int32_block_offsets(_count, dev=rr.require_cuda())
