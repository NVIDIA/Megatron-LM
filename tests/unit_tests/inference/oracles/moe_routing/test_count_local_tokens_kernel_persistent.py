# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_count_local_tokens_kernel_persistent`` (megatron/core/inference/moe/permute.py).

Launcher: ``compute_local_tokens_per_expert(..., persistent=True, pre_initialized_output_buffer=)``.
Its production caller is the default vLLM BF16 MoE path
(``vllm_fused_moe._moe_align_block_size_cuda_graphable``), which zeroes a fixed counter buffer
inside the same CUDA graph and replays it with ``valid`` changing every step.

Contract (from the launcher's documented API and that caller, written before reading the body):

* The same counts as the non-persistent kernel: for each local expert ``e``,
  ``counts[e] = |{(t, k) : t < min(valid, rows), routing_map[t, k] - start == e}|``, ``-1``
  and other ranks' ids skipped, duplicates counted per pair, rows ``>= valid`` and anything past
  the map's ``rows`` never read.
* The persistent grid (one program per SM, each looping over its share of 1024-pair blocks,
  the work bound read from ``valid`` on device) is an implementation choice: the result must
  not depend on the SM count.
* Output: ``[E_l]`` int32, launcher-zeroed or caller-zeroed (counts added into it), nothing
  outside it written. Graph replay: each replay of ``zero; count`` gives the counts for the
  ``valid`` current at replay.
* Tolerance class: **exact** (integer counts).

Oracle: ``torch.bincount`` of the live pairs on CPU (``routing_reference.local_counts``);
metamorphic: invariance to the SM count (``permute._NUM_SMS`` monkeypatched), and every graph
replay equal to the bincount at that replay's ``valid``.

Structure (read after the contract): one compiled variant per ``E_l`` (constexpr) and map
dtype; ``BLOCK_SIZE = 1024``; grid ``num_sms`` (cached process-wide from the device);
``total_blocks = cdiv(valid * topk, 1024)``, ``blocks_per_cta = cdiv(total_blocks, num_sms)``.
Inputs therefore cover ``num_sms`` in {1, 2, total_blocks-1, total_blocks, total_blocks+1, 132
(H100), 1000} against ``valid * topk`` at block boundaries, ``valid`` in {0, 1, rows-1, rows},
the ``E_l`` / slice / topk / dtype variants pairwise, ``-1`` ids, production sizes and, opt-in
(``large``), live pair counts at the int32 limits (the ``cdiv`` adds 1023 first) and a map of
more than 2^31 pairs behind a front guard.

Memory: canary-tailed routing maps (reads past the rows change a count), guard bands around the
caller-zeroed output, the map checked unchanged; one ``sanitizer`` case.

Known finding F1 (audit family 01): with ``valid > rows`` this variant reads
``(valid - rows) * topk`` entries past the map, with no grid cap.
``test_valid_above_rows_counts_only_the_map_rows`` and
``test_valid_above_rows_in_cuda_graph_replay`` therefore **fail on origin/main** and pass with
#8563 (``fix/d3-s6-mtp-moe-metadata``), which bounds the count kernels by the routing-map rows.
"""

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.moe_routing import routing_reference as rr

permute = pytest.importorskip("megatron.core.inference.moe.permute")
pytest.importorskip("triton")

H100_SMS = 132


@pytest.fixture(autouse=True)
def _sm_count_without_cuda(monkeypatch):
    """The launcher reads the SM count from the CUDA device; give the interpreter H100's."""
    if not torch.cuda.is_available():
        monkeypatch.setattr(permute, "_NUM_SMS", H100_SMS)


def _count(routing_map, start, num_local, valid, out=None):
    return permute.compute_local_tokens_per_expert(
        routing_map, start, num_local, valid, persistent=True, pre_initialized_output_buffer=out
    )


def _check(routing_map, start, num_local, valid_n, *, what, dev, guarded=True):
    before = routing_map.clone()
    out = None
    if guarded:
        out = ou.guarded_empty((num_local,), torch.int32, dev, poison=False)
        out.tensor.zero_()
    counts = _count(
        routing_map,
        start,
        num_local,
        rr.valid_tensor(valid_n, dev),
        out=out.tensor if guarded else None,
    )
    if guarded:
        ou.check_guards(out=out)
    ou.assert_exact(
        counts, rr.local_counts(routing_map, valid_n, start, num_local), what=f"counts ({what})"
    )
    ou.assert_exact(routing_map, before, what="routing_map is read-only")


def _sm_cases():
    cases = []
    topk = 8
    rows = 5 * rr.COUNT_BLOCK // topk  # 5 blocks of pairs
    for valid in [rows, rows - 1, 3 * rr.COUNT_BLOCK // topk + 1]:
        blocks = -(-valid * topk // rr.COUNT_BLOCK)
        for sms in sorted({1, 2, blocks - 1, blocks, blocks + 1, H100_SMS, 1000} - {0}):
            cases.append(pytest.param(rows, topk, valid, sms, id=f"v{valid}[{blocks}blk]-sms{sms}"))
    return cases


@pytest.mark.parametrize("rows,topk,valid,num_sms", _sm_cases())
def test_counts_do_not_depend_on_sm_count(monkeypatch, rows, topk, valid, num_sms):
    """``num_sms`` below / at / above the number of 1024-pair blocks, and the H100 count."""
    dev = rr.device()
    monkeypatch.setattr(permute, "_NUM_SMS", num_sms)
    g = ou.make_generator()
    num_local, start = 16, 32
    head = rr.boundary_expert_ids(rows, topk, 64, start, num_local, generator=g, dev=dev)
    routing_map, _ = rr.with_canary_tail(head, rr.COUNT_BLOCK, start)
    _check(routing_map, start, num_local, valid, what=f"num_sms={num_sms}", dev=dev)


def _valid_cases():
    return [
        pytest.param(topk, valid, id=f"topk{topk}-v{valid}[pairs{valid * topk}]")
        for topk in (1, 6, 8)
        for valid in [0, 1] + rr.rows_around_pairs(rr.COUNT_BLOCK, topk)
    ]


@pytest.mark.parametrize("num_sms", [1, 3, H100_SMS])
@pytest.mark.parametrize("topk,valid", _valid_cases())
def test_valid_at_block_boundaries(monkeypatch, topk, valid, num_sms):
    """``valid * topk`` just below / at / above one and two blocks inside a 3-block map."""
    dev = rr.device()
    monkeypatch.setattr(permute, "_NUM_SMS", num_sms)
    g = ou.make_generator()
    rows = 3 * rr.COUNT_BLOCK // topk
    num_local, start = 8, 8
    head = rr.boundary_expert_ids(rows, topk, 32, start, num_local, generator=g, dev=dev)
    routing_map, _ = rr.with_canary_tail(head, rr.COUNT_BLOCK, start)
    _check(routing_map, start, num_local, valid, what=f"valid={valid}", dev=dev)


_VARIANTS = ou.pairwise_cases(
    {
        "num_local": [1, 2, 7, 16, 32, 128],
        "slice": ["first", "second", "last"],
        "topk": [1, 2, 6, 8, 16],
        "dtype": [torch.int64, torch.int32],
        "valid": ["rows", "rows-1", "half", "0", "1"],
        "output": ["launcher", "caller"],
    }
)


@pytest.mark.parametrize(
    "case",
    _VARIANTS,
    ids=[
        f"E{c['num_local']}-{c['slice']}-topk{c['topk']}-{str(c['dtype'])[6:]}-v{c['valid']}"
        f"-{c['output']}"
        for c in _VARIANTS
    ],
)
def test_variants_pairwise(case):
    """Every ``E_l`` variant x slice x topk x map dtype x live rows x output buffer, pairwise.

    Runs at the device's own SM count (the production launch); ids include ``-1`` and the
    slice edges.
    """
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
    _check(
        routing_map,
        start,
        num_local,
        valid,
        what=str(case),
        dev=dev,
        guarded=case["output"] == "caller",
    )


@pytest.mark.parametrize(
    "kind", ["all-on-first", "all-on-last", "none-local", "all-masked", "masked-tail-rows"]
)
def test_degenerate_routing(kind):
    """Atomic contention on one expert, nothing local, ``-1`` ids (whole map or padding rows)."""
    dev = rr.device()
    g = ou.make_generator()
    rows, topk, num_local, start = 2 * rr.COUNT_BLOCK + 3, 4, 8, 8
    if kind == "masked-tail-rows":
        head = rr.topk_distinct_ids(rows, topk, 32, masked_rows=rows // 3, generator=g, dev=dev)
    else:
        value = {
            "all-on-first": start,
            "all-on-last": start + num_local - 1,
            "none-local": start + num_local,
            "all-masked": -1,
        }[kind]
        head = torch.full((rows, topk), value, dtype=torch.int64, device=dev)
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
    """Production shapes at the device's SM count, CUDA-graph padding rows masked to ``-1``."""
    dev = rr.device()
    g = ou.make_generator()
    num_local = num_experts // ep
    start = (ep // 2) * num_local
    masked = rows // 7
    head = rr.topk_distinct_ids(rows, topk, num_experts, masked_rows=masked, generator=g, dev=dev)
    routing_map, _ = rr.with_canary_tail(head, rr.COUNT_BLOCK, start)
    valid = rows if live == "all" else rows - masked - 5
    _check(routing_map, start, num_local, valid, what=f"{rows}x{topk}", dev=dev)


@pytest.mark.sanitizer
def test_sanitizer_small_case():
    """Small representative case for the compute-sanitizer memcheck pass."""
    dev = rr.device()
    g = ou.make_generator()
    rows, topk, num_local, start = 300, 6, 8, 8
    head = rr.boundary_expert_ids(rows, topk, 32, start, num_local, generator=g, dev=dev)
    _check(head, start, num_local, rows - 7, what="sanitizer", dev=dev)


def _capture_zero_and_count(routing_map, start, num_local, valid):
    """Capture the production sequence: zero the fixed counter buffer, then count into it."""
    out = ou.guarded_empty((num_local,), torch.int32, "cuda", poison=False)
    out.tensor.zero_()
    _count(routing_map, start, num_local, valid, out=out.tensor)  # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.tensor.zero_()
        _count(routing_map, start, num_local, valid, out=out.tensor)
    return graph, out


@pytest.mark.parametrize("kind", ["mixed", "masked-padding"])
def test_cuda_graph_replay_follows_valid(kind):
    """One capture, replays with ``valid`` in {rows, 0, 1, 17, block boundaries, rows-1}.

    The default vLLM path's pattern: a fixed caller-zeroed buffer re-zeroed inside the graph.
    """
    dev = rr.require_cuda()
    g = ou.make_generator()
    rows, topk, num_local, start = 300, 8, 8, 8
    if kind == "mixed":
        head = rr.boundary_expert_ids(rows, topk, 32, start, num_local, generator=g, dev=dev)
    else:
        head = rr.topk_distinct_ids(rows, topk, 32, masked_rows=100, generator=g, dev=dev)
    routing_map, _ = rr.with_canary_tail(head, rr.COUNT_BLOCK, start)
    valid = rr.valid_tensor(rows, dev)
    graph, out = _capture_zero_and_count(routing_map, start, num_local, valid)
    for v in [rows, 0, 1, 17, 127, 128, 129, 255, 256, 257, rows - 1, 35, rows]:
        valid.fill_(v)
        graph.replay()
        torch.cuda.synchronize()
        ou.check_guards(out=out)
        ou.assert_exact(
            out.tensor, rr.local_counts(routing_map, v, start, num_local), what=f"replay valid={v}"
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

    Fails on origin/main: the persistent kernel derives its work from ``valid`` alone and reads
    ``(valid - rows) * topk`` entries of the canary tail. Passes with #8563.
    """
    dev = rr.device()
    g = ou.make_generator()
    num_local, start = 8, 0
    head = rr.topk_distinct_ids(rows, topk, num_local - 1, generator=g, dev=dev)
    routing_map, _ = rr.with_canary_tail(head, extra + rr.COUNT_BLOCK, num_local - 1)
    _check(routing_map, start, num_local, rows + extra, what="valid > rows", dev=dev)


def test_valid_above_rows_in_cuda_graph_replay():
    """F1 in its production form: a graph captured for an MTP-sized map replayed with the
    main forward's larger ``valid``. Fails on origin/main; passes with #8563."""
    dev = rr.require_cuda()
    g = ou.make_generator()
    rows, topk, num_local, start = 12, 6, 8, 0
    head = rr.topk_distinct_ids(rows, topk, num_local - 1, generator=g, dev=dev)
    routing_map, _ = rr.with_canary_tail(head, 4096, num_local - 1)
    valid = rr.valid_tensor(rows, dev)
    graph, out = _capture_zero_and_count(routing_map, start, num_local, valid)
    for v in [rows, rows + 1, 4 * rows, 4000, rows]:
        valid.fill_(v)
        graph.replay()
        torch.cuda.synchronize()
        ou.check_guards(out=out)
        ou.assert_exact(
            out.tensor,
            rr.local_counts(routing_map, v, start, num_local),
            what=f"replay valid={v} rows={rows}",
        )


@pytest.mark.parametrize("rows", rr.int32_pair_count_params(8))
def test_int32_pair_count(rows):
    """Live pair counts ``valid * topk`` at the int32 limits (opt-in ``large``; 16 GiB map).

    Suspected finding O1-N1 (GPU-confirmed for the non-persistent kernel at 2^31 pairs, dfw
    21220533): ``valid * topk`` is int32, and so is ``cdiv``'s ``valid_pairs + 1023``. Expected
    on origin/main and with #8563: ``count-fits`` passes; ``cdiv-wraps`` (2^31 - 8 pairs) and
    ``count-wraps`` (2^31 pairs) count nothing (``total_blocks`` < 0). Unreachable at
    production sizes (2^31 pairs is about 268M routing rows at top-8).
    """
    rr.check_int32_pair_count(_count, rows, dev=rr.require_cuda())


@pytest.mark.large
def test_int32_block_offsets():
    """Control for O1-N2: a map of 2^31 + 8 pairs behind a 2^31-entry front guard, 1000 live
    rows. This variant bounds its blocks by ``valid`` (``block_id * 1024`` stays small), so it
    must count only the live rows; expected to pass on origin/main. 32 GiB, opt-in ``large``.
    """
    rr.check_int32_block_offsets(_count, dev=rr.require_cuda())
