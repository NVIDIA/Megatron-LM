# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU checks of the comparison helpers of the DeepGEMM GPU tests.

``test_dsa_deepgemm_scorer_gpu.py`` and ``test_dsa_deepgemm_layer_gpu.py`` skip without SM 10.x
GPUs, so their bounds, selection checks and multi-rank failure handling are checked here, the
bounds against CPU emulations of the error models they encode:

* the exact BF16 and FP32 spacings and the FP32 ulp distance;
* G10's slot-order bound on the selected-score payload, against an emulation of the cuDNN FE
  1.31 recompute's FP32 row sum under random slot orders, and the chunking defect it must catch
  (a row normalized by another row's sum);
* the CUDA graph layer test's dkey bounds, against an emulation of cuDNN's atomic FP32 dKV sum,
  the BF16 partials and the 2-rank BF16 reduce-scatter, and the defects they must catch (one
  query's contribution moved to another key row, a stale inverse key reorder, a 2x error on 1%
  of the elements, a reduce-scatter that summed another rank's rows);
* the distinct-key checks, the layer test's top-k recorder (pinned buffers allocated before
  the capture, count-based slot check), its key-partial hook and its collected failures, which
  raise on every rank.
"""

import math

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import dsa, dsa_cudnn_kernels
from tests.unit_tests.transformer.experimental_attention_variant import (
    test_dsa_deepgemm_layer_gpu as layer_gpu,
)
from tests.unit_tests.transformer.experimental_attention_variant import (
    test_dsa_deepgemm_scorer_gpu as scorer_gpu,
)

_FLT_MIN = torch.finfo(torch.float32).tiny


# ------------------------------------------------------------------------------ spacings
def test_bf16_ulp_is_the_spacing_of_bf16_numbers():
    # Zero, the subnormals and every finite positive BF16 value below the largest.
    bits = torch.arange(0, 0x7F7F, dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16)
    spacing = (bits + 1).view(torch.bfloat16).float() - values.float()
    assert torch.equal(layer_gpu._bf16_ulp(values), spacing)
    assert torch.equal(layer_gpu._bf16_ulp(-values), spacing)
    assert layer_gpu._bf16_ulp(torch.tensor([704.0, 352.0, 1.0])).tolist() == [4.0, 2.0, 2**-7]


def test_fp32_ulp_is_the_spacing_and_the_ulp_distance_counts_steps():
    generator = torch.Generator().manual_seed(1)
    bits = torch.cat(
        [
            torch.arange(0, 4096, dtype=torch.int64),  # zero and subnormals
            torch.randint(0, 0x7F7FFFFF, (100000,), generator=generator),
        ]
    ).to(torch.int32)
    values = bits.view(torch.float32)
    following = (bits + 1).view(torch.float32)
    assert torch.equal(scorer_gpu._fp32_ulp(values), following - values)
    assert torch.equal(scorer_gpu._fp32_ulp(-values), following - values)
    ones = torch.ones(bits.shape, dtype=torch.int64)
    assert torch.equal(scorer_gpu._fp32_ulp_distance(following, values), ones)
    assert torch.equal(scorer_gpu._fp32_ulp_distance(-following, -values), ones)
    zero = torch.zeros(1)
    assert scorer_gpu._fp32_ulp_distance(zero, -zero).item() == 0
    smallest = torch.tensor([2.0**-149])
    assert scorer_gpu._fp32_ulp_distance(smallest, -smallest).item() == 2


# ------------------------------------------------------------------- G10 payload bound
def _recompute_row_sum(e):
    """The cuDNN FE 1.31 recompute's FP32 row sum of ``e`` (``[rows, 2048]``, indexer epilogue).

    Thread ``lane`` of warp ``w`` holds slots ``32 w + lane + 128 i`` (i = 0..15) and sums them
    in that order; a 32-lane xor butterfly sums each warp; the 4 warp sums are added in turn.
    """
    rows = e.size(0)
    per_thread = e.view(rows, 16, 4, 32)
    acc = torch.zeros(rows, 4, 32)
    for slot in range(16):
        acc = acc + per_thread[:, slot]
    lanes = torch.arange(32)
    for offset in (16, 8, 4, 2, 1):
        acc = acc + acc[:, :, lanes ^ offset]
    warps = acc[:, :, 0]
    return ((warps[:, 0] + warps[:, 1]) + warps[:, 2]) + warps[:, 3]


def _recompute_payload(logits, row_sum=_recompute_row_sum):
    """``log(max(p, FLT_MIN))`` with ``p = exp2((s - max) log2 e) * (1 / sum)``, all in FP32."""
    e = torch.exp2((logits - logits.max(dim=-1, keepdim=True).values) * math.log2(math.e))
    p = e * (1.0 / row_sum(e)).unsqueeze(-1)
    return torch.log(p.clamp_min(_FLT_MIN))


@pytest.mark.parametrize(
    "spread, lead", [(0.5, 0.0), (2.0, 0.0), (4.0, 12.0), (8.0, 0.0), (30.0, 0.0)]
)
def test_log_prob_order_bound_covers_the_recompute_slot_orders(spread, lead):
    """The same keys in other slot orders stay within the bound, including where ``|log p|``
    is near 0 (a dominant key), where the difference is many FP32 ulps of ``log p``."""
    generator = torch.Generator().manual_seed(int(spread * 10 + lead))
    logits = torch.randn(256, 2048, generator=generator) * spread
    logits[:, 0] += lead
    payload = _recompute_payload(logits)
    for _ in range(4):
        order = torch.argsort(torch.rand(256, 2048, generator=generator), dim=-1)
        reordered = _recompute_payload(logits.gather(1, order))
        expected = payload.gather(1, order)
        diff = (reordered - expected).abs()
        assert bool((diff <= scorer_gpu._log_prob_order_bound(reordered, expected)).all())


def test_log_prob_order_bound_catches_another_rows_sum_and_a_small_sum_error():
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(256, 2048, generator=generator) * 2
    payload = _recompute_payload(logits)
    for row_sum in (
        lambda e: _recompute_row_sum(e).roll(1),  # a chunk normalized by another row's sum
        lambda e: _recompute_row_sum(e) * (1 + 2.0**-15),  # 8 times the slot-order bound
    ):
        wrong = _recompute_payload(logits, row_sum=row_sum)
        over = (wrong - payload).abs() > scorer_gpu._log_prob_order_bound(wrong, payload)
        assert bool(over.all())


# --------------------------------------------------------------------- dkey bounds
_ROWS_PER_RANK, _D, _ADDS = 16, 576, 256


def _contributions(case, generator):
    """``[2 ranks, adds, 2 * _ROWS_PER_RANK key rows, D]`` FP32 dKV contributions."""
    shape = (2, _ADDS, 2 * _ROWS_PER_RANK, _D)
    if case == "coherent":
        return torch.rand(shape, generator=generator) * 2.7
    if case == "mixed":
        return torch.randn(shape, generator=generator) * 3 + 0.5
    if case == "zero-mean":
        return torch.randn(shape, generator=generator) * 3
    if case == "wide":
        return torch.randn(shape, generator=generator) * torch.logspace(-6, 1, _D)
    # Large partials of opposite signs on the two ranks: dkey is about 0.
    base = torch.randn(shape[1:], generator=generator) * 10
    noise = 1 + 1e-3 * torch.randn(shape[1:], generator=generator)
    return torch.stack([base, -base * noise])


def _atomic_partials(contributions, generator):
    """Each rank's BF16 partial: its FP32 sum of the adds in a random (atomic) order."""
    partials = []
    for per_rank in contributions:
        total = torch.zeros(per_rank.shape[1:])
        for add in torch.randperm(per_rank.size(0), generator=generator).tolist():
            total = total + per_rank[add]
        partials.append(total.bfloat16())
    return torch.stack(partials)


def _reduce_scatter(partials, rank):
    """Rank ``rank``'s dkey: the 2-rank BF16 reduce-scatter of its rows (one rounding)."""
    rows = slice(rank * _ROWS_PER_RANK, (rank + 1) * _ROWS_PER_RANK)
    return (partials[0, rows].float() + partials[1, rows].float()).bfloat16()


def _dkey_failures(replay, eager, replay_dkey=None):
    """The layer test's three dkey checks for both ranks: failure messages (empty: passed)."""
    failures = []
    for rank in range(2):
        failure, _ = layer_gpu._bounded_mismatch(
            replay[rank],
            eager[rank],
            layer_gpu._partial_order_bound(replay[rank], eager[rank]),
            "p",
        )
        failures.append(failure)
        rows = slice(rank * _ROWS_PER_RANK, (rank + 1) * _ROWS_PER_RANK)
        got = _reduce_scatter(replay, rank) if replay_dkey is None else replay_dkey[rank]
        want = _reduce_scatter(eager, rank)
        for dkey, partials in ((got, replay), (want, eager)):
            summed = partials[0, rows].float() + partials[1, rows].float()
            failure, _ = layer_gpu._bounded_mismatch(
                dkey, summed, layer_gpu._reduce_scatter_bound(dkey, summed), "rs"
            )
            failures.append(failure)
        diff = sum((replay[r, rows].float() - eager[r, rows].float()).abs() for r in range(2))
        failure, _ = layer_gpu._bounded_mismatch(
            got, want, layer_gpu._dkey_bound(got, want, diff), "dkey"
        )
        failures.append(failure)
    return [failure for failure in failures if failure]


@pytest.mark.parametrize("case", ["coherent", "mixed", "zero-mean", "wide", "cancelling"])
def test_dkey_bounds_cover_atomic_orders_and_catch_dkv_defects(case):
    generator = torch.Generator().manual_seed(17)
    contributions = _contributions(case, generator)
    eager = _atomic_partials(contributions, generator)
    replay = _atomic_partials(contributions, generator)
    assert not _dkey_failures(replay, eager)
    # The bound's assumption: sum |c| <= 64 M_row for each element of a key row.
    row_scale = torch.maximum(eager.float().abs(), replay.float().abs()).amax(dim=-1)
    assert float((contributions.abs().sum(dim=1).amax(dim=-1) / row_scale).max()) <= 64

    # One query's contribution on rank 0 moved from key row 3 to key row 21 (a dKV scatter
    # defect that leaves dquery and the output alone).
    moved = contributions.clone()
    moved[0, 7, 21] += moved[0, 7, 3]
    moved[0, 7, 3] = 0
    assert _dkey_failures(_atomic_partials(moved, generator), eager)
    # One element of rank 0's dkey rounded the wrong way (away from a sum that is not a tie).
    rows = slice(0, _ROWS_PER_RANK)
    summed = replay[0, rows].float() + replay[1, rows].float()
    dkey = _reduce_scatter(replay, 0)
    ulp = layer_gpu._bf16_ulp(dkey)
    clear = (dkey.float() > 0) & ((dkey.float() - summed).abs() < ulp / 4)
    element = int(clear.flatten().nonzero()[0])
    misrounded = dkey.clone().flatten()
    step = ulp.flatten()[element] * (1 if summed.flatten()[element] > misrounded[element] else -1)
    misrounded[element] = (misrounded[element].float() - step).bfloat16()
    misrounded = misrounded.view_as(dkey)
    assert _dkey_failures(replay, eager, replay_dkey=[misrounded, _reduce_scatter(replay, 1)])
    # A stale inverse key reorder: rank 0's partial with two key rows swapped.
    swapped = replay.clone()
    swapped[0, [5, 9]] = swapped[0, [9, 5]]
    assert _dkey_failures(swapped, eager)
    # A 2x error on 1% of rank 1's partial.
    scaled = replay.clone()
    picked = torch.randperm(scaled[1].numel(), generator=generator)[: scaled[1].numel() // 100]
    scaled[1].view(-1)[picked] *= 2
    assert _dkey_failures(scaled, eager)
    # A reduce-scatter that summed rank 1's rows for rank 0 (and rank 0's for rank 1).
    crossed = [_reduce_scatter(replay, 1), _reduce_scatter(replay, 0)]
    assert _dkey_failures(replay, eager, replay_dkey=crossed)


def test_bounded_mismatch_fails_non_finite_values():
    value = torch.tensor([1.0, 2.0])
    infinite = torch.tensor([1.0, float("inf")])
    unbounded = torch.full((2,), math.inf)
    for actual, expected in ((infinite, value), (value, infinite), (infinite, infinite)):
        failure, _ = layer_gpu._bounded_mismatch(actual, expected, unbounded, "x")
        assert failure is not None
    failure, _ = layer_gpu._bounded_mismatch(value, value + 0.5, torch.full((2,), 0.5), "x")
    assert failure is None
    failure, _ = layer_gpu._bounded_mismatch(value, value + 0.5, torch.full((2,), 0.25), "x")
    assert failure is not None and "2 elements past the bound" in failure


def test_mismatch_reports_statistics():
    a = torch.randn(4, 5).bfloat16()
    assert layer_gpu._mismatch(a, a.clone(), "x") is None
    b = a.clone()
    b[1, 1] += 1
    assert "1 of 20 elements differ" in layer_gpu._mismatch(b, a, "x")
    b[0, 0] = float("nan")
    assert layer_gpu._mismatch(b, a, "x") is not None
    assert layer_gpu._mismatch(a[:2], a, "x") is not None


# ------------------------------------------------------------------- selection checks
def test_distinct_key_checks_see_a_key_taken_twice():
    indices = torch.tensor([[3, 1, 2, -1], [0, 0, -1, -1], [5, 6, 6, 9]], dtype=torch.int32)
    valid_slots = torch.tensor([4, 4, 3])
    assert layer_gpu._repeated_key_rows(indices, valid_slots) == 2
    # Padding (-1) and slots past a row's valid count are not keys.
    assert layer_gpu._repeated_key_rows(indices, torch.tensor([4, 1, 2])) == 0
    with pytest.raises(AssertionError, match="2 rows repeat a key"):
        scorer_gpu._assert_distinct_keys(indices.unsqueeze(0), valid_slots.unsqueeze(0))
    scorer_gpu._assert_distinct_keys(indices.unsqueeze(0), torch.tensor([[4, 1, 2]]))


@pytest.fixture
def fake_capture(monkeypatch):
    """A switchable ``torch.cuda.is_current_stream_capturing`` for CPU runs."""
    state = {"on": False}
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: state["on"])
    return state


def _fake_topk(logits, lengths, topk_k, topk_backend):
    values, indices = logits.float().topk(topk_k, dim=-1)
    return indices.int(), values


def test_graph_topk_recorder_feeds_preallocated_pinned_buffers(monkeypatch, fake_capture):
    monkeypatch.setattr(dsa_cudnn_kernels, "_deepgemm_logits_topk", _fake_topk)
    recorder = layer_gpu._GraphTopkRecorder(monkeypatch)
    logits = torch.randn(8, 32).bfloat16()
    lengths = torch.full((8,), 32, dtype=torch.int32)
    dsa_cudnn_kernels._deepgemm_logits_topk(logits, lengths, 4, "cudnn")
    selection = recorder.take_eager()

    fake_capture["on"] = True
    with pytest.raises(AssertionError, match="before the capture"):
        dsa_cudnn_kernels._deepgemm_logits_topk(logits, lengths, 4, "cudnn")
    with pytest.raises(AssertionError, match="before the capture"):
        recorder.allocate_pinned(selection)
    fake_capture["on"] = False
    recorder.captured = None
    recorder.allocate_pinned(selection)
    pinned = recorder.pinned
    fake_capture["on"] = True
    assert dsa_cudnn_kernels._deepgemm_logits_topk(logits, lengths, 4, "cudnn") is pinned
    fake_capture["on"] = False
    recorder.pin(selection)
    assert all(torch.equal(a, b) for a, b in zip(pinned, selection[2:]))

    monkeypatch.setattr(dsa_cudnn_kernels, "_deepgemm_logits_topk", _fake_topk)  # unwrap
    other = layer_gpu._GraphTopkRecorder(monkeypatch)
    other.pinned = (torch.empty(8, 5, dtype=torch.int32), torch.empty(8, 5))
    fake_capture["on"] = True
    with pytest.raises(AssertionError, match="pinned"):
        dsa_cudnn_kernels._deepgemm_logits_topk(logits, lengths, 4, "cudnn")


def test_compare_replay_is_tie_aware_counts_slots_and_rejects_repeated_keys(monkeypatch):
    recorder = layer_gpu._GraphTopkRecorder(monkeypatch)
    seen = torch.tensor([[3.0, 3.0, 1.0, 0.0], [4.0, 2.0, 2.0, 1.0]]).bfloat16()
    lengths = torch.tensor([4, 4], dtype=torch.int32)
    eager = (seen, lengths, torch.tensor([[0, 1], [0, 1]], dtype=torch.int32), None)

    def replay(indices):
        recorder.captured = (seen, lengths, torch.tensor(indices, dtype=torch.int32), None)

    replay([[1, 0], [0, 1]])  # the same keys in other slots
    assert recorder.compare_replay(eager, "x") == 0
    replay([[0, 1], [0, 2]])  # the other key tied at row 1's cutoff
    assert recorder.compare_replay(eager, "x") == 1
    replay([[0, 1], [0, 3]])  # a key below the cutoff
    with pytest.raises(AssertionError, match="not ties"):
        recorder.compare_replay(eager, "x")
    replay([[0, -1], [0, 1]])
    with pytest.raises(AssertionError, match="another number of keys"):
        recorder.compare_replay(eager, "x")
    replay([[0, 0], [0, 1]])  # one tied key twice: the same value multiset and count
    with pytest.raises(AssertionError, match="repeat a key"):
        recorder.compare_replay(eager, "x")


def test_key_partials_record_the_gradient_at_the_key_gather(monkeypatch, fake_capture):
    monkeypatch.setattr(
        dsa, "gather_from_sequence_parallel_region", lambda x, *args, **kwargs: torch.cat([x, x])
    )
    key_dim = 576
    partials = layer_gpu._KeyPartials(monkeypatch, key_dim)
    key = torch.randn(layer_gpu._GRAPH_LOCAL_ROWS, 1, 1, key_dim, requires_grad=True)
    gathered = dsa.gather_from_sequence_parallel_region(key, group=None)
    grad = torch.randn_like(gathered)
    gathered.backward(grad)
    recorded = partials.take_eager()
    assert torch.equal(recorded, grad) and recorded.data_ptr() != grad.data_ptr()
    other = torch.randn(layer_gpu._GRAPH_LOCAL_ROWS, 1, 1, 128, requires_grad=True)
    dsa.gather_from_sequence_parallel_region(other, group=None).sum().backward()
    assert not partials.eager
    fake_capture["on"] = True
    dsa.gather_from_sequence_parallel_region(key, group=None).backward(grad)
    fake_capture["on"] = False
    assert torch.equal(partials.captured, grad) and not partials.eager


def test_collected_failures_raise_on_every_rank(monkeypatch):
    peer = {"failures": 0}

    def all_reduce(tensor, group=None):
        tensor += peer["failures"]

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    failures = layer_gpu._CollectedFailures(group=None, device="cpu")
    failures.add(None)
    failures.check(True, "unused")
    assert failures.run(lambda value: value + 1, 1) == 2
    failures.raise_on_any_rank("clean")  # no failure on any rank: no raise

    peer["failures"] = 1  # only the other rank failed
    with pytest.raises(AssertionError, match="1 failed checks on the CP ranks; on this rank: none"):
        failures.raise_on_any_rank("peer")

    peer["failures"] = 0
    boom = AssertionError("boom")

    def failing():
        raise boom

    assert failures.run(failing) is None
    failures.check(False, "count")
    with pytest.raises(AssertionError, match="here: 2 failed checks.*boom; count") as raised:
        failures.raise_on_any_rank("here")
    assert raised.value.__cause__ is boom
