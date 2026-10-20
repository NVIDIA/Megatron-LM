# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``dsa_indexer_topk_backend='deepselect'`` on a GLM-like ``DSAttention`` layer (real kernels).

The unit-test layer suites run at score widths no larger than the top-k, where every call takes
the cuDNN top-k ("top-k covers the whole row"). Here the packed THD indexer scores are at least
65536 columns wide, so the production width rule (:func:`dsa_deepselect_topk.min_score_width`)
sends every top-k of the layer to DeepSelect. Each layout runs one forward/backward of the
layer (GLM-5.2 attention geometry: 64 heads, 512+64 latent, indexer 32x128, top-k 2048, sparse
indexer loss, cuDNN kernels) in production (non-deterministic) mode:

1. ``cudnn``: the reference.
2. ``cudnn`` again: the run-to-run noise of the non-deterministic kernels.
3. ``cudnn`` with every row's selected slots shuffled where the top-k returns them with their
   scores (the sparse-loss path): the noise of a different slot order, which is what a
   different top-k kernel produces (DeepSelect's order is unrelated to cuDNN's). The sparse
   attention and the indexer loss both consume the slots in that order.
4. ``deepselect``: a spy checks that every top-k call of the layer ran DeepSelect (no
   fallback), on the expected ``rows x width`` scores and row stride.
5. ``deepselect`` again: DeepSelect is deterministic, so its raw top-k (slot order included)
   must repeat bitwise.

Per call, the top-k lengths must be equal and each row's selection, sorted by key index, must
select the same keys with bitwise-equal scores. The only allowed difference is the choice among
keys whose score equals the row's cutoff score, which no top-k kernel here specifies: at these
widths cuDNN itself picks other cutoff-tied keys from run to run in a few rows per call (up to
15 of 32768..82000 rows per rank on GB300; DeepSelect differs from cuDNN in up to 16). Such a row attends to another key with an
unrelated attention weight, so its output and query gradient may move by O(that weight)
(peaks up to 0.14 of the output's maximum were seen) and the two key rows trade its gradient.
The production-mode tolerances of ``test_dsa_cudnn_skip_score_fill_layer`` (relative L2 1e-2,
peak 2e-2, indexer loss 1e-3) therefore apply to the output and query gradient on the rows
without a tie, to the key gradient as relative L2 only, and in full to every other gradient and
the indexer loss (:func:`_tie_aware_mismatches`). Runs 2 and 3 must meet the same limits,
which is what shows that they describe the noise of the kernels rather than a wrong selection;
run 5 must meet them on every tensor.

Layouts follow the GLM-5.2 THD recipes (dummy tail sequence, fixed-capacity ``cu_seqlens``),
with the masked score prefill skipped as in production unless noted:

* CP1, three sequences: an unaligned score width (82000 columns, not a multiple of 1024 bytes),
  so the scorer writes into the row-padded ``out=`` buffer that DeepSelect reads in place;
* zigzag CP4, three sequences: rank widths 131072..81920 (the narrowest at the many-rows
  threshold);
* dynamic CP2 in a CP4 world, one sequence per slot;
* dynamic CP1 in a CP4 world, one sequence per slot, with the ``-inf`` prefill kept.

Needs 4 GB300 GPUs (DeepSelect runs only on SM 10.3), a DeepSelect build (the GLM-5.2
``ds_tune`` overlay or upstream ``deep_select`` from 512K columns), the cuDNN frontend DSA and
FlashMLA. Lines starting with ``[deepselect-layer]`` report what each rank compared.
"""

import dataclasses
import functools

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.transformer.experimental_attention_variant import (
    dsa_cudnn_kernels,
    dsa_deepselect_topk,
)
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossLoggingHelper
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_dsa_cudnn_skip_score_fill_layer import (  # noqa: E501
    _PRODUCTION_REL_L2,
    _build_step,
    _initialize_model_parallel,
    _is_topk_indices,
    _production_mismatches,
    _ratio,
    _skip_unless_supported,
    _slot_order_diffs,
    _sort_topk_by_key,
    _worst,
)

_GIB = 1024**3
_TOPK = 2048


@dataclasses.dataclass(frozen=True)
class _Layout:
    """One packed THD microbatch as every rank of the realized CP group sees it.

    The fields up to ``topk`` are the ones the skip-score-fill layer helpers read.
    """

    model_cp: int  # CP size the model parallel state is initialized with
    realized_cp: int  # CP size of the group that runs this microbatch
    dynamic: bool  # realized group is a dynamic-CP subgroup (local_cp_size/cp_group set)
    local_rows: int  # static per-rank slot length (pad target)
    seq_lens: tuple  # real sequences; the dummy tail fills the slot
    max_num_seqs: int  # thd_max_packed_sequences, including the dummy sequence
    topk: int
    skip_fill: bool  # dsa_indexer_skip_masked_score_fill


_LAYOUTS = [
    pytest.param(
        _Layout(1, 1, False, 82000, (40000, 30000, 8000), 5, _TOPK, True),
        id="cp1-3seq-dummy-unaligned",
    ),
    pytest.param(
        _Layout(4, 4, False, 32768, (65536, 40000, 16000), 5, _TOPK, True), id="cp4-3seq-dummy"
    ),
    pytest.param(_Layout(4, 2, True, 57344, (110000,), 2, _TOPK, True), id="dcp2of4-slot"),
    pytest.param(_Layout(4, 1, True, 81920, (80000,), 2, _TOPK, False), id="dcp1of4-slot-prefill"),
]


def _score_width(layout: _Layout, cp_rank: int) -> int:
    """Columns of the packed THD indexer scores on ``cp_rank`` (``_indexer_topk_packed_thd``)."""
    global_rows = layout.local_rows * layout.realized_cp
    if layout.realized_cp == 1:
        return global_rows
    half = global_rows // (2 * layout.realized_cp)
    return max((cp_rank + 1) * half, global_rows - cp_rank * half)


def _cp_rank(layout: _Layout) -> int:
    if layout.dynamic:
        group = parallel_state.get_dynamic_data_context_parallel_groups(
            group_size=layout.realized_cp
        )
        return group.rank()
    return parallel_state.get_context_parallel_rank()


def _require_deepselect_layer() -> None:
    _skip_unless_supported()
    sm = tuple(torch.cuda.get_device_capability())
    if sm not in dsa_deepselect_topk.SUPPORTED_SM:
        pytest.skip(f"DeepSelect runs on SM {sorted(dsa_deepselect_topk.SUPPORTED_SM)} only")
    if dsa_deepselect_topk._load_provider() is None:
        pytest.skip(dsa_deepselect_topk._IMPORT_STATE["error"])
    if torch.cuda.get_device_properties(0).total_memory < 200 * _GIB:
        pytest.skip("needs a GPU with at least 200 GiB (GB300)")


class _DeepSelectSpy:
    """Records every DeepSelect eligibility decision and launch of the DSA top-k."""

    def __init__(self, monkeypatch):
        self.reasons = []
        self.launches = []
        real_reason = dsa_deepselect_topk.fallback_reason
        real_topk = dsa_deepselect_topk.deepselect_topk

        @functools.wraps(real_reason)
        def fallback_reason(scores, seq_lens, topk_k, deterministic):
            reason = real_reason(scores, seq_lens, topk_k, deterministic)
            self.reasons.append(reason)
            return reason

        @functools.wraps(real_topk)
        def deepselect_topk(scores, seq_lens, topk_k, return_topk_scores):
            self.launches.append((tuple(scores.shape), scores.stride(0), topk_k))
            return real_topk(scores, seq_lens, topk_k, return_topk_scores)

        monkeypatch.setattr(dsa_deepselect_topk, "fallback_reason", fallback_reason)
        monkeypatch.setattr(dsa_deepselect_topk, "deepselect_topk", deepselect_topk)

    def clear(self) -> None:
        self.reasons.clear()
        self.launches.clear()


def _permute_valid_slots(indices, lengths, scores, generator):
    """Shuffle each row's valid prefix (``slot < length``), carrying the scores along."""
    slots = torch.arange(indices.size(-1), device=indices.device)
    valid = slots < lengths.unsqueeze(-1)
    keys = torch.rand(indices.shape, generator=generator, device=indices.device)
    order = torch.where(valid, keys, torch.full_like(keys, 2.0)).argsort(dim=-1)
    return torch.gather(indices, -1, order), torch.gather(scores, -1, order)


class _SlotPermuter:
    """``_indexer_topk_bshd`` wrapper that can shuffle the slot order of returned selections.

    Only selections returned with their top-k scores (the indexer loss path) are shuffled: that
    is the order a top-k kernel hands to the loss. Selections without scores are already sorted
    by key for FlashMLA, and full score matrices are not top-k payloads.
    """

    def __init__(self, real):
        self.real = real
        self.enabled = False
        self.shuffled_calls = 0
        self._generator = None

    def start(self, seed: int) -> None:
        self.enabled = True
        self.shuffled_calls = 0
        self._generator = torch.Generator(device="cuda").manual_seed(seed)

    def __call__(self, *args, **kwargs):
        returned = self.real(*args, **kwargs)
        indices, lengths, payload = returned
        if not self.enabled or payload is None or payload.shape != indices.shape:
            return returned
        indices, payload = _permute_valid_slots(indices, lengths, payload, self._generator)
        self.shuffled_calls += 1
        return indices.contiguous(), lengths, payload.contiguous()


def _selection_mismatches(actual: dict, expected: dict, label: str, *, max_tie_fraction=0.01):
    """Compare the raw top-k selections of two runs, allowing only cutoff ties to differ.

    Per call, ``topk.callN.1`` (lengths) must be equal. Each row of ``topk.callN.0`` sorted by
    key, with its ``topk.callN.2`` scores, must be bitwise equal, except rows where the two
    selections pick different keys among those scoring exactly the row's cutoff (the smallest
    selected score): there the selected score multisets must be equal and every key selected by
    only one run must score the cutoff. At most ``max_tie_fraction`` of the rows may differ.
    Returns the failures and the set of (flattened ``batch x query``) rows that differ through
    cutoff ties.
    """
    names = sorted(name for name in expected if _is_topk_indices(name))
    if names != sorted(name for name in actual if _is_topk_indices(name)):
        return [f"{label}: top-k calls differ: {sorted(actual)} != {sorted(expected)}"], set()
    failures, tie_rows = [], set()
    sorted_actual, sorted_expected = _sort_topk_by_key(actual), _sort_topk_by_key(expected)
    for name in names:
        call = name.rsplit(".", 1)[0]
        lengths = expected[f"{call}.1"]
        if not torch.equal(actual[f"{call}.1"], lengths):
            failures.append(f"{label}: {call} top-k lengths differ")
            continue
        topk = expected[name].size(-1)
        got = sorted_actual[name].reshape(-1, topk)
        want = sorted_expected[name].reshape(-1, topk)
        scores_name = f"{call}.2"
        if scores_name not in expected or expected[scores_name].shape != expected[name].shape:
            if not torch.equal(got, want):
                failures.append(f"{label}: {call} selects other keys (no scores to check ties)")
            continue
        got_scores = sorted_actual[scores_name].reshape(-1, topk)
        want_scores = sorted_expected[scores_name].reshape(-1, topk)
        differ = (got != want).any(dim=-1) | (
            got_scores.view(torch.int32) != want_scores.view(torch.int32)
        ).any(dim=-1)
        rows = differ.nonzero().flatten().tolist()
        if not rows:
            continue
        if len(rows) > max_tie_fraction * got.size(0):
            failures.append(f"{label}: {call} selects other keys in {len(rows)}/{got.size(0)} rows")
            continue
        row_lengths = lengths.reshape(-1)
        for row in rows:
            n = int(row_lengths[row])
            a, a_scores = want[row, :n], want_scores[row, :n]
            b, b_scores = got[row, :n], got_scores[row, :n]
            only_a, only_b = ~torch.isin(a, b), ~torch.isin(b, a)
            cutoff = a_scores.min()
            if not (
                bool(only_a.any())
                and torch.equal(a_scores.sort().values, b_scores.sort().values)
                and torch.equal(a_scores[~only_a], b_scores[~only_b])
                and bool((a_scores[only_a] == cutoff).all())
                and bool((b_scores[only_b] == cutoff).all())
            ):
                failures.append(
                    f"{label}: {call} row {row} selects other keys or scores, not cutoff ties "
                    f"({int(only_a.sum())} keys differ, cutoff {float(cutoff)!r})"
                )
                break
            tie_rows.add(row)
    return failures, tie_rows


def _without_topk(result: dict) -> dict:
    return {name: value for name, value in result.items() if not name.startswith("topk.")}


# Tensors whose dim 0 is the query row: a query row's output and query gradient depend only on
# that row's own selection.
_QUERY_ROW_TENSORS = ("output", "grad.query")
# Tensors that sum over query rows per key row: a query row that selects another key moves its
# contribution from one key row to another.
_KEY_ROW_TENSORS = ("grad.key",)


def _tie_aware_mismatches(actual: dict, expected: dict, label: str, tie_rows: set):
    """Production-mode float comparison of two runs that may differ in cutoff-tie rows.

    A query row whose selection differs by a cutoff tie attends to another key with an
    unrelated attention weight (the tie is in the indexer score, not in the attention logit),
    so its output and query gradient may differ by O(that weight), and the key rows involved
    trade that row's contribution. cuDNN breaks such ties differently from run to run as well.
    So, with tie rows: the query-row tensors meet the production limits on the other rows, the
    key-row tensors meet the relative L2 limit (the peak is reported), and every other tensor
    (parameter gradients, indexer loss) meets the production limits. Without tie rows this is
    :func:`_production_mismatches` on all floats.
    """
    actual, expected = _without_topk(actual), _without_topk(expected)
    if not tie_rows:
        return _production_mismatches(actual, expected, label)
    keep = None
    key_rows_actual, key_rows_expected = {}, {}
    for name in list(expected):
        if name in _KEY_ROW_TENSORS:
            key_rows_actual[name] = actual.pop(name, None)
            key_rows_expected[name] = expected.pop(name)
        elif name in _QUERY_ROW_TENSORS and name in actual:
            if keep is None:
                rows = torch.ones(expected[name].size(0), dtype=torch.bool)
                rows[sorted(tie_rows)] = False
                keep = rows.nonzero().flatten().to(expected[name].device)
            expected[name] = expected[name].index_select(0, keep)
            actual[name] = actual[name].index_select(0, keep)
    failures, errors = _production_mismatches(actual, expected, f"{label} (non-tie rows)")
    for name, reference in key_rows_expected.items():
        value = key_rows_actual[name]
        if value is None or value.shape != reference.shape or value.dtype != reference.dtype:
            failures.append(f"{label}: {name} is missing or has another shape")
            continue
        value64, reference64 = value.double(), reference.double()
        if not (torch.isfinite(value64).all() and torch.isfinite(reference64).all()):
            failures.append(f"{label}: {name} has non-finite values")
            continue
        diff = (value64 - reference64).abs()
        rel_l2 = _ratio(diff.norm(), reference64.norm())
        errors[name] = (rel_l2, _ratio(diff.max(), reference64.abs().max()))
        if rel_l2 > _PRODUCTION_REL_L2:
            failures.append(f"{label}: {name} rel L2 {rel_l2:.3e} (limit {_PRODUCTION_REL_L2:.0e})")
    return failures, errors


def _run(step, attention, scorer_spy, deepselect_spy, layout, backend):
    """One forward/backward with ``backend``; checks which top-k kernel ran and where."""
    attention.config.dsa_indexer_topk_backend = backend
    scorer_spy.clean_logits_calls.clear()
    deepselect_spy.clear()
    result = step(skip_fill=layout.skip_fill)
    assert scorer_spy.clean_logits_calls, "the packed THD scorer did not run"
    assert all(
        call == (not layout.skip_fill) for call in scorer_spy.clean_logits_calls
    ), f"clean_logits {scorer_spy.clean_logits_calls} with skip_fill={layout.skip_fill}"
    assert any(_is_topk_indices(name) for name in result), "no top-k was selected"
    if backend == "cudnn":
        assert not deepselect_spy.reasons and not deepselect_spy.launches
        return result
    reasons = [reason for reason in deepselect_spy.reasons if reason is not None]
    assert deepselect_spy.reasons and not reasons, f"DeepSelect fell back: {reasons}"
    assert len(deepselect_spy.launches) == len(deepselect_spy.reasons)
    width = _score_width(layout, _cp_rank(layout))
    expected = ((layout.local_rows, width), dsa_deepselect_topk.score_row_stride(width), _TOPK)
    assert set(deepselect_spy.launches) == {
        expected
    }, f"DeepSelect launches {deepselect_spy.launches}, expected {expected}"
    return result


@pytest.mark.parametrize("layout", _LAYOUTS)
def test_deepselect_layer_matches_cudnn_in_production_mode(layout, monkeypatch):
    _require_deepselect_layer()
    _initialize_model_parallel(layout)
    try:
        cp_rank = _cp_rank(layout)
        width = _score_width(layout, cp_rank)
        reason = dsa_deepselect_topk.static_fallback_reason(
            n_rows=layout.local_rows,
            width=width,
            topk_k=_TOPK,
            dtype=torch.float32,
            device=torch.device("cuda", torch.cuda.current_device()),
            deterministic=False,
        )
        assert reason is None, f"layout is not DeepSelect-eligible on CP rank {cp_rank}: {reason}"

        # Installed first, so the step's recording wrapper logs what the layer receives.
        permuter = _SlotPermuter(dsa_cudnn_kernels._indexer_topk_bshd)
        monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_topk_bshd", permuter)
        step, scorer_spy = _build_step(layout, monkeypatch, deterministic=False)
        attention = step.args[0]
        deepselect_spy = _DeepSelectSpy(monkeypatch)
        run = functools.partial(_run, step, attention, scorer_spy, deepselect_spy, layout)

        raw_reference = run("cudnn")
        reference = _sort_topk_by_key(raw_reference)
        failures, report = [], []

        def compare(raw, label):
            mismatches, ties = _selection_mismatches(raw, raw_reference, label)
            float_mismatches, errors = _tie_aware_mismatches(raw, raw_reference, label, ties)
            failures.extend(mismatches + float_mismatches)
            report.append(
                f"{label}: tie rows {len(ties)}, "
                f"slot diffs {_slot_order_diffs(raw, raw_reference)}, "
                f"{_worst(errors)}"
            )

        compare(run("cudnn"), "cudnn rerun")
        permuter.start(seed=17 + torch.distributed.get_rank())
        shuffled = run("cudnn")
        permuter.enabled = False
        assert permuter.shuffled_calls > 0, "no top-k selection reached the indexer loss"
        compare(shuffled, "cudnn shuffled slots")
        del shuffled

        raw_deepselect = run("deepselect")
        compare(raw_deepselect, "deepselect")
        repeat = run("deepselect")
        for name, value in raw_deepselect.items():
            if name.startswith("topk.") and not torch.equal(
                repeat[name].view(torch.uint8), value.view(torch.uint8)
            ):
                failures.append(f"deepselect rerun: raw {name} is not bitwise repeatable")
        float_mismatches, errors = _production_mismatches(
            _without_topk(repeat), _without_topk(raw_deepselect), "deepselect rerun"
        )
        failures += float_mismatches
        report.append(f"deepselect rerun: {_worst(errors)}")

        print(
            f"[deepselect-layer] layout={layout} rank={torch.distributed.get_rank()} "
            f"cp_rank={cp_rank} rows={layout.local_rows} width={width} "
            f"row_stride={dsa_deepselect_topk.score_row_stride(width)} "
            f"compared={len(reference)} tensors; " + "; ".join(report)
        )
        assert not failures, "\n".join(failures)
    finally:
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        Utils.destroy_model_parallel()


# CPU checks of the comparison helpers above (no GPU needed).


def _selection(rows):
    """``(indices, lengths, scores)`` of shape ``[1, len(rows), K]`` from per-row key->score."""
    topk = max(len(row) for row in rows)
    indices = torch.full((1, len(rows), topk), -1, dtype=torch.int32)
    scores = torch.full((1, len(rows), topk), torch.finfo(torch.float32).min)
    for r, row in enumerate(rows):
        for slot, (key, score) in enumerate(row.items()):
            indices[0, r, slot] = key
            scores[0, r, slot] = score
    lengths = torch.tensor([[len(row) for row in rows]], dtype=torch.int32)
    return {"topk.call0.0": indices, "topk.call0.1": lengths, "topk.call0.2": scores}


@pytest.mark.parametrize(
    "other_row,ok",
    [
        ({5: 3.0, 9: 2.0, 2: 1.0}, True),  # same keys, other slot order
        ({5: 3.0, 9: 2.0, 7: 1.0}, True),  # the other key that scores the cutoff
        ({5: 3.0, 9: 2.0, 7: 1.5}, False),  # other key above the cutoff
        ({5: 3.0, 4: 2.0, 2: 1.0}, False),  # same scores, but not a cutoff tie
        ({5: 3.0, 9: 2.0, 2: 1.25}, False),  # same keys, other score
    ],
)
def test_selection_comparison_allows_only_cutoff_ties(other_row, ok):
    rows = [{1: 0.5, 3: 0.25}] * 199
    expected = _selection([{2: 1.0, 5: 3.0, 9: 2.0}] + rows)
    actual = _selection([other_row] + rows[::-1])
    failures, ties = _selection_mismatches(actual, expected, "cpu")
    assert (not failures) == ok, failures
    assert ties == ({0} if ok and set(other_row) != {2, 5, 9} else set())


def test_selection_comparison_bounds_the_tie_rows():
    expected = _selection([{2: 1.0, 5: 3.0}] * 10)
    actual = _selection([{7: 1.0, 5: 3.0}] * 10)
    failures, _ = _selection_mismatches(actual, expected, "cpu", max_tie_fraction=0.5)
    assert failures and "in 10/10 rows" in failures[0]
    assert not _selection_mismatches(actual, expected, "cpu", max_tie_fraction=1.0)[0]


def test_lengths_must_match():
    expected = _selection([{2: 1.0, 5: 3.0}])
    actual = _selection([{2: 1.0}])
    failures, _ = _selection_mismatches(actual, expected, "cpu")
    assert failures and "lengths differ" in failures[0]


def test_slot_shuffle_keeps_each_valid_prefix_and_its_scores():
    generator = torch.Generator().manual_seed(0)
    indices = torch.arange(24, dtype=torch.int32).view(1, 3, 8)
    lengths = torch.tensor([[8, 5, 0]], dtype=torch.int32)
    indices = torch.where(
        torch.arange(8).view(1, 1, 8) < lengths.unsqueeze(-1), indices, torch.full_like(indices, -1)
    )
    scores = indices.float() * 0.5
    shuffled, shuffled_scores = _permute_valid_slots(indices, lengths, scores, generator)
    assert torch.equal(shuffled_scores, shuffled.float() * 0.5)
    assert not torch.equal(shuffled, indices)
    for row, n in enumerate(lengths[0].tolist()):
        assert torch.equal(shuffled[0, row, :n].sort().values, indices[0, row, :n])
        assert bool((shuffled[0, row, n:] == -1).all())


def test_score_width_follows_the_packed_thd_scorer():
    layouts = {param.id: param.values[0] for param in _LAYOUTS}
    assert _score_width(layouts["cp1-3seq-dummy-unaligned"], 0) == 82000
    assert [_score_width(layouts["cp4-3seq-dummy"], r) for r in range(4)] == [
        131072,
        114688,
        98304,
        81920,
    ]
    assert [_score_width(layouts["dcp2of4-slot"], r) for r in range(2)] == [114688, 86016]
    assert _score_width(layouts["dcp1of4-slot-prefill"], 0) == 81920
    for layout in layouts.values():
        widths = [_score_width(layout, r) for r in range(layout.realized_cp)]
        # Every rank is at or past the ds_tune minimum for its row count.
        assert min(widths) >= dsa_deepselect_topk.min_score_width("ds_tune", layout.local_rows)
        assert layout.local_rows > dsa_deepselect_topk.MANY_ROWS_MIN_SCORE_WIDTH["ds_tune"][0]
        # The dummy tail fills the slot and the zigzag split stays even.
        real_rows = sum(layout.seq_lens)
        global_rows = layout.local_rows * layout.realized_cp
        assert 0 < global_rows - real_rows and len(layout.seq_lens) + 1 <= layout.max_num_seqs
        assert all(length % (2 * layout.realized_cp) == 0 for length in layout.seq_lens) or (
            layout.realized_cp == 1
        )
        assert (global_rows - real_rows) % (2 * layout.realized_cp) == 0 or layout.realized_cp == 1
        # The poison-free scorer spy and the cuDNN out= binding need a width multiple of 4.
        assert all(w % 4 == 0 for w in widths)
    unaligned = layouts["cp1-3seq-dummy-unaligned"]
    assert (unaligned.local_rows * 4) % 1024, "the CP1 layout must exercise the padded out="
