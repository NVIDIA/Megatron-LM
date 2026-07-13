# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Edge-case unit tests for ``compute_golden_statistics``.

This module is a deliberately adversarial testbed: it does NOT re-assert the
happy path (a clean run of training JSON producing obvious stats) but pins the
*weird* behaviours — malformed inputs, mixed/unknown formats, NaN/inf/scientific
string coercion, out-of-order and non-numeric step keys, near-zero means,
warmup slicing, and result-file discovery when the ``.out`` marker lies. These
are the behaviours a refactor is most likely to silently break, so locking them
down gives future changes a solid guardrail.

Pure Python — no GPU, no TensorBoard — so it runs as an ordinary CPU pytest
module alongside ``test_common.py``.
"""

import json

import pytest

from tests.functional_tests.python_test_utils.compute_golden_statistics import (
    _aggregate_inference_results,
    _aggregate_training_results,
    _detect_result_format,
    _extract_result_path_from_log,
    _find_json_files_directly,
    _is_valid_numeric,
    _to_float,
    aggregate_results,
    compute_recommended_tolerances,
    compute_statistics,
    find_result_json_files,
    format_summary,
    load_result_file,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def write_json(path, obj):
    """Serialize ``obj`` to ``path`` and return the path as a string."""
    path.write_text(json.dumps(obj))
    return str(path)


def training_doc(metric="lm loss", values=None):
    values = values if values is not None else {"1": 1.0, "2": 2.0}
    return {metric: {"values": values}}


# ── _detect_result_format — malformed / ambiguous payloads ────────────────────


class TestDetectResultFormatEdgeCases:
    def test_empty_dict_is_unknown(self):
        assert _detect_result_format({}) == "unknown"

    def test_first_value_is_scalar_is_unknown(self):
        # A dict whose first value is not itself a dict cannot be classified.
        assert _detect_result_format({"k": 1.0}) == "unknown"

    def test_first_value_is_list_is_unknown(self):
        assert _detect_result_format({"k": [1, 2, 3]}) == "unknown"

    def test_dict_without_recognised_keys_is_unknown(self):
        assert _detect_result_format({"k": {"totally": "unrelated"}}) == "unknown"

    def test_classification_uses_only_the_first_key(self):
        # First key is inference-shaped, so the whole doc is called inference
        # even though later keys look like training. This first-key-only rule is
        # load-bearing and easy to regress.
        data = {"req0": {"latency": 0.1}, "lm loss": {"values": {"1": 1.0}}}
        assert _detect_result_format(data) == "inference"

    def test_values_key_wins_even_with_latency_present(self):
        # 'values' is checked before 'latency', so a hybrid dict reads training.
        assert _detect_result_format({"m": {"values": {}, "latency": 0.1}}) == "training"


# ── numeric coercion — the surprising truthy/inf/scientific cases ─────────────


class TestNumericCoercionQuirks:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("1e3", 1000.0),  # scientific notation
            ("-4.5", -4.5),  # negative string
            ("  2.0  ", 2.0),  # surrounding whitespace (float() tolerates it)
            ("inf", float("inf")),  # 'inf' string parses to infinity, NOT rejected
            ("Infinity", float("inf")),
            (True, 1.0),  # bool is an int subclass -> coerces to 1.0
            (False, 0.0),
        ],
    )
    def test_to_float_surprising_but_valid(self, value, expected):
        assert _to_float(value) == expected

    @pytest.mark.parametrize("value", ["nan", "NaN", float("nan")])
    def test_to_float_rejects_nan_forms(self, value):
        assert _to_float(value) is None

    @pytest.mark.parametrize("value", ["", "1,2", "0x10", None, [1], {"a": 1}, object()])
    def test_to_float_rejects_junk(self, value):
        assert _to_float(value) is None

    def test_is_valid_numeric_accepts_inf_string(self):
        # Mirrors _to_float: 'inf' is considered a valid numeric here.
        assert _is_valid_numeric("inf") is True

    @pytest.mark.parametrize("value", ["nan", float("nan"), "abc", None, [1]])
    def test_is_valid_numeric_rejects(self, value):
        assert _is_valid_numeric(value) is False


# ── _aggregate_training_results — dirty values & step ordering ────────────────


class TestAggregateTrainingResultsEdgeCases:
    def test_nan_and_non_numeric_steps_are_dropped(self):
        aggregated = {}
        _aggregate_training_results(
            training_doc(values={"1": "nan", "2": 2.0, "3": "oops"}), aggregated, 0
        )
        assert dict(aggregated["lm loss"]) == {"2": [2.0]}

    def test_entry_without_values_key_is_ignored(self):
        aggregated = {}
        _aggregate_training_results({"weird": {"nope": 1}}, aggregated, 0)
        assert aggregated == {}

    def test_non_dict_metric_entry_is_ignored(self):
        aggregated = {}
        _aggregate_training_results({"weird": 3.0}, aggregated, 0)
        assert aggregated == {}

    def test_all_empty_metric_yields_empty_bucket(self):
        aggregated = {}
        _aggregate_training_results({"lm loss": {"values": {}}}, aggregated, 0)
        # Bucket is created but holds no steps.
        assert aggregated["lm loss"] == {}

    def test_median_metric_sorts_numeric_steps_and_parks_non_numeric_last(self):
        # Non-digit step keys sort with key float('inf'), so they land at the end
        # of the _all_values_run_* list regardless of insertion order.
        aggregated = {}
        doc = {"iteration-time": {"values": {"10": 1.0, "2": 2.0, "foo": 3.0}}}
        _aggregate_training_results(doc, aggregated, 0)
        assert aggregated["iteration-time"]["_all_values_run_0"] == [2.0, 1.0, 3.0]

    def test_all_values_key_is_run_indexed(self):
        aggregated = {}
        _aggregate_training_results({"iteration-time": {"values": {"1": 1.0}}}, aggregated, 0)
        _aggregate_training_results({"iteration-time": {"values": {"1": 2.0}}}, aggregated, 7)
        assert "_all_values_run_0" in aggregated["iteration-time"]
        assert "_all_values_run_7" in aggregated["iteration-time"]


# ── _aggregate_inference_results — partial / empty request records ────────────


class TestAggregateInferenceResultsEdgeCases:
    def test_non_dict_request_records_are_skipped(self):
        aggregated = {}
        _aggregate_inference_results({"a": "notadict", "b": 5}, aggregated, 0)
        assert aggregated == {}

    def test_empty_logprob_lists_produce_no_metric(self):
        aggregated = {}
        _aggregate_inference_results(
            {"a": {"prompt_logprobs": [], "generated_log_probs": []}}, aggregated, 0
        )
        assert aggregated == {}

    def test_latency_only_records_skip_other_metrics(self):
        aggregated = {}
        _aggregate_inference_results({"a": {"latency": 2.0}, "b": {"latency": 4.0}}, aggregated, 0)
        assert set(aggregated) == {"latency"}
        assert aggregated["latency"]["mean"] == [3.0]
        assert aggregated["latency"]["total"] == [6.0]

    def test_no_extractable_metrics_leaves_aggregate_untouched(self):
        aggregated = {}
        _aggregate_inference_results({"a": {"unrelated": 1}}, aggregated, 0)
        assert aggregated == {}


# ── aggregate_results — corrupt files & mixed formats across files ────────────


class TestAggregateResultsEdgeCases:
    def test_corrupt_file_is_skipped_not_fatal(self, tmp_path):
        good = write_json(tmp_path / "good.json", training_doc())
        bad = tmp_path / "bad.json"
        bad.write_text("{ this is not json")
        aggregated = aggregate_results([str(bad), good])
        assert "lm loss" in aggregated

    def test_missing_file_is_skipped(self, tmp_path):
        good = write_json(tmp_path / "good.json", training_doc())
        aggregated = aggregate_results([str(tmp_path / "ghost.json"), good])
        assert "lm loss" in aggregated

    def test_unknown_format_file_contributes_nothing(self, tmp_path):
        unknown = write_json(tmp_path / "u.json", {"k": 1.0})
        aggregated = aggregate_results([unknown])
        assert aggregated == {}

    def test_mixed_formats_are_dispatched_per_file(self, tmp_path):
        # The logged "detected format" is taken from the FIRST file only, but
        # dispatch is per-file, so a training file and an inference file in the
        # same batch each still get aggregated under their own keys.
        train = write_json(tmp_path / "t.json", training_doc(values={"1": 1.0}))
        infer = write_json(tmp_path / "i.json", {"r": {"latency": 0.5}})
        aggregated = aggregate_results([train, infer])
        assert set(aggregated) == {"lm loss", "latency"}

    def test_double_encoded_jsonl_string_is_decoded(self, tmp_path):
        p = tmp_path / "r.json"
        p.write_text(json.dumps(json.dumps(training_doc(values={"1": 5.0}))))
        aggregated = aggregate_results([str(p)])
        assert aggregated["lm loss"]["1"] == [5.0]

    def test_empty_file_list_yields_empty(self):
        assert aggregate_results([]) == {}


# ── compute_statistics — degenerate sample sets ───────────────────────────────


class TestComputeStatisticsEdgeCases:
    def test_single_sample_std_is_zero_not_error(self):
        stats = compute_statistics({"m": {"1": [4.0]}})
        assert stats["m"]["values"]["1"]["std"] == 0.0
        assert stats["m"]["values"]["1"]["count"] == 1

    def test_empty_step_list_is_skipped(self):
        stats = compute_statistics({"m": {"1": []}})
        assert stats["m"]["values"] == {}
        assert stats["m"]["num_samples"] == 0

    def test_internal_all_values_keys_excluded_from_stats_and_count(self):
        aggregated = {"iteration-time": {"1": [1.0, 2.0], "_all_values_run_0": [9.0, 9.0, 9.0]}}
        stats = compute_statistics(aggregated)
        assert "_all_values_run_0" not in stats["iteration-time"]["values"]
        assert stats["iteration-time"]["num_samples"] == 2

    def test_ragged_sample_counts_take_the_max(self):
        # Different steps observed a different number of runs; num_samples is the max.
        aggregated = {"m": {"1": [1.0, 2.0, 3.0], "2": [1.0]}}
        stats = compute_statistics(aggregated)
        assert stats["m"]["num_samples"] == 3

    def test_completely_empty_metric_reports_zero_samples(self):
        stats = compute_statistics({"m": {}})
        assert stats["m"]["num_samples"] == 0
        assert stats["m"]["values"] == {}


# ── compute_recommended_tolerances — variance, floors & warmup slicing ────────


class TestComputeRecommendedTolerancesEdgeCases:
    def test_zero_variance_hits_the_minimum_floor(self):
        aggregated = {"lm loss": {"1": [1.0, 1.0]}}
        stats = compute_statistics(aggregated)
        tol = compute_recommended_tolerances(stats, aggregated)
        assert tol["lm loss"]["relative_tolerance"] == 0.001  # floor

    def test_warmup_steps_below_start_step_are_excluded(self):
        # Step 1 is wildly noisy; with start_step=5 it must not inflate the tolerance.
        aggregated = {"lm loss": {"1": [1.0, 100.0], "5": [1.0, 1.0]}}
        stats = compute_statistics(aggregated)
        tol = compute_recommended_tolerances(stats, aggregated, start_step=5)
        assert tol["lm loss"]["steps_included"] == 1
        assert tol["lm loss"]["relative_tolerance"] == 0.001

    def test_non_numeric_step_key_is_always_included(self):
        # Inference-style 'mean'/'total' keys can't parse as int, so the warmup
        # filter keeps them rather than dropping them.
        aggregated = {"latency": {"mean": [1.0, 1.2]}}
        stats = compute_statistics(aggregated)
        tol = compute_recommended_tolerances(stats, aggregated, start_step=99)
        assert tol["latency"]["steps_included"] == 1

    def test_median_based_metric_uses_per_run_medians(self):
        aggregated = {
            "iteration-time": {
                "_all_values_run_0": [10.0, 1.0, 1.0, 1.0],  # median of tail = 1.0
                "_all_values_run_1": [10.0, 2.0, 2.0, 2.0],  # median of tail = 2.0
            }
        }
        stats = compute_statistics(aggregated)
        tol = compute_recommended_tolerances(stats, aggregated, start_step=1)
        # medians 1.0 & 2.0 -> mean 1.5, max rel var = 0.5/1.5 ≈ 0.333.
        assert tol["iteration-time"]["max_observed_relative_variance"] == pytest.approx(
            0.3333, abs=1e-3
        )

    def test_max_based_metric_skips_warmup_then_takes_max(self):
        aggregated = {
            "mem-allocated-bytes": {
                "_all_values_run_0": [5.0, 10.0, 8.0],  # warmup 5 dropped, max 10
                "_all_values_run_1": [5.0, 12.0, 9.0],  # max 12
            }
        }
        stats = compute_statistics(aggregated)
        tol = compute_recommended_tolerances(stats, aggregated)
        # maxes 10 & 12 -> mean 11, max rel var = 1/11 ≈ 0.0909.
        assert tol["mem-allocated-bytes"]["max_observed_relative_variance"] == pytest.approx(
            0.0909, abs=1e-3
        )

    def test_near_zero_mean_switches_to_absolute_variance(self):
        aggregated = {"grad": {"1": [0.0, 0.0]}}
        stats = compute_statistics(aggregated)
        # Force a near-zero mean with a nonzero sample to exercise the abs branch.
        stats["grad"]["values"]["1"]["mean"] = 0.0
        stats["grad"]["values"]["1"]["samples"] = [0.0, 1e-4]
        tol = compute_recommended_tolerances(stats, aggregated)
        assert tol["grad"]["max_observed_absolute_variance"] == pytest.approx(1e-4)
        # Absolute tolerance never drops below the 1e-6 floor.
        assert tol["grad"]["absolute_tolerance"] >= 1e-6

    def test_empty_stats_yields_empty_tolerances(self):
        assert compute_recommended_tolerances({}, {}) == {}


# ── load_result_file — non-dict payloads & failures ───────────────────────────


class TestLoadResultFileEdgeCases:
    def test_missing_file_returns_none(self, tmp_path):
        assert load_result_file(str(tmp_path / "nope.json")) is None

    def test_malformed_json_returns_none(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{oops")
        assert load_result_file(str(p)) is None

    def test_json_list_payload_is_returned_verbatim(self, tmp_path):
        # The loader does not enforce a dict; a top-level list round-trips.
        p = write_json(tmp_path / "l.json", [1, 2, 3])
        assert load_result_file(p) == [1, 2, 3]

    def test_double_encoded_dict_is_unwrapped(self, tmp_path):
        p = tmp_path / "d.json"
        p.write_text(json.dumps(json.dumps({"a": 2})))
        assert load_result_file(str(p)) == {"a": 2}


# ── result-file discovery — lying / missing markers ──────────────────────────


class TestResultDiscoveryEdgeCases:
    def test_missing_results_dir_returns_empty(self, tmp_path):
        assert find_result_json_files(str(tmp_path / "absent")) == []

    def test_marker_pointing_at_missing_dir_returns_none(self, tmp_path):
        out = tmp_path / "j.out"
        out.write_text("This test wrote results into /opt/megatron-lm/runs/ghost\n")
        assert _extract_result_path_from_log(out, str(tmp_path)) is None

    def test_no_marker_line_returns_none(self, tmp_path):
        out = tmp_path / "j.out"
        out.write_text("lots of logs\nbut no marker line\n")
        assert _extract_result_path_from_log(out, str(tmp_path)) is None

    def test_marker_resolves_container_path_to_workspace(self, tmp_path):
        run_dir = tmp_path / "runs" / "abc"
        run_dir.mkdir(parents=True)
        gv = write_json(run_dir / "golden_values.json", training_doc())
        out = tmp_path / "j.out"
        out.write_text("This test wrote results into /opt/megatron-lm/runs/abc\n")
        assert _extract_result_path_from_log(out, str(tmp_path)) == gv

    def test_find_falls_back_to_direct_search_without_out_files(self, tmp_path):
        # No .out files at all -> the direct-glob fallback path is taken.
        (tmp_path / "sub").mkdir()
        write_json(tmp_path / "sub" / "golden_values_x.json", training_doc())
        found = find_result_json_files(str(tmp_path), workspace_root=str(tmp_path))
        assert len(found) == 1
        assert found[0].endswith("golden_values_x.json")

    def test_direct_search_dedupes_overlapping_patterns(self, tmp_path):
        (tmp_path / "sub").mkdir()
        write_json(tmp_path / "sub" / "golden_values_a.json", training_doc())
        write_json(tmp_path / "sub" / "test_results_b.json", training_doc())
        found = _find_json_files_directly(str(tmp_path))
        assert len(found) == len(set(found)) == 2


# ── format_summary — missing tolerance entries & non-int step keys ────────────


class TestFormatSummaryEdgeCases:
    def test_missing_tolerance_entry_does_not_crash(self):
        stats = compute_statistics({"lm loss": {"1": [1.0]}})
        summary = format_summary(stats, {})  # no tolerances supplied
        assert "lm loss" in summary
        assert "Samples: 1" in summary

    def test_summary_renders_inference_style_string_step_keys(self):
        aggregated = {"latency": {"mean": [1.0, 2.0], "total": [3.0]}}
        stats = compute_statistics(aggregated)
        tol = compute_recommended_tolerances(stats, aggregated)
        summary = format_summary(stats, tol)
        assert "latency" in summary

    def test_metrics_are_listed_in_sorted_order(self):
        aggregated = {"zebra": {"1": [1.0]}, "alpha": {"1": [1.0]}}
        stats = compute_statistics(aggregated)
        summary = format_summary(stats, {})
        assert summary.index("alpha") < summary.index("zebra")
