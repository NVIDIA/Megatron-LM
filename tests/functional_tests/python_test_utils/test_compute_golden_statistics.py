# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for ``compute_golden_statistics``.

These exercise the pure-Python aggregation, statistics, tolerance-recommendation
and result-file-discovery helpers. They need neither a GPU nor TensorBoard, so
they run as an ordinary CPU pytest module alongside ``test_common.py``.
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
    path.write_text(json.dumps(obj))
    return str(path)


def training_doc(metric="lm loss", values=None):
    values = values or {"1": 1.0, "2": 2.0, "3": 3.0}
    return {metric: {"values": values}}


# ── _detect_result_format ─────────────────────────────────────────────────────


class TestDetectResultFormat:
    def test_empty_is_unknown(self):
        assert _detect_result_format({}) == "unknown"

    def test_training_via_values_key(self):
        assert _detect_result_format({"lm loss": {"values": {"1": 1.0}}}) == "training"

    def test_inference_via_latency(self):
        assert _detect_result_format({"req0": {"latency": 0.5}}) == "inference"

    def test_inference_via_generated_text(self):
        assert _detect_result_format({"req0": {"generated_text": "hi"}}) == "inference"

    def test_scalar_first_value_is_unknown(self):
        assert _detect_result_format({"k": 1.0}) == "unknown"

    def test_dict_without_known_keys_is_unknown(self):
        assert _detect_result_format({"k": {"other": 1}}) == "unknown"


# ── _is_valid_numeric ─────────────────────────────────────────────────────────


class TestIsValidNumeric:
    @pytest.mark.parametrize("value", [1, 1.5, "2.0", "3", 0, -4.2])
    def test_valid(self, value):
        assert _is_valid_numeric(value) is True

    @pytest.mark.parametrize("value", ["nan", "abc", None, [1], {"a": 1}, float("nan")])
    def test_invalid(self, value):
        assert _is_valid_numeric(value) is False


# ── _to_float ─────────────────────────────────────────────────────────────────


class TestToFloat:
    def test_int_and_float(self):
        assert _to_float(3) == 3.0
        assert _to_float(2.5) == 2.5

    def test_numeric_string(self):
        assert _to_float("4.25") == 4.25

    @pytest.mark.parametrize("value", ["nan", "notnum", None, [1], float("nan")])
    def test_invalid_returns_none(self, value):
        assert _to_float(value) is None

    def test_returns_float_type(self):
        result = _to_float(7)
        assert isinstance(result, float)


# ── _aggregate_training_results ───────────────────────────────────────────────


class TestAggregateTrainingResults:
    def test_basic_accumulation_across_runs(self):
        aggregated = {}
        _aggregate_training_results(training_doc(values={"1": 1.0, "2": 2.0}), aggregated, 0)
        _aggregate_training_results(training_doc(values={"1": 1.5, "2": 2.5}), aggregated, 1)
        assert aggregated["lm loss"]["1"] == [1.0, 1.5]
        assert aggregated["lm loss"]["2"] == [2.0, 2.5]

    def test_skips_nan_and_non_numeric(self):
        aggregated = {}
        _aggregate_training_results(
            training_doc(values={"1": "nan", "2": 2.0, "3": "oops"}), aggregated, 0
        )
        assert "1" not in aggregated["lm loss"]
        assert "3" not in aggregated["lm loss"]
        assert aggregated["lm loss"]["2"] == [2.0]

    def test_ignores_entries_without_values_key(self):
        aggregated = {}
        _aggregate_training_results({"weird": {"nope": 1}}, aggregated, 0)
        assert aggregated == {}

    def test_median_metric_stores_all_values_in_step_order(self):
        aggregated = {}
        # steps intentionally out of order; must be sorted numerically.
        doc = {"iteration-time": {"values": {"3": 0.3, "1": 0.1, "2": 0.2}}}
        _aggregate_training_results(doc, aggregated, 0)
        assert aggregated["iteration-time"]["_all_values_run_0"] == [0.1, 0.2, 0.3]


# ── _aggregate_inference_results ──────────────────────────────────────────────


class TestAggregateInferenceResults:
    def test_latency_mean_and_total(self):
        aggregated = {}
        data = {"a": {"latency": 1.0}, "b": {"latency": 3.0}}
        _aggregate_inference_results(data, aggregated, 0)
        assert aggregated["latency"]["mean"] == [2.0]
        assert aggregated["latency"]["total"] == [4.0]

    def test_step_count_and_logprob_means(self):
        aggregated = {}
        data = {
            "a": {
                "step_count": 10,
                "prompt_logprobs": [-1.0, -3.0],
                "generated_log_probs": [-2.0, -2.0],
            }
        }
        _aggregate_inference_results(data, aggregated, 0)
        assert aggregated["step_count"]["mean"] == [10.0]
        assert aggregated["prompt_logprob_mean"]["mean"] == [-2.0]
        assert aggregated["generated_logprob_mean"]["mean"] == [-2.0]

    def test_ignores_non_dict_and_empty_logprobs(self):
        aggregated = {}
        data = {"a": "notadict", "b": {"prompt_logprobs": []}}
        _aggregate_inference_results(data, aggregated, 0)
        assert aggregated == {}


# ── aggregate_results (end-to-end over files) ─────────────────────────────────


class TestAggregateResults:
    def test_training_files(self, tmp_path):
        f1 = write_json(tmp_path / "r1.json", training_doc(values={"1": 1.0}))
        f2 = write_json(tmp_path / "r2.json", training_doc(values={"1": 3.0}))
        aggregated = aggregate_results([f1, f2])
        assert aggregated["lm loss"]["1"] == [1.0, 3.0]

    def test_skips_unreadable_file(self, tmp_path):
        good = write_json(tmp_path / "good.json", training_doc())
        bad = tmp_path / "bad.json"
        bad.write_text("{ not json")
        aggregated = aggregate_results([str(bad), good])
        assert "lm loss" in aggregated

    def test_jsonl_double_encoded(self, tmp_path):
        # A JSONL-style file whose content is a JSON *string* of the payload.
        p = tmp_path / "r.json"
        p.write_text(json.dumps(json.dumps(training_doc(values={"1": 5.0}))))
        aggregated = aggregate_results([str(p)])
        assert aggregated["lm loss"]["1"] == [5.0]


# ── compute_statistics ────────────────────────────────────────────────────────


class TestComputeStatistics:
    def test_basic_stats(self):
        aggregated = {"lm loss": {"1": [1.0, 2.0, 3.0]}}
        stats = compute_statistics(aggregated)
        s = stats["lm loss"]["values"]["1"]
        assert s["min"] == 1.0
        assert s["max"] == 3.0
        assert s["mean"] == 2.0
        assert s["count"] == 3
        assert s["std"] == pytest.approx(1.0)
        assert stats["lm loss"]["num_samples"] == 3

    def test_single_sample_std_is_zero(self):
        stats = compute_statistics({"m": {"1": [4.0]}})
        assert stats["m"]["values"]["1"]["std"] == 0.0

    def test_internal_keys_excluded_from_num_samples(self):
        aggregated = {"iteration-time": {"1": [1.0, 2.0], "_all_values_run_0": [1.0, 2.0, 3.0]}}
        stats = compute_statistics(aggregated)
        assert stats["iteration-time"]["num_samples"] == 2
        assert "_all_values_run_0" not in stats["iteration-time"]["values"]

    def test_empty_step_list_skipped(self):
        stats = compute_statistics({"m": {"1": []}})
        assert stats["m"]["values"] == {}


# ── compute_recommended_tolerances ────────────────────────────────────────────


class TestComputeRecommendedTolerances:
    def test_per_step_relative_variance(self):
        aggregated = {"lm loss": {"1": [1.0, 1.1]}}
        stats = compute_statistics(aggregated)
        tol = compute_recommended_tolerances(stats, aggregated, confidence_multiplier=3.0)
        # mean=1.05; max rel var = |1.1-1.05|/1.05 ≈ 0.0476; *3 ≈ 0.1429.
        assert tol["lm loss"]["relative_tolerance"] == pytest.approx(0.1429, abs=1e-3)
        assert tol["lm loss"]["steps_included"] == 1

    def test_minimum_relative_tolerance_floor(self):
        aggregated = {"lm loss": {"1": [1.0, 1.0]}}  # zero variance
        stats = compute_statistics(aggregated)
        tol = compute_recommended_tolerances(stats, aggregated)
        assert tol["lm loss"]["relative_tolerance"] == 0.001

    def test_warmup_steps_below_start_step_skipped(self):
        aggregated = {"lm loss": {"1": [1.0, 100.0], "5": [1.0, 1.0]}}
        stats = compute_statistics(aggregated)
        tol = compute_recommended_tolerances(stats, aggregated, start_step=5)
        # Step 1 (the noisy one) is skipped, only step 5 counts.
        assert tol["lm loss"]["steps_included"] == 1
        assert tol["lm loss"]["relative_tolerance"] == 0.001

    def test_median_based_metric_uses_run_medians(self):
        aggregated = {
            "iteration-time": {
                "_all_values_run_0": [10.0, 1.0, 1.0, 1.0],  # median of [1,1,1] after start=1
                "_all_values_run_1": [10.0, 2.0, 2.0, 2.0],
            }
        }
        stats = compute_statistics(aggregated)
        tol = compute_recommended_tolerances(stats, aggregated, start_step=1)
        # run medians: 1.0 and 2.0 -> mean 1.5, max rel var = 0.5/1.5 = 0.333.
        assert tol["iteration-time"]["max_observed_relative_variance"] == pytest.approx(
            0.3333, abs=1e-3
        )

    def test_max_based_metric_uses_run_maxes(self):
        aggregated = {
            "mem-allocated-bytes": {
                "_all_values_run_0": [5.0, 10.0, 8.0],  # warmup 5 skipped, max=10
                "_all_values_run_1": [5.0, 12.0, 9.0],  # max=12
            }
        }
        stats = compute_statistics(aggregated)
        tol = compute_recommended_tolerances(stats, aggregated)
        # maxes 10 and 12 -> mean 11, max rel var = 1/11 ≈ 0.0909.
        assert tol["mem-allocated-bytes"]["max_observed_relative_variance"] == pytest.approx(
            0.0909, abs=1e-3
        )

    def test_near_zero_mean_tracks_absolute_variance(self):
        aggregated = {"grad": {"1": [0.0, 0.0]}}
        stats = compute_statistics(aggregated)
        # Force a near-zero mean with nonzero samples to exercise absolute branch.
        stats["grad"]["values"]["1"]["mean"] = 0.0
        stats["grad"]["values"]["1"]["samples"] = [0.0, 1e-4]
        tol = compute_recommended_tolerances(stats, aggregated)
        assert tol["grad"]["max_observed_absolute_variance"] == pytest.approx(1e-4)


# ── load_result_file ──────────────────────────────────────────────────────────


class TestLoadResultFile:
    def test_loads_dict(self, tmp_path):
        p = write_json(tmp_path / "x.json", {"a": 1})
        assert load_result_file(p) == {"a": 1}

    def test_double_encoded_string(self, tmp_path):
        p = tmp_path / "x.json"
        p.write_text(json.dumps(json.dumps({"a": 2})))
        assert load_result_file(str(p)) == {"a": 2}

    def test_missing_file_returns_none(self, tmp_path):
        assert load_result_file(str(tmp_path / "nope.json")) is None

    def test_bad_json_returns_none(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{oops")
        assert load_result_file(str(p)) is None


# ── file discovery ────────────────────────────────────────────────────────────


class TestFindResultJsonFiles:
    def test_missing_dir_returns_empty(self, tmp_path):
        assert find_result_json_files(str(tmp_path / "absent")) == []

    def test_extract_path_from_log_marker(self, tmp_path):
        run_dir = tmp_path / "runs" / "abc"
        run_dir.mkdir(parents=True)
        gv = write_json(run_dir / "golden_values.json", training_doc())
        out = tmp_path / "job.out"
        out.write_text(
            "some output\nThis test wrote results into /opt/megatron-lm/runs/abc\nmore\n"
        )
        found = _extract_result_path_from_log(out, str(tmp_path))
        assert found == gv

    def test_extract_path_no_marker_returns_none(self, tmp_path):
        out = tmp_path / "job.out"
        out.write_text("no marker here\n")
        assert _extract_result_path_from_log(out, str(tmp_path)) is None

    def test_find_via_out_files_end_to_end(self, tmp_path):
        run_dir = tmp_path / "runs" / "abc"
        run_dir.mkdir(parents=True)
        write_json(run_dir / "golden_values.json", training_doc())
        out = tmp_path / "job.out"
        out.write_text("This test wrote results into /opt/megatron-lm/runs/abc\n")
        found = find_result_json_files(str(tmp_path), workspace_root=str(tmp_path))
        assert len(found) == 1
        assert found[0].endswith("golden_values.json")

    def test_direct_fallback_dedupes(self, tmp_path):
        (tmp_path / "sub").mkdir()
        write_json(tmp_path / "sub" / "golden_values_a.json", training_doc())
        write_json(tmp_path / "sub" / "test_results_b.json", training_doc())
        found = _find_json_files_directly(str(tmp_path))
        assert len(found) == 2
        assert len(set(found)) == 2


# ── format_summary ────────────────────────────────────────────────────────────


class TestFormatSummary:
    def test_includes_metric_and_tolerance_fields(self):
        aggregated = {"lm loss": {"1": [1.0, 1.1]}}
        stats = compute_statistics(aggregated)
        tol = compute_recommended_tolerances(stats, aggregated)
        summary = format_summary(stats, tol)
        assert "lm loss" in summary
        assert "Recommended relative tolerance" in summary
        assert "Golden Values Statistics Summary" in summary

    def test_handles_missing_tolerance_entry(self):
        stats = compute_statistics({"lm loss": {"1": [1.0]}})
        summary = format_summary(stats, {})  # no tolerances provided
        assert "lm loss" in summary
        assert "Samples: 1" in summary
