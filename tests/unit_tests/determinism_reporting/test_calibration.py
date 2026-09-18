# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Calibration contracts with synthetic records; these do not execute GPU work."""

import copy
import json
import shutil
import statistics

import pytest

from tests.unit_tests.determinism_reporting.test_paired_performance import SCRIPTS, load_module
from tests.unit_tests.determinism_reporting.test_performance_baseline import (
    REVISION,
    artifacts,
    write_json,
)


@pytest.fixture
def calibration(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return load_module("calibration")


def bundle(
    calibration, root, *, gpu="GPU-fixture-0", overhead=1.2, mutate=None, limit=None, has_base=False
):
    covered, leaderboard = artifacts(root)

    def visit(value):
        if isinstance(value, dict):
            if "device_uuid" in value:
                value["device_uuid"] = gpu
            if "machine" in value:
                value["machine"] = {"host": gpu + "-host", "gpus": [gpu]}
            if mutate is not None:
                mutate(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    cover = json.loads(covered.read_text())
    visit(cover)
    write_json(covered, cover)
    reports = json.loads(leaderboard.read_text())
    for report in reports:
        if has_base:
            report["sources"]["base"] = {"revision": "b" * 40, "dirty": False}
            base_runs = copy.deepcopy(report["runs"])
            for run in base_runs:
                run["revision_label"] = "base"
            report["runs"].extend(base_runs)
        visit(report)
        for run in report["runs"]:
            ratio = overhead if run["revision_label"] == "head" else 1.2
            samples = [10.0 * ratio if run["mode"] == "det" else 10.0] * 3
            run.update(samples_ms=samples, median_ms=statistics.median(samples))
            run["kernel"]["samples_ms"] = samples
            case = leaderboard.parent / (
                "weighted_swiglu-bfloat16-" + report["measurement"]["phase"]
            )
            arm = case / f"pair-{run['pair']}" / f"{run['revision_label']}-{run['mode']}"
            write_json(arm / "kernel.json", run["kernel"])
            (arm / "launcher.log").write_text("Synthetic CPU fixture, not GPU evidence\n")
        report["comparisons"] = calibration.baseline.author_evidence.benchmark.summarize(
            report["runs"], 3, has_base, limit, None
        )
        report["status"] = "reported" if limit is None else "pass"
        write_json(case / "benchmark.json", report)
    write_json(leaderboard, reports)
    result = calibration.baseline.publish(
        covered, leaderboard, root / "store", cover["context"]["revision"], str(root)
    )
    return root / "store" / result["baseline_id"], result["baseline_id"]


def test_preserve_each_run_and_observed_range_without_pooling(calibration, tmp_path):
    first = bundle(calibration, tmp_path / "first", overhead=1.2)
    second = bundle(calibration, tmp_path / "second", gpu="GPU-fixture-1", overhead=0.9)
    result = calibration.compare([second, first])
    assert result["status"] == "report_only" and result["performance_gate"] == "not_gated"
    assert len(result["groups"]) == 2  # Forward and backward stay separate.
    for group in result["groups"]:
        assert group["run_count"] == group["distinct_timing_gpus"] == 2
        assert group["repeat_status"] == "repeated_measurements"
        comparison = group["comparisons"]["head_overhead"]
        assert comparison["minimum_ratio"] == 0.9 and comparison["maximum_ratio"] == 1.2
        assert comparison["median_ratio"] == pytest.approx(1.05)
        assert comparison["observed_range_percentage_points"] == pytest.approx(30)
        assert "bootstrap_95_percent_interval" not in comparison
        assert all(
            "bootstrap_95_percent_interval" in run["comparisons"]["head_overhead"]
            for run in group["runs"]
        )
    assert calibration.compare([first, second]) == result
    assert "1.200000" in calibration.markdown_report(result)


def test_same_gpu_repeats_are_not_counted_as_different_devices(calibration, tmp_path):
    first = bundle(calibration, tmp_path / "first", overhead=1.1)
    second = bundle(calibration, tmp_path / "second", overhead=1.2)
    result = calibration.compare([first, second])
    assert all(
        group["run_count"] == 2 and group["distinct_timing_gpus"] == 1 for group in result["groups"]
    )
    single = calibration.compare([first])
    assert all(group["repeat_status"] == "single_measurement" for group in single["groups"])


def test_revision_regressions_remain_separate_from_mode_overhead(calibration, tmp_path):
    first = bundle(calibration, tmp_path, overhead=1.3, has_base=True)
    result = calibration.compare([first])
    for group in result["groups"]:
        values = group["comparisons"]
        assert set(values) == {
            "head_overhead",
            "base_overhead",
            "default_regression",
            "det_regression",
        }
        assert values["head_overhead"]["median_ratio"] == 1.3
        assert values["base_overhead"]["median_ratio"] == 1.2
        assert values["default_regression"]["median_ratio"] == 1
        assert values["det_regression"]["median_ratio"] == pytest.approx(1.3 / 1.2)


@pytest.mark.parametrize(
    "field", ["gpu", "driver", "torch", "environment", "source", "base", "input", "adapter"]
)
def test_changed_contexts_do_not_form_one_cohort(calibration, tmp_path, field):
    first = bundle(calibration, tmp_path / "first", has_base=field == "base")

    def mutate(value):
        if "runtime" in value:
            runtime = value["runtime"]
            if field in ("gpu", "torch"):
                runtime[field] = "changed-fixture"
            elif field == "driver":
                runtime[field] = ["changed-driver"]
            elif field == "environment":
                runtime[field]["NCCL_PROTO"] = "changed-protocol"
        if field == "source" and value.get("revision") == REVISION:
            value["revision"] = "c" * 40
        if field == "base" and value.get("revision") == "b" * 40:
            value["revision"] = "c" * 40
        if field == "input" and "sha256" in value:
            value["sha256"] = "1" * 64
        if field == "adapter" and "adapter_sha256" in value:
            value["adapter_sha256"] = "2" * 64

    second = bundle(
        calibration,
        tmp_path / "second",
        gpu="GPU-fixture-1",
        mutate=mutate,
        has_base=field == "base",
    )
    result = calibration.compare([first, second])
    assert len(result["groups"]) == 4 and all(group["run_count"] == 1 for group in result["groups"])


def test_cache_location_comparison_is_explicit_and_preserves_paths(calibration, tmp_path):
    def cache(path):
        def mutate(value):
            if "runtime" in value:
                value["runtime"]["environment"]["TRITON_CACHE_DIR"] = path

        return mutate

    first = bundle(calibration, tmp_path / "first", mutate=cache("/first/cache"))
    second = bundle(
        calibration, tmp_path / "second", gpu="GPU-fixture-1", mutate=cache("/second/cache")
    )
    assert len(calibration.compare([first, second])["groups"]) == 4
    compared = calibration.compare([first, second], compare_cache_locations=True)
    assert len(compared["groups"]) == 2 and compared["compare_cache_locations"]
    assert {
        run["cache_locations"]["det"] for group in compared["groups"] for run in group["runs"]
    } == {"/first/cache", "/second/cache"}
    unset = bundle(calibration, tmp_path / "unset", gpu="GPU-fixture-2")
    assert len(calibration.compare([first, unset], compare_cache_locations=True)["groups"]) == 4


@pytest.mark.parametrize("copy_kind", ["same_id", "relocated", "relabelled"])
def test_duplicate_measurements_are_rejected(calibration, tmp_path, copy_kind):
    first = bundle(calibration, tmp_path / "first")
    if copy_kind == "same_id":
        second = first
    elif copy_kind == "relocated":
        relocated = tmp_path / "relocated"
        shutil.copytree(first[0], relocated)
        second = (relocated, first[1])
    else:
        second = bundle(calibration, tmp_path / "different-origin")
    with pytest.raises(ValueError, match="Repeated"):
        calibration.compare([first, second])


def test_budgeted_selection_cannot_become_calibration_evidence(calibration, tmp_path):
    selected = bundle(calibration, tmp_path, limit=1.35)
    with pytest.raises(ValueError, match="unbudgeted"):
        calibration.compare([selected])


@pytest.mark.parametrize("problem", ["wrong_id", "changed_raw", "missing_raw", "changed_summary"])
def test_reverify_artifacts_before_using_their_summaries(calibration, tmp_path, problem):
    directory, identity = bundle(calibration, tmp_path)
    raw = next((directory / "performance").rglob("kernel.json"))
    if problem == "wrong_id":
        identity = "0" * 64
    elif problem == "changed_raw":
        raw.write_text("{}")
    elif problem == "missing_raw":
        raw.unlink()
    else:
        (directory / "performance/leaderboard.json").write_text("[]")
    with pytest.raises(ValueError):
        calibration.compare([(directory, identity)])


def test_cli_writes_both_reports_and_preserves_earlier_output(calibration, tmp_path):
    directory, identity = bundle(calibration, tmp_path)
    output = tmp_path / "result.json"
    args = ["--baseline", str(directory), identity, "--output", str(output)]
    assert calibration.main(args) == 0
    original = output.read_bytes()
    assert json.loads(original)["performance_gate"] == "not_gated"
    assert output.with_suffix(".md").is_file()
    assert calibration.main(args) == 1 and output.read_bytes() == original
    with pytest.raises(ValueError, match="pinned"):
        calibration.compare([])
