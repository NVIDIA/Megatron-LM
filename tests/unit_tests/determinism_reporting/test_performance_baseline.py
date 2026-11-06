# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Portable artifact contracts using synthetic CPU-only timing/replay fixtures."""

import hashlib
import json
import shutil
import subprocess
import sys

import pytest

from tests.unit_tests.determinism_reporting.test_author_performance import coverage, timings
from tests.unit_tests.determinism_reporting.test_paired_performance import (
    SCRIPTS,
    benchmark,
    load_module,
)

REVISION = "a" * 40


@pytest.fixture
def baseline(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monkeypatch.setitem(sys.modules, "benchmark", benchmark)
    monkeypatch.setitem(sys.modules, "author_evidence", load_module("author_evidence"))
    return load_module("baseline")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def artifacts(tmp_path):
    root = tmp_path / "downloaded"
    coverage_path = root / "unit-logs/coverage.json"
    leaderboard = root / "perf-logs/leaderboard.json"
    write_json(coverage_path, coverage())
    reports = [timings(phase) for phase in ("forward", "backward")]
    for report in reports:
        phase = report["measurement"]["phase"]
        case = leaderboard.parent / f"weighted_swiglu-bfloat16-{phase}"
        for run in report["runs"]:
            relative = f"pair-{run['pair']}/{run['revision_label']}-{run['mode']}"
            # Original runner paths are deliberately unavailable after transport.
            run["log_directory"] = "/old-CI-runner/" + phase + "/" + relative
            run["timing_log"] = run["log_directory"] + "/kernel.json"
            write_json(case / relative / "kernel.json", run["kernel"])
            (case / relative / "launcher.log").write_text(
                "Synthetic CPU fixture, not GPU evidence\n"
            )
        write_json(case / "benchmark.json", report)
    write_json(leaderboard, reports)
    return coverage_path, leaderboard


def test_publish_relocate_and_verify_without_the_original_runner(baseline, tmp_path):
    covered, leaderboard = artifacts(tmp_path)
    store = tmp_path / "store"
    result = baseline.publish(covered, leaderboard, store, REVISION, "CPU-contract-fixture")
    assert result["created"] and result["status"] == "not_gated"
    assert result["author_cases"] == 1
    directory = store / result["baseline_id"]
    archive = shutil.make_archive(str(tmp_path / "transport"), "zip", directory)
    relocated = tmp_path / "relocated"
    shutil.unpack_archive(archive, relocated)
    shutil.rmtree(covered.parents[1])
    shutil.rmtree(store)
    verified = baseline.verify(relocated, result["baseline_id"])
    assert verified["status"] == "not_gated"
    assert verified["files_verified"] == result["files_verified"]


def test_publication_reuses_identical_content_without_overwriting(baseline, tmp_path):
    covered, leaderboard = artifacts(tmp_path)
    store = tmp_path / "store"
    first = baseline.publish(covered, leaderboard, store, REVISION, "CPU-contract-fixture")
    path = store / first["baseline_id"] / "baseline.json"
    before = path.stat().st_mtime_ns
    second = baseline.publish(covered, leaderboard, store, REVISION, "CPU-contract-fixture")
    assert not second["created"] and first["baseline_id"] == second["baseline_id"]
    assert path.stat().st_mtime_ns == before


@pytest.mark.parametrize("location", ["report", "embedded", "raw"])
def test_diagnostic_results_cannot_be_published_even_if_status_is_relabelled(
    baseline, tmp_path, location
):
    covered, leaderboard = artifacts(tmp_path)
    reports = json.loads(leaderboard.read_text())
    if location == "report":
        reports[0]["measurement"]["diagnostic_only"] = True
    elif location == "embedded":
        reports[0]["runs"][0]["kernel"]["diagnostics"] = {"status": "observed"}
    else:
        path = next(leaderboard.parent.rglob("kernel.json"))
        raw = json.loads(path.read_text())
        raw["measurement"]["diagnostic_only"] = True
        raw["diagnostics"] = {"status": "observed"}
        write_json(path, raw)
    write_json(leaderboard, reports)
    joined = baseline.author_evidence.join(json.loads(covered.read_text()), reports, REVISION)
    assert joined["evidence_complete"] is (location == "raw")
    with pytest.raises(ValueError):
        baseline.publish(covered, leaderboard, tmp_path / "store", REVISION, "CPU-fixture")
    assert not (tmp_path / "store").exists()


@pytest.mark.parametrize(
    "problem", ["missing_raw", "raw_changed", "separate_report_changed", "duplicate_attempt"]
)
def test_separate_artifacts_must_support_the_embedded_report(baseline, tmp_path, problem):
    covered, leaderboard = artifacts(tmp_path)
    case = leaderboard.parent / "weighted_swiglu-bfloat16-forward"
    if problem == "missing_raw":
        (case / "pair-0/head-det/kernel.json").unlink()
    elif problem == "raw_changed":
        path = case / "pair-0/head-det/kernel.json"
        raw = json.loads(path.read_text())
        raw["samples_ms"][0] *= 2
        write_json(path, raw)
    elif problem == "separate_report_changed":
        path = case / "benchmark.json"
        report = json.loads(path.read_text())
        report["runs"][0]["median_ms"] *= 2
        write_json(path, report)
    else:
        shutil.copytree(case, leaderboard.parent / "retry")
    # Existing JSON-only joins cannot notice these independent-file changes.
    joined = baseline.author_evidence.join(
        json.loads(covered.read_text()), json.loads(leaderboard.read_text()), REVISION
    )
    assert joined["evidence_complete"]
    with pytest.raises(ValueError):
        baseline.publish(covered, leaderboard, tmp_path / "store", REVISION, "CPU-fixture")
    assert not (tmp_path / "store").exists()


def test_different_phase_allocations_cannot_form_one_baseline(baseline, tmp_path):
    covered, leaderboard = artifacts(tmp_path)
    reports = json.loads(leaderboard.read_text())
    backward = reports[1]
    case = leaderboard.parent / "weighted_swiglu-bfloat16-backward"
    for run in backward["runs"]:
        run["kernel"]["device_uuid"] = "different-synthetic-allocation"
        write_json(
            case / f"pair-{run['pair']}" / f"head-{run['mode']}" / "kernel.json", run["kernel"]
        )
    write_json(case / "benchmark.json", backward)
    write_json(leaderboard, reports)
    assert baseline.author_evidence.join(json.loads(covered.read_text()), reports, REVISION)[
        "evidence_complete"
    ]
    with pytest.raises(ValueError, match="different timing GPUs"):
        baseline.publish(covered, leaderboard, tmp_path / "store", REVISION, "CPU-fixture")


@pytest.mark.parametrize("problem", ["changed", "missing", "extra", "symlink", "wrong_id"])
def test_transport_damage_and_unexpected_files_are_rejected(baseline, tmp_path, problem):
    covered, leaderboard = artifacts(tmp_path)
    result = baseline.publish(covered, leaderboard, tmp_path / "store", REVISION, "CPU-fixture")
    directory = tmp_path / "store" / result["baseline_id"]
    path = next((directory / "performance").rglob("kernel.json"))
    expected = result["baseline_id"]
    if problem == "changed":
        path.write_text("{}")
    elif problem == "missing":
        path.unlink()
    elif problem == "extra":
        (directory / "unlisted.txt").write_text("extra")
    elif problem == "symlink":
        path.unlink()
        path.symlink_to(covered)
    else:
        expected = "0" * 64
    with pytest.raises(ValueError):
        baseline.verify(directory, expected)


def test_rehashed_raw_corruption_still_fails_semantic_verification(baseline, tmp_path):
    covered, leaderboard = artifacts(tmp_path)
    result = baseline.publish(covered, leaderboard, tmp_path / "store", REVISION, "CPU-fixture")
    directory = tmp_path / "store" / result["baseline_id"]
    path = next((directory / "performance").rglob("kernel.json"))
    raw = json.loads(path.read_text())
    raw["samples_ms"][0] *= 2
    write_json(path, raw)
    manifest = json.loads((directory / "baseline.json").read_text())
    content = path.read_bytes()
    manifest["files"][path.relative_to(directory).as_posix()] = {
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }
    write_json(directory / "baseline.json", manifest)
    with pytest.raises(ValueError, match="Separate raw timing"):
        baseline.verify(directory)


def test_rejected_author_evidence_cannot_be_published(baseline, tmp_path):
    covered, leaderboard = artifacts(tmp_path)
    report = json.loads(covered.read_text())
    report["cases"][0]["checks"][0]["status"] = "failed"
    write_json(covered, report)
    with pytest.raises(ValueError, match="complete, nonfailing"):
        baseline.publish(covered, leaderboard, tmp_path / "store", REVISION, "CPU-fixture")


def test_store_cannot_nest_inside_the_artifact_source(baseline, tmp_path):
    covered, leaderboard = artifacts(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        baseline.publish(
            covered, leaderboard, leaderboard.parent / "store", REVISION, "CPU-fixture"
        )


def test_cli_round_trip_uses_only_downloaded_files(tmp_path):
    covered, leaderboard = artifacts(tmp_path)
    command = [
        sys.executable,
        str(SCRIPTS / "baseline.py"),
        "publish",
        "--coverage",
        str(covered),
        "--leaderboard",
        str(leaderboard),
        "--store",
        str(tmp_path / "store"),
        "--revision",
        REVISION,
        "--origin",
        "CPU-fixture",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    published = json.loads(result.stdout)
    verified = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "baseline.py"),
            "verify",
            published["path"],
            "--expected-id",
            published["baseline_id"],
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(verified.stdout)["status"] == "not_gated"
