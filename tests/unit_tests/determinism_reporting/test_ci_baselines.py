# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CI transport contracts with synthetic CPU fixtures, never GPU acceptance."""

import json
import os
import shutil
import subprocess

import pytest
import yaml

from tests.unit_tests.determinism_reporting.test_paired_performance import (
    ROOT,
    SCRIPTS,
    load_module,
)
from tests.unit_tests.determinism_reporting.test_performance_baseline import (
    REVISION,
    artifacts,
    write_json,
)

REPOSITORY = "example/synthetic-fixture"
RUN = 123
ATTEMPT = 2


@pytest.fixture
def ci(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return load_module("ci_artifacts")


def inputs(ci, tmp_path, platform="dgx_h100"):
    covered, leaderboard = artifacts(tmp_path)
    root = tmp_path / "downloaded"
    covered.rename(covered.with_name("determinism-coverage.json"))
    if platform == "dgx_gb200":

        def blackwell(value):
            if isinstance(value, dict):
                if "gpu" in value and "capability" in value:
                    value.update(gpu="GB200 synthetic fixture", capability=[10, 0])
                    value["environment"]["CUDA_DEVICE_MAX_CONNECTIONS"] = "32"
                for child in value.values():
                    blackwell(child)
            elif isinstance(value, list):
                for child in value:
                    blackwell(child)

        for path in root.rglob("*.json"):
            value = json.loads(path.read_text())
            blackwell(value)
            write_json(path, value)
    for directory, case in zip((covered.parent, leaderboard.parent), ci.CASES):
        record = ci.stamp(
            directory, REPOSITORY, REVISION, RUN, ATTEMPT, platform, case, "success", 0
        )
        directory.rename(root / record["artifact_name"])
    return root


def consume(ci, root, output, platforms=None):
    return ci.collect(root, output, REPOSITORY, REVISION, RUN, ATTEMPT, platforms or ["dgx_h100"])


def metadata(root, kind):
    return next(
        path
        for path in root.glob("*/determinism-ci-artifact.json")
        if f"-{kind}-" in path.parent.name
    )


def test_both_platforms_publish_independently_and_remain_unbudgeted(ci, tmp_path):
    root = inputs(ci, tmp_path / "h100")
    blackwell = inputs(ci, tmp_path / "gb200", "dgx_gb200")
    for directory in blackwell.iterdir():
        shutil.move(directory, root / directory.name)
    report = consume(ci, root, tmp_path / "result", ["dgx_h100", "dgx_gb200"])
    assert report["status"] == "complete"
    assert len(report["inputs"]) == 4
    assert {row["status"] for row in report["platforms"]} == {"not_gated"}
    assert len({row["baseline_id"] for row in report["platforms"]}) == 2
    # Verify the derived artifact after another archive/download cycle.
    archive = shutil.make_archive(str(tmp_path / "derived"), "zip", tmp_path / "result")
    shutil.unpack_archive(archive, tmp_path / "relocated")
    shutil.rmtree(tmp_path / "result")
    shutil.rmtree(root)
    for row in report["platforms"]:
        assert (
            ci.baseline.verify(tmp_path / "relocated" / row["path"], row["baseline_id"])["status"]
            == "not_gated"
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("repository", "another/repository"),
        ("revision", "b" * 40),
        ("run_id", RUN + 1),
        ("run_attempt", ATTEMPT - 1),
        ("run_attempt", str(ATTEMPT)),
        ("producer_outcome", "failure"),
        ("producer_outcome", "skipped"),
        ("producer_exit_code", 1),
        ("producer_exit_code", False),
        ("artifact_name", "renamed"),
        ("test_case", "unrelated_test"),
    ],
)
def test_wrong_provenance_or_failed_producer_cannot_publish(ci, tmp_path, key, value):
    root = inputs(ci, tmp_path)
    path = metadata(root, "coverage")
    record = json.loads(path.read_text())
    record[key] = value
    write_json(path, record)
    report = consume(ci, root, tmp_path / "result")
    assert report["status"] == "not_verified" and report["errors"]
    assert not (tmp_path / "result/baselines").exists()
    assert (tmp_path / "result/report.json").is_file()


@pytest.mark.parametrize(
    "problem",
    [
        "missing_provenance",
        "missing_timing",
        "upload_retry",
        "second_producer",
        "failed_duplicate",
        "symlink",
    ],
)
def test_missing_or_ambiguous_attempts_are_not_silently_selected(ci, tmp_path, problem):
    root = inputs(ci, tmp_path)
    path = metadata(root, "performance")
    if problem == "missing_provenance":
        path.unlink()
    elif problem == "missing_timing":
        shutil.rmtree(path.parent)
    elif problem == "symlink":
        (path.parent / "link").symlink_to(path)
    else:
        duplicate = root / (path.parent.name + "-retry")
        shutil.copytree(path.parent, duplicate)
        if problem in ("second_producer", "failed_duplicate"):
            record = ci.stamp(
                duplicate,
                REPOSITORY,
                REVISION,
                RUN,
                ATTEMPT,
                "dgx_h100",
                "determinism_kernel_perf",
                "success" if problem == "second_producer" else "failure",
                0,
            )
            duplicate.rename(root / record["artifact_name"])
    report = consume(ci, root, tmp_path / "result")
    assert report["status"] == "not_verified"
    assert not (tmp_path / "result/baselines").exists()


def test_single_transport_retry_is_accepted_without_picking_between_copies(ci, tmp_path):
    root = inputs(ci, tmp_path)
    directory = metadata(root, "performance").parent
    directory.rename(root / (directory.name + "-retry"))
    assert consume(ci, root, tmp_path / "result")["status"] == "complete"


def test_verification_preserves_previous_output_and_downloaded_inputs(ci, tmp_path):
    root = inputs(ci, tmp_path)
    consume(ci, root, tmp_path / "result")
    previous = (tmp_path / "result/report.json").read_bytes()
    with pytest.raises(ValueError, match="empty output"):
        consume(ci, root, tmp_path / "result")
    assert (tmp_path / "result/report.json").read_bytes() == previous
    with pytest.raises(ValueError, match="outside downloaded"):
        consume(ci, root, root / "output")
    assert not (root / "output").exists()


@pytest.mark.parametrize(
    "problem", ["changed_raw", "stale_inner_source", "nondeterministic", "wrong_gpu"]
)
def test_successful_producer_metadata_does_not_replace_evidence_validation(ci, tmp_path, problem):
    root = inputs(ci, tmp_path)
    if problem == "changed_raw":
        path = next(root.rglob("kernel.json"))
        value = json.loads(path.read_text())
        value["samples_ms"][0] *= 2
    elif problem in ("stale_inner_source", "nondeterministic"):
        path = next(root.rglob("determinism-coverage.json"))
        value = json.loads(path.read_text())
        if problem == "stale_inner_source":
            value["context"]["revision"] = "c" * 40
        else:
            value["cases"][0]["status"] = "verified_nondeterministic"
    else:
        path = next(root.rglob("leaderboard.json"))
        value = json.loads(path.read_text())
        value[0]["runs"][0]["kernel"]["case_signature"]["runtime"]["gpu"] = "another GPU"
    write_json(path, value)
    report = consume(ci, root, tmp_path / "result")
    assert report["status"] == "not_verified"
    assert report["platforms"][0]["reason"]
    assert report["platforms"][0]["reason"] in (tmp_path / "result/report.md").read_text()
    assert not (tmp_path / "result/baselines").exists()


def test_selected_platform_cannot_disappear_but_unselected_producer_is_recorded(ci, tmp_path):
    root = inputs(ci, tmp_path)
    report = consume(ci, root, tmp_path / "both", ["dgx_h100", "dgx_gb200"])
    assert report["status"] == "not_verified"
    assert report["platforms"][1]["status"] == "not_verified"
    directory = root / "staging"
    record = ci.stamp(
        directory,
        REPOSITORY,
        REVISION,
        RUN,
        ATTEMPT,
        "dgx_gb200",
        next(iter(ci.CASES)),
        "failure",
        1,
    )
    directory.rename(root / record["artifact_name"])
    report = consume(ci, root, tmp_path / "h100-only")
    assert report["status"] == "complete"
    assert report["ignored_inputs"][0]["provenance"]["producer_outcome"] == "failure"


def test_actual_ci_shell_consumes_named_artifacts_and_retains_failure_report(ci, tmp_path):
    root = inputs(ci, tmp_path)
    root.rename(tmp_path / "determinism-inputs")
    # The workflow uses this source-tree path; only artifact data are downloaded.
    (tmp_path / "tests").symlink_to(ROOT / "tests", target_is_directory=True)
    workflow = yaml.safe_load((ROOT / ".github/workflows/cicd-main.yml").read_text())
    job = workflow["jobs"]["cicd-determinism-baselines"]
    step = next(
        step for step in job["steps"] if step["name"] == "Verify replay and timing artifacts"
    )
    environment = {
        **os.environ,
        "GITHUB_REPOSITORY": REPOSITORY,
        "GITHUB_RUN_ID": str(RUN),
        "GITHUB_RUN_ATTEMPT": str(ATTEMPT),
        "REVISION": REVISION,
        "H100_SELECTED": "true",
        "GB200_SELECTED": "false",
        "H100_COLLECTIVE_SELECTED": "false",
        "GB200_COLLECTIVE_SELECTED": "false",
    }
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", step["run"]],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        json.loads((tmp_path / "determinism-baselines/report.json").read_text())["status"]
        == "complete"
    )
    shutil.rmtree(tmp_path / "determinism-baselines")
    environment["GB200_SELECTED"] = "true"
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", step["run"]],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert (
        json.loads((tmp_path / "determinism-baselines/report.json").read_text())["status"]
        == "not_verified"
    )


@pytest.mark.parametrize("test_case", ["determinism_kernel_perf", "determinism_collective_perf"])
def test_actual_producer_shell_stamps_failure_without_changing_it(tmp_path, test_case):
    action = yaml.safe_load((ROOT / ".github/actions/action.yml").read_text())
    step = next(
        step for step in action["runs"]["steps"] if step.get("id") == "determinism-artifact"
    )
    environment = {
        **os.environ,
        "LOG_BASE": str(tmp_path / "logs"),
        "TEST_CASE": test_case,
        "PLATFORM": "dgx_h100",
        "PRODUCER_OUTCOME": "failure",
        "PRODUCER_EXIT_CODE": "1",
        "GITHUB_REPOSITORY": REPOSITORY,
        "GITHUB_RUN_ID": str(RUN),
        "GITHUB_RUN_ATTEMPT": str(ATTEMPT),
        "GITHUB_OUTPUT": str(tmp_path / "outputs"),
    }
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", step["run"]],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads((tmp_path / "logs/determinism-ci-artifact.json").read_text())
    assert record["producer_outcome"] == "failure" and record["producer_exit_code"] == 1
    assert (
        record["revision"]
        == subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    )
    assert (tmp_path / "outputs").read_text() == f"name={record['artifact_name']}\n"
