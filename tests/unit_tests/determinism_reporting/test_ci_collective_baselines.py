# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CI collective transport checks; synthetic metadata never substitutes for a CI run."""

import json
import os
import shutil
import subprocess
import sys

import pytest
import yaml

from tests.unit_tests.determinism_reporting.test_ci_baselines import ATTEMPT, REPOSITORY, RUN
from tests.unit_tests.determinism_reporting.test_ci_baselines import inputs as activation_inputs
from tests.unit_tests.determinism_reporting.test_collective_baseline import (
    REVISION,
    artifacts,
    consumer,
    write_json,
)
from tests.unit_tests.determinism_reporting.test_paired_performance import (
    ROOT,
    SCRIPTS,
    load_module,
)


@pytest.fixture
def ci(monkeypatch, consumer):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return load_module("ci_artifacts")


def inputs(ci, consumer, root, platform="dgx_h100"):
    directory = root / platform
    artifacts(consumer, directory / "collective-performance", platform=platform)
    record = ci.stamp(
        directory,
        REPOSITORY,
        REVISION,
        RUN,
        ATTEMPT,
        platform,
        "determinism_collective_perf",
        "success",
        0,
    )
    target = root / record["artifact_name"]
    directory.rename(target)
    return target


def collect(ci, root, output, platforms=None):
    return ci.collect(
        root, output, REPOSITORY, REVISION, RUN, ATTEMPT, [], platforms or ["dgx_h100"]
    )


def test_both_collective_platforms_verify_after_transport_without_torch(ci, consumer, tmp_path):
    root = tmp_path / "inputs"
    for platform in ("dgx_h100", "dgx_gb200"):
        inputs(ci, consumer, root, platform)
    command = [
        sys.executable,
        "-S",
        str(SCRIPTS / "ci_artifacts.py"),
        "collect",
        "--artifacts",
        str(root),
        "--output",
        str(tmp_path / "output"),
        "--repository",
        REPOSITORY,
        "--revision",
        REVISION,
        "--run-id",
        str(RUN),
        "--attempt",
        str(ATTEMPT),
        "--collective-platform",
        "dgx_h100",
        "--collective-platform",
        "dgx_gb200",
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "output/report.json").read_text())
    assert report["status"] == "complete" and len(report["platforms"]) == 2
    assert all(
        row["kind"] == "collective" and row["status"] == "not_gated" for row in report["platforms"]
    )
    archive = shutil.make_archive(str(tmp_path / "transport"), "zip", tmp_path / "output")
    shutil.unpack_archive(archive, tmp_path / "relocated")
    shutil.rmtree(root)
    shutil.rmtree(tmp_path / "output")
    for row in report["platforms"]:
        result = consumer.baseline.verify(tmp_path / "relocated" / row["path"], row["baseline_id"])
        assert result["status"] == "not_gated" and result["rank_samples"] == 144


def test_activation_and_collective_evidence_are_selected_separately(ci, consumer, tmp_path):
    root = activation_inputs(ci, tmp_path / "activation")
    inputs(ci, consumer, root)
    report = ci.collect(
        root, tmp_path / "both", REPOSITORY, REVISION, RUN, ATTEMPT, ["dgx_h100"], ["dgx_h100"]
    )
    assert report["status"] == "complete" and len(report["platforms"]) == 2
    assert {row["kind"] for row in report["platforms"]} == {"activation", "collective"}
    assert len({row["baseline_id"] for row in report["platforms"]}) == 2
    scoped = collect(ci, root, tmp_path / "collectives-only")
    assert scoped["status"] == "complete" and len(scoped["ignored_inputs"]) == 2


def test_activation_artifacts_cannot_replace_missing_collective_evidence(ci, tmp_path):
    root = activation_inputs(ci, tmp_path)
    report = collect(ci, root, tmp_path / "output")
    assert report["status"] == "not_verified"
    assert "Expected one successful collective artifact" in report["platforms"][0]["reason"]


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate",
        "failed",
        "wrong_revision",
        "wrong_attempt",
        "wrong_gpu",
        "missing_capture",
        "changed_blob",
        "second_benchmark",
        "relabelled_activation",
    ],
)
def test_selected_collective_artifacts_require_matching_provenance_and_bytes(
    ci, consumer, tmp_path, fault
):
    root = tmp_path / "inputs"
    directory = inputs(ci, consumer, root)
    metadata = directory / ci.METADATA
    record = json.loads(metadata.read_text())
    if fault == "duplicate":
        retry = root / (directory.name + "-retry")
        shutil.copytree(directory, retry)
        (retry / "changed.log").write_text("different bytes")
    elif fault == "failed":
        record["producer_outcome"] = "failure"
    elif fault == "wrong_revision":
        record["revision"] = "b" * 40
    elif fault == "wrong_attempt":
        record["run_attempt"] -= 1
    elif fault == "missing_capture":
        shutil.rmtree(directory / "collective-performance/capture")
    elif fault == "changed_blob":
        next(directory.rglob("*.bin")).write_bytes(b"changed")
    elif fault == "second_benchmark":
        write_json(directory / "retry/benchmark.json", {})
    elif fault == "relabelled_activation":
        record["test_case"] = "determinism_kernel_perf"
    else:
        path = directory / "collective-performance/timing/benchmark.json"
        value = json.loads(path.read_text())
        value["capture"]["context"]["gpu"] = "Synthetic GB200"
        write_json(path, value)
    write_json(metadata, record)
    report = collect(ci, root, tmp_path / "output")
    assert report["status"] == "not_verified"
    assert not (tmp_path / "output/collective-baselines").exists()


def test_single_retried_upload_is_accepted_without_selecting_between_duplicates(
    ci, consumer, tmp_path
):
    root = tmp_path / "inputs"
    directory = inputs(ci, consumer, root)
    directory.rename(root / (directory.name + "-retry"))
    assert collect(ci, root, tmp_path / "output")["status"] == "complete"


@pytest.mark.parametrize("selected", [["dgx_h100"], ["dgx_gb200"], ["dgx_h100", "dgx_gb200"]])
def test_actual_workflow_shell_selects_collectives_and_rejects_missing_uploads(
    ci, consumer, tmp_path, selected
):
    root = tmp_path / "determinism-inputs"
    for platform in selected:
        inputs(ci, consumer, root, platform)
    (tmp_path / "tests").symlink_to(ROOT / "tests", target_is_directory=True)
    workflow = yaml.safe_load((ROOT / ".github/workflows/cicd-main.yml").read_text())
    step = next(
        row
        for row in workflow["jobs"]["cicd-determinism-baselines"]["steps"]
        if row["name"] == "Verify replay and timing artifacts"
    )
    environment = {
        **os.environ,
        "GITHUB_REPOSITORY": REPOSITORY,
        "GITHUB_RUN_ID": str(RUN),
        "GITHUB_RUN_ATTEMPT": str(ATTEMPT),
        "REVISION": REVISION,
        "H100_SELECTED": "false",
        "GB200_SELECTED": "false",
        "H100_COLLECTIVE_SELECTED": "true" if "dgx_h100" in selected else "false",
        "GB200_COLLECTIVE_SELECTED": "true" if "dgx_gb200" in selected else "false",
    }

    def execute():
        result = subprocess.run(
            ["bash", "-euo", "pipefail", "-c", step["run"]],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
        )
        report = json.loads((tmp_path / "determinism-baselines/report.json").read_text())
        return result, report

    result, report = execute()
    assert result.returncode == 0, result.stdout + result.stderr
    assert report["status"] == "complete" and len(report["platforms"]) == len(selected)
    assert all(
        row["kind"] == "collective" and row["status"] == "not_gated" for row in report["platforms"]
    )
    shutil.rmtree(tmp_path / "determinism-baselines")
    shutil.rmtree(next(root.iterdir()))
    result, report = execute()
    assert result.returncode == 1 and report["status"] == "not_verified"
    assert any(row["status"] == "not_verified" for row in report["platforms"])
