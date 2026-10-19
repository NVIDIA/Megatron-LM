# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise the GPU runner's consumption and fallback of CPU-selected plans."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("has_config_digest", [True, False])
def test_build_exports_config_identity_instead_of_manifest_digest(tmp_path, has_config_digest):
    workflow = yaml.safe_load((ROOT / ".github/workflows/_build_ci_container.yml").read_text())
    step = next(
        step
        for step in workflow["jobs"]["build-container"]["steps"]
        if step.get("id") == "unit-test-image"
    )
    metadata = {"containerimage.digest": "sha256:" + "b" * 64}
    if has_config_digest:
        metadata["containerimage.config.digest"] = "sha256:" + "a" * 64
    result = subprocess.run(
        ["bash", "-e", "-u", "-o", "pipefail"],
        input=step["run"],
        env={**os.environ, "BUILD_METADATA": json.dumps(metadata), "RUNNER_TEMP": str(tmp_path)},
        text=True,
        capture_output=True,
        timeout=10,
    )
    artifact = tmp_path / "unit-test-image/image-id.txt"
    assert (result.returncode == 0) == has_config_digest, result.stderr
    if has_config_digest:
        assert artifact.read_text().strip() == metadata["containerimage.config.digest"]
    else:
        assert not artifact.exists()


@pytest.mark.parametrize(
    ("validation_status", "test_status", "expected_status", "full_runs", "selected_runs"),
    [(0, 0, 0, 0, 2), (1, 0, 0, 1, 0), (0, 1, 1, 0, 1)],
)
def test_cpu_plan_execution(
    tmp_path, validation_status, test_status, expected_status, full_runs, selected_runs
):
    cache = tmp_path / "cache"
    for phase, content in (
        ("prod", "tests/unit_tests/test_example.py::test_one\n"),
        ("experimental", ""),
    ):
        directory = cache / ".testmon-work" / phase
        directory.mkdir(parents=True)
        (directory / "selected-tests").write_text(content)
        (directory / "mandatory-tests").write_text("")
    (tmp_path / ".coverage.unit_tests").touch()
    runner = (ROOT / "tests/unit_tests/run_ci_test.sh").read_text()
    functions = "\n".join(
        name + "() {" + runner.split(name + "() {", 1)[1].split("\n}\n", 1)[0] + "\n}"
        for name in ("run_selected_phase", "write_testmon_summary", "run_enforced_tests")
    )
    script = "\n".join(
        (
            "set -euo pipefail",
            "DISTRIBUTED_ARGS=() IGNORE_ARGS=()",
            "MARKER_ARG='not flaky_in_dev'",
            "UNIT_TEST_REPEAT=2 NUM_NODES=1 GPUS_PER_NODE=8 PLATFORM=h100",
            "BUCKET=tests/unit_tests/test_example.py",
            "git() { echo aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; }",
            'run_full_tests() { echo full >> "$EVENTS"; }',
            "run_testmon_phase() { echo 'unexpected GPU selection' >&2; return 99; }",
            "coverage() { return 0; }",
            "uv() {",
            '  if [[ "$1" == pip ]]; then return 0; fi',
            '  if [[ "$*" == *testmon_plan.py* ]]; then return "$VALIDATION_STATUS"; fi',
            '  echo "$*" >> "$EVENTS"',
            '  return "$TEST_STATUS"',
            "}",
            functions,
            "run_enforced_tests",
        )
    )
    events = tmp_path / "events"
    result = subprocess.run(
        ["bash"],
        input=script,
        cwd=tmp_path,
        env={
            **os.environ,
            "UNIT_TESTMON_MODE": "preselected",
            "UNIT_TESTMON_CACHE_DIR": str(cache),
            "EVENTS": str(events),
            "VALIDATION_STATUS": str(validation_status),
            "TEST_STATUS": str(test_status),
        },
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == expected_status, result.stdout + result.stderr
    executed = events.read_text().splitlines()
    assert executed.count("full") == full_runs
    assert sum("torch.distributed.run" in event for event in executed) == selected_runs
    assert all("test_example.py::test_one" in event for event in executed if event != "full")
    assert "unexpected GPU selection" not in result.stderr
