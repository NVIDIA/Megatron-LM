# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]
WRAPPER_PATH = ROOT / "tests/unit_tests/testmon_selector.py"
for name, filename in (
    ("testmon_cache", "testmon_cache.py"),
    ("unit_testmon_wrapper", "testmon_selector.py"),
):
    spec = importlib.util.spec_from_file_location(name, WRAPPER_PATH.with_name(filename))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
wrapper = sys.modules["unit_testmon_wrapper"]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text()


def _function(source: str, name: str) -> str:
    return source.split(f"{name}() {{", 1)[1].split("\n}\n", 1)[0]


def _invoke(project, cache, mode, phase="prod", rank=0):
    return subprocess.run(
        [
            sys.executable,
            str(WRAPPER_PATH),
            "--mode",
            mode,
            "--cache-dir",
            str(cache),
            "--phase",
            phase,
            "--",
            "-q",
            "-c",
            str(project / "pytest.ini"),
            "tests",
        ],
        cwd=project,
        env={
            **os.environ,
            "RANK": str(rank),
            "WORLD_SIZE": "4",
            "PYTHONPATH": str(project),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        text=True,
        capture_output=True,
        timeout=30,
    )


@pytest.fixture
def project(tmp_path):
    pytest.importorskip("testmon")
    project = tmp_path / "project"
    (project / "tests").mkdir(parents=True)
    (project / "pytest.ini").write_text("[pytest]\n")
    (project / "app.py").write_text(
        "def active(value):\n    return value + 1\n\n" "def unused(value):\n    return value - 1\n"
    )
    (project / "tests/test_app.py").write_text(
        "from app import active\n\ndef test_active():\n    assert active(1) == 2\n"
    )
    (project / "tests/test_other.py").write_text("def test_other():\n    assert 1 + 1 == 2\n")
    return project


@pytest.fixture
def cache(project, tmp_path):
    cache = tmp_path / "cache"
    result = _invoke(project, cache, "baseline")
    assert result.returncode == 0, result.stdout + result.stderr
    return cache


def _snapshot(directory):
    return {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in directory.iterdir()
        if path.is_file()
    }


def test_dependency_override_tracks_only_selected_packages(monkeypatch):
    monkeypatch.setattr(
        wrapper,
        "distributions",
        lambda: (
            SimpleNamespace(metadata={"Name": name})
            for name in (
                "numpy",
                "pytest",
                "torch",
                "Transformer_Engine",
                "Transformer_Engine_Torch",
                "transformers",
                "triton",
                "megatron-core",
                "pytest-testmon",
            )
        ),
    )
    assert (
        wrapper._testmon_dependency_override()
        == "testmon_ignore_dependencies=megatron-core pytest-testmon transformers"
    )


def test_successful_baseline_records_each_phase(project, cache):
    result = _invoke(project, cache, "baseline", "experimental")
    assert result.returncode == 0, result.stdout + result.stderr
    for phase in ("prod", "experimental"):
        assert (cache / phase / ".testmondata").is_file()
        assert (cache / phase / "metadata.json").is_file()


def test_failed_baseline_cannot_reuse_old_metadata(project, cache):
    (project / "tests/test_app.py").write_text("def test_failure():\n    assert False\n")
    result = _invoke(project, cache, "baseline")
    assert result.returncode == 1
    assert not (cache / "prod/metadata.json").exists()


def test_empty_baseline_phase_is_recorded(project, tmp_path):
    for test_file in (project / "tests").iterdir():
        test_file.unlink()
    cache = tmp_path / "empty-cache"
    result = _invoke(project, cache, "baseline")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (cache / "prod/metadata.json").is_file()
    selected = _invoke(project, cache, "select")
    assert selected.returncode == 0, selected.stdout + selected.stderr
    assert (cache / ".testmon-work/prod/rank-0/selected-tests").read_text() == ""


def test_nonzero_baseline_rank_runs_without_recording(project, tmp_path):
    cache = tmp_path / "rank-three-cache"
    result = _invoke(project, cache, "baseline", rank=3)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not cache.exists()


@pytest.mark.parametrize("rank", (0, 3))
def test_zero_selection_uses_private_copy_and_keeps_cache_readonly(project, cache, rank):
    phase = cache / "prod"
    before = _snapshot(phase)
    for path in phase.iterdir():
        path.chmod(0o444)
    phase.chmod(0o555)
    try:
        result = _invoke(project, cache, "select", rank=rank)
        assert result.returncode == 0, result.stdout + result.stderr
        private = cache / f".testmon-work/prod/rank-{rank}"
        assert (private / ".testmondata").is_file()
        assert (private / "selected-tests").read_text() == ""
        assert _snapshot(phase) == before
    finally:
        phase.chmod(0o755)


def test_changed_dependency_is_selected_without_learning_or_execution(project, cache):
    before = _snapshot(cache / "prod")
    # Executing this test would fail; selection must only collect it.
    (project / "app.py").write_text("def active(value):\n    return value + 2\n")
    for _ in range(2):
        result = _invoke(project, cache, "select")
        assert result.returncode == 0, result.stdout + result.stderr
        selected = cache / ".testmon-work/prod/rank-0/selected-tests"
        assert selected.read_text().splitlines() == ["tests/test_app.py::test_active"]
        assert _snapshot(cache / "prod") == before


def test_new_test_file_is_discovered_from_old_baseline(project, cache):
    (project / "tests/test_new.py").write_text(
        "def test_new():\n    raise AssertionError('selection must not execute tests')\n"
    )
    result = _invoke(project, cache, "select")
    assert result.returncode == 0, result.stdout + result.stderr
    selected = cache / ".testmon-work/prod/rank-0/selected-tests"
    assert selected.read_text().splitlines() == ["tests/test_new.py::test_new"]


def test_changed_function_selects_and_runs_only_affected_cases_in_same_file(project, tmp_path):
    parameter_ids = ["space value", 'quote " and $value']
    (project / "tests/test_app.py").write_text(
        "import pytest\nfrom app import active, unused\n\n"
        "class TestApp:\n"
        f"    @pytest.mark.parametrize('value', [1, 2], ids={parameter_ids!r})\n"
        "    def test_active(self, value):\n"
        "        assert active(value) == value + 1\n\n"
        "    def test_unused(self):\n"
        "        assert unused(1) == 0\n"
    )
    cache = tmp_path / "cache"
    baseline = _invoke(project, cache, "baseline")
    assert baseline.returncode == 0, baseline.stdout + baseline.stderr
    before = _snapshot(cache / "prod")
    (project / "app.py").write_text(
        "def active(value):\n    return 1 + value\n\n" "def unused(value):\n    return value - 1\n"
    )

    result = _invoke(project, cache, "select")
    assert result.returncode == 0, result.stdout + result.stderr
    expected = sorted(
        f"tests/test_app.py::TestApp::test_active[{parameter_id}]" for parameter_id in parameter_ids
    )
    phase = cache / ".testmon-work/prod"
    rank_zero = phase / "rank-0/selected-tests"
    assert rank_zero.read_text().splitlines() == expected
    assert _snapshot(cache / "prod") == before

    # Merge overlapping, differently ordered rank results without dropping node IDs.
    rank_zero.write_text("\n".join(reversed(expected)) + "\n")
    (phase / "rank-1").mkdir()
    (phase / "rank-1/selected-tests").write_text(expected[0] + "\n")
    runner = _source("tests/unit_tests/run_ci_test.sh")
    result = subprocess.run(
        ["bash", "-e", "-u", "-o", "pipefail"],
        input="\n".join(
            (
                "NUM_NODES=1",
                "GPUS_PER_NODE=2",
                "DISTRIBUTED_ARGS=()",
                "IGNORE_ARGS=()",
                "MARKER_ARG='not flaky'",
                # Replace only uv/torchrun; run the actual coverage/pytest command locally.
                "uv() {",
                '    [[ "$1 $2 $3 $4 $5" == "run --no-sync python -m torch.distributed.run" ]]',
                "    shift 5",
                '    "$TEST_PYTHON" "$@"',
                "}",
                "merge_rank_selections() {" + _function(runner, "merge_rank_selections") + "\n}",
                "run_selected_phase() {" + _function(runner, "run_selected_phase") + "\n}",
                "merge_rank_selections prod",
                "run_selected_phase prod",
            )
        ),
        cwd=project,
        env={
            **os.environ,
            "TEST_PYTHON": sys.executable,
            "PYTHONPATH": str(project),
            "PYTHONDONTWRITEBYTECODE": "1",
            "UNIT_TESTMON_CACHE_DIR": str(cache),
        },
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (phase / "selected-tests").read_text().splitlines() == expected
    assert "collected 2 items" in result.stdout
    assert "2 passed" in result.stdout
    assert _snapshot(cache / "prod") == before


@pytest.mark.parametrize("problem", ("missing", "corrupt", "schema", "runtime"))
def test_invalid_baseline_falls_back_before_private_copy(project, cache, problem):
    database = cache / "prod/.testmondata"
    if problem == "missing":
        database.unlink()
    elif problem == "corrupt":
        database.write_bytes(b"not a sqlite database")
    elif problem == "schema":
        with sqlite3.connect(database) as connection:
            connection.execute("PRAGMA user_version = 999")
        connection.close()
    else:
        metadata_path = cache / "prod/metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["runtime"]["python"] = "0.0.0"
        metadata_path.write_text(json.dumps(metadata))
    before = _snapshot(cache / "prod")
    runner = _source("tests/unit_tests/run_ci_test.sh")
    script = "\n".join(
        (
            "set -euo pipefail",
            'BUCKET="$PROJECT/tests"',
            "IGNORE_ARGS=()",
            "MARKER_ARG='not flaky'",
            "UNIT_TEST_REPEAT=1",
            'run_full_tests() { echo FULL_BUCKET; }',
            'write_testmon_summary() { printf "%s\\n" "$1"; }',
            "merge_rank_selections() { return 0; }",
            'run_testmon_phase() { RANK=0 WORLD_SIZE=1 "$TEST_PYTHON" "$WRAPPER" '
            '--mode "$1" --phase "$2" --cache-dir "$UNIT_TESTMON_CACHE_DIR" '
            '-- -q -c "$PROJECT/pytest.ini" "$PROJECT/tests"; }',
            "run_enforced_tests() {" + _function(runner, "run_enforced_tests") + "\n}",
            "run_enforced_tests",
        )
    )
    result = subprocess.run(
        ["bash"],
        input=script,
        cwd=project,
        env={
            **os.environ,
            "PROJECT": str(project),
            "PYTHONPATH": str(project),
            "TEST_PYTHON": sys.executable,
            "WRAPPER": str(WRAPPER_PATH),
            "UNIT_TESTMON_CACHE_DIR": str(cache),
        },
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "full fallback: Testmon selection failed" in result.stdout
    assert result.stdout.count("FULL_BUCKET") == 1
    assert "Testmon wrapper (select/prod):" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (cache / ".testmon-work").exists()
    assert _snapshot(cache / "prod") == before


def test_empty_phase_does_not_launch_pytest(tmp_path):
    phase = tmp_path / ".testmon-work/prod"
    phase.mkdir(parents=True)
    (phase / "selected-tests").write_text("")
    runner = _source("tests/unit_tests/run_ci_test.sh")
    result = subprocess.run(
        ["bash", "-e", "-u", "-o", "pipefail"],
        input="\n".join(
            (
                "DISTRIBUTED_ARGS=()",
                "uv() { echo UNEXPECTED_EXECUTION; return 99; }",
                "run_selected_phase() {" + _function(runner, "run_selected_phase") + "\n}",
                "run_selected_phase prod",
            )
        ),
        env={**os.environ, "UNIT_TESTMON_CACHE_DIR": str(tmp_path)},
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Testmon selected no prod tests." in result.stdout
    assert "UNEXPECTED_EXECUTION" not in result.stdout


def test_selected_test_failure_does_not_run_full_bucket(tmp_path):
    runner = _source("tests/unit_tests/run_ci_test.sh")
    result = subprocess.run(
        ["bash", "-e", "-u", "-o", "pipefail"],
        input="\n".join(
            (
                "BUCKET=tests/unit_tests/example",
                "IGNORE_ARGS=()",
                "MARKER_ARG='not flaky'",
                "UNIT_TEST_REPEAT=1",
                "run_testmon_phase() { return 0; }",
                'merge_rank_selections() { mkdir -p "$UNIT_TESTMON_CACHE_DIR/.testmon-work/$1"; '
                'echo tests/unit_tests/test_example.py > "$UNIT_TESTMON_CACHE_DIR/.testmon-work/$1/selected-tests"; }',
                "run_selected_phase() { return 1; }",
                "run_full_tests() { echo UNEXPECTED_FULL_BUCKET; }",
                "run_enforced_tests() {" + _function(runner, "run_enforced_tests") + "\n}",
                "run_enforced_tests",
            )
        ),
        env={**os.environ, "UNIT_TESTMON_CACHE_DIR": str(tmp_path)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "UNEXPECTED_FULL_BUCKET" not in result.stdout


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, "true"),
        ({"HAS_UNIT_TESTMON": "false"}, "false"),
        ({"EVENT_NAME": "merge_group"}, "false"),
        ({"REF": "refs/heads/main"}, "false"),
        ({"LABELS_VALID": "false"}, "false"),
        ({"HAS_RUN_TESTS": "true"}, "false"),
        ({"HAS_RUN_FUNCTIONAL": "true"}, "false"),
        ({"FORCE_RUN_ALL": "true"}, "false"),
        ({"HAS_LTS": "true"}, "false"),
    ],
)
def test_pr_label_gate(overrides, expected):
    workflow = _source(".github/workflows/cicd-main.yml")
    start = workflow.index("          UNIT_TESTMON_ELIGIBLE=false")
    end = workflow.index('\n\n          echo "scope=', start)
    gate = textwrap.dedent(workflow[start:end])
    environment = {
        **os.environ,
        "LABELS_VALID": "true",
        "HAS_UNIT_TESTMON": "true",
        "EVENT_NAME": "push",
        "REF": "refs/heads/pull-request/123",
        "HAS_RUN_TESTS": "false",
        "HAS_RUN_FUNCTIONAL": "false",
        "FORCE_RUN_ALL": "false",
        "HAS_LTS": "false",
        **overrides,
    }
    result = subprocess.run(
        ["bash", "-e", "-u", "-o", "pipefail"],
        input=gate + '\nprintf "%s\\n" "$UNIT_TESTMON_ELIGIBLE"\n',
        text=True,
        capture_output=True,
        check=True,
        env=environment,
    )
    assert result.stdout.strip() == expected


def test_mode_is_passed_as_container_environment():
    launcher = _source("tests/test_utils/python_scripts/launch_nemo_run_workload.py")
    h100_recipe = _source("tests/test_utils/recipes/h100/unit-tests.yaml")
    gb200_recipe = _source("tests/test_utils/recipes/gb200/unit-tests.yaml")

    assert '"UNIT_TESTMON_MODE": unit_testmon_mode' in launcher
    assert '["full", "enforce", "baseline"]' in launcher
    assert "{unit_testmon_mode}" not in h100_recipe + gb200_recipe
