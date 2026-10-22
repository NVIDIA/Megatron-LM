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


def _invoke(project, cache, mode, phase="prod", rank=0, pytest_args=()):
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
            *pytest_args,
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
def test_baseline_can_start_spawned_checkpoint_workers(project, tmp_path, rank):
    (project / "tests/test_app.py").write_text(textwrap.dedent(f"""\
            import multiprocessing
            import sys
            from pathlib import Path

            def test_spawned_checkpoint_worker():
                script_dir = Path({str(WRAPPER_PATH.parent)!r}).resolve()
                assert all(Path(entry or '.').resolve() != script_dir for entry in sys.path)
                assert 'tests.unit_tests' not in sys.modules
                with multiprocessing.get_context('spawn').Manager() as manager:
                    queue = manager.Queue()
                    queue.put('checkpoint-worker-ready')
                    assert queue.get(timeout=5) == 'checkpoint-worker-ready'
            """))
    result = _invoke(project, tmp_path / "spawn-cache", "baseline", rank=rank)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout


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
                'apply_mandatory_tests() { : > "$UNIT_TESTMON_CACHE_DIR/.testmon-work/$1/mandatory-tests"; }',
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
        ({"UNIT_TESTMON_REQUESTED": "false"}, "false"),
        ({"EVENT_NAME": "merge_group"}, "false"),
        ({"REF": "refs/heads/main"}, "false"),
        ({"LABELS_VALID": "false"}, "false"),
        ({"HAS_RUN_TESTS": "true"}, "false"),
        ({"HAS_RUN_FUNCTIONAL": "true"}, "false"),
        ({"FORCE_RUN_ALL": "true"}, "false"),
        ({"HAS_LTS": "true"}, "false"),
    ],
)
def test_pr_resolved_testmon_gate(overrides, expected):
    workflow = _source(".github/workflows/cicd-main.yml")
    start = workflow.index("          UNIT_TESTMON_ELIGIBLE=false")
    end = workflow.index('\n\n          echo "scope=', start)
    gate = textwrap.dedent(workflow[start:end])
    environment = {
        **os.environ,
        "LABELS_VALID": "true",
        "UNIT_TESTMON_REQUESTED": "true",
        "IS_CI_WORKLOAD": "false",
        "IS_MERGE_GROUP": "false",
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
    assert '["full", "enforce", "preselected", "baseline"]' in launcher
    assert "{unit_testmon_mode}" not in h100_recipe + gb200_recipe


@pytest.fixture
def cpu_baseline(project, tmp_path):
    pytest.importorskip("yaml")
    cache_module = sys.modules["testmon_cache"]
    directory = project / "tests/unit_tests/example"
    directory.mkdir(parents=True)
    for path in (project / "tests").glob("test_*.py"):
        path.rename(directory / path.name)
    (directory / "test_app.py").write_text(
        (directory / "test_app.py").read_text() + "\nimport pytest\n\n@pytest.mark.experimental\n"
        "def test_experimental():\n    assert active(2) == 3\n"
    )
    (project / "pytest.ini").write_text("[pytest]\nmarkers = experimental: experimental test\n")
    bucket, platform = "tests/unit_tests/example/**/*.py", "dgx_h100"
    for filename in cache_module.COMPATIBILITY_FILES:
        path = project / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n")
    recipe = project / cache_module.PLATFORMS[platform]["recipe"]
    recipe.parent.mkdir(parents=True)
    recipe.write_text(json.dumps({"products": [{"test_case": [bucket]}]}))
    (project / "tests/unit_tests/testmon_mandatory_tests.yaml").write_text(
        "mappings:\n  - source_dirs: [unobserved]\n    test_buckets:\n"
        f"      {platform}: [tests/unit_tests/example/test_other.py]\n"
    )
    cache_dir = tmp_path / "cpu-cache"
    for phase in cache_module.PHASES:
        marker = "not experimental" if phase == "prod" else "experimental"
        result = _invoke(project, cache_dir, "baseline", phase, pytest_args=("-m", marker))
        assert result.returncode == 0, result.stdout + result.stderr
    identity = cache_module.cache_identity(project, bucket, platform)
    cache_module.finalize(cache_dir, identity, "a" * 40, "123-1")
    pr_files = project / "pr-files"
    pr_files.mkdir()
    (pr_files / "changed-files").write_text("")
    (pr_files / "metadata.json").write_text(
        json.dumps({"tested_sha": "b" * 40, "changed_paths": 0})
    )
    return {
        "root": project,
        "cache_dir": cache_dir,
        "bucket": bucket,
        "recipe_platform": platform,
        "source_sha": "b" * 40,
        "matched_key": identity["cache_prefix"] + "123-1",
        "pr_files_dir": pr_files,
    }


@pytest.mark.parametrize(
    ("change", "mode", "selected"),
    [
        ("unchanged", "selected", []),
        ("dependency", "selected", ["test_app.py"]),
        ("new-test", "selected", ["test_app.py", "test_new.py"]),
        ("mandatory", "selected", ["test_other.py"]),
        ("missing-cache", "full", []),
        ("explicit-file", "selected", []),
    ],
)
def test_cpu_selection(cpu_baseline, change, mode, selected):
    arguments = cpu_baseline
    project, directory = arguments["root"], arguments["cache_dir"]
    if change == "unchanged":
        manifest = directory / "manifest.json"
        data = json.loads(manifest.read_text())
        data["image_id"] = "sha256:" + "c" * 64
        manifest.write_text(json.dumps(data))
        for phase in ("prod", "experimental"):
            metadata = directory / phase / "metadata.json"
            data = json.loads(metadata.read_text())
            data["runtime"].update(python="0.0.0", implementation="DifferentPython")
            metadata.write_text(json.dumps(data))
    elif change == "dependency":
        # Selection must not execute this now-failing test.
        (project / "app.py").write_text("def active(value):\n    return value + 2\n")
    elif change == "new-test":
        (project / "tests/unit_tests/example/test_new.py").write_text(
            "def test_new():\n    assert False\n"
        )
        existing = project / "tests/unit_tests/example/test_app.py"
        existing.write_text(existing.read_text() + "\ndef test_added():\n    assert False\n")
    elif change == "mandatory":
        (arguments["pr_files_dir"] / "changed-files").write_text("unobserved/source.py\n")
        (arguments["pr_files_dir"] / "metadata.json").write_text(
            json.dumps({"tested_sha": arguments["source_sha"], "changed_paths": 1})
        )
    elif change == "missing-cache":
        arguments["cache_dir"] = directory.with_name("missing-cache")
    elif change == "explicit-file":
        arguments["bucket"] = "tests/unit_tests/example/test_app.py"
        cache_module = sys.modules["testmon_cache"]
        identity = cache_module.cache_identity(
            project, arguments["bucket"], arguments["recipe_platform"]
        )
        cache_module.finalize(directory, identity, "a" * 40, "123-1")
        arguments["matched_key"] = identity["cache_prefix"] + "123-1"
    before = {phase: _snapshot(directory / phase) for phase in ("prod", "experimental")}
    common = [
        "--cache-dir",
        str(arguments["cache_dir"]),
        "--bucket",
        arguments["bucket"],
        "--platform",
        arguments["recipe_platform"],
        "--source-sha",
        arguments["source_sha"],
    ]
    plan_path = project / "plan.json"
    result = subprocess.run(
        [
            sys.executable,
            str(WRAPPER_PATH),
            "select",
            *common,
            "--matched-key",
            arguments["matched_key"],
            "--pr-files-dir",
            str(arguments["pr_files_dir"]),
            "--output",
            str(plan_path),
        ],
        cwd=project,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    plan = json.loads(plan_path.read_text())
    assert plan["mode"] == mode, plan["reason"]
    assert f"mode={mode}\n" in result.stdout
    assert f"has_tests={str(mode != 'selected' or bool(selected)).lower()}\n" in result.stdout
    if mode == "selected":
        selected = [f"tests/unit_tests/example/{filename}" for filename in selected]
        assert plan["phases"] == {phase: selected for phase in before}
        assert plan["mandatory_files"] == (selected if change == "mandatory" else [])
        result = subprocess.run(
            [sys.executable, str(WRAPPER_PATH), "prepare", *common, "--plan", str(plan_path)],
            cwd=project,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        for phase in before:
            output = directory / ".testmon-work" / phase
            assert (output / "selected-tests").read_text().splitlines() == selected
            assert (output / "mandatory-tests").read_text().splitlines() == plan["mandatory_files"]
    else:
        assert plan["phases"] == {} and plan["reason"]
    assert {phase: _snapshot(directory / phase) for phase in before} == before


@pytest.mark.parametrize(
    ("validation_status", "selected_runs"), [(0, 2), (1, 0)], ids=["valid-plan", "invalid-plan"]
)
def test_cpu_plan_execution(tmp_path, validation_status, selected_runs):
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
    runner = _source("tests/unit_tests/run_ci_test.sh")
    functions = "\n".join(
        name + "() {" + _function(runner, name) + "\n}"
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
            '  if [[ "$*" == *"testmon_selector.py prepare"* ]]; then return "$VALIDATION_STATUS"; fi',
            '  echo "$*" >> "$EVENTS"',
            "}",
            functions,
            "run_enforced_tests",
        )
    )
    events = tmp_path / "events"
    events.touch()
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
        },
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == validation_status, result.stdout + result.stderr
    executed = events.read_text().splitlines()
    assert "full" not in executed
    assert sum("torch.distributed.run" in event for event in executed) == selected_runs
    assert len(executed) == selected_runs
    assert all("test_example.py::test_one" in event for event in executed)
    assert "unexpected GPU selection" not in result.stderr
