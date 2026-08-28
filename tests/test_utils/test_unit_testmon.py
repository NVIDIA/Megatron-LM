# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]
WRAPPER_PATH = ROOT / "tests/unit_tests/testmon_selector.py"
SPEC = importlib.util.spec_from_file_location("unit_testmon_wrapper", WRAPPER_PATH)
assert SPEC is not None and SPEC.loader is not None
wrapper = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = wrapper
SPEC.loader.exec_module(wrapper)


def _source(relative: str) -> str:
    return (ROOT / relative).read_text()


def _function(source: str, name: str) -> str:
    return source.split(f"{name}() {{", 1)[1].split("\n}\n", 1)[0]


def _fake_pytest(
    monkeypatch: pytest.MonkeyPatch, *, return_code: int = 0, selected: tuple[str, ...] = ()
) -> list[list[str]]:
    calls: list[list[str]] = []

    def main(arguments, *_args, **kwargs):
        calls.append(list(arguments))
        if "--testmon-noselect" in arguments:
            database = Path(os.environ["TESTMON_DATAFILE"])
            database.parent.mkdir(parents=True, exist_ok=True)
            database.write_bytes(b"baseline")
        session = SimpleNamespace(items=[SimpleNamespace(nodeid=nodeid) for nodeid in selected])
        for plugin in kwargs.get("plugins", []):
            plugin.pytest_collection_finish(session)
        return return_code

    monkeypatch.setitem(sys.modules, "pytest", SimpleNamespace(main=main))
    return calls


def _run_wrapper(cache: Path, mode: str, phase: str = "prod") -> int:
    return wrapper.main(
        [
            "--mode",
            mode,
            "--cache-dir",
            str(cache),
            "--phase",
            phase,
            "--",
            "-m",
            "marker",
            "tests/unit_tests/example",
        ]
    )


def test_baseline_rank_zero_generates_one_database_per_phase(tmp_path, monkeypatch):
    calls = _fake_pytest(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "0")

    assert _run_wrapper(tmp_path, "baseline") == 0

    prod_database = tmp_path / "prod/.testmondata"
    assert prod_database.read_bytes() == b"baseline"
    assert {"--testmon", "--testmon-noselect"} <= set(calls[-1])

    assert _run_wrapper(tmp_path, "baseline", "experimental") == 0
    experimental_database = tmp_path / "experimental/.testmondata"
    assert experimental_database.read_bytes() == b"baseline"
    assert os.environ["TESTMON_DATAFILE"] == str(experimental_database)


def test_baseline_nonzero_rank_disables_testmon(tmp_path, monkeypatch):
    calls = _fake_pytest(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "3")

    assert _run_wrapper(tmp_path, "baseline") == 0

    assert not list(tmp_path.rglob(".testmondata"))
    assert {"-p", "no:testmon", "no:pytest-testmon"} <= set(calls[-1])
    assert "TESTMON_DATAFILE" not in os.environ


@pytest.mark.parametrize("rank", (0, 3))
def test_selection_uses_a_copy_and_records_nodeids(tmp_path, monkeypatch, rank):
    database = tmp_path / "experimental/.testmondata"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"trusted baseline")
    calls = _fake_pytest(
        monkeypatch,
        selected=("tests/unit_tests/test_a.py::test_a", "tests/unit_tests/test_b.py::test_b"),
    )
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", str(rank))

    assert _run_wrapper(tmp_path, "select", "experimental") == 0

    arguments = calls[0]
    assert {"--collect-only", "--testmon", "--testmon-nocollect", "--testmon-forceselect"} <= set(
        arguments
    )
    disposable = tmp_path / f".testmon-work/experimental/rank-{rank}/.testmondata"
    assert Path(os.environ["TESTMON_DATAFILE"]) == disposable
    assert disposable.read_bytes() == b"trusted baseline"
    assert database.read_bytes() == b"trusted baseline"
    assert (disposable.parent / "selected-tests").read_text().splitlines() == [
        "tests/unit_tests/test_a.py",
        "tests/unit_tests/test_b.py",
    ]


def test_missing_rank_database_is_an_error(tmp_path, monkeypatch, capsys):
    calls = _fake_pytest(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "1")

    assert _run_wrapper(tmp_path, "select") != 0
    assert calls == []
    assert "missing Testmon baseline" in capsys.readouterr().err


def test_empty_selection_is_success(tmp_path, monkeypatch):
    database = tmp_path / "prod/.testmondata"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"baseline")
    _fake_pytest(monkeypatch, return_code=5)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")

    assert _run_wrapper(tmp_path, "select") == 0


def test_runner_generates_databases_then_unions_rank_selections():
    runner = _source("tests/unit_tests/run_ci_test.sh")
    full = _function(runner, "run_full_tests")
    baseline = _function(runner, "run_baseline_tests")
    bootstrap = _function(runner, "run_bootstrap_tests")
    enforce = _function(runner, "run_enforced_tests")

    assert "-m coverage run" in full
    assert "--experimental" in full
    assert "testmon" not in full.lower()
    assert "run_testmon_phase baseline prod" in baseline
    assert "run_testmon_phase baseline experimental" in baseline
    assert bootstrap.index("run_full_tests") < bootstrap.index("run_baseline_tests")
    assert "run_testmon_phase select prod" in enforce
    assert "run_testmon_phase select experimental" in enforce
    assert "merge_rank_selections prod" in enforce
    assert "merge_rank_selections experimental" in enforce
    assert "run_full_tests" in enforce
    assert "CoverageData" in enforce
    assert "install_testmon" not in runner
    assert "selected_test_failed" not in runner


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


def test_cache_bootstraps_a_pr_scoped_baseline():
    main = _source(".github/workflows/cicd-main.yml")
    action = _source(".github/actions/action.yml")

    assert 'any(. == "Run selective unit tests")' in main
    assert "unit_testmon_mode: ${{ needs.configure.outputs.unit_testmon_eligible == 'true'" in main
    assert "unit_testmon_target_branch: ${{ needs.configure.outputs.target_branch }}" in main

    assert "unit-testmon-${TARGET_BRANCH_KEY}-${PLATFORM}-${BUCKET_HASH}" in action
    assert "uses: actions/cache/restore@" in action
    assert "uses: actions/cache/save@" in action
    assert 'test -s "$CACHE_DIR/prod/.testmondata"' in action
    assert 'test -s "$CACHE_DIR/experimental/.testmondata"' in action
    assert "EXPECTED_RANKS" not in action
    assert "restore-keys:" not in action
    assert "outputs.cache-hit" in action
    assert "EFFECTIVE_MODE=bootstrap" in action
    assert "steps.unit-testmon.outputs.mode == 'bootstrap'" in action
    assert "unit_testmon_head_sha" not in action + main
    assert "CREATED_AT" not in action
    assert "unit_testmon_base_sha" not in action
    assert "unit_testmon_pr_number" not in action


def test_mode_is_passed_as_container_environment():
    launcher = _source("tests/test_utils/python_scripts/launch_nemo_run_workload.py")
    h100_recipe = _source("tests/test_utils/recipes/h100/unit-tests.yaml")
    gb200_recipe = _source("tests/test_utils/recipes/gb200/unit-tests.yaml")
    pyproject = _source("pyproject.toml")

    assert '"UNIT_TESTMON_MODE": unit_testmon_mode' in launcher
    assert '["full", "enforce", "baseline", "bootstrap"]' in launcher
    assert "{unit_testmon_mode}" not in h100_recipe + gb200_recipe
    assert "pytest-testmon==2.2.0" in pyproject
