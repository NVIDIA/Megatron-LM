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
    monkeypatch: pytest.MonkeyPatch, return_code: int = 0, selected_count: int | None = None
) -> list[list[str]]:
    calls: list[list[str]] = []

    def main(arguments, *_args, **kwargs):
        calls.append(list(arguments))
        if selected_count is not None:
            session = SimpleNamespace(items=[object()] * selected_count)
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


def test_baseline_uses_testmon_only_on_rank_zero(tmp_path, monkeypatch):
    calls = _fake_pytest(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "0")

    assert _run_wrapper(tmp_path, "baseline") == 0
    assert {"--testmon", "--testmon-noselect"} <= set(calls[-1])
    assert os.environ["TESTMON_DATAFILE"] == str(tmp_path / "prod/.testmondata")

    monkeypatch.setenv("RANK", "3")
    assert _run_wrapper(tmp_path, "baseline", "experimental") == 0
    assert "--testmon" not in calls[-1]
    assert calls[-1][-4:] == ["-p", "no:testmon", "-p", "no:pytest-testmon"]
    assert "TESTMON_DATAFILE" not in os.environ


@pytest.mark.parametrize("rank", (0, 3))
def test_enforce_gives_every_rank_a_disposable_database(tmp_path, monkeypatch, rank):
    database = tmp_path / "prod/.testmondata"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"trusted baseline")
    calls = _fake_pytest(monkeypatch, selected_count=rank + 1)
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", str(rank))

    assert _run_wrapper(tmp_path, "enforce") == 0

    arguments = calls[0]
    assert {"--testmon", "--testmon-nocollect", "--testmon-forceselect"} <= set(arguments)
    disposable = Path(os.environ["TESTMON_DATAFILE"])
    assert disposable == tmp_path / f".testmon-work/prod/rank-{rank}/.testmondata"
    assert disposable.read_bytes() == b"trusted baseline"
    assert database.read_bytes() == b"trusted baseline"
    count = tmp_path / f".testmon-work/prod/rank-{rank}/selected-count"
    assert count.read_text() == f"{rank + 1}\n"
    assert (count.parent / "pytest-exit-code").read_text() == "0\n"


def test_selected_test_failure_is_recorded(tmp_path, monkeypatch):
    database = tmp_path / "prod/.testmondata"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"baseline")
    _fake_pytest(monkeypatch, return_code=1, selected_count=1)
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "0")

    assert _run_wrapper(tmp_path, "enforce") == 1
    status = tmp_path / ".testmon-work/prod/rank-0/pytest-exit-code"
    assert status.read_text() == "1\n"


@pytest.mark.parametrize(("mode", "rank"), (("baseline", 0), ("enforce", 2)))
def test_empty_pytest_selection_is_a_success(tmp_path, monkeypatch, mode, rank):
    if mode == "enforce":
        database = tmp_path / "prod/.testmondata"
        database.parent.mkdir(parents=True)
        database.write_bytes(b"baseline")
    _fake_pytest(monkeypatch, return_code=5)
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", str(rank))
    assert _run_wrapper(tmp_path, mode) == 0


def test_missing_baseline_is_an_error_for_the_runner_to_fail_open(tmp_path, monkeypatch, capsys):
    calls = _fake_pytest(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "1")
    assert _run_wrapper(tmp_path, "enforce") != 0
    assert calls == []
    assert "missing Testmon baseline" in capsys.readouterr().err


def test_full_runner_keeps_the_existing_commands_and_never_loads_testmon():
    runner = _source("tests/unit_tests/run_ci_test.sh")
    full = _function(runner, "run_full_tests")

    assert "-m coverage run" in full
    assert "--data-file=.coverage.unit_tests" in full
    assert "not experimental and ${MARKER_ARG}" in full
    assert "--experimental" in full
    assert "experimental and ${MARKER_ARG}" in full
    assert "coverage combine -q" in full
    assert "testmon" not in full.lower()


def test_enforce_failures_fall_back_with_the_plugin_disabled():
    runner = _source("tests/unit_tests/run_ci_test.sh")
    fallback = _function(runner, "run_full_fallback")
    enforce = _function(runner, "run_enforced_tests")
    always = _function(runner, "run_always_tests")

    assert "-p no:testmon -p no:pytest-testmon" in fallback
    assert "run_full_tests" in fallback
    assert "run_full_fallback" in enforce
    assert "install_testmon" in enforce
    assert "run_testmon_phase enforce" in enforce
    assert "selected_count prod" in enforce
    assert "rm -f -- .coverage.unit_tests*" in enforce
    assert "selected_test_failed" in enforce
    assert "run_always_tests" in enforce
    assert "test_basic.py" in always
    assert "test_data_parallel_inference_coordinator.py" in always
    assert "-p no:testmon" in always


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, "true"),
        ({"ENABLE_PR_UNIT_TESTMON": "false"}, "false"),
        ({"EVENT_NAME": "merge_group"}, "false"),
        ({"EVENT_NAME": "workflow_dispatch"}, "false"),
        ({"REF": "refs/heads/deploy-release/1"}, "false"),
        ({"LABELS_VALID": "false"}, "false"),
        ({"EVENT_SHA": "d" * 40}, "false"),
        ({"HAS_RUN_TESTS": "true"}, "false"),
        ({"HAS_RUN_FUNCTIONAL": "true"}, "false"),
        ({"FORCE_RUN_ALL": "true"}, "false"),
        ({"HAS_LTS": "true"}, "false"),
        ({"RESOLVED_SHA": "d" * 40}, "false"),
        ({"CHANGED_FILES_VALID": "false"}, "false"),
        ({"CHANGED_FILES_SAFE": "false"}, "false"),
    ],
)
def test_pr_eligibility_decision_table(overrides, expected):
    workflow = _source(".github/workflows/cicd-main.yml")
    start = workflow.index("          UNIT_TESTMON_ELIGIBLE=false")
    metadata_start = workflow.index("          METADATA_VALID=false", start)
    metadata_condition = workflow.index('          if [[ "$LABELS_VALID"', metadata_start)
    metadata_end = workflow.index("\n          fi", metadata_condition) + len("\n          fi")
    decision_start = workflow.index('          if [[ "$ENABLE_PR_UNIT_TESTMON"', metadata_end)
    end = workflow.index('\n\n          echo "scope=', start)
    gate = "UNIT_TESTMON_ELIGIBLE=false\nMETADATA_VALID=false\n"
    gate += textwrap.dedent(workflow[metadata_condition:metadata_end]) + "\n"
    gate += textwrap.dedent(workflow[decision_start:end])
    environment = {
        **os.environ,
        "LABELS_VALID": "true",
        "PR_BASE_SHA": "a" * 40,
        "PR_HEAD_SHA": "b" * 40,
        "PR_MERGE_SHA": "c" * 40,
        "RESOLVED_SHA": "c" * 40,
        "ENABLE_PR_UNIT_TESTMON": "true",
        "EVENT_NAME": "push",
        "REF": "refs/heads/pull-request/123",
        "EVENT_SHA": "b" * 40,
        "HAS_RUN_TESTS": "false",
        "HAS_RUN_FUNCTIONAL": "false",
        "FORCE_RUN_ALL": "false",
        "HAS_LTS": "false",
        "CHANGED_FILES_VALID": "true",
        "CHANGED_FILES_SAFE": "true",
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


def test_workflows_use_a_main_only_cache_and_keep_merge_queue_full():
    main = _source(".github/workflows/cicd-main.yml")
    baseline = _source(".github/workflows/unit-testmon-baseline.yml")
    action = _source(".github/actions/action.yml")

    assert 'cron: "0 10 * * *"' in main
    assert 'test("^megatron/' in main and 'test("^tests/unit_tests/' in main
    assert "unit_testmon_mode: ${{ needs.configure.outputs.unit_testmon_eligible == 'true'" in main
    assert "workflows: [\"CICD Megatron-LM\"]" in baseline
    assert "github.event.workflow_run.conclusion == 'success'" in baseline
    assert "github.event.workflow_run.event == 'schedule'" in baseline
    assert "merge_group:" not in baseline

    assert "uses: actions/cache/restore@" in action
    assert "uses: actions/cache/save@" in action
    assert "uses: actions/download-artifact@" not in action
    assert "unit-testmon-v1-" in action + baseline
    assert "restore-keys:" in action
    assert "cache-matched-key" in action
    assert "EFFECTIVE_MODE=full" in action
    assert "hashFiles(" in action
    assert "NOW - CREATED_AT <= 259200" in action
    assert 'compare/$PRODUCER_SHA...$BASE_SHA' in action
    assert '.status == "ahead" or .status == "identical"' in action


def test_baseline_and_pr_cache_paths_do_not_publish_from_pr_jobs():
    baseline = _source(".github/workflows/unit-testmon-baseline.yml")
    action = _source(".github/actions/action.yml")

    assert "unit_testmon_mode: baseline" in baseline
    assert "unit_testmon_mode: enforce" not in baseline
    assert "sha: ${{ needs.prepare.outputs.producer_sha }}" in baseline
    save = action.split("- name: Save unit Testmon baseline", 1)[1]
    assert "inputs.unit_testmon_mode == 'baseline'" in save
    assert "steps.prepare-unit-testmon-cache.outcome == 'success'" in save
    assert "steps.run-main-script.outcome == 'success'" in action
    assert "RUN_ID" in action and "RUN_ATTEMPT" in action
    assert "inputs.unit_testmon_mode == 'enforce'" not in save.split("- name:", 1)[0]
    assert "unit_testmon_baseline_run_id" not in action
    assert "actions/runs/" not in action
