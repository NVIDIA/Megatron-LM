# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib.util
import json
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
RUNNER = REPO_ROOT / "tests/unit_tests/run_ci_test.sh"
PLUGIN_PATH = REPO_ROOT / "tests/unit_tests/testmon_selected_plugin.py"
RECIPES = (
    REPO_ROOT / "tests/test_utils/recipes/h100/unit-tests.yaml",
    REPO_ROOT / "tests/test_utils/recipes/gb200/unit-tests.yaml",
)
LOCAL_JOB_GENERATOR = REPO_ROOT / "tests/test_utils/python_scripts/generate_local_jobs.py"
MAIN_WORKFLOW = REPO_ROOT / ".github/workflows/cicd-main.yml"
BASELINE_WORKFLOW = REPO_ROOT / ".github/workflows/unit-testmon-baseline.yml"
COMPOSITE_ACTION = REPO_ROOT / ".github/actions/action.yml"
COMMON_INSTALL = REPO_ROOT / "docker/common/install.sh"

PLUGIN_SPEC = importlib.util.spec_from_file_location("testmon_selected_plugin", PLUGIN_PATH)
assert PLUGIN_SPEC is not None and PLUGIN_SPEC.loader is not None
plugin = importlib.util.module_from_spec(PLUGIN_SPEC)
sys.modules[PLUGIN_SPEC.name] = plugin
PLUGIN_SPEC.loader.exec_module(plugin)


def test_runner_is_valid_bash():
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)


def test_full_runner_path_keeps_the_original_non_testmon_commands():
    source = RUNNER.read_text()
    full_body = source.split("run_full_tests() {", 1)[1].split("\n}\n\nrun_full_fallback", 1)[0]

    assert "testmon" not in full_body.lower()
    assert "-m coverage run" in full_body
    assert "--data-file=.coverage.unit_tests" in full_body
    assert "--source=megatron/core" in full_body
    assert "-m pytest" in full_body
    assert "-m \"'not experimental and ${MARKER_ARG}'\"" in full_body
    assert "--experimental" in full_body
    assert "coverage combine -q" in full_body


def test_testmon_modes_use_required_pytest_options_and_disable_execution_plugin():
    plugin_source = (REPO_ROOT / "tests/unit_tests/testmon_selected_plugin.py").read_text()
    runner_source = RUNNER.read_text()

    assert '("--testmon", "--testmon-noselect")' in plugin_source
    for option in ("--collect-only", "--testmon-nocollect", "--testmon-forceselect"):
        assert option in plugin_source
    assert "-p no:pytest-testmon" in runner_source
    assert (
        "-m coverage run"
        not in runner_source.split("run_baseline_tests() {", 1)[1].split(
            "\n}\n\nrun_enforced_tests", 1
        )[0]
    )


def test_enforcement_binds_validation_to_the_staged_trusted_index_record():
    source = RUNNER.read_text()
    enforce_body = source.split("run_enforced_tests() {", 1)[1].split(
        '\n}\n\ncase "$UNIT_TESTMON_MODE"', 1
    )[0]

    assert "tests/unit_tests/testmon/tooling.py validate-manifest" in enforce_body
    assert '--index-record "$UNIT_TESTMON_CACHE_DIR/expected-index-record.json"' in enforce_body


def test_rank_database_is_fresh_for_baseline_and_disposable_for_selection(tmp_path):
    baseline = plugin._prepare_database(tmp_path, "prod", rank=3, mode="baseline")
    baseline.write_bytes(b"baseline")
    Path(f"{baseline}-wal").write_bytes(b"wal")

    refreshed = plugin._prepare_database(tmp_path, "prod", rank=3, mode="baseline")
    assert refreshed == baseline
    assert not baseline.exists()
    assert not Path(f"{baseline}-wal").exists()

    baseline.write_bytes(b"trusted")
    baseline.chmod(stat.S_IRUSR)
    disposable = plugin._prepare_database(tmp_path, "prod", rank=3, mode="select")

    assert disposable == tmp_path / ".selection-work/prod/rank-3.testmondata"
    assert disposable.read_bytes() == b"trusted"
    assert disposable.stat().st_mode & stat.S_IWUSR
    assert baseline.read_bytes() == b"trusted"


def test_distributed_identity_uses_global_rank(monkeypatch):
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")

    assert plugin._distributed_identity() == (7, 8)


def test_rank_node_outputs_separate_baseline_from_pr_selection(tmp_path):
    assert plugin._node_output_path(tmp_path, "prod", 2, "baseline") == (
        tmp_path / "collected/prod/rank-2.json"
    )
    assert plugin._node_output_path(tmp_path, "prod", 2, "select") == (
        tmp_path / "selection/prod/rank-2.json"
    )


def test_pytest_import_path_does_not_shadow_installed_testmon(monkeypatch):
    script_directory = PLUGIN_PATH.parent.resolve()
    monkeypatch.setattr(sys, "path", [str(script_directory), "/python/site-packages"])

    plugin._prepare_pytest_import_path()

    assert Path(sys.path[0]) == REPO_ROOT.resolve()
    assert str(script_directory) not in sys.path


def test_collection_plugin_writes_sorted_node_ids(tmp_path):
    output = tmp_path / "selection/prod/rank-1.json"
    recorder = plugin.SelectedNodesPlugin(output, "prod", rank=1, world_size=8)
    session = type(
        "Session",
        (),
        {"items": [type("Item", (), {"nodeid": nodeid})() for nodeid in ("z::test", "a::test")]},
    )()

    recorder.pytest_collection_finish(session)

    assert json.loads(output.read_text()) == {
        "schema_version": 1,
        "phase": "prod",
        "rank": 1,
        "world_size": 8,
        "nodeids": ["a::test", "z::test"],
    }


def _write_baseline_rank(cache_dir: Path, phase: str, rank: int, nodeids: list[str]):
    database = cache_dir / phase / f"rank-{rank}.testmondata"
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE baseline (value INTEGER)")
    output = cache_dir / "collected" / phase / f"rank-{rank}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"schema_version": 1, "phase": phase, "rank": rank, "world_size": 2, "nodeids": nodeids}
        )
    )


def test_baseline_verification_checks_databases_and_equal_rank_collections(tmp_path):
    for phase in plugin.PHASES:
        for rank in range(2):
            _write_baseline_rank(tmp_path, phase, rank, ["tests/unit_tests/test_a.py::test_a"])

    plugin.verify_baseline_artifacts(tmp_path, world_size=2)


def test_baseline_verification_rejects_rank_collection_disagreement(tmp_path):
    for phase in plugin.PHASES:
        for rank in range(2):
            nodeids = [f"tests/unit_tests/test_a.py::test_rank_{rank}"]
            _write_baseline_rank(tmp_path, phase, rank, nodeids)

    with pytest.raises(ValueError, match="differ across ranks"):
        plugin.verify_baseline_artifacts(tmp_path, world_size=2)


def test_selected_manifest_accepts_canonical_sorted_repo_files(tmp_path):
    repo = tmp_path / "repo"
    test_dir = repo / "tests/unit_tests"
    test_dir.mkdir(parents=True)
    (test_dir / "test_a.py").write_text("")
    (test_dir / "test_b.py").write_text("")
    manifest = tmp_path / "selected.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selected_test_files": ["tests/unit_tests/test_a.py", "tests/unit_tests/test_b.py"],
                "selected_test_file_count": 2,
                "eligible_test_file_count": 4,
                "selection_ratio": 0.5,
            }
        )
    )

    assert plugin.load_selected_manifest(manifest, repo) == (
        ["tests/unit_tests/test_a.py", "tests/unit_tests/test_b.py"],
        4,
        0.5,
    )


def test_selected_manifest_cli_emits_summary_metrics_before_files(tmp_path, capsys):
    repo = tmp_path / "repo"
    test_file = repo / "tests/unit_tests/test_a.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("")
    manifest = tmp_path / "selected.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selected_test_files": ["tests/unit_tests/test_a.py"],
                "selected_test_file_count": 1,
                "eligible_test_file_count": 3,
                "selection_ratio": 1 / 3,
            }
        )
    )

    assert (
        plugin.main(
            [
                "selected-files",
                "--manifest",
                str(manifest),
                "--repo-root",
                str(repo),
                "--include-summary-metrics",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.splitlines() == ["3", "33.33%", "tests/unit_tests/test_a.py"]


@pytest.mark.parametrize(
    "metrics",
    [
        {},
        {"selected_test_file_count": True, "eligible_test_file_count": 1, "selection_ratio": 1.0},
        {"selected_test_file_count": 0, "eligible_test_file_count": 1, "selection_ratio": 1.0},
        {"selected_test_file_count": 1, "eligible_test_file_count": True, "selection_ratio": 1.0},
        {"selected_test_file_count": 1, "eligible_test_file_count": -1, "selection_ratio": 1.0},
        {"selected_test_file_count": 1, "eligible_test_file_count": 0, "selection_ratio": 0.0},
        {"selected_test_file_count": 1, "eligible_test_file_count": 2, "selection_ratio": "0.5"},
        {
            "selected_test_file_count": 1,
            "eligible_test_file_count": 2,
            "selection_ratio": float("nan"),
        },
        {"selected_test_file_count": 1, "eligible_test_file_count": 2, "selection_ratio": 0.75},
    ],
)
def test_selected_manifest_rejects_malformed_or_inconsistent_metrics(tmp_path, metrics):
    repo = tmp_path / "repo"
    test_file = repo / "tests/unit_tests/test_a.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("")
    manifest = tmp_path / "selected.json"
    manifest.write_text(
        json.dumps(
            {"schema_version": 1, "selected_test_files": ["tests/unit_tests/test_a.py"], **metrics}
        )
    )

    with pytest.raises(ValueError):
        plugin.load_selected_manifest(manifest, repo)


@pytest.mark.parametrize(
    "selected",
    [
        ["/tmp/test_escape.py"],
        ["tests/unit_tests/../test_escape.py"],
        ["tests/unit_tests/test_escape.txt"],
        ["tests/functional_tests/test_escape.py"],
        ["tests/unit_tests/test_b.py", "tests/unit_tests/test_a.py"],
        ["tests/unit_tests/test_a.py", "tests/unit_tests/test_a.py"],
        ["tests/unit_tests/test_a.py\npytest-option"],
    ],
)
def test_selected_manifest_rejects_unsafe_or_nondeterministic_files(tmp_path, selected):
    repo = tmp_path / "repo"
    test_dir = repo / "tests/unit_tests"
    test_dir.mkdir(parents=True)
    (test_dir / "test_a.py").write_text("")
    (test_dir / "test_b.py").write_text("")
    manifest = tmp_path / "selected.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selected_test_files": selected,
                "selected_test_file_count": len(selected),
                "eligible_test_file_count": len(selected),
                "selection_ratio": 1.0,
            }
        )
    )

    with pytest.raises(ValueError):
        plugin.load_selected_test_files(manifest, repo)


def test_runner_summary_reports_selected_over_eligible_and_percentage(tmp_path):
    source = RUNNER.read_text()
    summary_function = (
        "write_enforce_summary() {"
        + source.split("write_enforce_summary() {", 1)[1].split("\n}\n\nrun_baseline_tests", 1)[0]
        + "\n}"
    )
    subprocess.run(
        [
            "bash",
            "-c",
            f'{summary_function}\nwrite_enforce_summary 7 "" 5 40.00% test_a.py test_b.py',
        ],
        check=True,
        env={
            "PATH": "/usr/bin:/bin",
            "UNIT_TESTMON_CACHE_DIR": str(tmp_path),
            "PLATFORM": "h100",
            "BUCKET": "tests/unit_tests/foo/**/*.py",
        },
    )

    summary = (tmp_path / "summary.md").read_text()
    assert "- Selected files: `2/5`" in summary
    assert "- Selection ratio: `40.00%`" in summary
    assert "unavailable" not in summary


def test_unit_recipes_skip_coverage_export_when_no_data_exists():
    for recipe in RECIPES:
        source = recipe.read_text()
        assert "--unit-testmon-mode {unit_testmon_mode}" in source
        assert "--unit-testmon-cache-dir {unit_testmon_cache_dir}" in source
        assert "--unit-testmon-selected-manifest {unit_testmon_selected_manifest}" in source
        assert "if [[ -f .coverage ]]; then" in source
        assert "No unit-test coverage data was produced" in source
        assert source.index("if [[ -f .coverage ]]; then") < source.index(
            "/opt/venv/bin/coverage xml"
        )


def test_local_recipe_generation_defaults_to_exhaustive_mode():
    source = LOCAL_JOB_GENERATOR.read_text()

    assert 'magic_values["unit_testmon_mode"] = "full"' in source
    assert 'magic_values["unit_testmon_cache_dir"]' in source
    assert 'magic_values["unit_testmon_selected_manifest"]' in source


def test_pr_eligibility_gate_is_inline_and_never_executes_pr_code_with_pat():
    source = MAIN_WORKFLOW.read_text()
    configure = source.split("  configure:\n", 1)[1].split("\n  linting:\n", 1)[0]

    assert "tests/unit_tests/testmon/tooling.py" not in configure
    assert "ENABLE_PR_UNIT_TESTMON" in configure
    assert "^refs/heads/pull-request/[0-9]+$" in configure
    assert '"$PR_HEAD_SHA" != "$EVENT_SHA"' in configure
    assert '"$FORCE_RUN_ALL" == "true"' in configure
    assert '"$HAS_LTS" == "true"' in configure


def test_cicd_daily_producer_runs_at_10_utc():
    source = MAIN_WORKFLOW.read_text()

    assert 'schedule:\n    - cron: "0 10 * * *"' in source


def test_baseline_workflow_and_dependency_install_are_trusted_and_isolated():
    baseline = BASELINE_WORKFLOW.read_text()
    common_install = COMMON_INSTALL.read_text()
    action = COMPOSITE_ACTION.read_text()

    assert "workflow_run:" in baseline
    assert '.event == "schedule"' in baseline
    assert '.conclusion == "success"' in baseline
    assert 'producer_run_id:' in baseline
    assert '--producer-time "$PRODUCER_TIME"' in baseline
    assert "-${GITHUB_RUN_ID}" in baseline
    assert "merge_group:" not in baseline
    assert "--all-groups \\\n            --no-group testmon" in common_install
    assert "continue-on-error: true" in action
    assert "expected-index-record.json" in action
    assert "changed-test/index-record staging failed" in action
