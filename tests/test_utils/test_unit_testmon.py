# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import argparse
import datetime as dt
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

SELECTOR_PATH = Path(__file__).parents[1] / "unit_tests/testmon_selector.py"
SPEC = importlib.util.spec_from_file_location("unit_testmon_selector", SELECTOR_PATH)
assert SPEC is not None and SPEC.loader is not None
selector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = selector
SPEC.loader.exec_module(selector)


def _write(path: Path, contents: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for relative in selector._config_paths(root, "h100"):
        _write(root / relative, relative + "\n")
    _write(
        root / "tests/test_utils/recipes/h100/unit-tests.yaml",
        """products:
  - test_case: [tests/unit_tests/foo/nested/**/*.py]
  - test_case: [tests/unit_tests/foo/**/*.py]
  - test_case: [tests/unit_tests/**/*.py]
""",
    )
    for relative in (
        "megatron/core/a.py",
        "tests/unit_tests/foo/test_a.py",
        "tests/unit_tests/foo/nested/test_b.py",
        "tests/unit_tests/test_basic.py",
    ):
        _write(root / relative, "# baseline\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    return root


def _commit(root: Path, message: str) -> str:
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", message], cwd=root, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True, capture_output=True
    ).stdout.strip()


def _database(path: Path, nodeid: str, dependency: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE test_execution (id INTEGER PRIMARY KEY, test_name TEXT);
            CREATE TABLE file_fp (id INTEGER PRIMARY KEY, filename TEXT);
            CREATE TABLE test_execution_file_fp (
                test_execution_id INTEGER, fingerprint_id INTEGER
            );
            """)
        connection.execute("INSERT INTO test_execution VALUES (1, ?)", (nodeid,))
        if dependency:
            connection.execute("INSERT INTO file_fp VALUES (1, ?)", (dependency,))
            connection.execute("INSERT INTO test_execution_file_fp VALUES (1, 1)")


def _nodes(path: Path, phase: str, nodeids: list[str]) -> None:
    _write(
        path,
        json.dumps(
            {"schema_version": selector.SCHEMA_VERSION, "phase": phase, "nodeids": sorted(nodeids)}
        ),
    )


def _baseline(root: Path, bucket: str, producer: str, *, attributed: bool = True):
    cache = root.parent / "cache"
    nodeid = "tests/unit_tests/foo/test_a.py::test_a"
    for phase in selector.PHASES:
        _database(
            cache / f"{phase}.testmondata", nodeid, "megatron/core/a.py" if attributed else None
        )
        _nodes(cache / "collected" / f"{phase}.json", phase, [nodeid])
    digest = selector.config_hash(root, "h100", 8, bucket)
    args = argparse.Namespace(
        repo_root=root,
        cache_dir=cache,
        platform="h100",
        world_size=8,
        bucket=bucket,
        config_hash=digest,
        producer_sha=producer,
        producer_time=dt.datetime.now(dt.timezone.utc).isoformat(),
        output=None,
    )
    selector.finalize(args)
    return cache, digest


def _select_args(root: Path, cache: Path, digest: str, base: str, head: str, bucket: str):
    return argparse.Namespace(
        repo_root=root,
        cache_dir=cache,
        metadata=None,
        platform="h100",
        world_size=8,
        bucket=bucket,
        config_hash=digest,
        base_sha=base,
        head_sha=head,
        max_age_hours=72.0,
        validate_only=False,
        output=None,
    )


def test_hashes_are_stable_and_configuration_is_content_addressed(tmp_path):
    root = _repo(tmp_path)
    bucket = "tests/unit_tests/foo/**/*.py"
    first = selector.config_hash(root, "h100", 8, bucket)
    assert first == selector.config_hash(root, "h100", 8, bucket)
    assert len(first) == 64
    assert len(selector.bucket_hash(bucket)) == 16
    assert selector.config_hash(root, "h100", 4, bucket) != first
    _write(root / "tests/unit_tests/run_ci_test.sh", "changed\n")
    assert selector.config_hash(root, "h100", 8, bucket) != first


def test_bucket_precedence_matches_nested_recipe_buckets(tmp_path):
    root = _repo(tmp_path)
    assert selector._bucket_files(root, "h100", "tests/unit_tests/foo/**/*.py") == {
        "tests/unit_tests/foo/test_a.py"
    }
    assert selector._bucket_files(root, "h100", "tests/unit_tests/**/*.py") == {
        "tests/unit_tests/test_basic.py"
    }


def test_finalize_records_two_rank_zero_databases_and_unattributed_tests(tmp_path):
    root = _repo(tmp_path)
    producer = _commit(root, "baseline")
    cache, _ = _baseline(root, "tests/unit_tests/foo/**/*.py", producer, attributed=False)
    metadata = json.loads((cache / "metadata.json").read_text())

    assert set(metadata["databases"]) == set(selector.PHASES)
    assert {record["path"] for record in metadata["databases"].values()} == {
        "prod.testmondata",
        "experimental.testmondata",
    }
    assert metadata["world_size"] == 8
    assert metadata["always_run_files"] == ["tests/unit_tests/foo/test_a.py"]


def test_finalize_always_runs_subprocess_tests_even_when_attributed(tmp_path):
    root = _repo(tmp_path)
    _write(root / "tests/unit_tests/foo/test_a.py", "import subprocess\n")
    producer = _commit(root, "baseline")
    cache, _ = _baseline(root, "tests/unit_tests/foo/**/*.py", producer)
    metadata = json.loads((cache / "metadata.json").read_text())
    assert metadata["always_run_files"] == ["tests/unit_tests/foo/test_a.py"]


def test_select_validates_metadata_and_unions_phases(tmp_path):
    root = _repo(tmp_path)
    producer = _commit(root, "baseline")
    bucket = "tests/unit_tests/foo/**/*.py"
    cache, digest = _baseline(root, bucket, producer)
    _write(root / "megatron/core/a.py", "# modified\n")
    head = _commit(root, "modify dependency")
    _nodes(cache / "selection/prod.json", "prod", [])
    _nodes(
        cache / "selection/experimental.json",
        "experimental",
        ["tests/unit_tests/foo/test_a.py::test_experimental"],
    )

    result = selector.select(_select_args(root, cache, digest, producer, head, bucket))

    assert result == ["tests/unit_tests/foo/test_a.py"]
    manifest = json.loads((cache / "selected.json").read_text())
    assert manifest["selected_files"] == result
    assert manifest["eligible_file_count"] == 1
    assert manifest["selection_ratio"] == 1.0


def test_modified_test_is_direct_only_in_its_effective_bucket(tmp_path):
    root = _repo(tmp_path)
    producer = _commit(root, "baseline")
    parent = "tests/unit_tests/foo/**/*.py"
    cache, digest = _baseline(root, parent, producer)
    _write(root / "tests/unit_tests/foo/test_a.py", "# directly modified\n")
    head = _commit(root, "modify test")
    for phase in selector.PHASES:
        _nodes(cache / "selection" / f"{phase}.json", phase, [])
    assert selector.select(_select_args(root, cache, digest, producer, head, parent)) == [
        "tests/unit_tests/foo/test_a.py"
    ]


@pytest.mark.parametrize(
    ("path", "status"),
    [
        ("megatron/core/new.py", "A"),
        ("README.md", "M"),
        ("megatron/core/native.cu", "M"),
        ("tests/unit_tests/foo/helper.py", "M"),
    ],
)
def test_classifier_rejects_unsafe_changes(tmp_path, path, status):
    root = _repo(tmp_path)
    producer = _commit(root, "baseline")
    _write(root / path, "changed\n")
    if status == "M":
        _commit(root, "seed changed path")
        producer = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
        ).stdout.strip()
        _write(root / path, "modified again\n")
    head = _commit(root, "unsafe change")
    with pytest.raises(selector.SelectionError, match="requires full testing"):
        selector._changed_paths(root, producer, head)


def test_validation_rejects_stale_metadata_and_corrupt_database(tmp_path):
    root = _repo(tmp_path)
    producer = _commit(root, "baseline")
    bucket = "tests/unit_tests/foo/**/*.py"
    cache, digest = _baseline(root, bucket, producer)
    args = _select_args(root, cache, digest, producer, producer, bucket)
    args.validate_only = True
    metadata = json.loads((cache / "metadata.json").read_text())
    metadata["producer_time"] = "2020-01-01T00:00:00Z"
    _write(cache / "metadata.json", json.dumps(metadata))
    with pytest.raises(selector.SelectionError, match="stale"):
        selector.select(args)

    metadata["producer_time"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _write(cache / "metadata.json", json.dumps(metadata))
    (cache / "prod.testmondata").write_text("not sqlite")
    with pytest.raises(selector.SelectionError, match="checksum"):
        selector.select(args)


def test_validation_rejects_producer_that_is_not_an_ancestor(tmp_path):
    root = _repo(tmp_path)
    producer = _commit(root, "baseline")
    bucket = "tests/unit_tests/foo/**/*.py"
    cache, digest = _baseline(root, bucket, producer)
    unrelated = subprocess.run(
        ["git", "commit-tree", "HEAD^{tree}"],
        cwd=root,
        input="unrelated\n",
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    metadata = json.loads((cache / "metadata.json").read_text())
    metadata["producer_sha"] = unrelated
    _write(cache / "metadata.json", json.dumps(metadata))
    args = _select_args(root, cache, digest, producer, producer, bucket)
    args.validate_only = True
    with pytest.raises(selector.SelectionError, match="merge-base"):
        selector.select(args)


def test_selected_path_outside_effective_bucket_is_rejected(tmp_path):
    root = _repo(tmp_path)
    producer = _commit(root, "baseline")
    bucket = "tests/unit_tests/foo/**/*.py"
    cache, digest = _baseline(root, bucket, producer)
    for phase in selector.PHASES:
        _nodes(
            cache / "selection" / f"{phase}.json",
            phase,
            ["tests/unit_tests/foo/nested/test_b.py::test_b"],
        )
    with pytest.raises(selector.SelectionError, match="escape effective bucket"):
        selector.select(_select_args(root, cache, digest, producer, producer, bucket))


@pytest.mark.parametrize("path", ("../../test_bad.py", "/tmp/test_bad.py", "test_bad.txt"))
def test_unsafe_selected_paths_are_rejected(tmp_path, path):
    with pytest.raises(selector.SelectionError):
        selector._node_file(path, tmp_path)


def test_run_enables_testmon_only_on_global_rank_zero(tmp_path, monkeypatch):
    calls = []
    return_codes = [0]

    def fake_main(arguments, plugins):
        calls.append((arguments, plugins))
        if plugins:
            session = SimpleNamespace(
                items=[SimpleNamespace(nodeid="tests/unit_tests/test_a.py::x")]
            )
            plugins[0].pytest_collection_finish(session)
        return return_codes[-1]

    monkeypatch.setitem(sys.modules, "pytest", SimpleNamespace(main=fake_main))
    args = argparse.Namespace(
        cache_dir=tmp_path, phase="prod", mode="baseline", pytest_args=["--", "-q", "tests"]
    )
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    assert selector._run(args) == 0
    assert "--testmon-noselect" in calls[-1][0]
    assert (tmp_path / "collected/prod.json").is_file()

    monkeypatch.setenv("RANK", "3")
    assert selector._run(args) == 0
    assert "--testmon" not in calls[-1][0]
    assert "no:testmon" in calls[-1][0]
    assert not calls[-1][1]

    args.mode = "select"
    return_codes.append(5)
    assert selector._run(args) == 0
    assert "--collect-only" in calls[-1][0]
    assert "--testmon" not in calls[-1][0]
    assert "TESTMON_DATAFILE" not in os.environ


def test_rank_zero_selection_uses_disposable_database(tmp_path, monkeypatch):
    source = tmp_path / "prod.testmondata"
    source.write_bytes(b"trusted")
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "pytest",
        SimpleNamespace(main=lambda arguments, plugins: calls.append(arguments) or 0),
    )
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    args = argparse.Namespace(
        cache_dir=tmp_path, phase="prod", mode="select", pytest_args=["--", "-m", "marker"]
    )
    assert selector._run(args) == 0
    assert "--testmon-forceselect" in calls[0]
    assert os.environ["TESTMON_DATAFILE"].endswith(".selection-work/prod.testmondata")
    assert source.read_bytes() == b"trusted"


def test_ci_wires_artifacts_and_keeps_the_full_path_testmon_free():
    root = Path(__file__).parents[2]
    main = (root / ".github/workflows/cicd-main.yml").read_text()
    baseline = (root / ".github/workflows/unit-testmon-baseline.yml").read_text()
    action = (root / ".github/actions/action.yml").read_text()
    launcher = (root / "tests/test_utils/python_scripts/launch_nemo_run_workload.py").read_text()
    runner = (root / "tests/unit_tests/run_ci_test.sh").read_text()

    assert 'cron: "0 10 * * *"' in main
    assert 'workflows: ["CICD Megatron-LM"]' in baseline
    assert "Resolve trusted producer" in baseline
    assert "actions/workflows/unit-testmon-baseline.yml/runs" in main
    assert "actions/runs/$CANDIDATE/artifacts" in main
    assert "any(.expired == false)" in main
    assert "-f status=success" in main
    assert "uses: actions/download-artifact@" in action
    assert "run-id: ${{ inputs.unit_testmon_baseline_run_id }}" in action
    assert "actions/cache" not in baseline + action
    for obsolete in (
        "unit_testmon_cache_key",
        "unit_testmon_selected_manifest",
        "expected-index-record",
    ):
        assert obsolete not in main + baseline + action
    for option in (
        "--unit-testmon-mode",
        "--unit-testmon-cache-dir",
        "--unit-testmon-base-sha",
        "--unit-testmon-config-hash",
    ):
        assert option in launcher
    full_body = runner.split("run_full_tests() {", 1)[1].split("\n}\n\ninstall_testmon", 1)[0]
    assert "testmon" not in full_body.lower()
    assert "testmon_selector.py" not in full_body


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
    ],
)
def test_pr_eligibility_decision_table(overrides, expected):
    workflow = (Path(__file__).parents[2] / ".github/workflows/cicd-main.yml").read_text()
    start = workflow.index("          UNIT_TESTMON_ELIGIBLE=false")
    end = workflow.index('\n\n          echo "scope=', start)
    gate = textwrap.dedent(workflow[start:end])
    sha = {"base": "a" * 40, "head": "b" * 40, "merge": "c" * 40}
    environment = {
        **os.environ,
        "LABELS_VALID": "true",
        "PR_BASE_SHA": sha["base"],
        "PR_HEAD_SHA": sha["head"],
        "PR_MERGE_SHA": sha["merge"],
        "RESOLVED_SHA": sha["merge"],
        "ENABLE_PR_UNIT_TESTMON": "true",
        "EVENT_NAME": "push",
        "REF": "refs/heads/pull-request/123",
        "EVENT_SHA": sha["head"],
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
