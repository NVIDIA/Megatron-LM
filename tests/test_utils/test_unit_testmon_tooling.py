# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

# Import by path so this pure-Python test does not execute the existing
# tests/unit_tests/__init__.py, which imports torch._dynamo.
TOOLING_PATH = Path(__file__).parents[1] / "unit_tests/testmon/tooling.py"
SPEC = importlib.util.spec_from_file_location("unit_testmon_tooling", TOOLING_PATH)
assert SPEC is not None and SPEC.loader is not None
tooling = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tooling
SPEC.loader.exec_module(tooling)

Change = tooling.Change
ValidationError = tooling.TestmonValidationError


def _write(path: Path, contents: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)


def _environment_repo(tmp_path: Path, platform: str = "h100") -> Path:
    root = tmp_path / "repo"
    for relative in tooling._default_environment_paths(platform):
        _write(root / relative, f"contents of {relative}\n")
    _write(root / "tests/unit_tests/testmon/tooling.py", "selector\n")
    return root


def _selection_repo(tmp_path: Path) -> tuple[Path, list[str]]:
    root = tmp_path / "repo"
    files = (
        "tests/unit_tests/test_basic.py",
        "tests/unit_tests/foo/test_a.py",
        "tests/unit_tests/foo/test_b.py",
        "tests/unit_tests/bar/test_c.py",
        "megatron/core/a.py",
    )
    for relative in files:
        _write(root / relative, "# test\n")
    buckets = ["tests/unit_tests/foo/**/*.py", "tests/unit_tests/**/*.py"]
    return root, buckets


def _sqlite(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript("""
        PRAGMA user_version = 14;
        CREATE TABLE test_execution (
            id INTEGER PRIMARY KEY, environment_id INTEGER, test_name TEXT,
            duration FLOAT, failed BIT, forced BIT
        );
        CREATE TABLE file_fp (
            id INTEGER PRIMARY KEY, filename TEXT, method_checksums BLOB,
            mtime FLOAT, fsha TEXT
        );
        CREATE TABLE test_execution_file_fp (
            test_execution_id INTEGER, fingerprint_id INTEGER
        );
        INSERT INTO test_execution VALUES (
            1, 1, 'tests/unit_tests/foo/test_a.py::test_a', 1.0, 0, 0
        );
        INSERT INTO file_fp VALUES (1, 'megatron/core/a.py', NULL, 0.0, 'abc');
        INSERT INTO test_execution_file_fp VALUES (1, 1);
        """)
    connection.commit()
    connection.close()


def _baseline_cache(
    root: Path,
    *,
    world_size: int = 2,
    phases: tuple[str, ...] = ("prod", "experimental"),
    disagree: bool = False,
) -> Path:
    cache = root / "cache"
    for phase in phases:
        for rank in range(world_size):
            _sqlite(cache / phase / f"rank-{rank}.testmondata")
            nodeids = ["tests/unit_tests/foo/test_a.py::test_a"]
            if disagree and phase == "prod" and rank == world_size - 1:
                nodeids = ["tests/unit_tests/foo/test_b.py::test_b"]
            _write(cache / "collected" / phase / f"rank-{rank}.json", json.dumps(nodeids))
    return cache


def _union_cache(
    cache: Path,
    selections: dict[tuple[str, int], list[str]],
    *,
    collected_by_phase: dict[str, list[str]] | None = None,
) -> None:
    for phase in tooling.DEFAULT_PHASES:
        collected = (
            collected_by_phase.get(phase, [])
            if collected_by_phase is not None
            else sorted(
                {
                    nodeid
                    for (selection_phase, _), nodeids in selections.items()
                    if selection_phase == phase
                    for nodeid in nodeids
                }
            )
        )
        _write(cache / "collected" / phase / "rank-0.json", json.dumps(collected))
        for rank in range(2):
            _write(
                cache / "selection" / phase / f"rank-{rank}.json",
                json.dumps(selections.get((phase, rank), [])),
            )


def _manifest(tmp_path: Path, *, producer_time: str = "2026-08-21T00:00:00Z"):
    root, _ = _selection_repo(tmp_path)
    cache = _baseline_cache(root)
    manifest = tooling.build_bucket_manifest(
        cache,
        repo_root=root,
        platform="h100",
        world_size=2,
        bucket="tests/unit_tests/foo/**/*.py",
        producer_sha="a" * 40,
        producer_time=producer_time,
        environment_hash="b" * 64,
        container_identity="container@sha256:abc",
        dependency_identity="uv-lock:b",
        topology_identity=tooling.expected_topology_identity("h100", 2),
        tracked_python_files=[
            "megatron/core/a.py",
            "tests/unit_tests/foo/test_a.py",
            "tests/unit_tests/foo/test_b.py",
        ],
    )
    manifest_path = cache / "manifest.json"
    _write(manifest_path, json.dumps(manifest))
    return root, cache, manifest, manifest_path


def test_environment_hash_is_deterministic_and_content_addressed(tmp_path):
    root = _environment_repo(tmp_path)
    first = tooling.compute_environment_hash(root, "h100")
    assert first == tooling.compute_environment_hash(root, "h100")
    assert len(first) == 64

    _write(root / "tests/unit_tests/run_ci_test.sh", "changed\n")
    assert tooling.compute_environment_hash(root, "h100") != first
    assert tooling.compute_environment_hash(
        root, "h100", identities={"container": "new"}
    ) != tooling.compute_environment_hash(root, "h100", identities={"container": "old"})

    before_nested_conftest = tooling.compute_environment_hash(root, "h100")
    _write(root / "tests/unit_tests/nested/conftest.py", "collection hook\n")
    assert tooling.compute_environment_hash(root, "h100") != before_nested_conftest


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        ({}, None),
        ({"enabled": False}, "ENABLE_PR_UNIT_TESTMON"),
        ({"event_name": "merge_group"}, "not a synthetic PR push"),
        ({"event_name": "workflow_dispatch"}, "not a synthetic PR push"),
        ({"ref": "refs/heads/pull-request/not-a-number"}, "not a synthetic numeric PR ref"),
        ({"metadata_valid": False}, "metadata is invalid"),
        ({"pr_head_sha": "2" * 40}, "does not match"),
        ({"labels": ["Run tests"]}, "full-test label"),
        ({"labels": ["Run functional tests"]}, "full-test label"),
        ({"force_run_all": True}, "force_run_all"),
        ({"container": "lts"}, "not the default dev container"),
    ],
)
def test_pr_eligibility_decision_table(overrides, expected_reason):
    arguments = {
        "enabled": True,
        "event_name": "push",
        "ref": "refs/heads/pull-request/123",
        "github_sha": "1" * 40,
        "pr_head_sha": "1" * 40,
        "metadata_valid": True,
        "labels": [],
        "force_run_all": False,
        "container": "dev",
    }
    arguments.update(overrides)
    result = tooling.decide_pr_eligibility(**arguments)
    if expected_reason is None:
        assert result == {
            "mode": "enforce",
            "eligible": True,
            "reason": "eligible synthetic PR push",
            "reasons": [],
        }
    else:
        assert result["mode"] == "full"
        assert not result["eligible"]
        assert expected_reason in result["reason"]


def test_manifest_validates_equal_rank_sets_checksums_and_always_run(tmp_path):
    _, cache, manifest, _ = _manifest(tmp_path)
    assert manifest["collected_node_counts"] == {"prod": 1, "experimental": 1}
    assert manifest["always_run_files"] == ["tests/unit_tests/test_basic.py"]
    assert len(manifest["databases"]) == 4
    tooling.validate_bucket_manifest(
        manifest,
        cache_dir=cache,
        expected_platform="h100",
        expected_world_size=2,
        expected_bucket="tests/unit_tests/foo/**/*.py",
        expected_environment_hash="b" * 64,
    )


def test_manifest_rejects_rank_collection_disagreement(tmp_path):
    root, _ = _selection_repo(tmp_path)
    cache = _baseline_cache(root, disagree=True)
    with pytest.raises(ValidationError, match="collected node sets disagree"):
        tooling.build_bucket_manifest(
            cache,
            repo_root=root,
            platform="h100",
            world_size=2,
            bucket="tests/unit_tests/foo/**/*.py",
            producer_sha="a" * 40,
            producer_time="2026-08-21T00:00:00Z",
            environment_hash="b" * 64,
            container_identity="container",
            dependency_identity="dependencies",
            topology_identity="topology",
        )


def test_manifest_rejects_corrupt_restored_database(tmp_path):
    _, cache, manifest, _ = _manifest(tmp_path)
    database = cache / manifest["databases"][0]["path"]
    with database.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValidationError, match="checksum mismatch"):
        tooling.validate_bucket_manifest(manifest, cache_dir=cache)


def test_platform_index_requires_every_bucket(tmp_path):
    _, _, manifest, manifest_path = _manifest(tmp_path)
    with pytest.raises(ValidationError, match="incomplete"):
        tooling.build_platform_index(
            [(manifest, manifest_path)],
            expected_buckets=[manifest["bucket"], "tests/unit_tests/**/*.py"],
            platform="h100",
            world_size=2,
            environment_hash="b" * 64,
        )

    index = tooling.build_platform_index(
        [(manifest, manifest_path)],
        expected_buckets=[manifest["bucket"]],
        platform="h100",
        world_size=2,
        environment_hash="b" * 64,
    )
    record = index["buckets"][manifest["bucket"]]
    assert record["cache_key"] == manifest["cache_key"]
    assert record["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def test_restored_manifest_and_databases_are_bound_to_trusted_index_record(tmp_path):
    _, cache, manifest, manifest_path = _manifest(tmp_path)
    index = tooling.build_platform_index(
        [(manifest, manifest_path)],
        expected_buckets=[manifest["bucket"]],
        platform="h100",
        world_size=2,
        environment_hash="b" * 64,
    )
    record = index["buckets"][manifest["bucket"]]
    tooling.validate_bucket_manifest(
        manifest, cache_dir=cache, manifest_path=manifest_path, trusted_index_record=record
    )

    bad_manifest_record = {**record, "manifest_sha256": "0" * 64}
    with pytest.raises(ValidationError, match="manifest checksum does not match"):
        tooling.validate_bucket_manifest(
            manifest,
            cache_dir=cache,
            manifest_path=manifest_path,
            trusted_index_record=bad_manifest_record,
        )

    bad_database_record = {**record, "database_checksums": ["0" * 64] * 4}
    with pytest.raises(ValidationError, match="database checksum list does not match"):
        tooling.validate_bucket_manifest(
            manifest,
            cache_dir=cache,
            manifest_path=manifest_path,
            trusted_index_record=bad_database_record,
        )

    manifest_path.write_text(manifest_path.read_text() + "\n")
    with pytest.raises(ValidationError, match="manifest checksum does not match"):
        tooling.validate_bucket_manifest(
            manifest, cache_dir=cache, manifest_path=manifest_path, trusted_index_record=record
        )


def test_platform_index_trusted_time_allows_reused_bucket_manifests(tmp_path):
    _, _, first, first_path = _manifest(tmp_path / "first", producer_time="2026-08-19T00:00:00Z")
    second = json.loads(json.dumps(first))
    second["bucket"] = "tests/unit_tests/**/*.py"
    second["bucket_hash"] = tooling.bucket_hash(second["bucket"])
    second["cache_key"] = tooling._cache_key(
        "h100", 2, second["bucket"], second["environment_hash"], second["producer_sha"]
    )
    second["producer_time"] = "2026-08-20T00:00:00Z"
    second_path = tmp_path / "second/manifest.json"
    _write(second_path, json.dumps(second))
    manifests = [(first, first_path), (second, second_path)]
    expected_buckets = [first["bucket"], second["bucket"]]

    with pytest.raises(ValidationError, match="inconsistent baseline field producer_time"):
        tooling.build_platform_index(
            manifests,
            expected_buckets=expected_buckets,
            platform="h100",
            world_size=2,
            environment_hash="b" * 64,
        )

    index = tooling.build_platform_index(
        manifests,
        expected_buckets=expected_buckets,
        platform="h100",
        world_size=2,
        environment_hash="b" * 64,
        producer_time_override="2026-08-21T03:04:05+00:00",
    )
    assert index["producer_time"] == "2026-08-21T03:04:05Z"


def test_platform_index_rejects_stale_baseline(tmp_path, monkeypatch):
    root, _, manifest, manifest_path = _manifest(tmp_path)
    index = tooling.build_platform_index(
        [(manifest, manifest_path)],
        expected_buckets=[manifest["bucket"]],
        platform="h100",
        world_size=2,
        environment_hash="b" * 64,
    )
    monkeypatch.setattr(tooling, "_git_is_ancestor", lambda *_: True)
    with pytest.raises(ValidationError, match="stale"):
        tooling.validate_platform_index(
            index,
            repo_root=root,
            base_sha="c" * 40,
            platform="h100",
            world_size=2,
            environment_hash="b" * 64,
            expected_buckets=[manifest["bucket"]],
            now=dt.datetime(2026, 8, 25, tzinfo=dt.timezone.utc),
        )


@pytest.mark.parametrize(
    ("changes", "expected_fragment"),
    [
        ([Change("M", "pyproject.toml")], "selection input changed"),
        ([Change("M", ".github/workflows/cicd-main.yml")], "selection input changed"),
        (
            [Change("M", "tests/unit_tests/transformer/moe/conftest.py")],
            "pytest collection hook changed",
        ),
        ([Change("M", "megatron/core/kernel.cu")], "native or compiled source"),
        ([Change("M", "unknown/config.json")], "not safely classified"),
        ([Change("A", "megatron/core/new.py")], "absent from the baseline"),
        (
            [Change("R100", "megatron/core/renamed.py", "megatron/core/old.py")],
            "absent from the baseline",
        ),
    ],
)
def test_diff_classifier_falls_back_for_unsafe_or_unknown_changes(changes, expected_fragment):
    result = tooling.classify_changes(
        changes, tracked_python_files=["megatron/core/a.py"], buckets=["tests/unit_tests/**/*.py"]
    )
    assert result["mode"] == "full"
    assert expected_fragment in result["reason"]


def test_diff_classifier_handles_modified_added_deleted_and_renamed_tests():
    result = tooling.classify_changes(
        [
            Change("M", "megatron/core/a.py"),
            Change("M", "tests/unit_tests/foo/test_a.py"),
            Change("A", "tests/unit_tests/foo/test_new.py"),
            Change("D", "tests/unit_tests/foo/test_deleted.py"),
            Change(
                "R100", "tests/unit_tests/foo/test_renamed.py", "tests/unit_tests/foo/test_old.py"
            ),
        ],
        tracked_python_files=["megatron/core/a.py"],
        buckets=["tests/unit_tests/foo/**/*.py", "tests/unit_tests/**/*.py"],
    )
    assert result["mode"] == "select"
    assert result["direct_tests"] == [
        "tests/unit_tests/foo/test_a.py",
        "tests/unit_tests/foo/test_new.py",
        "tests/unit_tests/foo/test_renamed.py",
    ]
    assert result["deleted_tests"] == ["tests/unit_tests/foo/test_deleted.py"]
    assert (
        result["direct_tests_by_bucket"]["tests/unit_tests/foo/**/*.py"] == result["direct_tests"]
    )


def test_bucket_ownership_matches_find_test_cases_parent_child_precedence():
    buckets = [
        "tests/unit_tests/models/test_gpt_model.py",
        "tests/unit_tests/models/**/*.py",
        "tests/unit_tests/**/*.py",
    ]
    assert (
        tooling.owning_bucket("tests/unit_tests/models/test_gpt_model.py", buckets)
        == "tests/unit_tests/models/test_gpt_model.py"
    )
    assert (
        tooling.owning_bucket("tests/unit_tests/models/test_bert_model.py", buckets)
        == "tests/unit_tests/models/**/*.py"
    )
    assert (
        tooling.owning_bucket("tests/unit_tests/test_basic.py", buckets)
        == "tests/unit_tests/**/*.py"
    )


def test_union_selection_unions_rank_and_phase_dependencies(tmp_path):
    root, buckets = _selection_repo(tmp_path)
    cache = root / "cache"
    selections = {
        ("prod", 0): ["tests/unit_tests/foo/test_a.py::test_a"],
        ("prod", 1): ["tests/unit_tests/foo/test_b.py::test_nonzero_rank_dependency"],
        ("experimental", 0): [],
        ("experimental", 1): ["tests/unit_tests/foo/test_a.py::test_experimental"],
    }
    _union_cache(cache, selections)

    result = tooling.union_rank_selections(
        cache,
        repo_root=root,
        bucket="tests/unit_tests/foo/**/*.py",
        platform="h100",
        world_size=2,
        buckets=buckets,
        direct_tests=["tests/unit_tests/bar/test_c.py"],
        always_run_files=["tests/unit_tests/test_basic.py"],
    )
    assert result["selected_test_files"] == [
        "tests/unit_tests/foo/test_a.py",
        "tests/unit_tests/foo/test_b.py",
    ]
    assert result["selected_test_file_count"] == 2
    assert result["eligible_test_file_count"] == 2
    assert result["selection_ratio"] == 1.0
    assert result["selected_node_count"] == 3


def test_union_selection_allows_empty_bucket(tmp_path):
    root, buckets = _selection_repo(tmp_path)
    cache = root / "cache"
    _union_cache(cache, {})
    result = tooling.union_rank_selections(
        cache,
        repo_root=root,
        bucket="tests/unit_tests/foo/**/*.py",
        platform="h100",
        world_size=2,
        buckets=buckets,
    )
    assert result["selected_test_files"] == []
    assert result["selected_test_file_count"] == 0
    assert result["eligible_test_file_count"] == 0
    assert result["selection_ratio"] == 0.0


def test_union_selection_counts_direct_tests_in_eligible_universe(tmp_path):
    root, buckets = _selection_repo(tmp_path)
    cache = root / "cache"
    direct_test = "tests/unit_tests/foo/test_new.py"
    _write(root / direct_test, "# new test\n")
    selections = {
        ("prod", 0): ["tests/unit_tests/foo/test_a.py::test_a"],
        ("prod", 1): [],
        ("experimental", 0): [],
        ("experimental", 1): [],
    }
    _union_cache(
        cache,
        selections,
        collected_by_phase={
            "prod": [
                "tests/unit_tests/foo/test_a.py::test_a",
                "tests/unit_tests/foo/test_b.py::test_b",
            ],
            "experimental": [],
        },
    )

    result = tooling.union_rank_selections(
        cache,
        repo_root=root,
        bucket="tests/unit_tests/foo/**/*.py",
        platform="h100",
        world_size=2,
        buckets=buckets,
        direct_tests=[direct_test],
    )

    assert result["selected_test_files"] == ["tests/unit_tests/foo/test_a.py", direct_test]
    assert result["selected_test_file_count"] == 2
    assert result["eligible_test_file_count"] == 3
    assert result["selection_ratio"] == 2 / 3


def test_union_selection_requires_baseline_collections(tmp_path):
    root, buckets = _selection_repo(tmp_path)
    cache = root / "cache"
    for phase in tooling.DEFAULT_PHASES:
        for rank in range(2):
            _write(cache / "selection" / phase / f"rank-{rank}.json", "[]")

    with pytest.raises(ValidationError, match="cannot read JSON"):
        tooling.union_rank_selections(
            cache,
            repo_root=root,
            bucket="tests/unit_tests/foo/**/*.py",
            platform="h100",
            world_size=2,
            buckets=buckets,
        )


@pytest.mark.parametrize(
    "nodeid",
    [
        "/tmp/test_escape.py::test_escape",
        "../tests/unit_tests/foo/test_a.py::test_escape",
        "tests/unit_tests/foo/helper.txt::test_escape",
    ],
)
def test_union_selection_rejects_unsafe_paths(tmp_path, nodeid):
    root, buckets = _selection_repo(tmp_path)
    cache = root / "cache"
    selections = {
        ("prod", 0): [nodeid],
        ("prod", 1): [],
        ("experimental", 0): [],
        ("experimental", 1): [],
    }
    _union_cache(cache, selections)
    with pytest.raises(ValidationError):
        tooling.union_rank_selections(
            cache,
            repo_root=root,
            bucket="tests/unit_tests/foo/**/*.py",
            platform="h100",
            world_size=2,
            buckets=buckets,
        )


def test_union_selection_rejects_path_owned_by_another_bucket(tmp_path):
    root, buckets = _selection_repo(tmp_path)
    cache = root / "cache"
    selections = {
        ("prod", 0): ["tests/unit_tests/bar/test_c.py::test_c"],
        ("prod", 1): [],
        ("experimental", 0): [],
        ("experimental", 1): [],
    }
    _union_cache(cache, selections)
    with pytest.raises(ValidationError, match="outside effective bucket"):
        tooling.union_rank_selections(
            cache,
            repo_root=root,
            bucket="tests/unit_tests/foo/**/*.py",
            platform="h100",
            world_size=2,
            buckets=buckets,
        )


def test_parse_git_diff_preserves_rename_old_and_new_paths():
    changes = tooling._parse_git_diff_z(
        b"M\0megatron/core/a.py\0R100\0old.py\0megatron/core/new.py\0"
    )
    assert changes == [
        Change("M", "megatron/core/a.py"),
        Change("R100", "megatron/core/new.py", "old.py"),
    ]


def test_manifest_identifies_unattributed_and_subprocess_like_tests(tmp_path):
    root, _ = _selection_repo(tmp_path)
    cache = _baseline_cache(root)
    for database in cache.glob("*/rank-*.testmondata"):
        with sqlite3.connect(database) as connection:
            connection.execute("DELETE FROM test_execution_file_fp")
    manifest = tooling.build_bucket_manifest(
        cache,
        repo_root=root,
        platform="h100",
        world_size=2,
        bucket="tests/unit_tests/foo/**/*.py",
        producer_sha="a" * 40,
        producer_time="2026-08-21T00:00:00Z",
        environment_hash="b" * 64,
        container_identity="container",
        dependency_identity="dependencies",
        topology_identity="topology",
        tracked_python_files=["megatron/core/a.py"],
    )
    assert manifest["unattributed_tests"] == ["tests/unit_tests/foo/test_a.py"]
    assert manifest["always_run_files"] == [
        "tests/unit_tests/foo/test_a.py",
        "tests/unit_tests/test_basic.py",
    ]


def test_dependency_on_only_one_rank_is_still_attributed(tmp_path):
    root, _ = _selection_repo(tmp_path)
    cache = _baseline_cache(root)
    for phase in tooling.DEFAULT_PHASES:
        with sqlite3.connect(cache / phase / "rank-0.testmondata") as connection:
            connection.execute("DELETE FROM test_execution_file_fp")
    manifest = tooling.build_bucket_manifest(
        cache,
        repo_root=root,
        platform="h100",
        world_size=2,
        bucket="tests/unit_tests/foo/**/*.py",
        producer_sha="a" * 40,
        producer_time="2026-08-21T00:00:00Z",
        environment_hash="b" * 64,
        container_identity="container",
        dependency_identity="dependencies",
        topology_identity="topology",
        tracked_python_files=["megatron/core/a.py"],
    )
    assert manifest["unattributed_tests"] == []
    assert manifest["always_run_files"] == ["tests/unit_tests/test_basic.py"]


def test_prepare_matrix_malformed_index_falls_back_instead_of_raising(tmp_path):
    root = _environment_repo(tmp_path)
    recipe = root / "tests/test_utils/recipes/h100/unit-tests.yaml"
    _write(recipe, "products:\n  - test_case: [tests/unit_tests/**/*.py]\n")
    result = tooling.prepare_matrix(
        {"buckets": []},
        repo_root=root,
        platform="h100",
        world_size=8,
        base_sha="a" * 40,
        current_sha="b" * 40,
        recipe=recipe,
    )
    assert result["mode"] == "full"
    assert "schema version" in result["reason"]
    assert result["cache_age_hours"] is None


def test_prepare_matrix_propagates_trusted_bucket_binding(tmp_path, monkeypatch):
    root = _environment_repo(tmp_path)
    recipe = root / "tests/test_utils/recipes/h100/unit-tests.yaml"
    bucket = "tests/unit_tests/**/*.py"
    _write(recipe, f"products:\n  - test_case: [{bucket}]\n")
    index_record = {
        "cache_key": "unit-testmon-cache",
        "manifest_sha256": "c" * 64,
        "database_checksums": ["d" * 64, "e" * 64],
    }
    index = {
        "producer_sha": "a" * 40,
        "producer_time": "2026-08-21T03:00:00Z",
        "buckets": {bucket: index_record},
        "tracked_python_files": [],
    }
    monkeypatch.setattr(tooling, "compute_environment_hash", lambda *_args, **_kwargs: "f" * 64)
    monkeypatch.setattr(tooling, "validate_platform_index", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tooling, "_git_is_ancestor", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(tooling, "git_diff_changes", lambda *_args, **_kwargs: [])

    result = tooling.prepare_matrix(
        index,
        repo_root=root,
        platform="h100",
        world_size=8,
        base_sha="b" * 40,
        current_sha="c" * 40,
        recipe=recipe,
        now=dt.datetime(2026, 8, 21, 6, tzinfo=dt.timezone.utc),
    )
    assert result["mode"] == "enforce"
    assert result["cache_age_hours"] == 3.0
    assert result["buckets"][bucket] == {**index_record, "direct_tests": []}


def test_validate_manifest_cli_accepts_index_record_contract():
    args = tooling._build_parser().parse_args(
        [
            "validate-manifest",
            "--manifest",
            "manifest.json",
            "--cache-dir",
            "cache",
            "--platform",
            "h100",
            "--world-size",
            "8",
            "--bucket",
            "tests/unit_tests/**/*.py",
            "--index-record",
            "expected-index-record.json",
        ]
    )
    assert args.trusted_index_record == "expected-index-record.json"
