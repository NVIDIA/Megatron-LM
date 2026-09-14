# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from testmon.db import DB

ROOT = Path(__file__).parents[2]
HELPER = ROOT / "tests/unit_tests/testmon_cache.py"
SPEC = importlib.util.spec_from_file_location("unit_testmon_cache", HELPER)
assert SPEC is not None and SPEC.loader is not None
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)
IMAGE_ID = "sha256:" + "a" * 64
BUCKET = "tests/unit_tests/pipeline_parallel/**/*.py"


@pytest.fixture
def source_tree(tmp_path):
    root = tmp_path / "source"
    for name in (
        *cache.COMPATIBILITY_FILES,
        "docker/.ngc_version.dev",
        "docker/Dockerfile.ci.dev",
        ".dockerignore",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    return root


@pytest.fixture
def generation(tmp_path, source_tree):
    identity = cache.cache_identity(source_tree, BUCKET, "dgx_h100", IMAGE_ID)
    directory = tmp_path / "assets_dir/testmon"
    for phase in cache.PHASES:
        path = directory / phase / ".testmondata"
        path.parent.mkdir(parents=True)
        database = DB(str(path))
        database.con.close()
        cache.record_phase(directory, phase)
    cache.finalize(directory, identity, "b" * 40, "123-1")
    return directory, identity


def _snapshot(directory):
    return {
        str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_source_edits_preserve_identity_but_build_inputs_invalidate(source_tree):
    before = cache.cache_identity(source_tree, BUCKET, "dgx_h100", IMAGE_ID)
    (source_tree / "megatron/core/ordinary.py").write_text("changed source")
    assert cache.cache_identity(source_tree, BUCKET, "dgx_h100", IMAGE_ID) == before
    (source_tree / "uv.lock").write_text("changed dependency")
    assert (
        cache.cache_identity(source_tree, BUCKET, "dgx_h100", IMAGE_ID)["cache_prefix"]
        != before["cache_prefix"]
    )


@pytest.mark.parametrize(
    "changed", ["docker/.ngc_version.dev", ".dockerignore", "tests/unit_tests/find_test_cases.py"]
)
def test_execution_and_hidden_container_inputs_invalidate(source_tree, changed):
    before = cache.cache_identity(source_tree, BUCKET, "dgx_h100", IMAGE_ID)
    (source_tree / changed).write_text("changed")
    assert cache.cache_identity(source_tree, BUCKET, "dgx_h100", IMAGE_ID) != before


def test_image_platform_and_bucket_are_isolated(source_tree):
    identities = [
        cache.cache_identity(source_tree, BUCKET, "dgx_h100", IMAGE_ID),
        cache.cache_identity(source_tree, BUCKET, "dgx_gb200", IMAGE_ID),
        cache.cache_identity(source_tree, "tests/unit_tests/other.py", "dgx_h100", IMAGE_ID),
        cache.cache_identity(source_tree, BUCKET, "dgx_h100", "sha256:" + "b" * 64),
    ]
    assert len({identity["cache_prefix"] for identity in identities}) == 4
    with pytest.raises(ValueError, match="immutable"):
        cache.cache_identity(source_tree, BUCKET, "dgx_h100", "image:latest")


def test_runtime_tracks_normalized_exact_versions_and_duplicate_distributions(monkeypatch):
    monkeypatch.setattr(
        cache,
        "distributions",
        lambda: [
            SimpleNamespace(metadata={"Name": name}, version=value)
            for name, value in [
                ("torch", "2.10.0"),
                ("torch", "2.11.0"),
                ("Transformer_Engine_CU12", "2.5.0"),
                ("megatron-core", "dev"),
            ]
        ],
    )
    identity = cache.runtime_identity()
    assert identity["packages"] == [
        ["torch", "2.10.0"],
        ["torch", "2.11.0"],
        ["transformer-engine-cu12", "2.5.0"],
    ]
    assert identity["testmon"] == "2.2.0"
    assert identity["python"]


def test_valid_generation_is_read_only(generation):
    directory, identity = generation
    before = _snapshot(directory)
    manifest = cache.validate_cache(directory, identity, identity["cache_prefix"] + "123-1")
    assert manifest["source_sha"] == "b" * 40
    for phase in cache.PHASES:
        cache.validate_phase(directory, phase)
    assert _snapshot(directory) == before


@pytest.mark.parametrize(
    "mutation", ["missing", "corrupt", "schema", "wal", "metadata", "runtime", "checksum"]
)
def test_invalid_phase_is_rejected_without_repair(generation, mutation):
    directory, _ = generation
    database = directory / "prod/.testmondata"
    metadata_path = directory / "prod/metadata.json"
    if mutation == "missing":
        database.unlink()
    elif mutation == "corrupt":
        database.write_bytes(b"not SQLite")
    elif mutation == "schema":
        connection = sqlite3.connect(database)
        connection.execute("PRAGMA user_version=13")
        connection.close()
    elif mutation == "wal":
        database.with_name(database.name + "-wal").write_bytes(b"unfinished transaction")
    elif mutation == "metadata":
        metadata_path.write_text("[]")
    else:
        metadata = json.loads(metadata_path.read_text())
        if mutation == "runtime":
            metadata["runtime"]["python"] = "0.0.0"
        else:
            metadata["database_sha256"] = "0" * 64
        metadata_path.write_text(json.dumps(metadata))
    before = _snapshot(directory)
    with pytest.raises((ValueError, OSError)):
        cache.validate_phase(directory, "prod")
    assert _snapshot(directory) == before


def test_manifest_rejects_wrong_key_compatibility_and_incomplete_phase(generation):
    directory, identity = generation
    with pytest.raises(ValueError, match="generation"):
        cache.validate_cache(directory, identity, identity["cache_prefix"] + "999-1")
    with pytest.raises(ValueError, match="compatibility"):
        cache.validate_cache(
            directory, {**identity, "image_id": "different"}, identity["cache_prefix"] + "123-1"
        )
    (directory / "manifest.json").unlink()
    (directory / "experimental/metadata.json").unlink()
    with pytest.raises(OSError):
        cache.finalize(directory, identity, "c" * 40, "456-1")
    assert not (directory / "manifest.json").exists()


def test_manifest_requires_timezone_for_age_reporting(generation):
    directory, identity = generation
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["created_at"] = "2026-09-14T00:00:00"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="timezone"):
        cache.validate_cache(directory, identity, identity["cache_prefix"] + "123-1")


def _action_script(name):
    action = yaml.safe_load((ROOT / ".github/actions/action.yml").read_text())
    return next(step["run"] for step in action["runs"]["steps"] if step["name"] == name)


@pytest.mark.parametrize(
    "mode,publication,expected",
    [
        ("baseline", "failure", "false"),
        ("baseline", "skipped", "false"),
        ("baseline", "success", "true"),
        ("enforce", "skipped", "true"),
    ],
)
def test_producer_result_requires_cache_publication(mode, publication, expected):
    script = _action_script("Check result")
    script = script[script.index('EXIT_CODE="${MAIN_EXIT_CODE') :]
    script = script.split('if [[ "$IS_SUCCESS" == "false"', 1)[0]
    result = subprocess.run(
        ["bash", "-e", "-u", "-o", "pipefail", "-c", script + 'printf "%s" "$IS_SUCCESS"'],
        env={
            **os.environ,
            "MAIN_EXIT_CODE": "0",
            "MAIN_CONCLUSION": "success",
            "TESTMON_MODE": mode,
            "CACHE_PUBLICATION": publication,
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == expected


@pytest.mark.parametrize("restore", ["valid", "miss", "error", "invalid", "identity-error"])
def test_action_resolver_uses_prefix_restores_and_never_bootstraps(generation, tmp_path, restore):
    directory, identity = generation
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    identity_file = runtime_dir / "unit-testmon-identity.json"
    identity_file.write_text(json.dumps(identity))
    helper = tmp_path / "tests/unit_tests/testmon_cache.py"
    helper.parent.mkdir(parents=True)
    shutil.copy2(HELPER, helper)
    if restore == "miss":
        shutil.rmtree(directory)
    elif restore == "invalid":
        (directory / "manifest.json").write_text("[]")
    before = _snapshot(directory) if directory.exists() else {}
    output = tmp_path / "output"
    summary = tmp_path / "summary"
    result = subprocess.run(
        ["bash", "-e", "-u", "-o", "pipefail", "-c", _action_script("Resolve unit Testmon mode")],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
            "REQUESTED_MODE": "enforce",
            "IDENTITY_OUTCOME": "failure" if restore == "identity-error" else "success",
            "RESTORE_OUTCOME": "failure" if restore == "error" else "success",
            "MATCHED_KEY": "" if restore == "miss" else identity["cache_prefix"] + "123-1",
            "CACHE_HIT": "false",
            "RUNNER_TEMP": str(runtime_dir),
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(summary),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text().strip() == ("mode=enforce" if restore == "valid" else "mode=full")
    after = _snapshot(directory)
    after.pop("summary.md", None)
    assert after == before
    if restore == "valid":
        assert "b" * 40 in summary.read_text()
    else:
        assert "without recording or saving" in summary.read_text()


@pytest.mark.parametrize(
    "override,allowed",
    [
        ({}, True),
        ({"SOURCE_REF": "refs/heads/pull-request/6934"}, False),
        ({"SOURCE_REPOSITORY": "fork/Megatron-LM"}, False),
        ({"SOURCE_EVENT": "pull_request"}, False),
        ({"REQUESTED_SHA": "c" * 40}, False),
    ],
)
def test_action_baseline_guard_rejects_untrusted_producers(tmp_path, override, allowed):
    fake_git = tmp_path / "git"
    fake_git.write_text("#!/bin/sh\nprintf '%s\\n' \"$SOURCE_SHA\"\n")
    fake_git.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "-e",
            "-u",
            "-o",
            "pipefail",
            "-c",
            _action_script("Validate unit Testmon baseline producer"),
        ],
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "SOURCE_REPOSITORY": "NVIDIA/Megatron-LM",
            "SOURCE_REF": "refs/heads/main",
            "SOURCE_EVENT": "schedule",
            "SOURCE_SHA": "b" * 40,
            "REQUESTED_SHA": "b" * 40,
            "TARGET_BRANCH": "main",
            "SUITE_TAG": "latest",
            **override,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) is allowed
