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

testmon_db = pytest.importorskip("testmon.db", reason="requires the testmon dependency group")
DB = testmon_db.DB

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


def test_source_edits_preserve_identity(source_tree):
    before = cache.cache_identity(source_tree, BUCKET, "dgx_h100", IMAGE_ID)
    (source_tree / "megatron/core/ordinary.py").write_text("changed source")
    assert cache.cache_identity(source_tree, BUCKET, "dgx_h100", IMAGE_ID) == before


@pytest.mark.parametrize(
    "changed",
    ["uv.lock", "docker/.ngc_version.dev", ".dockerignore", "tests/unit_tests/find_test_cases.py"],
)
def test_build_inputs_preserve_lookup_prefix_but_reject_restored_generation(
    source_tree, generation, changed
):
    directory, producer = generation
    (source_tree / changed).write_text("changed")
    consumer = cache.cache_identity(source_tree, BUCKET, "dgx_h100", IMAGE_ID)
    assert consumer["cache_prefix"] == producer["cache_prefix"]
    assert consumer["compatibility"] != producer["compatibility"]
    before = _snapshot(directory)
    with pytest.raises(ValueError, match="compatibility"):
        cache.validate_cache(directory, consumer, producer["cache_prefix"] + "123-1")
    assert _snapshot(directory) == before


def test_platform_and_bucket_are_isolated(source_tree):
    identities = [
        cache.cache_identity(source_tree, BUCKET, "dgx_h100", IMAGE_ID),
        cache.cache_identity(source_tree, BUCKET, "dgx_gb200", IMAGE_ID),
        cache.cache_identity(source_tree, "tests/unit_tests/other.py", "dgx_h100", IMAGE_ID),
    ]
    assert len({identity["cache_prefix"] for identity in identities}) == 3
    assert all(
        identity["cache_prefix"].startswith("unit-testmon-v1-main-") for identity in identities
    )


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


@pytest.mark.parametrize("consumer_image", [IMAGE_ID, "sha256:" + "c" * 64, "image:latest", None])
def test_valid_generation_accepts_optional_image_diagnostics_and_is_read_only(
    generation, source_tree, consumer_image
):
    directory, identity = generation
    if consumer_image is None:
        consumer = cache.cache_identity(source_tree, BUCKET, "dgx_h100")
        assert consumer["image_id"] == "unknown"
    else:
        consumer = cache.cache_identity(source_tree, BUCKET, "dgx_h100", consumer_image)
        assert consumer["image_id"] == consumer_image
    assert consumer["cache_prefix"] == identity["cache_prefix"]
    before = _snapshot(directory)
    manifest = cache.validate_cache(directory, consumer, identity["cache_prefix"] + "123-1")
    assert manifest["source_sha"] == "b" * 40
    assert manifest["image_id"] == IMAGE_ID
    assert manifest["identity"] == consumer["compatibility"]
    assert "image_id" not in manifest["identity"]
    for phase in cache.PHASES:
        cache.validate_phase(directory, phase)
    assert _snapshot(directory) == before


def test_run_and_attempt_generations_share_prefix_and_require_matching_key(generation):
    directory, identity = generation
    keys = []
    for generation_id in ("123-1", "123-2", "456-1"):
        cache.finalize(directory, identity, "b" * 40, generation_id)
        key = identity["cache_prefix"] + generation_id
        manifest = cache.validate_cache(directory, identity, key)
        assert manifest["generation"] == generation_id
        keys.append(key)
    assert len(set(keys)) == 3
    with pytest.raises(ValueError, match="generation"):
        cache.validate_cache(directory, identity, keys[0])


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "corrupt",
        "schema",
        "wal",
        "metadata",
        "unsupported-metadata-schema",
        "runtime",
        "checksum",
    ],
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
        if mutation == "unsupported-metadata-schema":
            metadata["schema"] = 2
        elif mutation == "runtime":
            metadata["runtime"]["python"] = "0.0.0"
        else:
            metadata["database_sha256"] = "0" * 64
        metadata_path.write_text(json.dumps(metadata))
    before = _snapshot(directory)
    with pytest.raises((ValueError, OSError)):
        cache.validate_phase(directory, "prod")
    assert _snapshot(directory) == before


def test_manifest_rejects_wrong_key_and_incomplete_phase(generation):
    directory, identity = generation
    with pytest.raises(ValueError, match="generation"):
        cache.validate_cache(directory, identity, identity["cache_prefix"] + "999-1")
    (directory / "manifest.json").unlink()
    (directory / "experimental/metadata.json").unlink()
    with pytest.raises(OSError):
        cache.finalize(directory, identity, "c" * 40, "456-1")
    assert not (directory / "manifest.json").exists()


def test_manifest_rejects_unsupported_cache_schema(generation):
    directory, identity = generation
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["schema"] = 2
    path.write_text(json.dumps(manifest))
    before = _snapshot(directory)
    with pytest.raises(ValueError, match="compatibility"):
        cache.validate_cache(directory, identity, identity["cache_prefix"] + "123-1")
    assert _snapshot(directory) == before


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
    "diagnostic,source_ref,source_event,namespace",
    [
        ("failed-inspection", "refs/heads/main", "schedule", "main"),
        ("omitted-cli-option", "refs/heads/main", "schedule", "main"),
        ("failed-inspection", "refs/heads/pull-request/7454", "push", "pr-7454"),
        ("failed-inspection", "refs/heads/pull-request/7454", "workflow_dispatch", "main"),
        ("failed-inspection", "refs/heads/pull-request/7455", "push", "main"),
    ],
)
def test_identity_preserves_usable_cache_and_isolates_pr_generation(
    generation, source_tree, tmp_path, diagnostic, source_ref, source_event, namespace
):
    directory, producer = generation
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    output = tmp_path / "output"
    before = _snapshot(directory)
    script = (
        _action_script("Compute unit Testmon cache identity")
        if diagnostic == "failed-inspection"
        else 'python tests/unit_tests/testmon_cache.py identity --bucket "$BUCKET" '
        '--platform "$RECIPE_PLATFORM" --output "$RUNNER_TEMP/unit-testmon-identity.json" '
        '| tee -a "$GITHUB_OUTPUT"'
    )
    result = subprocess.run(
        ["bash", "-e", "-u", "-o", "pipefail"],
        input="\n".join(
            (
                # Do not inspect Docker or delete anything; use only the temporary source tree.
                'docker() { [[ "$1 $2" == "image inspect" ]]; return 1; }',
                'sudo() { [[ "$*" == "rm -rf -- assets_dir/testmon" ]]; }',
                'python() { if [[ "$1" == "-" ]]; then "$TEST_PYTHON" "$@"; else '
                '[[ "$1" == "tests/unit_tests/testmon_cache.py" ]]; '
                'shift; "$TEST_PYTHON" "$TESTMON_HELPER" "$@"; fi; }',
                script,
            )
        ),
        cwd=source_tree,
        env={
            **os.environ,
            "TEST_PYTHON": sys.executable,
            "TESTMON_HELPER": str(HELPER),
            "SOURCE_REF": source_ref,
            "SOURCE_EVENT": source_event,
            "TARGET_BRANCH": "main",
            "SUITE_TAG": "latest",
            "BUCKET": BUCKET,
            "RECIPE_PLATFORM": "dgx_h100",
            "CONTAINER_IMAGE": "image:latest",
            "RUNNER_TEMP": str(runtime_dir),
            "GITHUB_OUTPUT": str(output),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    consumer = json.loads((runtime_dir / "unit-testmon-identity.json").read_text())
    assert consumer["image_id"] == "unknown"
    assert consumer["cache_prefix"].startswith(f"unit-testmon-v{cache.SCHEMA}-{namespace}-")
    assert consumer["compatibility"] == producer["compatibility"]
    assert output.read_text().strip() == f"cache_prefix={consumer['cache_prefix']}"
    if namespace == "main":
        assert consumer["cache_prefix"] == producer["cache_prefix"]
        assert consumer.get("source_ref", "refs/heads/main") == "refs/heads/main"
        manifest = cache.validate_cache(directory, consumer, producer["cache_prefix"] + "123-1")
        assert manifest["source_sha"] == "b" * 40
        assert _snapshot(directory) == before
    else:
        assert consumer["cache_prefix"] != producer["cache_prefix"]
        assert consumer["source_ref"] == source_ref
        with pytest.raises(ValueError, match="generation"):
            cache.validate_cache(directory, consumer, producer["cache_prefix"] + "123-1")
        merge_sha = "c" * 40
        cache.finalize(directory, consumer, merge_sha, "456-1")
        manifest = cache.validate_cache(directory, consumer, consumer["cache_prefix"] + "456-1")
        assert manifest["source_ref"] == source_ref
        assert manifest["source_sha"] == merge_sha
        with pytest.raises(ValueError, match="generation"):
            cache.validate_cache(directory, producer, consumer["cache_prefix"] + "456-1")
        # Even relabeling the PR cache key cannot make it a valid main baseline.
        with pytest.raises(ValueError, match="source"):
            cache.validate_cache(directory, producer, producer["cache_prefix"] + "456-1")
        after = _snapshot(directory)
        before.pop("manifest.json")
        after.pop("manifest.json")
        assert after == before


@pytest.mark.parametrize(
    "mode,publication,main_conclusion,main_exit_code,expected_success,expected_exit_code",
    [
        ("baseline", "failure", "success", "0", "false", "Testmon cache publication failed"),
        ("baseline", "skipped", "success", "0", "false", "Testmon cache publication failed"),
        ("baseline", "success", "success", "0", "true", "0"),
        ("enforce", "skipped", "success", "0", "true", "0"),
        ("baseline", "skipped", "failure", "1", "false", "1"),
        ("baseline", "skipped", "failure", "", "false", "failure"),
        ("baseline", "skipped", "cancelled", "", "false", "cancelled"),
        ("baseline", "skipped", "skipped", "", "false", "skipped"),
    ],
)
def test_producer_result_preserves_test_failures_and_requires_cache_publication(
    mode, publication, main_conclusion, main_exit_code, expected_success, expected_exit_code
):
    script = _action_script("Check result")
    script = script[script.index('EXIT_CODE="${MAIN_EXIT_CODE') :]
    script = script.split('if [[ "$IS_SUCCESS" == "false"', 1)[0]
    result = subprocess.run(
        [
            "bash",
            "-e",
            "-u",
            "-o",
            "pipefail",
            "-c",
            script + 'printf "%s\\n%s\\n" "$IS_SUCCESS" "$EXIT_CODE"',
        ],
        env={
            **os.environ,
            "MAIN_EXIT_CODE": main_exit_code,
            "MAIN_CONCLUSION": main_conclusion,
            "TESTMON_MODE": mode,
            "CACHE_PUBLICATION": publication,
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == [expected_success, expected_exit_code]


@pytest.mark.parametrize(
    "restore",
    [
        "valid",
        "different-image",
        "missing-image",
        "changed-config",
        "miss",
        "error",
        "invalid",
        "identity-error",
    ],
)
def test_action_resolver_uses_prefix_restores_and_never_bootstraps(
    generation, source_tree, tmp_path, restore
):
    directory, identity = generation
    if restore == "different-image":
        identity = cache.cache_identity(source_tree, BUCKET, "dgx_h100", "sha256:" + "c" * 64)
    elif restore == "missing-image":
        identity = cache.cache_identity(source_tree, BUCKET, "dgx_h100")
    elif restore == "changed-config":
        (source_tree / "tests/unit_tests/find_test_cases.py").write_text("changed")
        identity = cache.cache_identity(source_tree, BUCKET, "dgx_h100", IMAGE_ID)
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
    valid = restore in {"valid", "different-image", "missing-image"}
    assert output.read_text().strip() == ("mode=enforce" if valid else "mode=full")
    after = _snapshot(directory)
    after.pop("summary.md", None)
    assert after == before
    if valid:
        assert "b" * 40 in summary.read_text()
    else:
        assert "without recording or saving" in summary.read_text()


PR_BASELINE_SOURCE = {
    "SOURCE_REF": "refs/heads/pull-request/7454",
    "SOURCE_EVENT": "push",
    "REQUESTED_SHA": "c" * 40,
    "CHECKED_OUT_SHA": "c" * 40,
}


@pytest.mark.parametrize(
    "override,allowed",
    [
        ({}, True),
        ({"SOURCE_REF": "refs/heads/pull-request/6934"}, False),
        ({"SOURCE_REPOSITORY": "fork/Megatron-LM"}, False),
        ({"SOURCE_EVENT": "pull_request"}, False),
        ({"REQUESTED_SHA": "c" * 40}, False),
        ({"REQUESTED_SHA": "c" * 40, "CHECKED_OUT_SHA": "c" * 40}, False),
        ({"CHECKED_OUT_SHA": "c" * 40}, False),
        (PR_BASELINE_SOURCE, True),
        ({**PR_BASELINE_SOURCE, "SOURCE_REF": "refs/heads/pull-request/7455"}, False),
        ({**PR_BASELINE_SOURCE, "SOURCE_REPOSITORY": "fork/Megatron-LM"}, False),
        ({**PR_BASELINE_SOURCE, "SOURCE_EVENT": "schedule"}, False),
        ({**PR_BASELINE_SOURCE, "SOURCE_EVENT": "workflow_dispatch"}, False),
        ({**PR_BASELINE_SOURCE, "SOURCE_EVENT": "pull_request"}, False),
        ({**PR_BASELINE_SOURCE, "TARGET_BRANCH": "dev"}, False),
        ({**PR_BASELINE_SOURCE, "SUITE_TAG": "legacy"}, False),
        ({**PR_BASELINE_SOURCE, "CHECKED_OUT_SHA": "b" * 40}, False),
        ({**PR_BASELINE_SOURCE, "REQUESTED_SHA": "", "CHECKED_OUT_SHA": ""}, False),
    ],
)
def test_action_baseline_guard_rejects_untrusted_producers(tmp_path, override, allowed):
    fake_git = tmp_path / "git"
    fake_git.write_text("#!/bin/sh\nprintf '%s\\n' \"$CHECKED_OUT_SHA\"\n")
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
            "CHECKED_OUT_SHA": "b" * 40,
            "TARGET_BRANCH": "main",
            "SUITE_TAG": "latest",
            **override,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) is allowed
