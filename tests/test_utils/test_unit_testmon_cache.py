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
UNHASHED_INPUTS = (
    ".github/actions/action.yml",
    ".github/workflows/_build_ci_container.yml",
    "pyproject.toml",
    "uv.lock",
    "megatron/core/__init__.py",
    "megatron/core/package_info.py",
    "megatron/core/ordinary.py",
    "tests/unit_tests/run_ci_test.sh",
    "tests/unit_tests/find_test_cases.py",
    "tests/unit_tests/testmon_selector.py",
    "tests/unit_tests/testmon_cache.py",
    "tests/test_utils/python_scripts/launch_nemo_run_workload.py",
    "tests/test_utils/python_scripts/recipe_parser.py",
    "tests/test_utils/python_scripts/download_unit_tests_dataset.py",
    "docker/.ngc_version.dev",
    "docker/Dockerfile.ci.dev",
    "docker/nested/config.json",
    "tests/test_utils/recipes/h100/unit-tests.yaml",
    "tests/test_utils/recipes/gb200/unit-tests.yaml",
)


@pytest.fixture
def source_tree(tmp_path):
    root = tmp_path / "source"
    for name in (
        *cache.COMPATIBILITY_FILES,
        *UNHASHED_INPUTS,
        ".dockerignore",
        "tests/unit_tests/conftest.py",
        "tests/unit_tests/nested/conftest.py",
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


@pytest.mark.parametrize("changed", UNHASHED_INPUTS)
def test_unhashed_edits_preserve_restored_generation(source_tree, generation, changed):
    directory, producer = generation
    (source_tree / changed).write_text("changed")
    consumer = cache.cache_identity(source_tree, BUCKET, "dgx_h100", IMAGE_ID)
    assert consumer == producer
    before = _snapshot(directory)
    manifest = cache.validate_cache(directory, consumer, producer["cache_prefix"] + "123-1")
    assert manifest["identity"] == consumer["compatibility"]
    assert _snapshot(directory) == before


@pytest.mark.parametrize(
    "changed",
    [
        "README.md",
        "tests/unit_tests/testmon_mandatory.py",
        ".dockerignore",
        "tests/unit_tests/conftest.py",
        "tests/unit_tests/nested/conftest.py",
    ],
)
def test_compatibility_edits_preserve_lookup_prefix_but_reject_restored_generation(
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


def test_new_platform_uses_registry_without_hashing_its_recipe(source_tree, monkeypatch):
    recipe = "tests/test_utils/recipes/gb300/unit-tests.yaml"
    monkeypatch.setitem(cache.PLATFORMS, "dgx_gb300", {"cloud": "gb300-test", "recipe": recipe})
    (source_tree / recipe).parent.mkdir(parents=True)
    (source_tree / recipe).write_text("original recipe")

    before = cache.cache_identity(source_tree, BUCKET, "dgx_gb300", IMAGE_ID)
    assert before["cache_prefix"].startswith("unit-testmon-v1-main-dgx_gb300-")
    assert "tests/unit_tests/testmon_mandatory.py" in before["compatibility"]["inputs"]
    assert recipe not in before["compatibility"]["inputs"]

    (source_tree / recipe).write_text("changed recipe")
    after = cache.cache_identity(source_tree, BUCKET, "dgx_gb300", IMAGE_ID)
    assert after == before
    with pytest.raises(ValueError, match="unsupported Testmon platform"):
        cache.cache_identity(source_tree, BUCKET, "dgx_unknown", IMAGE_ID)


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


@pytest.mark.parametrize("diagnostic", ["failed-inspection", "omitted-cli-option"])
def test_identity_without_image_diagnostics_preserves_usable_cache(
    generation, source_tree, tmp_path, diagnostic
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
                'python() { [[ "$1" == "tests/unit_tests/testmon_cache.py" ]]; '
                'shift; "$TEST_PYTHON" "$TESTMON_HELPER" "$@"; }',
                script,
            )
        ),
        cwd=source_tree,
        env={
            **os.environ,
            "TEST_PYTHON": sys.executable,
            "TESTMON_HELPER": str(HELPER),
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
    assert consumer["cache_prefix"] == producer["cache_prefix"]
    assert consumer["compatibility"] == producer["compatibility"]
    assert output.read_text().strip() == f"cache_prefix={producer['cache_prefix']}"
    manifest = cache.validate_cache(directory, consumer, producer["cache_prefix"] + "123-1")
    assert manifest["source_sha"] == "b" * 40
    assert _snapshot(directory) == before


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
        "artifact-error",
        "artifact-skipped",
        "empty-artifact-dir",
        "artifact-missing",
        "missing-metadata",
        "missing-files",
        "invalid-metadata",
        "invalid-metadata-type",
        "string-file-count",
        "boolean-file-count",
        "negative-file-count",
        "fractional-file-count",
        "string-path-count",
        "boolean-path-count",
        "negative-path-count",
        "fractional-path-count",
        "missing-path-count",
        "truncated-files",
        "empty-pr",
        "renamed-files",
        "maximum-renames",
        "too-many-files",
        "too-many-paths",
        "wrong-sha",
    ],
)
def test_action_resolver_uses_prefix_restores_and_never_bootstraps(
    generation, source_tree, tmp_path, restore
):
    directory, identity = generation
    # Every bucket consumes the same immutable artifact without querying GitHub
    # or comparing the PR commit with the Testmon baseline.
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    command_log = tmp_path / "unexpected-commands.log"
    for name in ("gh", "git"):
        fake_command = fake_bin / name
        fake_command.write_text('#!/bin/sh\necho "$0 $*" >> "$COMMAND_LOG"\nexit 99\n')
        fake_command.chmod(0o755)
    if restore == "empty-pr":
        expected_files = []
    elif restore in {"maximum-renames", "too-many-files", "too-many-paths"}:
        count = {"maximum-renames": 6000, "too-many-files": 3001, "too-many-paths": 6001}[restore]
        expected_files = [f"megatron/core/file_{index}.py" for index in range(count)]
    else:
        expected_files = ["megatron/core/a.py", "tests/unit_tests/test_b.py"]
        if restore == "renamed-files":
            expected_files.append("megatron/core/old_a.py")
    pr_files_dir = tmp_path / "pr-files"
    pr_files_dir.mkdir()
    (pr_files_dir / "changed-files").write_text("".join(f"{path}\n" for path in expected_files))
    metadata = {
        "tested_sha": ("d" if restore == "wrong-sha" else "c") * 40,
        "changed_files": {
            "empty-pr": 0,
            "too-many-files": 3001,
            "maximum-renames": 3000,
            "too-many-paths": 3000,
            "string-file-count": "2",
            "boolean-file-count": True,
            "negative-file-count": -1,
            "fractional-file-count": 1.5,
        }.get(restore, 2),
        "changed_paths": {
            "truncated-files": 3,
            "string-path-count": "2",
            "boolean-path-count": True,
            "negative-path-count": -1,
            "fractional-path-count": 1.5,
        }.get(restore, len(expected_files)),
    }
    if restore == "missing-path-count":
        metadata.pop("changed_paths")
    metadata_file = pr_files_dir / "metadata.json"
    metadata_file.write_text(json.dumps(metadata))
    if restore == "artifact-missing":
        shutil.rmtree(pr_files_dir)
    elif restore == "missing-metadata":
        metadata_file.unlink()
    elif restore == "missing-files":
        (pr_files_dir / "changed-files").unlink()
    elif restore == "invalid-metadata":
        metadata_file.write_text("{")
    elif restore == "invalid-metadata-type":
        metadata_file.write_text("[]")
    artifact_before = _snapshot(pr_files_dir) if pr_files_dir.exists() else {}
    if restore == "different-image":
        identity = cache.cache_identity(source_tree, BUCKET, "dgx_h100", "sha256:" + "c" * 64)
    elif restore == "missing-image":
        identity = cache.cache_identity(source_tree, BUCKET, "dgx_h100")
    elif restore == "changed-config":
        (source_tree / "tests/unit_tests/testmon_mandatory.py").write_text("changed")
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
            "PATH": os.pathsep.join(
                (str(fake_bin), str(Path(sys.executable).parent), os.environ["PATH"])
            ),
            "COMMAND_LOG": str(command_log),
            "PR_FILES_OUTCOME": {"artifact-error": "failure", "artifact-skipped": "skipped"}.get(
                restore, "success"
            ),
            "PR_FILES_DIR": "" if restore == "empty-artifact-dir" else str(pr_files_dir),
            # The tested PR commit is deliberately independent of baseline b*40.
            "TESTED_SHA": "c" * 40,
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
    valid = restore in {
        "valid",
        "different-image",
        "missing-image",
        "empty-pr",
        "renamed-files",
        "maximum-renames",
    }
    assert output.read_text().strip() == ("mode=enforce" if valid else "mode=full")
    after = _snapshot(directory)
    after.pop("summary.md", None)
    changed_files = after.pop("changed-files", None)
    assert after == before
    assert not command_log.exists()
    assert (_snapshot(pr_files_dir) if pr_files_dir.exists() else {}) == artifact_before
    if valid:
        assert "b" * 40 in summary.read_text()
        assert f"Changed paths in PR: {len(expected_files)}" in summary.read_text()
        assert changed_files is not None
        assert changed_files[0].decode().splitlines() == expected_files
    else:
        assert "without recording or saving" in summary.read_text()
        assert changed_files is None


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
