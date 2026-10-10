# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("testmon", reason="requires the testmon dependency group")

ROOT = Path(__file__).parents[2]
HELPERS = ROOT / "tests/unit_tests"
for name in ("testmon_cache", "find_test_cases", "testmon_mandatory", "testmon_plan"):
    spec = importlib.util.spec_from_file_location(name, HELPERS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
cache = sys.modules["testmon_cache"]
planner = sys.modules["testmon_plan"]
BUCKET = "tests/unit_tests/example/**/*.py"
PLATFORM = "dgx_h100"
IMAGE_ID = "sha256:" + "a" * 64
SOURCE_SHA = "b" * 40
APP_TEST = "tests/unit_tests/example/test_app.py"
OTHER_TEST = "tests/unit_tests/example/test_other.py"


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    for filename in cache.COMPATIBILITY_FILES:
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n")
    for definition in cache.PLATFORMS.values():
        path = root / definition["recipe"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"products": [{"test_case": [BUCKET]}]}))
    (root / "tests/unit_tests/testmon_mandatory_tests.yaml").write_text("mappings: []\n")
    _pr_files(root, [])
    (root / "pytest.ini").write_text("[pytest]\nmarkers = experimental: experimental test\n")
    (root / "app.py").write_text(
        "def active(value):\n    return value + 1\n\n" "def unused(value):\n    return value - 1\n"
    )
    directory = root / "tests/unit_tests/example"
    directory.mkdir(parents=True)
    (directory / "test_app.py").write_text(
        "import pytest\nfrom app import active, unused\n\n"
        "def test_active():\n    assert active(1) == 2\n\n"
        "def test_unused():\n    assert unused(1) == 0\n\n"
        "@pytest.mark.experimental\ndef test_experimental():\n    assert active(2) == 3\n"
    )
    (directory / "test_other.py").write_text("def test_other():\n    assert 1 + 1 == 2\n")
    # Exercise Testmon's tracked-file hash path, without committing anything.
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    return root


def _baseline(project, directory, bucket=BUCKET):
    target = bucket.removesuffix("/**/*.py")
    for phase in cache.PHASES:
        result = subprocess.run(
            [
                sys.executable,
                str(HELPERS / "testmon_selector.py"),
                "--mode",
                "baseline",
                "--cache-dir",
                str(directory),
                "--phase",
                phase,
                "--",
                "-q",
                "-c",
                str(project / "pytest.ini"),
                "-m",
                "not experimental" if phase == "prod" else "experimental",
                target,
            ],
            cwd=project,
            env={
                **os.environ,
                "RANK": "0",
                "WORLD_SIZE": "1",
                "PYTHONPATH": str(project),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    identity = cache.cache_identity(project, bucket, PLATFORM, IMAGE_ID)
    cache.finalize(directory, identity, SOURCE_SHA, "123-1")
    return directory, identity["cache_prefix"] + "123-1"


@pytest.fixture
def generation(project, tmp_path):
    return _baseline(project, tmp_path / "cache")


def _select(project, generation, **overrides):
    directory, matched_key = generation
    arguments = {
        "root": project,
        "cache_dir": directory,
        "bucket": BUCKET,
        "recipe_platform": PLATFORM,
        "image_id": IMAGE_ID,
        "source_sha": SOURCE_SHA,
        "matched_key": matched_key,
        "pr_files_dir": project / "pr-files",
        **overrides,
    }
    return planner.select(**arguments)


def _pr_files(project, paths, *, changed_files=None, tested_sha=SOURCE_SHA):
    directory = project / "pr-files"
    directory.mkdir(exist_ok=True)
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "tested_sha": tested_sha,
                "changed_files": len(paths) if changed_files is None else changed_files,
                "changed_paths": len(paths),
            }
        )
    )
    (directory / "changed-files").write_text("".join(f"{path}\n" for path in paths))
    return directory


def _mapping(project, source="megatron/core/mandatory", patterns=None):
    (project / "tests/unit_tests/testmon_mandatory_tests.yaml").write_text(
        json.dumps(
            {
                "mappings": [
                    {
                        "source_dirs": [source],
                        "test_buckets": {
                            platform: patterns or [OTHER_TEST] for platform in cache.PLATFORMS
                        },
                    }
                ]
            }
        )
    )


def _metadata(generation, phase, update):
    path = generation[0] / phase / "metadata.json"
    metadata = json.loads(path.read_text())
    update(metadata)
    path.write_text(json.dumps(metadata))


def _snapshot(directory):
    return {
        str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_unchanged_selection_is_empty_and_keeps_cache_readonly(project, generation):
    before = _snapshot(generation[0])
    plan = _select(project, generation)
    assert plan["mode"] == "selected", plan["reason"]
    assert plan["phases"] == {"prod": [], "experimental": []}
    assert plan["world_size"] == 1
    assert plan["runtime"]["prod"] == cache.runtime_identity()
    assert _snapshot(generation[0]) == before


def test_cpu_selects_only_affected_files_in_both_phases(project, generation):
    (project / "app.py").write_text(
        "def active(value):\n    return value + 2\n\n" "def unused(value):\n    return value - 1\n"
    )
    plan = _select(project, generation)
    assert plan["mode"] == "selected", plan["reason"]
    assert plan["phases"] == {"prod": [APP_TEST], "experimental": [APP_TEST]}


def test_deleted_dependency_selects_its_test_file(project, generation):
    (project / "app.py").unlink()
    plan = _select(project, generation)
    assert plan["mode"] == "selected", plan["reason"]
    assert plan["phases"] == {"prod": [APP_TEST], "experimental": [APP_TEST]}


@pytest.mark.parametrize("change", ("new", "modified", "deleted", "helper"))
def test_collection_changes_fall_back_to_full_bucket(project, generation, change):
    path = project / OTHER_TEST
    if change == "new":
        path = path.with_name("test_new.py")
    elif change == "helper":
        path = path.with_name("helper.py")
    if change == "deleted":
        path.unlink()
    else:
        path.write_text("raise AssertionError('the CPU selector must never import tests')\n")
    plan = _select(project, generation)
    assert plan["mode"] == "full"
    assert "sources changed" in plan["reason"]


def test_tests_collected_only_on_other_ranks_are_always_selected(project, generation):
    extra = OTHER_TEST + "::test_rank_one"
    for phase in cache.PHASES:
        _metadata(generation, phase, lambda value: value["collection"].update(world_size=2))
    _metadata(generation, "prod", lambda value: value["collection"]["nodeids"].append(extra))
    plan = _select(project, generation)
    assert plan["mode"] == "selected", plan["reason"]
    assert plan["phases"] == {"prod": [OTHER_TEST], "experimental": [OTHER_TEST]}


@pytest.mark.parametrize("image_id", ("unknown", "sha256:" + "c" * 64))
def test_unproven_runtime_image_falls_back(project, generation, image_id):
    plan = _select(project, generation, image_id=image_id)
    assert plan["mode"] == "full"
    assert "image" in plan["reason"]


@pytest.mark.parametrize("problem", ("collection", "python", "database", "world_size", "key"))
def test_incomplete_evidence_falls_back(project, generation, problem):
    overrides = {}
    if problem == "collection":
        _metadata(generation, "prod", lambda value: value.pop("collection"))
    elif problem == "python":
        _metadata(generation, "prod", lambda value: value["runtime"].update(python="0.0.0"))
    elif problem == "world_size":
        _metadata(generation, "prod", lambda value: value["collection"].update(world_size=2))
    elif problem == "database":
        (generation[0] / "prod/.testmondata").write_bytes(b"invalid sqlite")
    else:
        overrides["matched_key"] = "wrong"
    plan = _select(project, generation, **overrides)
    assert plan["mode"] == "full"
    assert plan["reason"]


def test_cli_emits_empty_selection_without_gpu_collection(project, generation):
    output = project / "plan.json"
    result = subprocess.run(
        [
            sys.executable,
            str(HELPERS / "testmon_plan.py"),
            "select",
            "--cache-dir",
            str(generation[0]),
            "--bucket",
            BUCKET,
            "--platform",
            PLATFORM,
            "--image-id",
            IMAGE_ID,
            "--source-sha",
            SOURCE_SHA,
            "--matched-key",
            generation[1],
            "--pr-files-dir",
            str(project / "pr-files"),
            "--output",
            str(output),
        ],
        cwd=project,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout == "mode=selected\nhas_tests=false\n"
    assert json.loads(output.read_text())["phases"] == {"prod": [], "experimental": []}


def test_prepare_preserves_nodeids_and_validates_container_runtime(
    project, generation, monkeypatch
):
    plan = _select(project, generation)
    plan["phases"]["prod"] = [APP_TEST + '::test_active[space " and $value]']
    monkeypatch.chdir(project)
    planner.prepare(
        plan,
        generation[0],
        BUCKET,
        PLATFORM,
        SOURCE_SHA,
        image_id=IMAGE_ID,
        check_runtime=True,
        world_size=1,
    )
    for phase in cache.PHASES:
        selected = generation[0] / ".testmon-work" / phase / "selected-tests"
        assert selected.read_text().splitlines() == plan["phases"][phase]


def test_host_prepare_needs_only_the_standard_library(project, generation):
    plan = _select(project, generation)
    output = project / "plan.json"
    output.write_text(json.dumps(plan))
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(HELPERS / "testmon_plan.py"),
            "prepare",
            "--plan",
            str(output),
            "--cache-dir",
            str(generation[0]),
            "--bucket",
            BUCKET,
            "--platform",
            PLATFORM,
            "--image-id",
            IMAGE_ID,
            "--source-sha",
            SOURCE_SHA,
        ],
        cwd=project,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("problem", ("full", "source", "image", "runtime", "world_size", "path"))
def test_prepare_rejects_stale_or_invalid_selection(project, generation, monkeypatch, problem):
    plan = _select(project, generation)
    if problem == "full":
        plan["mode"] = "full"
    elif problem == "source":
        plan["source_sha"] = "c" * 40
    elif problem == "image":
        plan["image_id"] = "sha256:" + "c" * 64
    elif problem == "runtime":
        plan["runtime"]["prod"]["python"] = "0.0.0"
    elif problem == "world_size":
        plan["world_size"] = 2
    else:
        plan["phases"]["prod"] = ["tests/unit_tests/../../escape.py::test_bad"]
    monkeypatch.chdir(project)
    with pytest.raises(ValueError):
        planner.prepare(
            plan,
            generation[0],
            BUCKET,
            PLATFORM,
            SOURCE_SHA,
            image_id=IMAGE_ID,
            check_runtime=True,
            world_size=1,
        )
    assert not (generation[0] / ".testmon-work").exists()


def test_explicit_file_bucket_recollects_after_source_changes(project, tmp_path):
    generation = _baseline(project, tmp_path / "cache", APP_TEST)
    unchanged = _select(project, generation, bucket=APP_TEST)
    assert unchanged["mode"] == "selected", unchanged["reason"]
    changed = _select(project, generation, bucket=APP_TEST, source_sha="c" * 40)
    assert changed["mode"] == "full"
    assert "explicit test-file" in changed["reason"]


def test_affected_file_recollects_new_parameter_ids(project, tmp_path):
    (project / "app.py").write_text(
        "VALUES = [1]\n\ndef active(value):\n    return value + len(VALUES)\n"
    )
    (project / APP_TEST).write_text(
        "import pytest\nfrom app import VALUES, active\n\n"
        "@pytest.mark.parametrize('value', VALUES)\n"
        "def test_value(value):\n    assert active(value) == value + 1\n"
    )
    generation = _baseline(project, tmp_path / "cache")
    (project / "app.py").write_text(
        "VALUES = [1, 2]\n\ndef active(value):\n    return value + len(VALUES)\n"
    )
    plan = _select(project, generation)
    assert plan["mode"] == "selected", plan["reason"]
    assert plan["phases"]["prod"] == [APP_TEST]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-c",
            str(project / "pytest.ini"),
            "--collect-only",
            *plan["phases"]["prod"],
        ],
        cwd=project,
        env={**os.environ, "PYTHONPATH": str(project)},
        text=True,
        capture_output=True,
        check=True,
    )
    assert "test_value[1]" in result.stdout
    assert "test_value[2]" in result.stdout


def test_mandatory_tests_prevent_empty_gpu_selection(project, generation, monkeypatch):
    _mapping(project)
    _pr_files(project, ["megatron/core/mandatory/unobserved.py"])
    plan = _select(project, generation)
    assert plan["mode"] == "selected", plan["reason"]
    assert plan["phases"] == {phase: [OTHER_TEST] for phase in cache.PHASES}
    assert plan["mandatory_files"] == [OTHER_TEST]

    monkeypatch.chdir(project)
    planner.prepare(plan, generation[0], BUCKET, PLATFORM, SOURCE_SHA, image_id=IMAGE_ID)
    for phase in cache.PHASES:
        output = generation[0] / ".testmon-work" / phase
        assert (output / "mandatory-tests").read_text().splitlines() == [OTHER_TEST]
        assert (output / "selected-tests").read_text().splitlines() == [OTHER_TEST]


def test_mandatory_mapping_uses_old_paths_in_renamed_pr_files(project, generation):
    _mapping(project)
    _pr_files(
        project,
        ["megatron/core/unmapped/renamed.py", "megatron/core/mandatory/original.py"],
        changed_files=1,
    )
    plan = _select(project, generation)
    assert plan["mode"] == "selected", plan["reason"]
    assert plan["mandatory_files"] == [OTHER_TEST]


@pytest.mark.parametrize(
    "problem",
    (
        "missing",
        "missing-metadata",
        "missing-files",
        "malformed",
        "wrong-sha",
        "metadata-list",
        "truncated",
        "missing-newline",
        "string-count",
        "boolean-count",
        "negative-count",
        "fractional-count",
        "too-many-files",
        "too-many-paths",
    ),
)
def test_invalid_pr_changed_file_artifact_runs_full_bucket(project, generation, problem):
    directory = _pr_files(project, ["megatron/core/mandatory/source.py"])
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if problem == "missing":
        directory = project / "missing-artifact"
    elif problem == "missing-metadata":
        metadata_path.unlink()
    elif problem == "missing-files":
        (directory / "changed-files").unlink()
    elif problem == "malformed":
        metadata_path.write_text("{")
    elif problem == "metadata-list":
        metadata_path.write_text("[]")
    elif problem == "missing-newline":
        (directory / "changed-files").write_text("megatron/core/mandatory/source.py")
    else:
        if problem == "wrong-sha":
            metadata["tested_sha"] = "c" * 40
        else:
            field, value = {
                "truncated": ("changed_paths", 2),
                "string-count": ("changed_files", "1"),
                "boolean-count": ("changed_files", True),
                "negative-count": ("changed_paths", -1),
                "fractional-count": ("changed_files", 1.5),
                "too-many-files": ("changed_files", 3001),
                "too-many-paths": ("changed_paths", 6001),
            }[problem]
            metadata[field] = value
        metadata_path.write_text(json.dumps(metadata))
    plan = _select(project, generation, pr_files_dir=directory)
    assert plan["mode"] == "full"
    assert plan["reason"]


def test_valid_empty_pr_still_allows_empty_selection(project, generation):
    _mapping(project)
    _pr_files(project, [])
    plan = _select(project, generation)
    assert plan["mode"] == "selected", plan["reason"]
    assert plan["mandatory_files"] == []
    assert plan["phases"] == {phase: [] for phase in cache.PHASES}


def test_mandatory_files_respect_child_buckets_and_platform_markers(project):
    child = project / "tests/unit_tests/example/child/test_child.py"
    child.parent.mkdir()
    child.write_text("# launch_on_gb200\ndef test_child():\n    pass\n")
    (project / OTHER_TEST).write_text("# launch_on_gb200\ndef test_other():\n    pass\n")
    _mapping(project, patterns=[BUCKET])
    for definition in cache.PLATFORMS.values():
        (project / definition["recipe"]).write_text(
            json.dumps(
                {"products": [{"test_case": [BUCKET, "tests/unit_tests/example/child/**/*.py"]}]}
            )
        )
    changed = ["megatron/core/mandatory/source.py"]
    assert planner._mandatory_files(project, BUCKET, "dgx_h100", changed) == [APP_TEST, OTHER_TEST]
    assert planner._mandatory_files(project, BUCKET, "dgx_gb200", changed) == [OTHER_TEST]


@pytest.mark.parametrize("problem", ("missing", "nodeid", "unselected", "escape"))
def test_prepare_rejects_invalid_mandatory_files(project, generation, monkeypatch, problem):
    plan = _select(project, generation)
    if problem == "missing":
        plan.pop("mandatory_files")
    else:
        plan["mandatory_files"] = [
            {
                "nodeid": OTHER_TEST + "::test_other",
                "unselected": OTHER_TEST,
                "escape": "tests/unit_tests/../../escape.py",
            }[problem]
        ]
    monkeypatch.chdir(project)
    with pytest.raises(ValueError):
        planner.prepare(plan, generation[0], BUCKET, PLATFORM, SOURCE_SHA, image_id=IMAGE_ID)
    assert not (generation[0] / ".testmon-work").exists()
