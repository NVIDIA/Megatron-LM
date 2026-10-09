# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from pathlib import Path

import pytest
import yaml

from tests.test_utils.python_scripts import unit_compiler_cache as cache


@pytest.fixture
def source_tree(tmp_path):
    for name in (
        "pyproject.toml",
        "uv.lock",
        "docker/.ngc_version.dev",
        "docker/Dockerfile.ci.dev",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    return tmp_path


@pytest.mark.parametrize(
    "changed", ["uv.lock", "pyproject.toml", "docker/.ngc_version.dev", "docker/Dockerfile.ci.dev"]
)
def test_dependency_build_changes_invalidate_cache(source_tree, changed):
    before = cache.cache_prefix(source_tree, "dgx_h100", {"torch": "version-a"})
    (source_tree / changed).write_text("changed")
    assert cache.cache_prefix(source_tree, "dgx_h100", {"torch": "version-a"}) != before


def test_runtime_and_platform_separate_cache_namespaces(source_tree):
    keys = {
        cache.cache_prefix(source_tree, platform, {"torch": version, "gpu": gpu})
        for platform in ("dgx_h100", "dgx_gb200")
        for version in ("version-a", "version-b")
        for gpu in ("H100", "B200")
    }
    assert len(keys) == 8


def test_source_and_test_selection_do_not_invalidate_compiler_cache(source_tree):
    before = cache.cache_prefix(source_tree, "dgx_h100", {})
    for name in ("megatron/core/example.py", cache.GDN_BUCKET, "assets_dir/testmon/selected-tests"):
        path = source_tree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("a changed source file or a single selected nodeid")
    assert cache.cache_prefix(source_tree, "dgx_h100", {}) == before


@pytest.mark.parametrize(
    "scope,bucket,environment,tag",
    [
        ("unit-tests", "tests/unit_tests/ssm/**/*.py", "dev", "latest"),
        ("unit-tests", cache.GDN_BUCKET, "lts", "latest"),
        ("unit-tests", cache.GDN_BUCKET, "dev", "legacy"),
        ("mr-github", cache.GDN_BUCKET, "dev", "latest"),
    ],
)
def test_other_workloads_keep_their_compiler_settings(scope, bucket, environment, tag):
    assert cache.compiler_cache_env(scope, bucket, environment, tag) == {}


def test_action_restores_optional_cache_and_only_publishes_trusted_baselines():
    action = yaml.safe_load((Path(__file__).parents[2] / ".github/actions/action.yml").read_text())
    steps = {step.get("id"): step for step in action["runs"]["steps"] if "id" in step}
    restore = steps["restore-gdn-compiler"]
    assert restore["continue-on-error"]
    assert restore["with"]["path"] == "assets_dir/compiler-cache/gdn"
    assert "lookup-" in restore["with"]["key"]
    producer = steps["prepare-gdn-compiler"]["if"]
    for guard in (
        "steps.unit-testmon.outputs.mode == 'baseline'",
        "steps.run-main-script.outcome == 'success'",
        "github.repository == 'NVIDIA/Megatron-LM'",
        "github.ref == 'refs/heads/main'",
    ):
        assert guard in producer
    save = next(
        step for step in action["runs"]["steps"] if step["name"] == "Save GDN compiler cache"
    )
    assert save["if"] == "steps.prepare-gdn-compiler.outcome == 'success'"
    assert save["with"]["path"] == restore["with"]["path"]
