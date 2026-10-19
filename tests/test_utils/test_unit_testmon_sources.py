# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

HELPERS = Path(__file__).parents[2] / "tests/unit_tests"
for name in ("testmon_cache", "testmon_plan"):
    spec = importlib.util.spec_from_file_location(name, HELPERS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
planner = sys.modules["testmon_plan"]


def _git(root, *args, input=None):
    return subprocess.run(
        ["git", "-C", str(root), *args], input=input, text=True, capture_output=True, check=True
    ).stdout.strip()


def _commit(root, parent=None):
    _git(root, "add", "-A")
    tree = _git(root, "write-tree")
    commit = (
        f"tree {tree}\n"
        + (f"parent {parent}\n" if parent else "")
        + "author Test Fixture <fixture@example.test> 0 +0000\n"
        "committer Test Fixture <fixture@example.test> 0 +0000\n\nTest source fixture\n"
    )
    sha = _git(root, "hash-object", "-t", "commit", "-w", "--stdin", input=commit)
    _git(root, "update-ref", "HEAD", sha)
    return sha


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repository"
    tests = root / "tests/unit_tests"
    tests.mkdir(parents=True)
    (tests / "test_example.py").write_text("def test_example():\n    assert True\n")
    (tests / "shared_helper.py").write_text("VALUE = 1\n")
    (root / "app.py").write_text("VALUE = 1\n")
    _git(root, "init", "-q")
    return root, _commit(root)


def test_unchanged_tests_allow_production_changes(repository):
    root, baseline = repository
    planner._test_sources_unchanged(root, baseline)
    (root / "app.py").write_text("VALUE = 2\n")
    _commit(root, baseline)
    planner._test_sources_unchanged(root, baseline)


@pytest.mark.parametrize("staged", (False, True))
@pytest.mark.parametrize("change", ("modify", "add", "delete"))
def test_changed_test_sources_require_collection(repository, staged, change):
    root, baseline = repository
    helper = root / "tests/unit_tests/shared_helper.py"
    if change == "delete":
        helper.unlink()
    elif change == "add":
        helper.with_name("test_new.py").write_text("def test_new():\n    assert True\n")
    else:
        helper.write_text("VALUE = 2\n")
    if staged:
        _git(root, "add", "-A")
    with pytest.raises(ValueError, match="unit-test sources changed"):
        planner._test_sources_unchanged(root, baseline)


def test_bytecode_does_not_change_test_sources(repository):
    root, baseline = repository
    tests = root / "tests/unit_tests"
    bytecode = tests / "__pycache__/shared_helper.pyc"
    bytecode.parent.mkdir()
    bytecode.write_bytes(b"cached bytecode")
    planner._test_sources_unchanged(root, baseline)


@pytest.mark.parametrize("change", ("unchanged", "modify", "add", "delete"))
def test_recipe_must_match_baseline(repository, change):
    root, baseline = repository
    recipe_path = "tests/test_utils/recipes/h100/unit-tests.yaml"
    recipe = root / recipe_path
    recipe.parent.mkdir(parents=True)
    recipe.write_text("spec:\n  gpus: 8\n")
    if change != "add":
        baseline = _commit(root, baseline)
    if change == "modify":
        recipe.write_text("spec:\n  gpus: 4\n")
    elif change == "delete":
        recipe.unlink()
    if change == "unchanged":
        planner._test_sources_unchanged(root, baseline, recipe_path)
    else:
        with pytest.raises(ValueError, match="unit-test sources changed"):
            planner._test_sources_unchanged(root, baseline, recipe_path)


def test_shallow_checkout_fetches_baseline_without_changing_head(repository, tmp_path):
    root, baseline = repository
    (root / "app.py").write_text("VALUE = 2\n")
    current = _commit(root, baseline)
    checkout = tmp_path / "checkout"
    _git(tmp_path, "clone", "--depth=1", root.as_uri(), str(checkout))
    with pytest.raises(subprocess.CalledProcessError):
        _git(checkout, "cat-file", "-e", baseline)
    planner._test_sources_unchanged(checkout, baseline)
    assert _git(checkout, "rev-parse", "HEAD") == current
    _git(checkout, "cat-file", "-e", baseline)


def test_unavailable_baseline_fails_instead_of_accepting_sources(repository):
    root, _ = repository
    _git(root, "remote", "add", "origin", root.as_uri())
    with pytest.raises(subprocess.CalledProcessError) as error:
        planner._test_sources_unchanged(root, "f" * 40)
    assert error.value.cmd[1] == "fetch"
