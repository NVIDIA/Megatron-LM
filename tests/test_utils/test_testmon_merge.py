# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise local Testmon merges against real Git repositories."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / ".github/scripts/prepare_testmon_merge.py"


def git(root, *args):
    """Run Git inside a test repository."""
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def repository(tmp_path):
    """Create diverged main and PR branches, then clone the PR checkout."""
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "-b", "main")
    git(origin, "config", "user.name", "Test")
    git(origin, "config", "user.email", "test@example.com")
    git(origin, "config", "commit.gpgsign", "false")
    (origin / "base.py").write_text("base = 1\n")
    git(origin, "add", ".")
    git(origin, "commit", "-m", "base")
    git(origin, "checkout", "-b", "pr")
    (origin / "pr.py").write_text("pr = 1\n")
    git(origin, "add", ".")
    git(origin, "commit", "-m", "PR")
    source = git(origin, "rev-parse", "HEAD")
    git(origin, "checkout", "main")
    git(origin, "mv", "base.py", "renamed.py")
    git(origin, "commit", "-m", "main rename")
    checkout = tmp_path / "checkout"
    git(tmp_path, "clone", "-b", "pr", "--depth=1", origin.as_uri(), str(checkout))
    paths = tmp_path / "paths"
    paths.mkdir()
    (paths / "changed-files").write_text("pr.py\n")
    (paths / "metadata.json").write_text(json.dumps({"tested_sha": source, "changed_paths": 1}))
    return origin, checkout, paths, source


def run_merge(checkout, source, tmp_path, **environment):
    """Execute the same helper used by the CPU and GPU workflow steps."""
    output = tmp_path / "outputs"
    output.unlink(missing_ok=True)
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=checkout,
        env={**os.environ, "SOURCE_SHA": source, "GITHUB_OUTPUT": str(output), **environment},
        capture_output=True,
        text=True,
    )
    values = (
        dict(line.split("=", 1) for line in output.read_text().splitlines())
        if output.exists()
        else {}
    )
    return result, values


def test_cpu_and_gpu_use_same_workspace_when_main_advances(repository, tmp_path):
    """The GPU pins main and reproduces the CPU tree without pushing or committing."""
    origin, checkout, paths, source = repository
    main = git(origin, "rev-parse", "main")
    result, outputs = run_merge(checkout, source, tmp_path, PR_FILES_DIR=str(paths))
    assert result.returncode == 0, result.stderr
    assert outputs["main_sha"] == main
    assert git(checkout, "rev-parse", "HEAD") == source
    assert git(origin, "rev-parse", "pr") == source
    assert (paths / "changed-files").read_text().splitlines() == ["base.py", "pr.py", "renamed.py"]
    assert json.loads((paths / "metadata.json").read_text())["tested_sha"] == source
    (origin / "later.py").write_text("later = 1\n")
    git(origin, "add", ".")
    git(origin, "commit", "-m", "later main")
    gpu = tmp_path / "gpu"
    git(tmp_path, "clone", "-b", "pr", "--depth=1", origin.as_uri(), str(gpu))
    result, gpu_outputs = run_merge(
        gpu, source, tmp_path, MAIN_SHA=main, EXPECTED_TREE=outputs["tree"]
    )
    assert result.returncode == 0, result.stderr
    assert gpu_outputs == outputs
    assert not (gpu / "later.py").exists()


@pytest.mark.parametrize("failure", ["conflict", "wrong-source", "wrong-tree", "missing-main"])
def test_merge_failures_stop_selection(repository, tmp_path, failure):
    """Never publish selection inputs after a merge or identity failure."""
    origin, checkout, _, source = repository
    environment = {}
    if failure == "conflict":
        git(checkout, "config", "user.name", "Test")
        git(checkout, "config", "user.email", "test@example.com")
        git(checkout, "rm", "base.py")
        git(checkout, "-c", "commit.gpgsign=false", "commit", "-m", "delete")
        source = git(checkout, "rev-parse", "HEAD")
    elif failure == "wrong-source":
        source = "0" * 40
    elif failure == "wrong-tree":
        environment["EXPECTED_TREE"] = "0" * 40
    else:
        environment["MAIN_SHA"] = "0" * 40
    result, outputs = run_merge(checkout, source, tmp_path, **environment)
    assert result.returncode != 0
    assert outputs == {}


@pytest.mark.parametrize("metadata", [None, {"tested_sha": "stale", "changed_paths": 1}])
def test_invalid_pr_paths_keep_full_bucket_fallback(repository, tmp_path, metadata):
    """An invalid PR artifact must remain unusable after merging main."""
    _, checkout, paths, source = repository
    path = paths / "metadata.json"
    if metadata is None:
        path.unlink()
    else:
        path.write_text(json.dumps(metadata))
    result, outputs = run_merge(checkout, source, tmp_path, PR_FILES_DIR=str(paths))
    assert result.returncode == 0, result.stderr
    assert outputs["tree"] == git(checkout, "write-tree")
    assert not path.exists()


def test_workflow_merges_before_selection_and_after_gpu_checkout():
    """Keep both merge steps ordered before their workspace consumers."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/_unit-tests.yml").read_text())
    select = workflow["jobs"]["select"]
    assert "unit_testmon_target_branch == 'main'" in select["if"]
    steps = select["steps"]
    ids = [step.get("id") for step in steps]
    assert ids.index("checkout") < ids.index("merge-main") < ids.index("plan")
    action = yaml.safe_load((ROOT / ".github/actions/action.yml").read_text())
    steps = action["runs"]["steps"]
    checkout = next(i for i, step in enumerate(steps) if step["name"] == "Checkout repository")
    merge = next(i for i, step in enumerate(steps) if step["name"].startswith("Merge the same"))
    validate = next(
        i for i, step in enumerate(steps) if step.get("id") == "validate-unit-testmon-plan"
    )
    assert checkout < merge < validate
    assert steps[merge]["env"]["EXPECTED_TREE"] == "${{ inputs.unit_testmon_source_tree }}"


def test_main_already_in_checkout_is_unchanged(repository, tmp_path):
    """Merging an already included main is a no-op."""
    _, checkout, _, _ = repository
    git(checkout, "fetch", "origin", "main")
    git(checkout, "checkout", "--detach", "FETCH_HEAD")
    source = git(checkout, "rev-parse", "HEAD")
    tree = git(checkout, "rev-parse", "HEAD^{tree}")
    result, outputs = run_merge(checkout, source, tmp_path)
    assert result.returncode == 0, result.stderr
    assert outputs == {"main_sha": source, "tree": tree}
    assert git(checkout, "status", "--porcelain") == ""
