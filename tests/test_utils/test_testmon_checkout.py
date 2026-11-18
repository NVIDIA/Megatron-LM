# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exercise baseline checkout preparation with real Git objects and bundles."""

import argparse
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/testmon_checkout.py"
SPEC = importlib.util.spec_from_file_location("testmon_checkout", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
checkout = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checkout)


def git(repo, *args):
    return (
        subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE)
        .decode()
        .strip()
    )


def commit(repo, message):
    git(repo, "add", ".")
    git(repo, "-c", "commit.gpgsign=false", "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.name", "CI test")
    git(repo, "config", "user.email", "ci@example.com")
    (repo / "model.py").write_text("original\n")
    (repo / "removed.py").write_text("remove me\n")
    (repo / "pyproject.toml").write_text("# unchanged image\n")
    baseline = commit(repo, "Baseline")
    git(repo, "remote", "add", "origin", str(repo))
    monkeypatch.chdir(repo)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.setattr(checkout, "latest_baseline", lambda: (baseline, "123-1"))
    return repo, baseline


def arguments(repo, head):
    pr_files = repo.parent / "pr-files"
    pr_files.mkdir(exist_ok=True)
    (pr_files / "metadata.json").write_text(json.dumps({"tested_sha": head, "changed_paths": 3}))
    (pr_files / "changed-files").write_text("model.py\nremoved.py\nnew.bin\n")
    return argparse.Namespace(
        tested_sha=head, pr_files_dir=str(pr_files), output_dir=str(pr_files / "testmon-checkout")
    )


def test_prepared_checkout_preserves_tested_tree_and_restores_bundle(
    repository, tmp_path, monkeypatch
):
    repo, baseline = repository
    (repo / "unrelated.py").write_text("new main work\n")
    (repo / "pyproject.toml").write_text("# newer main dependencies\n")
    commit(repo, "Newer main")
    git(repo, "checkout", "-b", "pr")
    (repo / "model.py").write_text("PR change\n")
    (repo / "removed.py").unlink()
    (repo / "new.bin").write_bytes(b"\x00\xff\x80complete binary change")
    head = commit(repo, "Complete PR change")
    args = arguments(repo, head)
    original_metadata = json.loads((Path(args.pr_files_dir) / "metadata.json").read_text())
    checkout.prepare(args)
    output = Path(args.output_dir)
    manifest = json.loads((output / "manifest.json").read_text())
    tested = manifest["tested_sha"]
    assert manifest["source_sha"] == baseline
    assert manifest["original_sha"] == head
    assert git(repo, "rev-parse", "HEAD") == head
    assert git(repo, "status", "--porcelain") == ""
    assert json.loads((Path(args.pr_files_dir) / "metadata.json").read_text()) == {
        **original_metadata,
        "tested_sha": tested,
    }

    # A separate, shallow consumer has no synthetic commit until bundle restore.
    consumer = tmp_path / "consumer"
    git(repo, "clone", "--depth=1", "--no-local", str(repo), str(consumer))
    monkeypatch.chdir(consumer)
    with pytest.raises(ValueError, match="does not belong"):
        checkout.restore(argparse.Namespace(directory=str(output), expected_sha=baseline))
    assert git(consumer, "rev-parse", "HEAD") == head
    github_output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    # Exercise the consumer action against the same combined artifact layout as CI.
    shutil.copytree(args.pr_files_dir, tmp_path / "testmon-pr-files")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("ORIGINAL_SHA", head)
    action = yaml.safe_load((SCRIPT.parents[1] / "actions/testmon-checkout/action.yml").read_text())
    restore = action["runs"]["steps"][-1]["run"].replace(
        str(SCRIPT.relative_to(SCRIPT.parents[2])), str(SCRIPT)
    )
    subprocess.run(["bash", "-euo", "pipefail", "-c", restore], check=True)
    assert git(consumer, "rev-parse", "HEAD") == tested
    assert git(consumer, "rev-parse", "HEAD^") == baseline
    assert git(consumer, "rev-parse", "HEAD^{tree}") == git(
        consumer, "rev-parse", f"{head}^{{tree}}"
    )
    assert (consumer / "model.py").read_text() == "PR change\n"
    assert (consumer / "pyproject.toml").read_text() == "# newer main dependencies\n"
    assert (consumer / "unrelated.py").read_text() == "new main work\n"
    assert not (consumer / "removed.py").exists()
    assert (consumer / "new.bin").read_bytes() == b"\x00\xff\x80complete binary change"
    assert github_output.read_text() == f"sha={tested}\nbaseline_sha={baseline}\ngeneration=123-1\n"


def test_newer_main_changes_do_not_conflict_with_preparation(repository):
    repo, baseline = repository
    (repo / "model.py").write_text("different main implementation\n")
    commit(repo, "New main")
    (repo / "model.py").write_text("PR implementation\n")
    head = commit(repo, "PR")
    args = arguments(repo, head)
    checkout.prepare(args)
    manifest = json.loads((Path(args.output_dir) / "manifest.json").read_text())
    tested = manifest["tested_sha"]
    git(repo, "fetch", "--no-tags", str(Path(args.output_dir) / "checkout.bundle"), "HEAD")
    assert git(repo, "rev-parse", f"{tested}^{{tree}}") == git(
        repo, "rev-parse", f"{head}^{{tree}}"
    )
    assert git(repo, "rev-parse", f"{tested}^") == baseline
    assert git(repo, "rev-parse", "HEAD") == head
    assert git(repo, "status", "--porcelain") == ""


def test_missing_baseline_preserves_checkout_and_metadata(repository, monkeypatch, capsys):
    repo, head = repository
    args = arguments(repo, head)
    monkeypatch.setattr(checkout, "latest_baseline", lambda: ("a" * 40, "123-1"))
    metadata_path = Path(args.pr_files_dir) / "metadata.json"
    original_metadata = metadata_path.read_bytes()
    checkout.prepare(args)
    message = capsys.readouterr().out
    assert "::notice::Using original PR checkout:" in message
    assert git(repo, "rev-parse", "HEAD") == head
    assert git(repo, "status", "--porcelain") == ""
    assert metadata_path.read_bytes() == original_metadata
    assert not Path(args.output_dir).exists()


def test_restore_rejects_a_bundle_that_changes_the_tested_tree(repository):
    repo, baseline = repository
    (repo / "model.py").write_text("current implementation\n")
    head = commit(repo, "Current main")
    args = arguments(repo, head)
    checkout.prepare(args)
    output = Path(args.output_dir)
    manifest = json.loads((output / "manifest.json").read_text())
    wrong = git(repo, "commit-tree", f"{baseline}^{{tree}}", "-p", baseline, "-m", "Stale source")
    producer = repo.parent / "stale-producer"
    git(repo, "clone", "--shared", "--no-checkout", str(repo), str(producer))
    git(producer, "update-ref", "HEAD", wrong)
    (output / "checkout.bundle").unlink()
    git(producer, "bundle", "create", str(output / "checkout.bundle"), "HEAD", f"^{baseline}")
    manifest["tested_sha"] = wrong
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not match the original tested source tree"):
        checkout.restore(argparse.Namespace(directory=str(output), expected_sha=head))
    assert git(repo, "rev-parse", "HEAD") == head


def test_baseline_lookup_excludes_build_only_forks_and_retries(monkeypatch):
    candidate = {
        "event": "schedule",
        "head_repository": {"full_name": checkout.REPOSITORY},
        "run_attempt": 1,
        "id": 123,
        "head_sha": "a" * 40,
    }
    candidates = [
        {**candidate, "event": "push"},
        {**candidate, "run_attempt": 2},
        {**candidate, "head_repository": {"full_name": "fork/Megatron-LM"}},
        candidate,
    ]
    monkeypatch.setattr(checkout, "run", lambda *args: json.dumps({"workflow_runs": candidates}))
    assert checkout.latest_baseline() == ("a" * 40, "123-1")
    candidates.pop()
    with pytest.raises(ValueError, match="No successful first-attempt"):
        checkout.latest_baseline()
