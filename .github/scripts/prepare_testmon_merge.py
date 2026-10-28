# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Merge main into the workspace before Testmon selection or execution."""

import json
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def git(*args: str) -> str:
    """Run Git, propagating failures before any selection can be published."""
    return subprocess.check_output(["git", *args], text=True).rstrip("\n")


def prepare(source_sha: str, directory: Path | None, main_sha: str = "") -> tuple[str, str]:
    """Merge a pinned main locally, keeping HEAD and remote branches unchanged."""
    if git("rev-parse", "HEAD") != source_sha:
        raise ValueError("Testmon checkout does not match the requested source SHA")
    if git("rev-parse", "--is-shallow-repository") == "true":
        git("fetch", "--unshallow", "--no-tags", "origin")
    git("fetch", "--no-tags", "origin", main_sha or "refs/heads/main")
    main_sha = git("rev-parse", "FETCH_HEAD")
    git(
        "-c",
        "user.name=Testmon CI",
        "-c",
        "user.email=testmon-ci@users.noreply.github.com",
        "merge",
        "--no-commit",
        "--no-ff",
        main_sha,
    )
    tree = git("write-tree")
    logger.info("Workspace changes from main %s:\n%s", main_sha, git("diff", "HEAD", "--stat"))
    if directory is None:
        return main_sha, tree

    # Keep the existing full-bucket fallback for missing or stale PR metadata.
    # Add paths introduced by main (including both sides of renames) before
    # selecting against the merged workspace. HEAD remains the requested SHA.
    try:
        metadata = json.loads((directory / "metadata.json").read_text())
        paths = (directory / "changed-files").read_text().splitlines()
        if metadata["tested_sha"] != source_sha or metadata["changed_paths"] != len(paths):
            raise ValueError("PR paths do not match the original checkout")
        changed = git("diff", "--name-only", "--no-renames", "-z", source_sha)
        added_paths = changed.rstrip("\0").split("\0") if changed else []
        if any("\n" in path or "\r" in path for path in added_paths):
            raise ValueError("Merged paths cannot be represented in the PR file list")
        paths = sorted(set(paths) | set(added_paths))
        metadata.update(changed_paths=len(paths))
        (directory / "changed-files").write_text("".join(f"{path}\n" for path in paths))
        (directory / "metadata.json").write_text(json.dumps(metadata))
    except (OSError, ValueError, KeyError, TypeError) as error:
        (directory / "metadata.json").unlink(missing_ok=True)
        logger.warning("Merged PR paths unavailable; full buckets will run: %s", error)
    return main_sha, tree


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    original = os.environ["SOURCE_SHA"]
    main, tree = prepare(
        original,
        Path(os.environ["PR_FILES_DIR"]) if os.environ.get("PR_FILES_DIR") else None,
        os.environ.get("MAIN_SHA", ""),
    )
    expected_tree = os.environ.get("EXPECTED_TREE", "")
    if expected_tree and tree != expected_tree:
        raise ValueError("GPU workspace does not match the tree inspected by Testmon")
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.write(f"main_sha={main}\n")
        output.write(f"tree={tree}\n")
