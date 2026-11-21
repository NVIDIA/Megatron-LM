#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Apply a complete PR diff to a trusted Testmon baseline for unit-test CI."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

REPOSITORY = "NVIDIA/Megatron-LM"


def run(
    *args: str,
    cwd: Path | None = None,
    data: bytes | None = None,
    env: dict[str, str] | None = None,
) -> bytes:
    """Run a bounded command, retaining binary output and failure diagnostics."""
    return subprocess.run(
        args, cwd=cwd, input=data, env=env, check=True, capture_output=True, timeout=120
    ).stdout


def git(
    *args: str,
    cwd: Path | None = None,
    data: bytes | None = None,
    env: dict[str, str] | None = None,
) -> str:
    """Run Git and return its textual result."""
    return run("git", *args, cwd=cwd, data=data, env=env).decode().strip()


def sha(value: object) -> str:
    """Validate a full commit SHA before passing it to Git."""
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("Expected a full Git commit SHA")
    return value


def latest_baseline() -> tuple[str, str]:
    """Find a completed trusted main producer with a common cache generation."""
    # Cache entries prove that the run actually published data. Build-only
    # schedules and main pushes without a matching merge group create none.
    api = f"repos/{REPOSITORY}/actions"
    seen = set()
    newest = None
    page = 1
    while True:
        caches = json.loads(
            run(
                "gh",
                "api",
                f"{api}/caches?ref=refs%2Fheads%2Fmain&key=unit-testmon-v"
                f"&sort=created_at&direction=desc&per_page=100&page={page}",
            )
        )["actions_caches"]
        for cache in caches:
            # An older push can finish or retry after a newer push. Prefer the
            # newer producer. A run cannot start after it creates its caches.
            if newest and datetime.fromisoformat(cache["created_at"]) < newest[0]:
                return newest[2:]
            match = re.fullmatch(
                r"unit-testmon-v\d+-main-[\w-]+-[a-f0-9]{16}-([1-9]\d*)-([1-9]\d*)", cache["key"]
            )
            if not match or match.groups() in seen:
                continue
            seen.add(match.groups())
            run_id, attempt = match.groups()
            try:
                producer = json.loads(run("gh", "api", f"{api}/runs/{run_id}"))
            except subprocess.CalledProcessError as error:
                if b"HTTP 404" in (error.stderr or b""):
                    continue  # A deleted run can leave unexpired caches.
                raise
            if (
                producer["head_branch"] != "main"
                or (producer.get("head_repository") or {}).get("full_name") != REPOSITORY
                or producer["status"] != "completed"
                or str(producer["run_attempt"]) != attempt
            ):
                continue
            published = (
                producer["path"] == ".github/workflows/publish-unit-testmon.yml"
                and producer["event"] == "push"
                and producer["conclusion"] == "success"
            )
            # The manual GPU matrix (and earlier scheduled matrices) can publish
            # different generations across buckets after a partial retry.
            gpu_baseline = (
                producer["path"] == ".github/workflows/populate-build-cache.yml"
                and producer["event"] in {"schedule", "workflow_dispatch"}
                and producer["conclusion"] in {"success", "failure"}
                and attempt == "1"
            )
            if published or gpu_baseline:
                candidate = (
                    datetime.fromisoformat(producer["created_at"]),
                    int(run_id),
                    sha(producer["head_sha"]),
                    f"{run_id}-{attempt}",
                )
                if newest is None or candidate > newest:
                    newest = candidate
        if len(caches) < 100:
            if newest:
                return newest[2:]
            raise ValueError("No published main Testmon baseline found")
        page += 1


def outputs(manifest: dict[str, str]) -> None:
    """Publish the prepared checkout identity to Actions and its log."""
    result = (
        f'sha={manifest["tested_sha"]}\n'
        f'baseline_sha={manifest["source_sha"]}\n'
        f'generation={manifest["generation"]}\n'
    )
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write(result)
    sys.stdout.write(result)


def prepare(args: argparse.Namespace) -> None:
    """Bundle the PR diff on the selected baseline, preserving the original on failure."""
    output = Path(args.output_dir).resolve()
    bundle = str(output / "checkout.bundle")
    metadata_path = Path(args.pr_files_dir).resolve() / "metadata.json"
    original_metadata = None
    rewritten = False
    try:
        original = sha(args.tested_sha)
        head, base = sha(args.pr_head), sha(args.pr_base)
        if git("rev-parse", "HEAD") != original:
            raise ValueError("Checkout does not match the original tested commit")
        original_metadata = metadata_path.read_bytes()
        metadata = json.loads(original_metadata)
        if metadata["tested_sha"] != original:
            raise ValueError("PR files metadata does not match the original tested commit")
        source, generation = latest_baseline()
        git("fetch", "--no-tags", "origin", source)
        merge_base = git("merge-base", base, head)
        patch = run("git", "diff", "--binary", "--full-index", "--no-ext-diff", merge_base, head)
        with tempfile.TemporaryDirectory(prefix="testmon-checkout-") as temporary:
            checkout = Path(temporary) / "checkout"
            git("clone", "--shared", "--no-checkout", str(Path.cwd()), str(checkout))
            git("checkout", "--detach", source, cwd=checkout)
            if patch:
                git("apply", "--3way", "--index", cwd=checkout, data=patch)
            tree = git("write-tree", cwd=checkout)
            date = git("show", "-s", "--format=%cI", original)
            identity = {
                f"GIT_{role}_{field}": value
                for role in ("AUTHOR", "COMMITTER")
                for field, value in (
                    ("NAME", "Megatron CI"),
                    ("EMAIL", "megatron-ci@nvidia.com"),
                    ("DATE", date),
                )
            }
            tested = git(
                "commit-tree",
                tree,
                "-p",
                source,
                data=f"Apply PR {head} to Testmon baseline {source}\n".encode(),
                env={**os.environ, **identity},
                cwd=checkout,
            )
            git("update-ref", "HEAD", tested, cwd=checkout)
            output.mkdir(parents=True, exist_ok=True)
            git("bundle", "create", bundle, "HEAD", f"^{source}", cwd=checkout)
        manifest = {
            "source_sha": source,
            "tested_sha": tested,
            "original_sha": original,
            "generation": generation,
        }
        (output / "manifest.json").write_text(json.dumps(manifest) + "\n")
        metadata["tested_sha"] = tested
        rewritten = True
        metadata_path.write_text(json.dumps(metadata) + "\n")
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        if rewritten:
            assert original_metadata is not None
            metadata_path.write_bytes(original_metadata)
        shutil.rmtree(output, ignore_errors=True)
        detail = str(error)
        if (
            isinstance(error, (subprocess.CalledProcessError, subprocess.TimeoutExpired))
            and error.stderr
        ):
            detail += ": " + error.stderr.decode(errors="replace").strip()
        detail = detail.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        sys.stdout.write(f"::notice::Using original PR checkout: {detail}\n")
        return
    outputs(manifest)


def restore(args: argparse.Namespace) -> None:
    """Validate and check out the exact bundled commit for this CI run."""
    directory = Path(args.directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    original, source, tested = (
        sha(manifest[key]) for key in ("original_sha", "source_sha", "tested_sha")
    )
    if original != sha(args.expected_sha) or git("rev-parse", "HEAD") != original:
        raise ValueError("Prepared checkout does not belong to the original tested commit")
    if not re.fullmatch(r"[1-9][0-9]*-[1-9][0-9]*", manifest["generation"]):
        raise ValueError("Invalid Testmon baseline generation")
    bundle = str(directory / "checkout.bundle")
    if git("bundle", "list-heads", bundle) != f"{tested} HEAD":
        raise ValueError("Bundle does not match the prepared checkout")
    git("fetch", "--no-tags", "--depth=1", "origin", source)
    git("bundle", "verify", bundle)
    git("fetch", "--no-tags", bundle, "HEAD")
    if git("rev-list", "--parents", "-n", "1", tested) != f"{tested} {source}":
        raise ValueError("Prepared checkout does not have the expected baseline parent")
    git("checkout", "--detach", tested)
    outputs(manifest)


def main() -> None:
    """Dispatch preparation or restoration from the workflow CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    for option in ("pr-head", "pr-base", "tested-sha", "pr-files-dir", "output-dir"):
        prepare_parser.add_argument(f"--{option}", required=True)
    restore_parser = commands.add_parser("restore")
    restore_parser.add_argument("--directory", required=True)
    restore_parser.add_argument("--expected-sha", required=True)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else restore)(args)


if __name__ == "__main__":
    main()
