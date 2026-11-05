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
    """Find a successful trusted main producer with a common cache generation."""
    response = json.loads(
        run(
            "gh",
            "api",
            f"repos/{REPOSITORY}/actions/workflows/populate-build-cache.yml/"
            "runs?branch=main&status=success&per_page=100",
        )
    )
    for candidate in response["workflow_runs"]:
        # Retried runs can publish different generations for different buckets.
        if (
            candidate["event"] in {"schedule", "workflow_dispatch"}
            and candidate["head_repository"]["full_name"] == REPOSITORY
            and candidate["run_attempt"] == 1
        ):
            return sha(candidate["head_sha"]), f'{candidate["id"]}-1'
    raise ValueError("No successful first-attempt main Testmon baseline found")


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
    if not re.fullmatch(r"[1-9][0-9]*-1", manifest["generation"]):
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
