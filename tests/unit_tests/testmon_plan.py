# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Select unit tests on CPU without importing or collecting the GPU test suite."""

from __future__ import annotations

import argparse
import json
import platform
import re
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

from testmon_cache import PHASES, PLATFORMS, cache_identity, validate_cache


def _test_sources_unchanged(root: Path, source_sha: str, recipe_path: str | None = None) -> None:
    """Require unchanged test sources and recipe before reusing baseline test IDs."""

    def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, timeout=30, check=check
        )

    if git("cat-file", "-e", f"{source_sha}^{{commit}}", check=False).returncode:
        # Actions uses a shallow checkout; fetch the baseline without moving HEAD.
        git("fetch", "--no-tags", "--no-write-fetch-head", "--depth=1", "origin", source_sha)
    paths = [":(glob)tests/unit_tests/**/*.py"]
    if recipe_path is not None:
        paths.append(recipe_path)
    changed = git("diff", "--name-only", source_sha, "--", *paths).stdout
    added = git("ls-files", "--others", "--exclude-standard", "--", *paths).stdout
    if changed or added:
        raise ValueError("unit-test sources changed; GPU collection is required")


def _changed_files(directory: Path, source_sha: str) -> list[str]:
    # Configure validates API limits and paths before publishing this artifact.
    metadata = json.loads((directory / "metadata.json").read_text())
    paths = (directory / "changed-files").read_text().splitlines()
    if metadata["tested_sha"] != source_sha or metadata["changed_paths"] != len(paths):
        raise ValueError("PR changed-file artifact is incomplete or belongs to another commit")
    return paths


def _mandatory_files(
    root: Path, bucket: str, recipe_platform: str, changed: list[str]
) -> list[str]:
    import yaml
    from find_test_cases import (
        PLATFORM_MARKERS,
        expand_pattern,
        file_has_marker,
        is_child_of_bucket,
    )
    from testmon_mandatory import load_mappings, mandatory_files, triggered_patterns

    recipe_path = Path(PLATFORMS[recipe_platform]["recipe"])
    try:
        recipe = yaml.safe_load((root / recipe_path).read_text())
        mappings = load_mappings(root / "tests/unit_tests/testmon_mandatory_tests.yaml")
    except yaml.YAMLError as error:
        raise ValueError(f"invalid mandatory-test configuration: {error}") from error
    all_buckets = [case for product in recipe["products"] for case in product["test_case"]]
    bucket_files = set(expand_pattern(str(root / bucket)))
    ignored = set()
    for test_case in all_buckets:
        if test_case != bucket and is_child_of_bucket(test_case, bucket):
            ignored.update(expand_pattern(str(root / test_case)))
    marker = PLATFORM_MARKERS.get(recipe_path.parent.name)
    if marker:
        ignored.update(
            path
            for path in bucket_files
            if Path(path).name.startswith("test_") and not file_has_marker(path, marker)
        )
    _, patterns = triggered_patterns(mappings, changed, recipe_platform)
    return [
        str(Path(path).relative_to(root))
        for path in mandatory_files(
            [str(root / pattern) for pattern in patterns], str(root / bucket), ignored
        )
    ]


def _select_phase(root: Path, database: Path) -> list[str]:
    from testmon.process_code import Module, blob_to_checksums, read_source_sha

    with closing(
        sqlite3.connect(database.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    ) as connection:
        # Deselected placeholders have no duration. LEFT JOIN retains executed
        # tests without fingerprints so they are always selected.
        fingerprints = connection.execute(
            "SELECT te.test_name, te.failed, f.filename, f.fsha, f.method_checksums "
            "FROM test_execution te "
            "LEFT JOIN test_execution_file_fp tf ON tf.test_execution_id = te.id "
            "LEFT JOIN file_fp f ON f.id = tf.fingerprint_id "
            "WHERE te.duration IS NOT NULL"
        ).fetchall()

    selected = set()
    sources = {}
    for nodeid, failed, filename, recorded_sha, fingerprint in fingerprints:
        if failed != 0 or filename is None:
            selected.add(nodeid)
            continue
        if filename not in sources:
            path = root / filename
            source, source_sha = read_source_sha(path)
            checksums = (
                set(Module(source_code=source, ext=path.suffix.lstrip(".")).method_checksums)
                if source is not None
                else set()
            )
            sources[filename] = (source_sha, checksums)
        source_sha, checksums = sources[filename]
        if source_sha is None or (
            recorded_sha != source_sha
            and (not checksums or not set(blob_to_checksums(fingerprint)) <= checksums)
        ):
            selected.add(nodeid)
    return sorted(selected)


def select(
    root: Path,
    cache_dir: Path,
    bucket: str,
    recipe_platform: str,
    image_id: str,
    source_sha: str,
    matched_key: str,
    pr_files_dir: Path,
) -> dict:
    """Return a CPU selection, or a full-bucket plan if any evidence is unavailable."""
    plan = {
        "schema": 1,
        "source_sha": source_sha,
        "bucket": bucket,
        "platform": recipe_platform,
        "image_id": image_id,
        "mode": "full",
        "reason": "",
        "phases": {},
    }
    try:
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            raise ValueError("immutable container image identity is unavailable")
        identity = cache_identity(root, bucket, recipe_platform, image_id)
        manifest = validate_cache(cache_dir, identity, matched_key)
        if manifest.get("image_id") != image_id:
            # The CPU host cannot measure the packages in a different GPU image.
            # Equal image IDs prove the baseline runtime is still the execution runtime.
            raise ValueError("container image changed; GPU runtime validation is required")
        if not any(char in bucket for char in "*?[") and source_sha != manifest["source_sha"]:
            # Explicit-file pytest arguments bypass Testmon's stable-file collection skip.
            # They may discover new IDs from collection-only production dependencies.
            raise ValueError(
                "explicit test-file bucket requires GPU collection after source changes"
            )
        changed_files = _changed_files(pr_files_dir, source_sha)
        mandatory = _mandatory_files(root, bucket, recipe_platform, changed_files)
        _test_sources_unchanged(root, manifest["source_sha"], PLATFORMS[recipe_platform]["recipe"])
        runtimes = {}
        for phase in PHASES:
            runtime = json.loads((cache_dir / phase / "metadata.json").read_text())["runtime"]
            if (
                runtime["python"].split(".")[:2] != platform.python_version().split(".")[:2]
                or runtime["implementation"] != platform.python_implementation()
            ):
                raise ValueError("CPU Python differs from the Testmon fingerprint interpreter")
            runtimes[phase] = runtime
        phases = {
            phase: _select_phase(root, cache_dir / phase / ".testmondata") for phase in PHASES
        }
        # Recollect affected files in both modes: production dependencies can change
        # parameter IDs or experimental markers without changing the test source.
        affected_files = sorted(
            {nodeid.split("::", 1)[0] for nodes in phases.values() for nodeid in nodes}
            | set(mandatory)
        )
        phases = {phase: affected_files for phase in PHASES}
        plan.update(
            mode="selected",
            reason="selected on CPU from the compatible main baseline",
            phases=phases,
            runtime=runtimes,
            mandatory_files=mandatory,
        )
    except (
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        KeyError,
        sqlite3.Error,
        subprocess.SubprocessError,
    ) as error:
        plan["reason"] = str(error)
    return plan


def prepare(
    plan: dict,
    cache_dir: Path,
    bucket: str,
    recipe_platform: str,
    source_sha: str,
    *,
    image_id: str | None = None,
    check_runtime: bool = False,
) -> None:
    """Validate a selected plan before exposing its IDs to the distributed runner."""
    from testmon_cache import runtime_identity

    if plan["schema"] != 1 or plan["mode"] != "selected":
        raise ValueError("no valid CPU-selected Testmon plan")
    if (
        plan["source_sha"] != source_sha
        or plan["bucket"] != bucket
        or plan["platform"] != recipe_platform
    ):
        raise ValueError("Testmon plan does not match this checkout and bucket")
    if image_id is not None and plan["image_id"] != image_id:
        raise ValueError("Testmon plan container image changed")
    if check_runtime:
        runtime = runtime_identity()
        if any(plan["runtime"][phase] != runtime for phase in PHASES):
            raise ValueError("Testmon plan runtime environment changed")
    for phase in PHASES:
        output = cache_dir / ".testmon-work" / phase / "selected-tests"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(f"{nodeid}\n" for nodeid in plan["phases"][phase]))
        output.with_name("mandatory-tests").write_text(
            "".join(f"{filename}\n" for filename in plan["mandatory_files"])
        )


def main(argv: list[str] | None = None) -> int:
    """Write a selection artifact and Actions outputs; uncertainty always schedules GPUs."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select_parser = commands.add_parser("select")
    select_parser.add_argument("--matched-key", required=True)
    select_parser.add_argument("--pr-files-dir", type=Path, required=True)
    select_parser.add_argument("--output", type=Path, required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--plan", type=Path, required=True)
    prepare_parser.add_argument("--check-runtime", action="store_true")
    for child in (select_parser, prepare_parser):
        child.add_argument("--cache-dir", type=Path, required=True)
        child.add_argument("--bucket", required=True)
        child.add_argument("--platform", required=True)
        child.add_argument("--image-id", required=child is select_parser)
        child.add_argument("--source-sha", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        try:
            prepare(
                json.loads(args.plan.read_text()),
                args.cache_dir,
                args.bucket,
                args.platform,
                args.source_sha,
                image_id=args.image_id,
                check_runtime=args.check_runtime,
            )
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
            print(f"Unit Testmon plan unavailable: {error}", file=sys.stderr)
            return 2
        return 0
    plan = select(
        Path.cwd(),
        args.cache_dir,
        args.bucket,
        args.platform,
        args.image_id,
        args.source_sha,
        args.matched_key,
        args.pr_files_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n")
    has_tests = plan["mode"] == "full" or any(plan["phases"].values())
    print(f"mode={plan['mode']}")
    print(f"has_tests={str(has_tests).lower()}")
    print(f"Unit Testmon CPU selection: {plan['reason']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
