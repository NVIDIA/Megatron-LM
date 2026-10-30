# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Select unit tests on CPU without importing or collecting the GPU test suite."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

from testmon_cache import PHASES, PLATFORMS, cache_identity, validate_cache


def _changed_files(directory: Path, source_sha: str) -> list[str]:
    # Configure validates API limits and paths before publishing this artifact.
    metadata = json.loads((directory / "metadata.json").read_text())
    paths = (directory / "changed-files").read_text().splitlines()
    if metadata["tested_sha"] != source_sha or metadata["changed_paths"] != len(paths):
        raise ValueError("PR changed-file artifact is incomplete or belongs to another commit")
    return paths


def _bucket_files(
    root: Path, bucket: str, recipe_platform: str, changed: list[str]
) -> tuple[set[str], list[str]]:
    import yaml
    from find_test_cases import (
        PLATFORM_EXCLUSIONS,
        PLATFORM_MARKERS,
        expand_pattern,
        file_has_marker,
        is_child_of_bucket,
    )
    from gb200_placement import active_buckets
    from testmon_mandatory import load_mappings, mandatory_files, triggered_patterns

    recipe_path = Path(PLATFORMS[recipe_platform]["recipe"])
    try:
        recipe = yaml.safe_load((root / recipe_path).read_text())
        mappings = load_mappings(root / "tests/unit_tests/testmon_mandatory_tests.yaml")
    except yaml.YAMLError as error:
        raise ValueError(f"invalid mandatory-test configuration: {error}") from error
    all_buckets = [case for product in recipe["products"] for case in product["test_case"]]
    if recipe_platform == "dgx_gb200":
        all_buckets = active_buckets(
            all_buckets, "gb200", os.environ.get("UNIT_TEST_GB200_ENABLED") == "true"
        )
    bucket_files = set(expand_pattern(str(root / bucket)))
    ignored = {str(root / path) for path in PLATFORM_EXCLUSIONS.get(recipe_path.parent.name, ())}
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
    tests = {
        str(Path(path).relative_to(root))
        for path in bucket_files - ignored
        if Path(path).name.startswith("test_")
    }
    mandatory = [
        str(Path(path).relative_to(root))
        for path in mandatory_files(
            [str(root / pattern) for pattern in patterns], str(root / bucket), ignored
        )
    ]
    return tests, mandatory


def _select_phase(root: Path, database: Path, test_files: set[str]) -> set[str]:
    from testmon.process_code import Module, blob_to_checksums, read_source_sha

    with closing(
        sqlite3.connect(database.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    ) as connection:
        # Keep placeholders in the known-file inventory, but use only executed
        # tests for dependency selection.
        fingerprints = connection.execute(
            "SELECT te.test_name, te.duration, te.failed, f.filename, f.fsha, f.method_checksums "
            "FROM test_execution te "
            "LEFT JOIN test_execution_file_fp tf ON tf.test_execution_id = te.id "
            "LEFT JOIN file_fp f ON f.id = tf.fingerprint_id "
        ).fetchall()

    known_files = {row[0].split("::", 1)[0] for row in fingerprints}
    selected = test_files - known_files
    sources = {}
    for nodeid, duration, failed, filename, recorded_sha, fingerprint in fingerprints:
        if duration is None:
            continue
        test_file = nodeid.split("::", 1)[0]
        if failed != 0 or filename is None:
            selected.add(test_file)
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
        if filename == test_file and recorded_sha != source_sha:
            # Recollect changed test files to discover new tests and parameter IDs.
            selected.add(test_file)
        elif source_sha is None or (
            recorded_sha != source_sha
            and (not checksums or not set(blob_to_checksums(fingerprint)) <= checksums)
        ):
            selected.add(test_file)
    return selected & test_files


def select(
    root: Path,
    cache_dir: Path,
    bucket: str,
    recipe_platform: str,
    source_sha: str,
    matched_key: str,
    pr_files_dir: Path,
) -> dict:
    """Decide on CPU whether to execute selected files or the full bucket."""
    plan = {
        "schema": 1,
        "source_sha": source_sha,
        "bucket": bucket,
        "platform": recipe_platform,
        "mode": "full",
        "reason": "",
        "phases": {},
    }
    try:
        identity = cache_identity(root, bucket, recipe_platform)
        validate_cache(cache_dir, identity, matched_key)
        changed_files = _changed_files(pr_files_dir, source_sha)
        test_files, mandatory = _bucket_files(root, bucket, recipe_platform, changed_files)
        # Recollect affected files in both modes: production dependencies can change
        # parameter IDs or experimental markers without changing the test source.
        affected_files = set(mandatory) | (test_files & set(changed_files))
        for phase in PHASES:
            affected_files.update(
                _select_phase(root, cache_dir / phase / ".testmondata", test_files)
            )
        phases = {phase: sorted(affected_files) for phase in PHASES}
        plan.update(
            mode="selected",
            reason="selected on CPU from the compatible main baseline",
            phases=phases,
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
    ) as error:
        plan["reason"] = str(error)
    return plan


def prepare(
    plan: dict, cache_dir: Path, bucket: str, recipe_platform: str, source_sha: str
) -> None:
    """Validate the plan identity and write the distributed runner's selection files."""
    if plan["schema"] != 1 or plan["mode"] != "selected":
        raise ValueError("no valid CPU-selected Testmon plan")
    if (
        plan["source_sha"] != source_sha
        or plan["bucket"] != bucket
        or plan["platform"] != recipe_platform
    ):
        raise ValueError("Testmon plan does not match this checkout and bucket")
    for phase in PHASES:
        output = cache_dir / ".testmon-work" / phase / "selected-tests"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(f"{nodeid}\n" for nodeid in plan["phases"][phase]))
        output.with_name("mandatory-tests").write_text(
            "".join(f"{filename}\n" for filename in plan["mandatory_files"])
        )
