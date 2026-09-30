# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Source-directory-to-test-bucket mapping for selective unit testing.

Some execution paths (e.g. CUDA autograd callbacks) are not recorded by
Testmon's tracing.  This module reads the changed paths supplied by CI and,
when any changed file falls under a mapped source directory, returns the
test buckets that must run in full — regardless of Testmon's deselection.

Usage from CI::

    python tests/unit_tests/testmon_source_mapping.py \\
        --root . --platform dgx_h100 --changed-files-json changed-files.json

Or with an explicit file list on stdin::

    python tests/unit_tests/testmon_source_mapping.py \\
        --root . --platform dgx_h100 --changed-files-stdin < changed-files.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SOURCE_MAPPING_FILE = "tests/unit_tests/testmon_source_mapping.yml"
RECIPE_GLOB = "tests/test_utils/recipes/*/unit-tests.yaml"


def _load_yaml(path: Path):
    """Load a YAML file, falling back to a minimal parser when PyYAML is unavailable."""
    text = path.read_text()
    try:
        from yaml import safe_load

        return safe_load(text)
    except ImportError:
        return _parse_mapping_yaml(text)


def _parse_mapping_yaml(text: str) -> dict:
    """Parse testmon_source_mapping.yml without PyYAML.

    Handles only the block-style mapping format used by this file: a top-level
    ``mappings`` key containing a sequence of rules with ``source_dirs`` and
    ``test_buckets``.
    """
    mappings: list[dict] = []
    rule: dict | None = None
    section: str | None = None
    platform: str | None = None

    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip()
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        comment = stripped.find(" #")
        if comment > 0:
            stripped = stripped[:comment].rstrip()
        if stripped == "mappings:":
            continue
        if stripped == "- source_dirs:":
            rule = {"source_dirs": [], "test_buckets": {}}
            mappings.append(rule)
            section = "source_dirs"
            platform = None
        elif section == "source_dirs" and stripped.startswith("- "):
            if rule is None:
                raise ValueError(f"line {lineno}: list item outside a rule")
            rule["source_dirs"].append(stripped[2:].strip())
        elif stripped == "test_buckets:" and rule is not None:
            section = "test_buckets"
            platform = None
        elif (
            section == "test_buckets"
            and stripped.endswith(":")
            and not stripped.startswith("- ")
            and rule is not None
        ):
            platform = stripped[:-1].strip()
            rule["test_buckets"].setdefault(platform, [])
        elif section == "test_buckets" and stripped.startswith("- ") and platform is not None:
            if rule is None:
                raise ValueError(f"line {lineno}: list item outside a rule")
            rule["test_buckets"][platform].append(stripped[2:].strip())
        else:
            raise ValueError(f"line {lineno}: unexpected content in source mapping")
    return {"mappings": mappings}


def _discover_recipe_platforms(root: Path) -> frozenset[str]:
    """Extract supported platform names from unit-test recipe files."""
    platforms: set[str] = set()
    for recipe in sorted(root.glob(RECIPE_GLOB)):
        match = re.search(r"^\s+platforms:\s+(\S+)", recipe.read_text(), re.MULTILINE)
        if match:
            platforms.add(match.group(1))
    if not platforms:
        raise ValueError("no unit-test recipe platforms discovered")
    return frozenset(platforms)


def _validate_source_dir(source_dir: str) -> None:
    """Reject obviously unsafe source directory paths."""
    if (
        not source_dir
        or "\0" in source_dir
        or source_dir.startswith("/")
        or ".." in source_dir.split("/")
    ):
        raise ValueError(f"unsafe source directory path: {source_dir!r}")


def _load_source_mapping(root: Path, valid_platforms: frozenset[str] | None = None) -> list[dict]:
    """Load and validate the source mapping configuration."""
    path = root / SOURCE_MAPPING_FILE
    if not path.is_file():
        return []
    data = _load_yaml(path)
    if not isinstance(data, dict):
        raise ValueError(f"source mapping must be a YAML mapping: {path}")
    mappings = data.get("mappings")
    if mappings is None:
        return []
    if not isinstance(mappings, list):
        raise ValueError(f"'mappings' must be a sequence: {path}")
    if valid_platforms is None:
        valid_platforms = _discover_recipe_platforms(root)
    for i, rule in enumerate(mappings):
        if not isinstance(rule, dict):
            raise ValueError(f"mapping rule {i}: must be a mapping")
        dirs = rule.get("source_dirs")
        if not isinstance(dirs, list) or not dirs:
            raise ValueError(f"mapping rule {i}: 'source_dirs' must be a non-empty list")
        for d in dirs:
            if not isinstance(d, str):
                raise ValueError(f"mapping rule {i}: source directory must be a string")
            _validate_source_dir(d)
        buckets = rule.get("test_buckets")
        if not isinstance(buckets, dict):
            raise ValueError(f"mapping rule {i}: 'test_buckets' must be a mapping")
        for platform_name, bucket_list in buckets.items():
            if platform_name not in valid_platforms:
                raise ValueError(
                    f"mapping rule {i}: unknown platform {platform_name!r} "
                    f"(discovered: {sorted(valid_platforms)})"
                )
            if not isinstance(bucket_list, list):
                raise ValueError(f"mapping rule {i}: buckets for {platform_name!r} must be a list")
    return mappings


def forced_full_buckets(root: Path, changed_files: list[str], platform: str) -> list[str]:
    """Return test buckets that must run fully because changed files touch mapped source dirs.

    Args:
        root: Repository root directory.
        changed_files: Repo-relative paths of files changed in the PR.
        platform: Recipe platform name (e.g. ``dgx_h100``).

    Returns:
        Sorted list of test-bucket strings whose tests must run in full.
    """
    valid_platforms = _discover_recipe_platforms(root)
    if platform not in valid_platforms:
        return []
    mappings = _load_source_mapping(root, valid_platforms)
    if not mappings:
        return []
    buckets: set[str] = set()
    for rule in mappings:
        source_dirs = rule.get("source_dirs", [])
        triggered = False
        for changed in changed_files:
            for source_dir in source_dirs:
                if changed == source_dir or changed.startswith(source_dir + "/"):
                    triggered = True
                    break
            if triggered:
                break
        if triggered:
            platform_buckets = rule.get("test_buckets", {}).get(platform, [])
            buckets.update(platform_buckets)
    return sorted(buckets)


def load_changed_files(path: Path) -> list[str]:
    """Read a JSON array of paths without changing filename characters."""
    try:
        changed_files = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid changed-files JSON in {path}: {error}") from error
    if not isinstance(changed_files, list) or any(
        not isinstance(changed, str) or not changed for changed in changed_files
    ):
        raise ValueError(f"changed files must be a JSON array of non-empty strings: {path}")
    return changed_files


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: print test buckets that must run fully."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--platform", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--changed-files-json", type=Path, help="JSON file containing changed paths from CI"
    )
    group.add_argument(
        "--changed-files-stdin",
        action="store_true",
        help="Read changed file paths from stdin, one per line",
    )
    args = parser.parse_args(argv)

    try:
        if args.changed_files_json:
            files = load_changed_files(args.changed_files_json)
        else:
            files = [line.strip() for line in sys.stdin if line.strip()]
        buckets = forced_full_buckets(args.root, files, args.platform)
    except (OSError, ValueError) as error:
        print(f"Testmon source mapping: {error}", file=sys.stderr)
        return 1
    for bucket in buckets:
        print(bucket)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
