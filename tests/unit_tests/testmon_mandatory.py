# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Add the unit tests mapped to changed source directories to a Testmon selection."""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

import yaml
from find_test_cases import expand_pattern
from testmon_cache import PLATFORMS

DEFAULT_CONFIG = Path(__file__).resolve().with_name("testmon_mandatory_tests.yaml")


def load_mappings(config: Path) -> list[dict]:
    """Read and validate the `mappings` list of the mandatory-test configuration."""
    data = yaml.safe_load(config.read_text()) or {}
    mappings = data.get("mappings") if isinstance(data, dict) else None
    if not isinstance(mappings, list):
        raise ValueError(f"expected a top-level 'mappings' list: {config}")
    for mapping in mappings:
        source_dirs = mapping.get("source_dirs") if isinstance(mapping, dict) else None
        test_buckets = mapping.get("test_buckets") if isinstance(mapping, dict) else None
        if (
            not isinstance(source_dirs, list)
            or not all(isinstance(source, str) and source for source in source_dirs)
            or not isinstance(test_buckets, dict)
            or not all(
                platform in PLATFORMS
                and isinstance(patterns, list)
                and all(isinstance(pattern, str) and pattern for pattern in patterns)
                for platform, patterns in test_buckets.items()
            )
        ):
            raise ValueError(f"invalid mandatory-test mapping in {config}: {mapping!r}")
    return mappings


def _matches(changed_file: str, source: str) -> bool:
    source = source.rstrip("/")
    if any(character in source for character in "*?["):
        return fnmatch.fnmatch(changed_file, source)
    return changed_file == source or changed_file.startswith(source + "/")


def triggered_patterns(
    mappings: list[dict], changed_files: list[str], platform: str
) -> tuple[list[str], list[str]]:
    """Return the changed source entries that fired and the test patterns they require."""
    sources: list[str] = []
    patterns: list[str] = []
    for mapping in mappings:
        bucket_patterns = mapping["test_buckets"].get(platform) or []
        fired = [
            source
            for source in mapping["source_dirs"]
            if any(_matches(changed_file, source) for changed_file in changed_files)
        ]
        if fired and bucket_patterns:
            sources.extend(source for source in fired if source not in sources)
            patterns.extend(pattern for pattern in bucket_patterns if pattern not in patterns)
    return sources, patterns


def mandatory_files(patterns: list[str], bucket: str, ignored: set[str]) -> list[str]:
    """Expand test patterns to the pytest files that belong to the current bucket."""
    bucket_files = set(expand_pattern(bucket)) - ignored
    return sorted(
        {
            test_file
            for pattern in patterns
            for test_file in expand_pattern(pattern)
            if test_file in bucket_files and Path(test_file).name.startswith("test_")
        }
    )


def merge_selection(selection: Path, files: list[str]) -> list[str]:
    """Rewrite a Testmon selection so whole mandatory files replace their individual node IDs."""
    mandatory = set(files)
    selected = [
        nodeid
        for nodeid in selection.read_text().splitlines()
        if nodeid and nodeid.split("::", 1)[0] not in mandatory
    ]
    merged = sorted(set(selected) | mandatory)
    selection.write_text("".join(f"{nodeid}\n" for nodeid in merged))
    return merged


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--changed-files", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--platform", choices=PLATFORMS, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--ignore", action="append", default=[])
    return parser


def _run(args: argparse.Namespace) -> None:
    changed_files = [line for line in args.changed_files.read_text().splitlines() if line]
    sources, patterns = triggered_patterns(load_mappings(args.config), changed_files, args.platform)
    files = mandatory_files(patterns, args.bucket, set(args.ignore))
    (args.selection.parent / "mandatory-tests").write_text("".join(f"{path}\n" for path in files))
    merge_selection(args.selection, files)
    if sources:
        print(f"Mandatory tests: changed {', '.join(sources)}; added {len(files)} file(s).")
    else:
        print("Mandatory tests: no mapped source changes.")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _run(args)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"Testmon mandatory tests: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
