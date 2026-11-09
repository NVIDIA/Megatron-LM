# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import yaml

# Platforms whose unit-test selection is driven by a pytest marker rather than
# by the full recipe bucket. Only files carrying the marker are launched.
PLATFORM_MARKERS = {"gb200": "launch_on_gb200"}
# The dedicated FP8 launcher selects precision profiles instead of this marker.
PLATFORM_MARKER_EXEMPTIONS = {"gb200": {"tests/unit_tests/inference/test_batch_invariant_fp8.py"}}

# This bucket is scheduled exclusively on GB200. Keep it out of H100 catch-all
# buckets without changing the additive behavior of launch_on_gb200 elsewhere.
PLATFORM_EXCLUSIONS = {
    "h100": {
        "tests/unit_tests/core/ssm/test_gated_delta_net.py",
        "tests/unit_tests/ssm/test_gated_delta_net.py",
    }
}
BUCKET_MANIFEST = Path("tests/unit_tests/ci_buckets.yaml")
CATCH_ALL_BUCKET = "tests/unit_tests/**/*.py"


def file_has_marker(filepath, marker):
    """Return True if the test file references the given pytest marker.

    Args:
        filepath: Path to a Python test file.
        marker: The pytest marker name to look for (e.g. ``launch_on_gb200``).

    Returns:
        True if the marker name appears anywhere in the file, else False.
    """
    try:
        return marker in Path(filepath).read_text()
    except (OSError, UnicodeDecodeError):
        return False


def get_test_cases(yaml_file: str, tag: str = "latest") -> list[str]:
    """Return recipe buckets for one test-suite version, preserving their order."""
    with open(yaml_file) as stream:
        recipe = yaml.safe_load(stream)
    return list(
        dict.fromkeys(
            test_case
            for product in recipe["products"]
            if not product.get("products")
            or any(tag in params.get("tag", []) for params in product["products"])
            for test_case in product["test_case"]
        )
    )


def get_base_path(pattern):
    if '**' in pattern:
        return pattern.split('/**')[0]
    elif '*' in pattern:
        return pattern.rsplit('/', 1)[0]
    return pattern.rstrip('/')


def is_child_of_bucket(test_case, bucket):
    test_base = get_base_path(test_case)
    bucket_base = get_base_path(bucket)
    return test_base.startswith(bucket_base + '/')


def expand_pattern(pattern):
    if '**' in pattern:
        parts = pattern.split('/**/')
        if len(parts) == 2:
            base_dir, file_pattern = parts
        else:
            # Handle case like 'dir/**'
            base_dir = pattern.split('/**')[0]
            file_pattern = '*.py'
        return [str(f) for f in Path(base_dir).rglob(file_pattern) if f.is_file()]
    elif '*' in pattern:
        base_dir, file_pattern = pattern.rsplit('/', 1)
        return [str(f) for f in Path(base_dir).glob(file_pattern) if f.is_file()]
    elif Path(pattern).is_file():
        return [pattern]
    return []


def _directory_selection(bucket: str, platform: str, root: Path, tag: str) -> tuple[set, set]:
    """Retain the directory-based selection used by historical checkouts."""
    recipe = root / f"tests/test_utils/recipes/{platform}/unit-tests.yaml"
    all_test_cases = get_test_cases(str(recipe), tag)
    bucket_files = set(expand_pattern(str(root / bucket)))

    # Collect files from child test cases to ignore
    files_to_ignore = {str(root / path) for path in PLATFORM_EXCLUSIONS.get(platform, ())}
    for test_case in all_test_cases:
        if test_case != bucket and is_child_of_bucket(test_case, bucket):
            files_to_ignore.update(expand_pattern(str(root / test_case)))

    # On marker-driven platforms, ignore any test file that does not carry the
    # platform marker so only marked tests are launched. Restrict to pytest test
    # files (test_*.py) so conftest.py and helper modules stay collectable.
    marker = PLATFORM_MARKERS.get(platform)
    if marker:
        files_to_ignore.update(
            f
            for f in bucket_files
            if Path(f).name.startswith("test_") and not file_has_marker(f, marker)
        )

    return bucket_files, files_to_ignore & bucket_files


def _manifest_buckets(root: Path, platform: str, tag: str) -> dict[str, list[str]] | None:
    """Read explicit bucket memberships without treating their IDs as paths."""
    manifest = root / BUCKET_MANIFEST
    if tag != "latest" or not manifest.is_file():
        return None
    try:
        document = yaml.safe_load(manifest.read_text())
    except yaml.YAMLError as error:
        raise ValueError(f"invalid unit-test bucket manifest: {error}") from error
    buckets = document.get(platform) if isinstance(document, dict) else None
    if not isinstance(buckets, dict):
        raise ValueError(f"missing unit-test bucket mapping for {platform}")
    for bucket, targets in buckets.items():
        if not isinstance(bucket, str) or not isinstance(targets, list):
            raise ValueError(f"invalid unit-test bucket entry: {bucket!r}")
        for target in targets:
            if not isinstance(target, str):
                raise ValueError(f"invalid unit-test target: {target!r}")
            path = Path(target)
            if (
                path.parts[:2] != ("tests", "unit_tests")
                or ".." in path.parts
                or path.as_posix() != target
                or not path.name.startswith("test_")
                or path.suffix != ".py"
                or any(character in target for character in "\n\r\0*?[")
            ):
                raise ValueError(f"invalid unit-test target: {target!r}")
    return buckets


def resolve_bucket(
    bucket: str, platform: str, root: Path = Path("."), tag: str = "latest"
) -> list[str]:
    """Resolve an ordered bucket to repository-relative test files on one platform."""
    root = root.resolve()
    platform = platform.removeprefix("dgx_")
    buckets = _manifest_buckets(root, platform, tag)
    if buckets is None:
        files, ignored = _directory_selection(bucket, platform, root, tag)
        return sorted(
            str(Path(path).relative_to(root))
            for path in files - ignored
            if Path(path).name.startswith("test_") and Path(path).suffix == ".py"
        )
    if bucket not in buckets:
        raise ValueError(f"unknown unit-test bucket for {platform}: {bucket}")
    targets = list(dict.fromkeys(buckets[bucket]))
    if bucket == CATCH_ALL_BUCKET:
        assigned = {target for name, paths in buckets.items() if name != bucket for target in paths}
        targets = [target for target in targets if target not in assigned]
        assigned.update(targets)
        targets.extend(
            sorted(
                str(path.relative_to(root))
                for path in (root / "tests/unit_tests").rglob("test_*.py")
                if path.is_file() and str(path.relative_to(root)) not in assigned
            )
        )
    excluded = PLATFORM_EXCLUSIONS.get(platform, ())
    marker = (
        None
        if bucket in PLATFORM_MARKER_EXEMPTIONS.get(platform, ())
        else PLATFORM_MARKERS.get(platform)
    )
    selected = []
    for target in targets:
        path = root / target
        if not path.is_file() or not path.resolve().is_relative_to(root / "tests/unit_tests"):
            raise ValueError(f"missing or invalid unit-test target: {target}")
        if target not in excluded and (marker is None or file_has_marker(path, marker)):
            selected.append(target)
    return selected


def main():
    parser = argparse.ArgumentParser(description="Resolve CI unit-test bucket membership.")
    parser.add_argument("bucket")
    parser.add_argument("platform")
    parser.add_argument("--targets", action="store_true")
    args = parser.parse_args()
    tag = os.environ.get("UNIT_TEST_TAG", "latest")
    if args.targets:
        for target in resolve_bucket(args.bucket, args.platform, tag=tag):
            print(target)
    else:
        # Historical callers pass the bucket directory to pytest themselves.
        _, ignored = _directory_selection(args.bucket, args.platform, Path("."), tag)
        for path in sorted(ignored):
            print(f"--ignore={path}")


if __name__ == '__main__':
    main()
