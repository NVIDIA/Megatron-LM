#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Skip GPU unit tests only for a verified diff of functional assets or launch scripts."""

import argparse
import os
import re
import subprocess
import sys


def is_exempt_path(path: str) -> bool:
    """Return whether a repository-relative path is independent of GPU unit tests."""
    parts = path.split("/")
    if len(parts) == 6 and parts[:3] == ["tests", "functional_tests", "test_cases"]:
        return parts[-1] == "model_config.yaml" or (
            parts[-1].startswith("golden_values") and parts[-1].endswith(".json")
        )
    if len(parts) == 5 and parts[:3] == ["tests", "test_utils", "recipes"]:
        return (
            parts[3] in {"h100", "gb200"}
            and parts[-1].endswith(".yaml")
            and parts[-1] != "unit-tests.yaml"
        )
    return len(parts) >= 2 and parts[0] == "examples" and parts[-1].endswith(".sh")


def should_skip_unit_tests(base: str, head: str) -> bool:
    """Check the complete base-to-head diff, retaining unit tests on uncertainty."""
    if not all(re.fullmatch(r"[0-9a-fA-F]{40}", ref) for ref in (base, head)):
        print("Unit tests retained: missing or invalid base/head SHA.", file=sys.stderr)
        return False

    try:
        # The workflow supplies the merge base and the tested head. Reject stale or
        # unrelated metadata instead of considering a partial or misleading diff.
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", base, head],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        result = subprocess.run(
            ["git", "diff", "--name-only", "--no-renames", "-z", base, head, "--"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        print(
            f"Unit tests retained: unable to verify the changed paths ({error}).", file=sys.stderr
        )
        return False

    # Disabling rename detection includes both the deleted source and added target.
    # NUL delimiters preserve paths containing whitespace or newlines.
    paths = [os.fsdecode(path) for path in result.stdout.split(b"\0") if path]
    skip = bool(paths) and all(is_exempt_path(path) for path in paths)
    print(
        f"Unit tests {'skipped' if skip else 'retained'}: checked {len(paths)} changed paths.",
        file=sys.stderr,
    )
    return skip


def main() -> None:
    """Write one GitHub Actions output; missing refs default to running tests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="")
    parser.add_argument("--head", default="")
    args = parser.parse_args()
    skip = should_skip_unit_tests(args.base, args.head)
    print(f"skip_unit_tests={str(skip).lower()}")


if __name__ == "__main__":
    main()
