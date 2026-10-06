# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Select changed functional test cases from the configure job's validated PR artifact."""

import argparse
import json
import re
import sys
from pathlib import Path

PR_FILE_STATUSES = {"added", "removed", "modified", "renamed", "copied", "changed", "unchanged"}
SELECTED_STATUSES = {"added", "modified", "copied", "renamed"}
TEST_CASE_PREFIX = ["tests", "functional_tests", "test_cases"]


def load_changed_test_cases(artifact_dir: Path, tested_sha: str) -> list[str]:
    """Validate the complete PR artifact and return sorted, unique ACMR test case names.

    An empty list is valid only after all three artifact files have been validated.
    Missing or invalid artifacts raise OSError or ValueError so callers can use a
    conservative fallback instead of silently omitting changed tests.
    """
    if re.fullmatch(r"[0-9a-f]{40}", tested_sha) is None:
        raise ValueError("tested SHA must be a full lowercase Git commit SHA")

    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("tested_sha") != tested_sha:
        raise ValueError("metadata.json does not match the tested SHA")
    count = metadata.get("changed_files")
    if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= 3000:
        raise ValueError("metadata.json changed_files must be an integer between 0 and 3000")

    # Read bytes so universal newline conversion cannot conceal carriage returns.
    text = (artifact_dir / "changed-files").read_bytes().decode("utf-8")
    if text and not text.endswith("\n"):
        raise ValueError("changed-files must contain complete newline-terminated records")
    filenames = text.split("\n")[:-1] if text else []
    if len(filenames) != count:
        raise ValueError("changed-files count does not match metadata.json")
    if len(set(filenames)) != count:
        raise ValueError("changed-files contains duplicate paths")

    entries = json.loads((artifact_dir / "files.json").read_text(encoding="utf-8"))
    if not isinstance(entries, list) or len(entries) != count:
        raise ValueError("files.json must be an array matching the changed-files count")

    cases = set()
    for filename, entry in zip(filenames, entries):
        parts = filename.split("/")
        if any(part in {"", ".", ".."} for part in parts) or any(
            character in filename for character in "\n\r\0"
        ):
            raise ValueError(f"invalid relative canonical path in changed-files: {filename!r}")
        if not isinstance(entry, dict) or entry.get("filename") != filename:
            raise ValueError("files.json filenames must match changed-files in the same order")
        status = entry.get("status")
        if not isinstance(status, str) or status not in PR_FILE_STATUSES:
            raise ValueError(f"unsupported file status in files.json for {filename!r}: {status!r}")
        if status in SELECTED_STATUSES and len(parts) >= 6 and parts[:3] == TEST_CASE_PREFIX:
            case = parts[4]
            if "," in case:
                raise ValueError(f"functional test case name cannot contain a comma: {case!r}")
            cases.add(case)

    return sorted(cases)


def main(argv: list[str] | None = None) -> int:
    """Print the comma-separated cases, or return 2 when the artifact is unusable."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--tested-sha", required=True)
    args = parser.parse_args(argv)
    try:
        cases = load_changed_test_cases(args.artifact_dir, args.tested_sha)
    except (OSError, ValueError) as error:
        print(f"Changed functional tests: {error}", file=sys.stderr)
        return 2
    print(",".join(cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
