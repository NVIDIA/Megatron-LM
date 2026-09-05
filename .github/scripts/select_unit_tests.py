#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Select impacted unit-test files and map them to the existing CI buckets.

``pytest-impacted`` supplies the dependency analysis. This wrapper owns the
fail-closed policy required by Megatron-LM CI: anything ambiguous produces the
full bucket matrix instead of an empty or incomplete test run.
"""

from __future__ import annotations

import argparse
import base64
import fnmatch
import glob
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

UNIT_TEST_ROOT = Path("tests/unit_tests")
SOURCE_ROOT = Path("megatron")
SELECTOR_TIMEOUT_SECONDS = 300
SAFE_DOCUMENTATION_SUFFIXES = {".gif", ".jpeg", ".jpg", ".md", ".png", ".rst", ".svg"}
HIGH_IMPACT_PATTERNS = (
    ".github/**",
    ".gitlab/**",
    ".coveragerc",
    "docker/**",
    "megatron/__init__.py",
    "megatron/**/__init__.py",
    "pyproject.toml",
    "pytest.ini",
    "requirements*.txt",
    "requirements/**",
    "setup.cfg",
    "setup.py",
    "tests/test_utils/python_scripts/launch_nemo_run_workload.py",
    "tests/test_utils/python_scripts/recipe_parser.py",
    "tests/test_utils/recipes/**/unit-tests.yaml",
    "tests/unit_tests/__init__.py",
    "tests/unit_tests/**/__init__.py",
    "tests/unit_tests/conftest.py",
    "tests/unit_tests/**/conftest.py",
    "tests/unit_tests/find_test_cases.py",
    "tests/unit_tests/run_ci_test.sh",
    "tests/unit_tests/selective_test_guard.py",
    "tox.ini",
    "uv.lock",
)
UNSAFE_SELECTOR_OUTPUT = re.compile(
    r"(?:\bERROR\b|Traceback|could not be resolved to a known module|"
    r"Syntax error while parsing|not found in discovered submodules)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Change:
    """One path and status reported by git diff."""

    status: str
    path: str
    old_path: str | None = None

    @property
    def paths(self) -> tuple[str, ...]:
        """Return both paths for a rename/copy and the current path otherwise."""

        if self.old_path is None:
            return (self.path,)
        return (self.old_path, self.path)


@dataclass(frozen=True)
class CommandResult:
    """Small subprocess result interface used by the selector and its tests."""

    returncode: int
    stdout: str
    stderr: str


SelectorRunner = Callable[[Sequence[str], Path], CommandResult]


def _run_command(command: Sequence[str], cwd: Path) -> CommandResult:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=SELECTOR_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return CommandResult(
            124, "", f"pytest-impacted timed out after {SELECTOR_TIMEOUT_SECONDS} seconds"
        )
    return CommandResult(result.returncode, result.stdout, result.stderr)


def _validate_git_ref(ref: str) -> None:
    if re.fullmatch(r"[0-9a-fA-F]{7,64}", ref or "") is None:
        raise ValueError(f"base ref must be an exact commit SHA: {ref!r}")


def _parse_name_status(output: str) -> list[Change]:
    changes: list[Change] = []
    for line in output.splitlines():
        if not line:
            continue
        fields = line.split("\t")
        status = fields[0]
        kind = status[:1]
        if kind in {"C", "R"} and len(fields) == 3:
            changes.append(Change(status=kind, old_path=fields[1], path=fields[2]))
        elif len(fields) == 2:
            changes.append(Change(status=kind, path=fields[1]))
        else:
            raise ValueError(f"unexpected git diff entry: {line!r}")
    return changes


def find_changes(repo_root: Path, git_mode: str, base_ref: str | None) -> list[Change]:
    """Return changes using the same committed/unstaged modes as pytest-impacted."""

    if git_mode == "branch":
        if base_ref is None:
            raise ValueError("--base-ref is required in branch mode")
        _validate_git_ref(base_ref)
        subprocess.run(
            ["git", "rev-parse", "--verify", f"{base_ref}^{{commit}}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", base_ref, "HEAD"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if ancestor.returncode != 0:
            raise ValueError(f"base commit {base_ref} is not an ancestor of HEAD")
        result = subprocess.run(
            [
                "git",
                "diff",
                "--name-status",
                "--find-renames",
                "--find-copies",
                base_ref,
                "HEAD",
                "--",
            ],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return _parse_name_status(result.stdout)

    if git_mode != "unstaged":
        raise ValueError(f"unsupported git mode: {git_mode}")

    unstaged = subprocess.run(
        ["git", "diff", "--name-status", "--find-renames", "--find-copies", "--"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    changes = _parse_name_status(unstaged.stdout)
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    changes.extend(Change(status="A", path=path) for path in untracked.stdout.splitlines() if path)

    # pytest-impacted's unstaged mode does not inspect the index. A staged-only
    # change therefore cannot be selected safely and is represented as unknown.
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    if staged.stdout.strip():
        changes.append(Change(status="S", path="<staged changes>"))
    return changes


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _is_documentation_path(path: str) -> bool:
    candidate = Path(path)
    return candidate.suffix.lower() in SAFE_DOCUMENTATION_SUFFIXES or candidate.name in {
        "LICENSE",
        "NOTICE",
    }


def _matches_high_impact_path(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in HIGH_IMPACT_PATTERNS)


def _source_module_is_discoverable(repo_root: Path, relative_path: Path) -> bool:
    """Mirror pytest-impacted's package discovery constraint conservatively.

    ``megatron`` itself is a namespace package, but nested directories without
    ``__init__.py`` are omitted by the tool's production-module discovery.
    """

    parent = relative_path.parent
    while parent != SOURCE_ROOT:
        if not (repo_root / parent / "__init__.py").is_file():
            return False
        parent = parent.parent
        if parent == Path("."):
            return False
    return True


def full_run_reason(repo_root: Path, changes: Sequence[Change]) -> str | None:
    """Return a reason to force the full suite, or ``None`` when analyzable."""

    if not changes:
        return "no changed files were detected"

    for change in changes:
        if change.status not in {"A", "M"}:
            return f"git status {change.status!r} is not safe for selective analysis"

        for path_string in change.paths:
            if _matches_high_impact_path(path_string):
                return f"high-impact file changed: {path_string}"
            if _is_documentation_path(path_string):
                continue

            path = Path(path_string)
            if path.suffix != ".py":
                return f"unsupported non-Python file changed: {path_string}"
            if _is_relative_to(path, UNIT_TEST_ROOT):
                continue
            if _is_relative_to(path, SOURCE_ROOT):
                if not _source_module_is_discoverable(repo_root, path):
                    return f"pytest-impacted cannot discover namespace path: {path_string}"
                continue
            return f"Python file outside the analyzed package changed: {path_string}"
    return None


def get_base_path(pattern: str) -> str:
    """Return the non-glob parent used to compare nested recipe buckets."""

    if "**" in pattern:
        return pattern.split("/**", 1)[0]
    if "*" in pattern:
        return pattern.rsplit("/", 1)[0]
    return pattern.rstrip("/")


def is_child_bucket(test_case: str, bucket: str) -> bool:
    """Return whether a recipe entry is more specific than another bucket."""

    test_base = get_base_path(test_case)
    bucket_base = get_base_path(bucket)
    return test_base.startswith(bucket_base + "/")


def read_buckets(path: Path) -> list[str]:
    """Read and validate a JSON list of H100 unit-test recipe buckets."""

    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("bucket file must contain a JSON list")
    buckets: list[str] = []
    for entry in data:
        bucket = entry.get("bucket") if isinstance(entry, dict) else entry
        if (
            not isinstance(bucket, str)
            or re.fullmatch(r"tests/unit_tests/[A-Za-z0-9_./*+-]+", bucket) is None
            or ".." in Path(bucket).parts
        ):
            raise ValueError(f"invalid unit-test bucket: {entry!r}")
        if bucket in buckets:
            raise ValueError(f"duplicate unit-test bucket: {bucket}")
        buckets.append(bucket)
    if not buckets:
        raise ValueError("bucket file is empty")
    return buckets


def _expand_bucket(repo_root: Path, pattern: str) -> set[str]:
    absolute_pattern = str(repo_root / pattern)
    files: set[str] = set()
    for match in glob.glob(absolute_pattern, recursive=True):
        path = Path(match)
        if path.is_file() and path.name.startswith("test_") and path.suffix == ".py":
            files.add(path.resolve().relative_to(repo_root).as_posix())
    return files


def build_bucket_ownership(repo_root: Path, buckets: Sequence[str]) -> dict[str, set[str]]:
    """Build the same most-specific-bucket partition as find_test_cases.py."""

    expanded = {bucket: _expand_bucket(repo_root, bucket) for bucket in buckets}
    owned: dict[str, set[str]] = {}
    for bucket in buckets:
        child_files: set[str] = set()
        for candidate in buckets:
            if candidate != bucket and is_child_bucket(candidate, bucket):
                child_files.update(expanded[candidate])
        owned[bucket] = expanded[bucket] - child_files

    empty_buckets = sorted(bucket for bucket, files in owned.items() if not files)
    if empty_buckets:
        raise ValueError(f"unit-test recipe buckets do not own any tests: {empty_buckets}")

    owners: dict[str, list[str]] = {}
    for bucket, files in owned.items():
        for file_path in files:
            owners.setdefault(file_path, []).append(bucket)
    duplicates = {path: values for path, values in owners.items() if len(values) != 1}
    if duplicates:
        raise ValueError(f"unit-test files have ambiguous bucket ownership: {duplicates}")

    expected = {
        path.resolve().relative_to(repo_root).as_posix()
        for path in (repo_root / UNIT_TEST_ROOT).rglob("test_*.py")
        if path.is_file()
    }
    if not expected:
        raise ValueError("no unit-test files were found")
    missing = sorted(expected - owners.keys())
    if missing:
        raise ValueError(f"unit-test files are not owned by a recipe bucket: {missing}")
    return owned


def _normalize_selector_output(repo_root: Path, stdout: str) -> tuple[set[str], str | None]:
    selected: set[str] = set()
    unit_root = (repo_root / UNIT_TEST_ROOT).resolve()
    for raw_line in stdout.splitlines():
        raw_path = raw_line.strip()
        if not raw_path:
            continue
        candidate = Path(raw_path)
        resolved = (candidate if candidate.is_absolute() else repo_root / candidate).resolve()
        if not _is_relative_to(resolved, repo_root):
            return set(), f"selector returned a path outside the repository: {raw_path}"
        if not _is_relative_to(resolved, unit_root):
            # pytest-impacted also classifies source files ending in ``_test.py``
            # as tests. Those are valid tool output but outside this CI suite.
            continue
        relative = resolved.relative_to(repo_root).as_posix()
        if resolved.name == "conftest.py":
            return set(), f"pytest-impacted reported an impacted shared fixture: {relative}"
        if not resolved.is_file():
            return set(), f"selector returned a missing test file: {relative}"
        if not resolved.name.startswith("test_") or resolved.suffix != ".py":
            continue
        selected.add(relative)
    return selected, None


def _directly_changed_tests(changes: Sequence[Change]) -> set[str]:
    return {
        path
        for change in changes
        for path in change.paths
        if change.status in {"A", "M"}
        and _is_relative_to(Path(path), UNIT_TEST_ROOT)
        and Path(path).name.startswith("test_")
        and Path(path).suffix == ".py"
    }


def run_pytest_impacted(
    repo_root: Path, git_mode: str, base_ref: str | None, runner: SelectorRunner = _run_command
) -> CommandResult:
    """Invoke pytest-impacted with the repository's package and test roots."""

    command = [
        "impacted-tests",
        "--module",
        "megatron",
        "--tests-dir",
        "tests",
        "--root-dir",
        str(repo_root),
        "--git-mode",
        git_mode,
    ]
    if git_mode == "branch":
        if base_ref is None:
            raise ValueError("base ref is required in branch mode")
        command.extend(["--base-branch", base_ref])
    return runner(command, repo_root)


def _encode_test_files(test_files: Sequence[str]) -> str:
    payload = json.dumps(list(test_files), separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(payload).decode()


def _full_report(
    reason: str, buckets: Sequence[str], ownership: dict[str, set[str]], changes: Sequence[Change]
) -> dict[str, object]:
    return {
        "mode": "full",
        "reason": reason,
        "changed_files": sorted({path for change in changes for path in change.paths}),
        "selected_files": [],
        "selected_count": sum(len(files) for files in ownership.values()),
        "total_count": sum(len(files) for files in ownership.values()),
        "matrix": [{"bucket": bucket, "unit_test_files": ""} for bucket in buckets],
    }


def select_unit_tests(
    repo_root: Path,
    buckets: Sequence[str],
    changes: Sequence[Change],
    git_mode: str,
    base_ref: str | None,
    force_full: str | None = None,
    runner: SelectorRunner = _run_command,
) -> dict[str, object]:
    """Return a deterministic selective/full report for CI and local use."""

    ownership = build_bucket_ownership(repo_root, buckets)
    if force_full:
        return _full_report(force_full, buckets, ownership, changes)

    policy_reason = full_run_reason(repo_root, changes)
    if policy_reason:
        return _full_report(policy_reason, buckets, ownership, changes)

    result = run_pytest_impacted(repo_root, git_mode, base_ref, runner=runner)
    if result.returncode != 0:
        reason = f"pytest-impacted failed with exit code {result.returncode}"
        return _full_report(reason, buckets, ownership, changes)
    normalized_stderr = " ".join(result.stderr.split())
    if UNSAFE_SELECTOR_OUTPUT.search(normalized_stderr):
        return _full_report(
            "pytest-impacted reported an analysis error", buckets, ownership, changes
        )

    selected, output_error = _normalize_selector_output(repo_root, result.stdout)
    if output_error:
        return _full_report(output_error, buckets, ownership, changes)
    selected.update(_directly_changed_tests(changes))

    all_owned_files = set().union(*ownership.values())
    unknown = sorted(selected - all_owned_files)
    if unknown:
        return _full_report(
            f"selected tests are not owned by the H100 recipe: {', '.join(unknown)}",
            buckets,
            ownership,
            changes,
        )
    if not selected:
        return _full_report(
            "pytest-impacted returned no unit tests for an analyzable change",
            buckets,
            ownership,
            changes,
        )

    matrix = []
    for bucket in buckets:
        bucket_files = sorted(selected & ownership[bucket])
        if bucket_files:
            matrix.append({"bucket": bucket, "unit_test_files": _encode_test_files(bucket_files)})
    if not matrix:
        return _full_report("no selected test mapped to a CI bucket", buckets, ownership, changes)

    return {
        "mode": "selective",
        "reason": f"pytest-impacted selected {len(selected)} unit-test file(s)",
        "changed_files": sorted({path for change in changes for path in change.paths}),
        "selected_files": sorted(selected),
        "selected_count": len(selected),
        "total_count": len(all_owned_files),
        "matrix": matrix,
    }


def write_summary(report: dict[str, object], destination: Path) -> None:
    """Write a human-readable GitHub step summary for a selection report."""

    mode = str(report["mode"])
    selected_count = int(report["selected_count"])
    total_count = int(report["total_count"])
    matrix = report["matrix"]
    assert isinstance(matrix, list)
    duration = report.get("selector_duration_seconds", "not recorded")
    lines = [
        "## Unit-test selection",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| Mode | `{mode}` |",
        f"| Reason | {report['reason']} |",
        f"| Test files | `{selected_count}` / `{total_count}` |",
        f"| H100 buckets | `{len(matrix)}` |",
        f"| Selector overhead | `{duration}` seconds |",
        "| Persistent dependency cache | Not required (pytest-impacted is stateless across runs) |",
    ]
    selected_files = report.get("selected_files")
    if isinstance(selected_files, list) and selected_files:
        lines.extend(["", "<details><summary>Selected test files</summary>", "", "```text"])
        lines.extend(str(path) for path in selected_files)
        lines.extend(["```", "", "</details>"])
    destination.write_text("\n".join(lines) + "\n")


def run_local_tests(repo_root: Path, report: dict[str, object], nproc_per_node: int) -> int:
    """Run either the selected files or the complete unit-test directory."""

    selected_files = report.get("selected_files")
    if report["mode"] == "selective" and isinstance(selected_files, list):
        targets = [str(path) for path in selected_files]
    else:
        targets = [UNIT_TEST_ROOT.as_posix()]

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
        str(nproc_per_node),
        "-m",
        "pytest",
        "-q",
        *targets,
    ]
    sys.stdout.write(f"Selection mode: {report['mode']} ({report['reason']})\n")
    sys.stdout.write(
        f"Running {report['selected_count']} of {report['total_count']} unit-test files\n"
    )
    return subprocess.run(command, cwd=repo_root, check=False).returncode


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--buckets-file", type=Path, required=True)
    parser.add_argument("--git-mode", choices=("branch", "unstaged"), default="branch")
    parser.add_argument("--base-ref")
    parser.add_argument("--force-full", metavar="REASON")
    parser.add_argument("--output", type=Path, default=Path("unit-test-selection.json"))
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--run", action="store_true", help="Run the selected tests locally")
    parser.add_argument("--nproc-per-node", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    """Select tests, persist the report, and optionally launch them."""

    args = parse_args()
    repo_root = args.repo_root.resolve()
    buckets_file = args.buckets_file
    if not buckets_file.is_absolute():
        buckets_file = repo_root / buckets_file
    buckets = read_buckets(buckets_file)

    started_at = time.monotonic()
    try:
        # Full-suite CI triggers do not necessarily have a PR base SHA. Avoid
        # consulting git in that case; bucket validation is still performed by
        # ``select_unit_tests`` before the full matrix is emitted.
        changes = [] if args.force_full else find_changes(repo_root, args.git_mode, args.base_ref)
        report = select_unit_tests(
            repo_root=repo_root,
            buckets=buckets,
            changes=changes,
            git_mode=args.git_mode,
            base_ref=args.base_ref,
            force_full=args.force_full,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        ownership = build_bucket_ownership(repo_root, buckets)
        report = _full_report(f"selector setup failed: {error}", buckets, ownership, [])
    report["selector_duration_seconds"] = round(time.monotonic() - started_at, 3)

    output = args.output
    if not output.is_absolute():
        output = repo_root / output
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    sys.stderr.write(f"Unit-test selection mode: {report['mode']}\n")
    sys.stderr.write(f"Reason: {report['reason']}\n")
    sys.stderr.write(
        f"Selected {report['selected_count']} of {report['total_count']} test files "
        f"across {len(report['matrix'])} bucket(s)\n"
    )

    if args.summary:
        summary = args.summary
        if not summary.is_absolute():
            summary = repo_root / summary
        write_summary(report, summary)
    if args.run:
        return run_local_tests(repo_root, report, args.nproc_per_node)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
