# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bind existing CI log artifacts to one run attempt and verify baseline candidates.

The consumer reads data only. It does not execute downloaded code, schedule GPUs,
set performance budgets, or promote candidates into reviewed historical baselines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from pathlib import Path

import baseline

METADATA = "determinism-ci-artifact.json"
PLATFORMS = ("dgx_h100", "dgx_gb200")
CASES = {
    "tests/unit_tests/determinism/kernels/**/*.py": "coverage",
    "determinism_kernel_perf": "performance",
}
ERRORS = (OSError, AttributeError, IndexError, KeyError, TypeError, ValueError)


def _identity(repository: str, revision: str, run_id: int, attempt: int) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Expected a repository owner/name")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        raise ValueError("Expected a full source revision")
    if any(type(value) is not int or value < 1 for value in (run_id, attempt)):
        raise ValueError("Run and attempt must be positive integers")
    return {
        "repository": repository,
        "revision": revision,
        "run_id": run_id,
        "run_attempt": attempt,
    }


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def stamp(
    directory: Path,
    repository: str,
    revision: str,
    run_id: int,
    attempt: int,
    platform: str,
    test_case: str,
    outcome: str,
    exit_code: int | None,
) -> dict:
    """Record actual producer outcome alongside the existing uploaded log tree."""
    identity = _identity(repository, revision, run_id, attempt)
    if platform not in PLATFORMS or test_case not in CASES:
        raise ValueError("Unsupported determinism producer")
    if outcome not in ("success", "failure", "cancelled", "skipped"):
        raise ValueError("Invalid producer outcome")
    name = f"logs-determinism-{run_id}-a{attempt}-{platform}-{CASES[test_case]}-{uuid.uuid4()}"
    record = {
        "schema_version": 1,
        "kind": "determinism_ci_artifact",
        **identity,
        "platform": platform,
        "test_case": test_case,
        "producer_outcome": outcome,
        "producer_exit_code": exit_code,
        "artifact_name": name,
    }
    _write(directory / METADATA, record)
    return record


def _input(directory: Path, expected: dict) -> dict:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Expected a separate downloaded artifact directory")
    if any(path.is_symlink() for path in directory.rglob("*")):
        raise ValueError("Artifact contains a symlink")
    raw = (directory / METADATA).read_bytes()
    record = json.loads(raw)
    if record.get("schema_version") != 1 or record.get("kind") != "determinism_ci_artifact":
        raise ValueError("Missing or unsupported producer provenance")
    different = [
        key
        for key, value in expected.items()
        if type(record.get(key)) is not type(value) or record[key] != value
    ]
    if different:
        raise ValueError("Producer identity differs: " + ", ".join(different))
    platform, test_case = record["platform"], record["test_case"]
    if platform not in PLATFORMS or test_case not in CASES:
        raise ValueError("Unsupported producer platform or test case")
    prefix = f"logs-determinism-{expected['run_id']}-a{expected['run_attempt']}-{platform}-{CASES[test_case]}-"
    name = record["artifact_name"]
    if not isinstance(name, str) or not name.startswith(prefix):
        raise ValueError("Artifact name differs from producer identity")
    if str(uuid.UUID(name.removeprefix(prefix))) != name.removeprefix(prefix):
        raise ValueError("Invalid producer identifier")
    if directory.name not in (name, name + "-retry"):
        raise ValueError("Download must preserve the original artifact name")
    return {
        "artifact": directory.name,
        "metadata_sha256": hashlib.sha256(raw).hexdigest(),
        "provenance": record,
    }


def _one(directory: Path, filename: str) -> Path:
    found = list(directory.rglob(filename))
    if len(found) != 1 or not found[0].is_file():
        raise ValueError(f"Expected exactly one {filename}; missing or ambiguous attempts")
    return found[0]


def _platform(leaderboard: Path, platform: str) -> None:
    name, capability = {"dgx_h100": ("H100", [9, 0]), "dgx_gb200": ("GB200", [10, 0])}[platform]
    reports = json.loads(leaderboard.read_bytes())
    for report in reports:
        for run in report["runs"]:
            runtime = run["kernel"]["case_signature"]["runtime"]
            if name not in runtime["gpu"] or runtime["capability"] != capability:
                raise ValueError("Observed GPU does not match the selected CI platform")


def collect(
    artifacts: Path,
    output: Path,
    repository: str,
    revision: str,
    run_id: int,
    attempt: int,
    platforms: list[str],
) -> dict:
    """Verify every selected platform, retaining failures and source provenance."""
    expected = _identity(repository, revision, run_id, attempt)
    if (
        not platforms
        or len(set(platforms)) != len(platforms)
        or not set(platforms) <= set(PLATFORMS)
    ):
        raise ValueError("Select each expected platform exactly once")
    if output.resolve().is_relative_to(artifacts.resolve()):
        raise ValueError("Output must be outside downloaded artifacts")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory; preserve earlier verification attempts")
    output.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "schema_version": 1,
        "kind": "determinism_ci_baselines",
        **expected,
        "status": "not_verified",
        "inputs": [],
        "ignored_inputs": [],
        "errors": [],
        "platforms": [],
    }
    sources: dict[tuple[str, str], list[Path]] = {}
    for directory in sorted(artifacts.iterdir()) if artifacts.is_dir() else []:
        try:
            source = _input(directory, expected)
            record = source["provenance"]
            if record["platform"] not in platforms:
                report["ignored_inputs"].append(
                    {**source, "reason": "Platform was not selected for performance"}
                )
                continue
            if (
                record["producer_outcome"] != "success"
                or type(record["producer_exit_code"]) is not int
                or record["producer_exit_code"] != 0
            ):
                raise ValueError("Producer did not complete successfully")
            report["inputs"].append(source)
            sources.setdefault((record["platform"], CASES[record["test_case"]]), []).append(
                directory
            )
        except ERRORS as error:
            report["errors"].append({"artifact": directory.name, "reason": str(error)})
    origin = f"https://github.com/{repository}/actions/runs/{run_id}/attempts/{attempt}"
    for platform in platforms:
        row: dict = {"platform": platform, "status": "not_verified"}
        report["platforms"].append(row)
        try:
            if report["errors"]:
                raise ValueError("Input artifact validation failed; see errors")
            candidates = [sources.get((platform, kind), []) for kind in ("coverage", "performance")]
            if any(len(paths) != 1 for paths in candidates):
                raise ValueError(
                    "Expected one successful replay artifact and one timing artifact; missing or ambiguous uploads"
                )
            covered, timed = (paths[0] for paths in candidates)
            leaderboard = _one(timed, "leaderboard.json")
            _platform(leaderboard, platform)
            result = baseline.publish(
                _one(covered, "determinism-coverage.json"),
                leaderboard,
                output / "baselines" / platform,
                revision,
                origin,
            )
            # Save only relative paths so the derived artifact can itself move.
            result["path"] = Path(result["path"]).relative_to(output).as_posix()
            row.update(result, coverage_artifact=covered.name, performance_artifact=timed.name)
        except ERRORS as error:
            row["reason"] = str(error)
    if not report["errors"] and all(
        row["status"] in ("passed", "not_gated") for row in report["platforms"]
    ):
        report["status"] = "complete"
    _write(output / "report.json", report)
    (output / "report.md").write_text(markdown_report(report))
    return report


def markdown_report(report: dict) -> str:
    """Keep artifact completeness distinct from passing performance budgets."""

    def cell(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    lines = [
        "# Determinism CI artifact verification",
        "",
        f"Artifact verification: **{report['status']}**. Unbudgeted performance remains **not_gated**.",
        "",
        "| Platform | Status | Baseline ID |",
        "| --- | --- | --- |",
    ]
    for row in report["platforms"]:
        lines.append(
            f"| {row['platform']} | {row['status']} | {row.get('baseline_id', 'unavailable')} |"
        )
    reasons = [
        f"{row['platform']}: {row['reason']}" for row in report["platforms"] if "reason" in row
    ]
    reasons.extend(f"{row['artifact']}: {row['reason']}" for row in report["errors"])
    if reasons:
        lines.append("")
        lines.extend(f"- {cell(reason)}" for reason in reasons)
    lines.extend(
        [
            "",
            "See report.json for input provenance, hashes and rejection reasons.",
            "These are verified candidates under CI artifact retention, not reviewed baseline promotion.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Stamp producer logs or consume separately downloaded artifacts on a CPU runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("stamp", "collect"))
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", action="append", choices=PLATFORMS, required=True)
    parser.add_argument("--test-case", choices=CASES)
    parser.add_argument("--outcome", choices=("success", "failure", "cancelled", "skipped"))
    parser.add_argument("--exit-code", default="")
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "stamp":
            if len(args.platform) != 1:
                raise ValueError("A producer has exactly one platform")
            record = stamp(
                args.output,
                args.repository,
                args.revision,
                args.run_id,
                args.attempt,
                args.platform[0],
                args.test_case,
                args.outcome,
                int(args.exit_code) if args.exit_code else None,
            )
            print(record["artifact_name"])
            return 0
        if args.artifacts is None:
            raise ValueError("Downloaded artifacts are required")
        report = collect(
            args.artifacts,
            args.output,
            args.repository,
            args.revision,
            args.run_id,
            args.attempt,
            args.platform,
        )
        print(markdown_report(report))
        return 0 if report["status"] == "complete" else 1
    except ERRORS as error:
        print(json.dumps({"status": "not_verified", "error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
