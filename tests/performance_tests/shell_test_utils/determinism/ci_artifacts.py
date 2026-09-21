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
    "determinism_collective_perf": "collective",
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


def _tree_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(b"d" if path.is_dir() else b"f")
        if path.is_file():
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _one(directory: Path, filename: str) -> Path:
    found = list(directory.rglob(filename))
    if len(found) != 1 or not found[0].is_file():
        raise ValueError(f"Expected exactly one {filename}; missing or ambiguous attempts")
    return found[0]


def _platform(leaderboard: Path, platform: str) -> None:
    reports = json.loads(leaderboard.read_bytes())
    for report in reports:
        for run in report["runs"]:
            runtime = run["kernel"]["case_signature"]["runtime"]
            _context_platform(runtime, platform)


def _context_platform(context: dict, platform: str) -> None:
    name, capability = {"dgx_h100": ("H100", [9, 0]), "dgx_gb200": ("GB200", [10, 0])}[platform]
    if name not in context["gpu"] or context["capability"] != capability:
        raise ValueError("Observed GPU does not match the selected CI platform")


def _collective(directory: Path, platform: str, store: Path, revision: str, origin: str) -> dict:
    import collective_baseline

    timed = _one(directory, "benchmark.json")
    root = timed.parent.parent
    covered = _one(directory, "coverage.json")
    if timed.parent.name != "timing" or covered != root / "coverage.json":
        raise ValueError("Expected one capture/coverage/timing dataset in the collective artifact")
    report = json.loads(timed.read_bytes())
    _context_platform(report["capture"]["context"], platform)
    return collective_baseline.publish(root / "capture", covered, timed, store, revision, origin)


def collect(
    artifacts: Path,
    output: Path,
    repository: str,
    revision: str,
    run_id: int,
    attempt: int,
    platforms: list[str],
    collective_platforms: list[str] | None = None,
) -> dict:
    """Verify every selected platform, retaining failures and source provenance."""
    expected = _identity(repository, revision, run_id, attempt)
    collective_platforms = collective_platforms or []
    if not (platforms or collective_platforms) or any(
        len(set(selected)) != len(selected) or not set(selected) <= set(PLATFORMS)
        for selected in (platforms, collective_platforms)
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
    producers: dict[str, dict] = {}
    for directory in sorted(artifacts.iterdir()) if artifacts.is_dir() else []:
        try:
            source = _input(directory, expected)
            record = source["provenance"]
            selected = (
                collective_platforms if CASES[record["test_case"]] == "collective" else platforms
            )
            if record["platform"] not in selected:
                report["ignored_inputs"].append(
                    {**source, "reason": "Platform was not selected for this performance producer"}
                )
                continue
            if (
                record["producer_outcome"] != "success"
                or type(record["producer_exit_code"]) is not int
                or record["producer_exit_code"] != 0
            ):
                raise ValueError("Producer did not complete successfully")
            source["tree_sha256"] = _tree_hash(directory)
            previous = producers.get(record["artifact_name"])
            if previous is not None:
                if source["tree_sha256"] != previous["tree_sha256"]:
                    raise ValueError("Retry upload differs from the original producer tree")
                report["ignored_inputs"].append(
                    {
                        **source,
                        "reason": "Byte-identical retry upload",
                        "duplicate_of": previous["artifact"],
                    }
                )
                continue
            producers[record["artifact_name"]] = source
            report["inputs"].append(source)
            sources.setdefault((record["platform"], CASES[record["test_case"]]), []).append(
                directory
            )
        except ERRORS as error:
            report["errors"].append({"artifact": directory.name, "reason": str(error)})
    origin = f"https://github.com/{repository}/actions/runs/{run_id}/attempts/{attempt}"
    selection = [("activation", platform) for platform in platforms] + [
        ("collective", platform) for platform in collective_platforms
    ]
    for kind, platform in selection:
        row: dict = {"platform": platform, "kind": kind, "status": "not_verified"}
        report["platforms"].append(row)
        try:
            if report["errors"]:
                raise ValueError("Input artifact validation failed; see errors")
            if kind == "collective":
                collective_candidates = sources.get((platform, "collective"), [])
                if len(collective_candidates) != 1:
                    raise ValueError(
                        "Expected one successful collective artifact; missing or ambiguous uploads"
                    )
                result = _collective(
                    collective_candidates[0],
                    platform,
                    output / "collective-baselines" / platform,
                    revision,
                    origin,
                )
                row["collective_artifact"] = collective_candidates[0].name
            else:
                candidates = [
                    sources.get((platform, name), []) for name in ("coverage", "performance")
                ]
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
                row.update(coverage_artifact=covered.name, performance_artifact=timed.name)
            # Save only relative paths so the derived artifact can itself move.
            result["path"] = Path(result["path"]).relative_to(output).as_posix()
            row.update(result)
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
        "| Platform | Evidence | Status | Baseline ID |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["platforms"]:
        lines.append(
            f"| {row['platform']} | {row.get('kind', 'activation')} | {row['status']} | {row.get('baseline_id', 'unavailable')} |"
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
    parser.add_argument("--platform", action="append", choices=PLATFORMS, default=[])
    parser.add_argument("--collective-platform", action="append", choices=PLATFORMS, default=[])
    parser.add_argument("--test-case", choices=CASES)
    parser.add_argument("--outcome", choices=("success", "failure", "cancelled", "skipped"))
    parser.add_argument("--exit-code", default="")
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "stamp":
            if len(args.platform) != 1 or args.collective_platform:
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
            args.collective_platform,
        )
        print(markdown_report(report))
        return 0 if report["status"] == "complete" else 1
    except ERRORS as error:
        print(json.dumps({"status": "not_verified", "error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
