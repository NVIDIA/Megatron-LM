#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Report actual H100 unit-job timings alongside the persisted selection plan.

Inputs are the selector's JSON manifest and ``gh run view --json jobs`` output
for the same run attempt. This command performs no network requests.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

FAILURE_CONCLUSIONS = {"failure", "timed_out", "startup_failure", "action_required"}
CSV_FIELDS = (
    "schema_version",
    "repository",
    "run_id",
    "run_attempt",
    "sha",
    "event_name",
    "pr_number",
    "run_url",
    "selective_label_present",
    "mode",
    "selection_reason",
    "selected_file_count",
    "total_file_count",
    "selected_bucket_count",
    "total_bucket_count",
    "selector_duration_seconds",
    "planning_duration_seconds",
    "expected_job_count",
    "observed_job_count",
    "timed_job_count",
    "missing_job_count",
    "duplicate_job_count",
    "successful_job_count",
    "failed_job_count",
    "cancelled_job_count",
    "skipped_job_count",
    "pending_job_count",
    "other_job_count",
    "unit_test_outcome",
    "timing_complete",
    "sum_job_execution_seconds",
    "job_execution_span_seconds",
    "observed_job_execution_seconds",
)


def _number(value: object, *, integer: bool = False) -> int | float | None:
    types = (int,) if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, types) or value < 0:
        return None
    if isinstance(value, float) and not float("-inf") < value < float("inf"):
        return None
    return value


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    # GitHub uses year 0001 timestamps for jobs that have not started/finished.
    if timestamp.tzinfo is None or timestamp.year < 2000:
        return None
    return timestamp


def _job_measurement(job: dict[str, Any], bucket: str) -> dict[str, Any]:
    start = _timestamp(job.get("startedAt"))
    end = _timestamp(job.get("completedAt"))
    status = job.get("status")
    conclusion = job.get("conclusion") if isinstance(job.get("conclusion"), str) else None
    duration = None
    issue = None
    if status != "completed":
        issue = "job has not completed"
    elif conclusion == "skipped":
        issue = "job was skipped; execution duration is unavailable"
    elif not conclusion:
        issue = "completed job has no conclusion"
    elif start is None or end is None:
        issue = "job is missing valid execution timestamps"
    elif end < start:
        issue = "job completion precedes its start"
    else:
        duration = round((end - start).total_seconds(), 3)
    return {
        "bucket": bucket,
        "job_id": job.get("databaseId"),
        "name": job.get("name"),
        "url": job.get("url"),
        "status": status,
        "conclusion": conclusion,
        "started_at": job.get("startedAt"),
        "completed_at": job.get("completedAt"),
        "duration_seconds": duration,
        "timing_issue": issue,
    }


def build_report(
    selection: dict[str, Any] | None,
    jobs_payload: dict[str, Any] | None,
    metadata: dict[str, Any],
    input_issues: list[str] | None = None,
) -> dict[str, Any]:
    """Build a run-level record, preserving incomplete and unsuccessful runs."""

    issues = list(input_issues or ())
    selection = selection or {}
    mode = selection.get("mode")
    if not isinstance(mode, str) or mode not in {"full", "selective"}:
        issues.append("selection mode is unavailable or invalid")
        mode = "unknown"

    selected_count = _number(selection.get("selected_count"), integer=True)
    total_count = _number(selection.get("total_count"), integer=True)
    if (
        selected_count is None
        or total_count is None
        or selected_count <= 0
        or total_count <= 0
        or selected_count > total_count
    ):
        issues.append("selection file counts are unavailable or invalid")

    matrix = selection.get("matrix")
    buckets = []
    if not isinstance(matrix, list) or not matrix:
        issues.append("selection matrix is unavailable or empty")
    else:
        for entry in matrix:
            bucket = entry.get("bucket") if isinstance(entry, dict) else None
            if not isinstance(bucket, str) or not bucket.startswith("tests/unit_tests/"):
                issues.append("selection matrix contains an invalid bucket")
            elif bucket in buckets:
                issues.append(f"selection matrix repeats bucket: {bucket}")
            else:
                buckets.append(bucket)

    expected_names = {f"{bucket} - latest": bucket for bucket in buckets}
    source_jobs = (jobs_payload or {}).get("jobs")
    if not isinstance(source_jobs, list):
        issues.append("GitHub jobs data is unavailable or invalid")
        source_jobs = []
    jobs = [
        _job_measurement(job, expected_names[job["name"]])
        for job in source_jobs
        if isinstance(job, dict)
        and isinstance(job.get("name"), str)
        and job["name"] in expected_names
    ]
    jobs.sort(key=lambda job: (job["name"], str(job["job_id"])))
    names = Counter(job["name"] for job in jobs)
    missing = sorted(set(expected_names) - names.keys())
    duplicates = sum(count - 1 for count in names.values())
    if missing:
        issues.append(f"{len(missing)} planned H100 job(s) are missing from the jobs response")
    if duplicates:
        issues.append("duplicate H100 job names; supply jobs for one run attempt")
    timed_jobs = [job for job in jobs if job["duration_seconds"] is not None]
    if len(timed_jobs) != len(jobs):
        issues.append("some H100 jobs have incomplete execution timings")

    pending = sum(job["status"] != "completed" for job in jobs)
    conclusions = Counter(job["conclusion"] for job in jobs if job["status"] == "completed")
    failed = sum(conclusions[value] for value in FAILURE_CONCLUSIONS)
    successful = conclusions["success"]
    cancelled = conclusions["cancelled"]
    skipped = conclusions["skipped"]
    other = len(jobs) - pending - failed - successful - cancelled - skipped
    if failed:
        outcome = "failure"
    elif cancelled:
        outcome = "cancelled"
    elif not buckets or missing or duplicates or pending:
        outcome = "incomplete"
    elif successful == len(buckets):
        outcome = "success"
    elif skipped == len(buckets):
        outcome = "skipped"
    else:
        outcome = "mixed"

    timing_complete = bool(buckets) and not issues
    observed_seconds = (
        round(sum(job["duration_seconds"] for job in timed_jobs), 3) if timed_jobs else None
    )
    span_seconds = None
    if timing_complete:
        starts = [_timestamp(job["started_at"]) for job in jobs]
        ends = [_timestamp(job["completed_at"]) for job in jobs]
        span_seconds = round((max(ends) - min(starts)).total_seconds(), 3)

    return {
        "schema_version": 1,
        **metadata,
        "mode": mode,
        "selection_reason": selection.get("reason"),
        "selected_file_count": selected_count,
        "total_file_count": total_count,
        "selected_bucket_count": len(buckets) if buckets else None,
        "total_bucket_count": _number(selection.get("total_bucket_count"), integer=True),
        "selector_duration_seconds": _number(selection.get("selector_duration_seconds")),
        "planning_duration_seconds": _number(selection.get("planning_duration_seconds")),
        "expected_job_count": len(buckets) if buckets else None,
        "observed_job_count": len(jobs),
        "timed_job_count": len(timed_jobs),
        "missing_job_count": len(missing),
        "duplicate_job_count": duplicates,
        "successful_job_count": successful,
        "failed_job_count": failed,
        "cancelled_job_count": cancelled,
        "skipped_job_count": skipped,
        "pending_job_count": pending,
        "other_job_count": other,
        "unit_test_outcome": outcome,
        "timing_complete": timing_complete,
        "sum_job_execution_seconds": observed_seconds if timing_complete else None,
        "job_execution_span_seconds": span_seconds,
        "observed_job_execution_seconds": observed_seconds,
        "missing_jobs": missing,
        "jobs": jobs,
        "issues": issues,
    }


def render_summary(report: dict[str, Any]) -> str:
    """Render measurements with an explicit distinction between sum and span."""

    def value(key: str) -> str:
        item = report.get(key)
        if item is None:
            return "unavailable"
        return str(item).replace("|", "\\|").replace("\n", " ")

    lines = [
        "## H100 unit-test selection measurements",
        "",
        "| Measurement | Value |",
        "|---|---|",
        f"| Run / attempt | {value('run_id')} / {value('run_attempt')} |",
        f"| Run selective unit tests label present | {value('selective_label_present')} |",
        f"| Actual selection mode | {value('mode')} |",
        f"| Selection reason | {value('selection_reason')} |",
        "| Planned test files / available files | "
        f"{value('selected_file_count')} / {value('total_file_count')} |",
        "| Planned H100 buckets / observed jobs | "
        f"{value('selected_bucket_count')} / {value('observed_job_count')} |",
        f"| Selector overhead (seconds) | {value('selector_duration_seconds')} |",
        f"| Planning incl. dependency sync (seconds) | {value('planning_duration_seconds')} |",
        f"| H100 outcome | {value('unit_test_outcome')} |",
        f"| Complete execution timings | {value('timing_complete')} |",
        f"| Sum of job execution durations (seconds) | {value('sum_job_execution_seconds')} |",
        f"| First job start to last job end (seconds) | {value('job_execution_span_seconds')} |",
        "| Observed timed-job sum, possibly partial (seconds) | "
        f"{value('observed_job_execution_seconds')} |",
        "",
        "Durations include each H100 job's setup and execution. The span reflects parallel jobs; "
        "neither metric measures queue time. Missing or incomplete timings remain unavailable. "
        "Planning time excludes setup-uv and runner provisioning. "
        "File counts describe the selection plan, not collected pytest cases. "
        "These are observed runs, not an estimate of what the other mode would have cost.",
    ]
    if report["issues"]:
        lines.extend(["", "Measurement issues:", ""])
        lines.extend(f"- {issue}" for issue in report["issues"])
    return "\n".join(lines) + "\n"


def _read_json(path: Path, label: str, issues: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as error:
        issues.append(f"{label} could not be read: {error}")
        return None


def main() -> int:
    """Write durable JSON, one-row CSV, and Markdown artifacts for this run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-attempt", type=int, required=True)
    parser.add_argument("--sha", default="")
    parser.add_argument("--event-name", default="")
    parser.add_argument("--pr-number", default="")
    parser.add_argument(
        "--selective-label-present", choices=("true", "false", "unknown"), default="unknown"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("unit-test-metrics"))
    args = parser.parse_args()

    issues: list[str] = []
    selection = _read_json(args.selection, "selection manifest", issues)
    jobs = _read_json(args.jobs, "GitHub jobs response", issues)
    report = build_report(
        selection,
        jobs,
        {
            "repository": args.repository,
            "run_id": args.run_id,
            "run_attempt": args.run_attempt,
            "sha": args.sha,
            "event_name": args.event_name,
            "pr_number": args.pr_number or None,
            "run_url": (
                f"https://github.com/{args.repository}/actions/runs/{args.run_id}"
                f"/attempts/{args.run_attempt}"
            ),
            "selective_label_present": {"true": True, "false": False, "unknown": None}[
                args.selective_label_present
            ],
        },
        issues,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "unit-test-metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output_dir / "unit-test-metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                key: str(report[key]).lower() if isinstance(report[key], bool) else report[key]
                for key in CSV_FIELDS
            }
        )
    (args.output_dir / "unit-test-metrics.md").write_text(render_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
