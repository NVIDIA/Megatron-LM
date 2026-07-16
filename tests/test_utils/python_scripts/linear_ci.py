# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron-LM adapters for nemo-ci-triage's Linear workflow.

The triage package owns Linear status, reconciliation, configuration, mutation,
and Slack follow-up logic, but its NeMo pipeline summarizer is repository-specific.
This module converts Megatron-LM child-job results into the package's
``pipeline_summaries.json`` and ``failure_buckets.json`` contracts.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

LINEAR_MODULE = "megatron_lm"
_FUNCTIONAL_PREFIX = "functional:run_"


def _variant_name(pipeline_name: str) -> str:
    """Return the stable environment/platform suffix of a functional bridge."""
    return pipeline_name.removeprefix(_FUNCTIONAL_PREFIX).replace("_", "-")


def _recipe_name(pipeline_name: str, config_name: str) -> str:
    """Disambiguate the same recipe across dev/LTS and GPU child pipelines."""
    return f"{config_name}@{_variant_name(pipeline_name)}"


def _bucket_label(category: str, subtype: str) -> str:
    """Build a readable, stable label for one exact error fingerprint."""
    signature = f"{category}\n{subtype}"
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:8]
    slug = re.sub(r"[^a-z0-9]+", "-", category.lower()).strip("-") or "unknown"
    return f"{slug[:48]}-{digest}"


def _job_url(project_url: str, job: dict) -> str:
    return job.get("web_url") or f"{project_url}/-/jobs/{job['id']}"


def _failure_record(pipeline_name: str, job: dict, report: dict | None, project_url: str) -> dict:
    report = report or {}
    category = report.get("category") or report.get("error_type") or job.get("error_type")
    category = category or "Unknown"
    subtype = report.get("error_subtype") or job.get("error_type")
    subtype = subtype or f"No structured error report was available for {job['config_name']}"
    rationale = subtype if subtype == category else f"{category}: {subtype}"
    return {
        "test_name": _recipe_name(pipeline_name, job["config_name"]),
        "module": LINEAR_MODULE,
        "category": category,
        "summary": rationale,
        "excerpt": report.get("excerpt"),
        "job_url": _job_url(project_url, job),
        "subtype": subtype,
    }


def build_pipeline_reports(
    pipeline_id: int,
    scope: str,
    pipeline_jobs: list[tuple[str, int, list[dict]]],
    load_error_report: Callable[[int], dict | None],
    project_url: str,
) -> tuple[dict, dict]:
    """Build the two JSON contracts consumed by nemo-ci-triage reconciliation.

    Each recipe is qualified by its child-pipeline variant. A recipe is only
    included in ``passed_tests`` when that exact variant completed successfully;
    failed, canceled, and ambiguous allow-failure jobs can therefore never close
    a live Linear issue accidentally.
    """
    passed: set[str] = set()
    unknown: set[str] = set()
    failures: list[dict] = []
    failed_jobs = 0

    for pipeline_name, _, jobs in sorted(pipeline_jobs, key=lambda item: item[0]):
        for job in sorted(jobs, key=lambda item: (item["config_name"], item["id"])):
            recipe = _recipe_name(pipeline_name, job["config_name"])
            status = job.get("status")
            report = None

            if status == "failed" or (status == "success" and job.get("allow_failure")):
                report = load_error_report(job["id"])

            suppressed_failure = bool(
                status == "success" and report and report.get("exit_code_training") not in (None, 0)
            )
            if status == "failed" or suppressed_failure:
                failed_jobs += 1
                failures.append(_failure_record(pipeline_name, job, report, project_url))
            elif status == "success" and (not job.get("allow_failure") or report is not None):
                passed.add(recipe)
            else:
                unknown.add(recipe)

    failed_recipes = {failure["test_name"] for failure in failures}
    passed_tests = sorted(passed - failed_recipes - unknown)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for failure in failures:
        grouped[(failure["category"], failure["subtype"])].append(failure)

    buckets = []
    for (category, subtype), records in sorted(grouped.items()):
        tests_by_name = {record["test_name"]: record["job_url"] for record in records}
        buckets.append(
            {
                "label": _bucket_label(category, subtype),
                "rationale": records[0]["summary"],
                "category": category,
                "module": LINEAR_MODULE,
                "sample_excerpt": next(
                    (record["excerpt"] for record in records if record.get("excerpt")), None
                ),
                "tests": [
                    {"name": name, "job_url": url} for name, url in sorted(tests_by_name.items())
                ],
            }
        )

    module_stats = {
        "passed": len(passed_tests),
        "failed": failed_jobs,
        "passed_tests": passed_tests,
    }
    summaries = {
        "pipeline_id": pipeline_id,
        "scope": scope,
        "modules": {LINEAR_MODULE: module_stats},
        "digest": (
            f"Megatron-LM functional CI: {len(passed_tests)} passing recipe variants, "
            f"{failed_jobs} failed jobs, and {len(unknown)} unknown recipe variants."
        ),
        "failures": [
            {key: value for key, value in failure.items() if key != "subtype"}
            for failure in failures
        ],
    }
    failure_buckets = {"pipeline_id": pipeline_id, "bucketing_failed": False, "buckets": buckets}
    return summaries, failure_buckets


def fetch_error_report(project: Any, job_id: int) -> dict | None:
    """Fetch one child job's structured report, degrading safely if absent."""
    try:
        raw = project.jobs.get(job_id, lazy=True).artifact("error_report.json")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)
    except Exception as exc:
        print(f"WARNING: job {job_id}: could not read error_report.json: {exc}", file=sys.stderr)
        return None


def write_pipeline_reports(
    pipeline_id: int,
    scope: str,
    pipeline_jobs: list[tuple[str, int, list[dict]]],
    project: Any,
    project_url: str,
    summaries_path: Path,
    buckets_path: Path,
) -> None:
    summaries, buckets = build_pipeline_reports(
        pipeline_id,
        scope,
        pipeline_jobs,
        lambda job_id: fetch_error_report(project, job_id),
        project_url,
    )
    summaries_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    buckets_path.write_text(json.dumps(buckets, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summaries_path} and {buckets_path}")
