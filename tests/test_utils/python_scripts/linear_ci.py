# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron-LM adapters for nemo-ci-triage's failure-reporting workflow.

The triage package owns LLM summarization, Linear reconciliation, and Slack
follow-up logic. This module only converts Megatron-LM's direct child-pipeline
jobs into the generic failure records consumed by the package summarizer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

from nemo_ci_triage.agent import summarize_pipeline_failures as summarizer

LINEAR_MODULE = "megatron_lm"
_FUNCTIONAL_PREFIX = "functional:run_"


def _variant_name(pipeline_name: str) -> str:
    """Return the stable environment/platform suffix of a functional bridge."""
    return pipeline_name.removeprefix(_FUNCTIONAL_PREFIX).replace("_", "-")


def _recipe_name(pipeline_name: str, config_name: str) -> str:
    """Disambiguate the same recipe across dev/LTS and GPU child pipelines."""
    return f"{config_name}@{_variant_name(pipeline_name)}"


def _job_url(project_url: str, job: dict) -> str:
    return job.get("web_url") or f"{project_url}/-/jobs/{job['id']}"


def _failure_record(pipeline_name: str, job: dict, report: dict | None, project_url: str) -> dict:
    """Return the raw failure shape accepted by the upstream LLM summarizer."""
    return {
        "test_name": _recipe_name(pipeline_name, job["config_name"]),
        "module": LINEAR_MODULE,
        "report": report,
        "job_url": _job_url(project_url, job),
        "job_error_type": job.get("error_type"),
    }


def _fallback_summary(failure: dict) -> dict:
    """Preserve a failed test when its per-test LLM summary is unavailable."""
    report = failure.get("report") or {}
    category = (
        report.get("error_type")
        or report.get("category")
        or failure.get("job_error_type")
        or "Unknown"
    )
    subtype = report.get("error_subtype") or failure.get("job_error_type")
    subtype = subtype or (f"No structured error report was available for {failure['test_name']}")
    summary = subtype if subtype == category else f"{category}: {subtype}"
    return {
        "test_name": failure["test_name"],
        "module": failure["module"],
        "category": category,
        "summary": summary,
        "excerpt": report.get("excerpt"),
        "job_url": failure["job_url"],
    }


def _summarize_failures(raw_failures: list[dict]) -> list[dict]:
    """Use upstream LLM summaries, falling back without dropping failures."""
    with_reports = [failure for failure in raw_failures if failure.get("report")]
    summarized = summarizer._summarize_failures(
        with_reports, summarizer._SUMMARIZER_PROMPT.read_text(encoding="utf-8").strip()
    )
    by_job = {(failure["test_name"], failure["job_url"]): failure for failure in summarized}
    return [
        by_job.get((failure["test_name"], failure["job_url"]), _fallback_summary(failure))
        for failure in raw_failures
    ]


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
    raw_failures: list[dict] = []
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
                raw_failures.append(_failure_record(pipeline_name, job, report, project_url))
            elif status == "success" and (not job.get("allow_failure") or report is not None):
                passed.add(recipe)
            else:
                unknown.add(recipe)

    failed_recipes = {failure["test_name"] for failure in raw_failures}
    passed_tests = sorted(passed - failed_recipes - unknown)

    failures = _summarize_failures(raw_failures)
    buckets, failed_stage = summarizer._subcategorize(failures)
    bucketing_failed = buckets is None
    if bucketing_failed:
        print(
            f"WARNING: LLM categorizer failed at {failed_stage}; "
            "Linear reconciliation will skip this report",
            file=sys.stderr,
        )
        buckets = []
    else:
        summarizer._attach_categories(buckets, failures)

    digest = summarizer._digest(
        failures, {LINEAR_MODULE: {"passed": len(passed_tests), "failed": failed_jobs}}
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
        "digest": digest,
        "failures": failures,
    }
    failure_buckets = {
        "pipeline_id": pipeline_id,
        "bucketing_failed": bucketing_failed,
        "buckets": summarizer._denormalize_buckets(buckets, failures),
    }
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
