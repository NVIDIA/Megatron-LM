# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import logging
import os
from pathlib import Path
from typing import Any

import click
import gitlab
from nemo_ci_triage.slack_notification import notification
from nemo_ci_triage.slack_notification.utils import repository_settings

from tests.test_utils.python_scripts import linear_ci

TRIAGE_CONFIG = Path(os.getenv("NEMO_CI_TRIAGE_CONFIG", ".gitlab/nemo-ci-triage.yml"))
PROJECT_ID, REPO_NAME = repository_settings(TRIAGE_CONFIG)
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
SLACK_BOT_TOKEN = os.getenv("MCORE_SLACK_BOT_TOKEN") or os.getenv("ALERTMANAGER_TOKEN", "")
SLACK_CHANNEL_ID = os.getenv("MCORE_SLACK_CHANNEL_ID", "")
GITLAB_ENDPOINT = os.getenv("GITLAB_ENDPOINT")
if not GITLAB_ENDPOINT:
    raise ValueError("GITLAB_ENDPOINT is required")
SERVER_URL = f"https://{GITLAB_ENDPOINT}"
PROJECT_URL = os.getenv("CI_PROJECT_URL", f"{SERVER_URL}/{REPO_NAME}")
TAG_TEAM = os.getenv("TAG_TEAM", "0") == "1"
TEAM_SLUG = os.getenv("TEAM_SLUG", "")

JOB_PREFIXES = {
    "unit-tests": "test:unit_tests",
    "integration-tests": "integration:run_",
    "functional-tests": ("functional:run_", "functional:smoke-"),
    "smoke-tests": "functional:smoke-",
}

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_gitlab_handle() -> gitlab.Gitlab:
    return gitlab.Gitlab(SERVER_URL, private_token=os.getenv("RO_API_TOKEN"))


def get_project() -> Any:
    """Return the configured Megatron-LM GitLab project."""
    return get_gitlab_handle().projects.get(PROJECT_ID)


def _bridge_gpu(bridge_name: str) -> str:
    for gpu in ("GB200", "H100", "A100"):
        if gpu.lower() in bridge_name.lower():
            return gpu
    return "Unknown"


def get_pipeline_jobs(
    pipeline_id: int, job_prefix: str | tuple[str, ...], project: Any | None = None
) -> list[tuple[str, int, list[dict]]]:
    """Collect Megatron-LM's direct child pipelines using nemo-ci-triage-2."""
    project = project or get_project()
    root_pipeline = project.pipelines.get(pipeline_id)
    pipeline_jobs = []

    for bridge in root_pipeline.bridges.list(get_all=True):
        downstream = bridge.attributes.get("downstream_pipeline")
        if not bridge.name.startswith(job_prefix) or downstream is None:
            continue

        child_pipeline_id = downstream["id"]
        jobs = notification.get_jobs_from_pipeline(project, child_pipeline_id)
        bridge_gpu = _bridge_gpu(bridge.name)
        for job in jobs:
            if job["gpu"] == "Unknown":
                job["gpu"] = bridge_gpu
        pipeline_jobs.append((bridge.name, child_pipeline_id, jobs))

    return pipeline_jobs


def write_slack_context(output: Path | None, thread_timestamp: str | None) -> None:
    """Persist the non-secret Slack coordinates needed by a follow-up job."""
    if output is None:
        return
    context = {"channel_id": SLACK_CHANNEL_ID or None, "thread_timestamp": thread_timestamp}
    output.write_text(json.dumps(context, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote Slack thread context to %s", output)


@click.command()
@click.option("--pipeline-id", required=True, type=int, help="PipelineID")
@click.option(
    "--check-for",
    required=True,
    type=click.Choice(["unit-tests", "integration-tests", "functional-tests", "smoke-tests"]),
)
@click.option("--pipeline-context", required=True, type=str)
@click.option("--pipeline-created-at", required=True, type=str, expose_value=False)
@click.option("--summary-output", type=click.Path(path_type=Path), default=None)
@click.option("--failure-buckets-output", type=click.Path(path_type=Path), default=None)
@click.option("--slack-output", type=click.Path(path_type=Path), default=None)
def main(
    pipeline_id: int,
    check_for: str,
    pipeline_context: str,
    summary_output: Path | None,
    failure_buckets_output: Path | None,
    slack_output: Path | None,
) -> None:
    if bool(summary_output) != bool(failure_buckets_output):
        raise click.UsageError(
            "--summary-output and --failure-buckets-output must be provided together"
        )

    project = get_project()
    pipeline_jobs = get_pipeline_jobs(pipeline_id, JOB_PREFIXES[check_for], project=project)

    if summary_output:
        linear_ci.write_pipeline_reports(
            pipeline_id,
            pipeline_context,
            pipeline_jobs,
            project,
            PROJECT_URL,
            summary_output,
            failure_buckets_output,
        )

    if check_for == "smoke-tests":
        if all(job["status"] == "success" for _, _, jobs in pipeline_jobs for job in jobs):
            logger.info("All smoke tests passed, skipping Slack notification")
            write_slack_context(slack_output, None)
            return

    use_bot = bool(SLACK_BOT_TOKEN and SLACK_CHANNEL_ID)
    if bool(SLACK_BOT_TOKEN) != bool(SLACK_CHANNEL_ID):
        logger.warning(
            "Both MCORE_SLACK_BOT_TOKEN (or ALERTMANAGER_TOKEN) and "
            "MCORE_SLACK_CHANNEL_ID are required for threaded Slack replies"
        )

    if not WEBHOOK_URL and not use_bot:
        logger.info("No Slack bot or webhook configured, skipping Slack notification")
        write_slack_context(slack_output, None)
        return

    slack_mentions = f"{TEAM_SLUG} <!subteam^S0A7B4U1T3P> <@U09TX0DHZ97>" if TAG_TEAM else None
    thread_timestamp = notification.send_slack_notification(
        "megatron-lm",
        pipeline_context,
        pipeline_jobs,
        slack_mentions,
        webhook_url=WEBHOOK_URL or None,
        slack_bot_token=SLACK_BOT_TOKEN if use_bot else None,
        slack_channel_id=SLACK_CHANNEL_ID if use_bot else None,
        config=TRIAGE_CONFIG,
    )
    write_slack_context(slack_output, thread_timestamp)


if __name__ == "__main__":
    main()
