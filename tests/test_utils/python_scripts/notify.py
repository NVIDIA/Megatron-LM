# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import logging
import os
from pathlib import Path
from typing import Any

import click
import gitlab
<<<<<<< HEAD
from cerno.slack_notification import notification
from cerno.slack_notification.utils import repository_settings
=======
import pandas as pd
import requests
import slack_sdk
>>>>>>> origin/dev

from tests.test_utils.python_scripts import linear_ci

CERNO_CONFIG = Path(os.getenv("CERNO_CONFIG", ".gitlab/cerno.yml"))
PROJECT_ID, REPO_NAME = repository_settings(CERNO_CONFIG)
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
<<<<<<< HEAD
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
=======
GITLAB_ENDPOINT = os.getenv('GITLAB_ENDPOINT')
TAG_TEAM = bool(os.getenv('TAG_TEAM', 0))
TEAM_SLUG = str(os.getenv('TEAM_SLUG'))
>>>>>>> origin/dev

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_gitlab_handle():
    return gitlab.Gitlab(f"https://{GITLAB_ENDPOINT}", private_token=os.getenv("RO_API_TOKEN"))


<<<<<<< HEAD
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
    """Collect Megatron-LM's direct child pipelines using Cerno."""
    project = project or get_project()
    root_pipeline = project.pipelines.get(pipeline_id)
    pipeline_jobs = []

    for bridge in root_pipeline.bridges.list(get_all=True):
        downstream = bridge.attributes.get("downstream_pipeline")
        if not bridge.name.startswith(job_prefix) or downstream is None:
=======
def get_jobs_per_bridge(pipeline_id: int, type_of_job: str):
    bridge = {}
    for pipeline_bridge in (
        get_gitlab_handle()
        .projects.get(PROJECT_ID)
        .pipelines.get(pipeline_id)
        .bridges.list(get_all=True)
    ):
        if (
            not pipeline_bridge.name.startswith(type_of_job)
            or pipeline_bridge.attributes['downstream_pipeline'] is None
        ):
>>>>>>> origin/dev
            continue

        if pipeline_bridge.name not in bridge:
            bridge[pipeline_bridge.name] = []

<<<<<<< HEAD
    return pipeline_jobs


def write_slack_context(output: Path | None, thread_timestamp: str | None) -> None:
    """Persist the non-secret Slack coordinates needed by a follow-up job."""
    if output is None:
        return
    context = {"channel_id": SLACK_CHANNEL_ID or None, "thread_timestamp": thread_timestamp}
    output.write_text(json.dumps(context, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote Slack thread context to %s", output)
=======
        for job in (
            get_gitlab_handle()
            .projects.get(PROJECT_ID)
            .pipelines.get(pipeline_bridge.attributes['downstream_pipeline']['id'])
            .jobs.list(get_all=True)
        ):
            bridge[pipeline_bridge.name].append(job)
    return bridge
>>>>>>> origin/dev


@click.command()
@click.option("--pipeline-id", required=True, type=int, help="PipelineID")
@click.option(
    "--check-for",
    required=True,
    type=click.Choice(["unit-tests", "integration-tests", "functional-tests", "smoke-tests"]),
)
@click.option("--pipeline-context", required=True, type=str)
<<<<<<< HEAD
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
=======
@click.option("--pipeline-created-at", required=True, type=str)
def main(pipeline_id: int, check_for: str, pipeline_context: str, pipeline_created_at: str):
    if check_for == "unit-tests":
        bridges = get_jobs_per_bridge(pipeline_id, "test:unit_tests")

    if check_for == "integration-tests":
        bridges = get_jobs_per_bridge(pipeline_id, "integration:run_")

    if check_for == "functional-tests":
        bridges = get_jobs_per_bridge(pipeline_id, "functional:run_")
>>>>>>> origin/dev

    if check_for == "smoke-tests":
        bridges = get_jobs_per_bridge(pipeline_id, "functional:smoke-")
        if all(job.status == "success" for jobs in bridges.values() for job in jobs):
            logger.info("All smoke tests passed, skipping Slack notification")
            write_slack_context(slack_output, None)
            return

<<<<<<< HEAD
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
        config=CERNO_CONFIG,
    )
    write_slack_context(slack_output, thread_timestamp)
=======
    pipeline_created_at_day = pd.Timestamp(pipeline_created_at).strftime("%Y-%m-%d")

    messages = []

    for bridge_name in bridges.keys():

        total_num_jobs = len(bridges[bridge_name])
        if all(job.status == "success" for job in bridges[bridge_name]):
            messages.append(
                f":doge3d: <https://{GITLAB_ENDPOINT}/ADLR/megatron-lm/-/pipelines/{pipeline_id}|Report - {pipeline_created_at_day} - {pipeline_context} - {bridge_name}>: All {total_num_jobs} passed."
            )
            continue

        unsuccessful_jobs = [job for job in bridges[bridge_name] if job.status != "success"]
        messages.append(
            f":doctorge: <https://{GITLAB_ENDPOINT}/ADLR/megatron-lm/-/pipelines/{pipeline_id}|Report - {pipeline_created_at_day} - {pipeline_context} - {bridge_name}>: {len(unsuccessful_jobs)} of {total_num_jobs} failed."
        )
        if TAG_TEAM:
            messages.append(
                f"cc {TEAM_SLUG} <!subteam^S0A7B4U1T3P> <@U09TX0DHZ97>: Critical event, please react as soon as possible."
            )

        for job in unsuccessful_jobs:
            messages.append(
                f"\tJob: <https://{GITLAB_ENDPOINT}/ADLR/megatron-lm/-/jobs/{job.id}|{job.name}>"
            )

    messages.append("===============================================")

    if not WEBHOOK_URL:
        logger.info("No webhook URL configured, skipping Slack notification")
        return

    for message in messages:
        response = slack_sdk.webhook.WebhookClient(WEBHOOK_URL).send(text=message)
        logger.info(response.status_code)
>>>>>>> origin/dev


if __name__ == "__main__":
    main()
