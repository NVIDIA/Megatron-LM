# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
import os

import click
import gitlab
from nemo_ci_triage.slack_notification import notification

PROJECT_ID = int(os.getenv("CI_PROJECT_ID", 19378))
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
GITLAB_ENDPOINT = os.getenv("GITLAB_ENDPOINT")
if not GITLAB_ENDPOINT:
    raise ValueError("GITLAB_ENDPOINT is required")
SERVER_URL = f"https://{GITLAB_ENDPOINT}"
PROJECT_URL = os.getenv("CI_PROJECT_URL", f"{SERVER_URL}/ADLR/megatron-lm")
TAG_TEAM = os.getenv("TAG_TEAM", "0") == "1"
TEAM_SLUG = os.getenv("TEAM_SLUG", "")

JOB_PREFIXES = {
    "unit-tests": "test:unit_tests",
    "integration-tests": "integration:run_",
    "functional-tests": "functional:run_",
    "smoke-tests": "functional:smoke-",
}

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_gitlab_handle() -> gitlab.Gitlab:
    return gitlab.Gitlab(SERVER_URL, private_token=os.getenv("RO_API_TOKEN"))


def _bridge_gpu(bridge_name: str) -> str:
    for gpu in ("GB200", "H100", "A100"):
        if gpu.lower() in bridge_name.lower():
            return gpu
    return "Unknown"


def get_pipeline_jobs(pipeline_id: int, job_prefix: str) -> list[tuple[str, int, list[dict]]]:
    """Collect Megatron-LM's direct child pipelines using nemo-ci-triage-2."""
    project = get_gitlab_handle().projects.get(PROJECT_ID)
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


def configure_notification_urls() -> None:
    """Point nemo-ci-triage-2's notification links at Megatron-LM."""
    notification.JOB_URL_TEMPLATE = f"{PROJECT_URL}/-/jobs/{{}}"
    notification.PIPELINE_URL_TEMPLATE = f"{PROJECT_URL}/-/pipelines/{{}}"


@click.command()
@click.option("--pipeline-id", required=True, type=int, help="PipelineID")
@click.option(
    "--check-for",
    required=True,
    type=click.Choice(["unit-tests", "integration-tests", "functional-tests", "smoke-tests"]),
)
@click.option("--pipeline-context", required=True, type=str)
@click.option("--pipeline-created-at", required=True, type=str, expose_value=False)
def main(pipeline_id: int, check_for: str, pipeline_context: str) -> None:
    pipeline_jobs = get_pipeline_jobs(pipeline_id, JOB_PREFIXES[check_for])

    if check_for == "smoke-tests":
        if all(job["status"] == "success" for _, _, jobs in pipeline_jobs for job in jobs):
            logger.info("All smoke tests passed, skipping Slack notification")
            return

    if not WEBHOOK_URL:
        logger.info("No webhook URL configured, skipping Slack notification")
        return

    configure_notification_urls()
    slack_mentions = f"{TEAM_SLUG} <!subteam^S0A7B4U1T3P> <@U09TX0DHZ97>" if TAG_TEAM else None
    notification.send_slack_notification(
        "megatron-lm", pipeline_context, pipeline_jobs, slack_mentions, webhook_url=WEBHOOK_URL
    )


if __name__ == "__main__":
    main()
