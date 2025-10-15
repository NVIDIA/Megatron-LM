import logging
import os

import click
import gitlab
import pandas as pd
import requests
import slack_sdk

PROJECT_ID = int(os.getenv("CI_PROJECT_ID", 19378))
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
GITLAB_ENDPOINT = os.getenv('GITLAB_ENDPOINT')
TAG_TEAM = bool(os.getenv('TAG_TEAM', 0))
TEAM_SLUG = str(os.getenv('TEAM_SLUG'))

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_gitlab_handle():
    return gitlab.Gitlab(f"https://{GITLAB_ENDPOINT}", private_token=os.getenv("RO_API_TOKEN"))


def extract_surrounding_text(text, keyword="error", context=400, fallback_length=800):
    index = text.rfind(keyword)  # Find the last occurrence
    if index == -1:
        return text[-fallback_length:]  # Return last 800 chars if keyword is not found

    start = max(0, index - context)  # Ensure we don't go below 0
    end = min(len(text), index + len(keyword))  # Ensure we don't exceed the text length

    return text[start:end]


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
            continue

        if pipeline_bridge.name not in bridge:
            bridge[pipeline_bridge.name] = []

        for job in (
            get_gitlab_handle()
            .projects.get(PROJECT_ID)
            .pipelines.get(pipeline_bridge.attributes['downstream_pipeline']['id'])
            .jobs.list(get_all=True)
        ):
            bridge[pipeline_bridge.name].append(job)
    return bridge


@click.command()
@click.option("--pipeline-id", required=True, type=int, help="PipelineID")
@click.option(
    "--check-for",
    required=True,
    type=click.Choice(["unit-tests", "integration-tests", "functional-tests"]),
)
@click.option("--pipeline-context", required=True, type=str)
@click.option("--pipeline-created-at", required=True, type=str)
def main(pipeline_id: int, check_for: str, pipeline_context: str, pipeline_created_at: str):
    if check_for == "unit-tests":
        bridges = get_jobs_per_bridge(pipeline_id, "test:unit_tests")

    if check_for == "integration-tests":
        bridges = get_jobs_per_bridge(pipeline_id, "integration:run_")

    if check_for == "functional-tests":
        bridges = get_jobs_per_bridge(pipeline_id, "functional:run_")
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
            messages.append(f"cc {TEAM_SLUG}: Critical event, please react as soon as possible.")

        for job in unsuccessful_jobs:
            messages.append(
                f"\tJob: <https://{GITLAB_ENDPOINT}/ADLR/megatron-lm/-/jobs/{job.id}|{job.name}>"
            )

    messages.append("===============================================")

    for message in messages:
        response = slack_sdk.webhook.WebhookClient(WEBHOOK_URL).send(text=message)
        logger.info(response.status_code)


if __name__ == "__main__":
    main()
