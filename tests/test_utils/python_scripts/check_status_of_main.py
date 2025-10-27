# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import logging
import os
import sys
import time

import click
import gitlab

PROJECT_ID = int(os.getenv("CI_PROJECT_ID", 19378))
GITLAB_ENDPOINT = os.getenv('GITLAB_ENDPOINT')
RO_API_TOKEN = os.getenv("RO_API_TOKEN")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_gitlab_handle():
    return gitlab.Gitlab(f"https://{GITLAB_ENDPOINT}", private_token=os.getenv("RO_API_TOKEN"))


def most_recent_pipeline(target_branch: str):
    logger.info(f"Getting most recent pipeline for branch {target_branch}")
    n_attempts = 0
    while n_attempts < 3:
        try:
            pipelines = (
                get_gitlab_handle()
                .projects.get(PROJECT_ID)
                .pipelines.list(ref=target_branch, source="push", get_all=False)
            )
            break
        except Exception as e:
            logger.error(f"Network error, retrying... ({n_attempts}/3)")
            time.sleep(10 * (2**n_attempts))  # Exponential backoff: 10s, 20s, 40s
            n_attempts += 1

    logger.info(f"Pipelines: {pipelines}")

    return pipelines[0]


def is_pending(target_branch: str):
    pipeline = most_recent_pipeline(target_branch)
    PENDING_STATUSES = [
        "created",
        "waiting_for_resource",
        "preparing",
        "pending",
        "running",
        "canceled",
        "skipped",
        "manual",
        "scheduled",
    ]

    is_pending = pipeline.attributes['status'] in PENDING_STATUSES

    if not is_pending:
        logger.info(
            f"Main pipeline {pipeline.id} finished with status {pipeline.attributes['status']}"
        )

    return is_pending


@click.command()
@click.option("--target-branch", type=str, help="Target branch to check")
@click.option("--continuous/--once", is_flag=True, help="Continuous mode", default=True)
def main(target_branch: str, continuous: bool):
    while is_pending(target_branch):
        logger.info(f"Waiting for branch {target_branch} to finish")
        if not continuous:
            break
        time.sleep(60)

    pipeline = most_recent_pipeline(target_branch)

    if pipeline.attributes['status'] == 'failed':
        logger.error(
            "Main is broken, we're therefore blocking your merge. Please wait until main is fixed again by checking the repo's front page. If the status is green again, you can re-attempt the merge. Feel free to ping the team if you have any questions."
        )
        sys.exit(1)

    if pipeline.attributes['status'] == 'running':
        logger.info("Main is running, we won't cancel the deployment.")
        sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
