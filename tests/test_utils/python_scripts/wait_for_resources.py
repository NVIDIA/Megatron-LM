from __future__ import annotations

import logging
import os
import re
import time
from typing import Literal

import click
import gitlab
import requests

PROJECT_ID = int(os.getenv("CI_PROJECT_ID", 19378))
GITLAB_ENDPOINT = os.getenv("GITLAB_ENDPOINT")
RO_API_TOKEN = os.getenv("RO_API_TOKEN")
NUM_CONCURRENT_JOBS = int(os.getenv("NUM_CONCURRENT_JOBS", 2)) // 2  # for main and dev branch

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_gitlab_handle():
    return gitlab.Gitlab(f"https://{GITLAB_ENDPOINT}", private_token=os.getenv("RO_API_TOKEN"))


def ci_is_busy(pipeline, target_branch: str):
    """List all merge request pipelines created before the given pipeline that are still pending or running."""
    mr_pipelines = (
        get_gitlab_handle()
        .projects.get(PROJECT_ID)
        .pipelines.list(
            source="merge_request_event", per_page=100, page=1, order_by="id", sort="desc"
        )
    )

    pipeline_time = pipeline.attributes["created_at"]
    in_queue = len(
        [
            p
            for p in mr_pipelines
            if p.attributes["created_at"] < pipeline_time
            if (
                get_gitlab_handle()
                .projects.get(PROJECT_ID)
                .mergerequests.get(
                    int(re.search(r'merge-requests/(\d+)', p.attributes["ref"]).group(1))
                )
                .target_branch
                == target_branch
            )
            and p.attributes["status"] in ("pending", "running")
        ]
    )
    logger.info(f"Position in queue: {in_queue+1}. Waiting for resources...")
    return in_queue > NUM_CONCURRENT_JOBS


@click.command()
@click.option("--pipeline-id", required=True, type=int, help="CI pipeline ID to check")
@click.option("--target-branch", required=True, type=str, help="Target branch to check")
def main(pipeline_id, target_branch):
    pipeline = get_gitlab_handle().projects.get(PROJECT_ID).pipelines.get(pipeline_id)
    logger.info(f"Job concurrency: {NUM_CONCURRENT_JOBS}")

    while True:
        try:
            is_busy = ci_is_busy(pipeline, target_branch)
            if not is_busy:
                break
            time.sleep(60)

        except (requests.exceptions.ConnectionError, gitlab.exceptions.GitlabListError) as e:
            logger.info(f"Network error. Retrying... {e}")
            time.sleep(15)
            continue
        except Exception as e:
            logger.error(f"Error: {e}")
            break


if __name__ == "__main__":
    main()
