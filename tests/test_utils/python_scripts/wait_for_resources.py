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

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_gitlab_handle():
    return gitlab.Gitlab(f"https://{GITLAB_ENDPOINT}", private_token=os.getenv("RO_API_TOKEN"))


def ci_is_busy(pipeline):
    """List all merge request pipelines created before the given pipeline that are still pending or running."""
    mr_pipelines = (
        get_gitlab_handle().projects.get(PROJECT_ID).pipelines.list(source="merge_request_event")
    )

    pipeline_time = pipeline.attributes['created_at']
    return (
        len(
            [
                p
                for p in mr_pipelines
                if p.attributes['created_at'] < pipeline_time
                and p.attributes['status'] in ('pending', 'running')
            ]
        )
        > 0
    )


@click.command()
@click.option('--pipeline-id', required=True, type=int, help='CI pipeline ID to check')
def main(pipeline_id):
    pipeline = get_gitlab_handle().projects.get(PROJECT_ID).pipelines.get(pipeline_id)

    while ci_is_busy(pipeline):
        logger.info("Waiting for resources...")
        time.sleep(60)


if __name__ == "__main__":
    main()
