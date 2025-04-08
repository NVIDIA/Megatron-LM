import logging
import os
import sys
import time

import click
import gitlab

PROJECT_ID = int(os.getenv("CI_PROJECT_ID", 19378))
GITLAB_ENDPOINT = os.getenv('GITLAB_ENDPOINT')

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_gitlab_handle():
    return gitlab.Gitlab(f"https://{GITLAB_ENDPOINT}", private_token=os.getenv("RO_API_TOKEN"))


def most_recent_pipeline(target_branch: str):
    pipelines = get_gitlab_handle().projects.get(PROJECT_ID).pipelines.list(ref=target_branch)
    return pipelines[0]


def is_pending(pipeline: gitlab.v4.objects.Pipeline):
    return pipeline.attributes['status'] == 'pending'


def is_sucess(pipeline: gitlab.v4.objects.Pipeline):
    return pipeline.attributes['status'] == 'success'


@click.command()
@click.option("--target-branch", type=str, help="Target branch to check")
def main(target_branch: str):
    pipeline = most_recent_pipeline(target_branch)
    while is_pending(pipeline):
        logger.info(f"Waiting for branch {target_branch} to finish")
        time.sleep(10)

    logger.info(f"Main pipeline {pipeline.id} finished with status {pipeline.attributes['status']}")

    if not is_sucess(pipeline):
        sys.exit(1)


if __name__ == "__main__":
    main()
