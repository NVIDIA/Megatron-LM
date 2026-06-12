# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import hashlib
import logging
import os
import sys
import time
from typing import Dict, List, Optional

import click
import jetclient
from jetclient.services.dtos.pipeline import PipelineStatus

from tests.test_utils.python_scripts import launch_jet_workload, recipe_parser

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _local_image_workload_name(source_image: str, local_path: str) -> str:
    digest = hashlib.sha1(f"{source_image}\n{local_path}".encode()).hexdigest()[:12]
    return f"prepare-sqsh-{digest}"


def build_prepare_workload(
    build: str, source_image: str, local_path: str, time_limit: int
) -> Dict:
    workload_name = _local_image_workload_name(source_image=source_image, local_path=local_path)
    return {
        "type": "basic",
        "format_version": 1,
        "maintainers": ["mcore"],
        "loggers": ["stdout"],
        "spec": {
            "name": workload_name,
            "build": build,
            "image_source": {"image_tag": source_image},
            "nodes": 1,
            "gpus": 0,
            "time_limit": time_limit,
            "script": "\n".join(
                [
                    "set -euo pipefail",
                    "echo 'Creating a local SquashFS image with Pyxis --container-save'",
                    "true",
                ]
            ),
        },
    }


def submit_prepare_workload(
    build: str,
    source_image: str,
    local_path: str,
    cluster: str,
    account: str,
    partition: Optional[str],
    time_limit: int,
) -> jetclient.JETPipeline:
    cluster_config = {"account": account, "srun_additional_flags": {"container_save": local_path}}
    if partition is not None:
        cluster_config["partition"] = partition

    n_submission_attempts = 0
    while n_submission_attempts < 3:
        try:
            pipeline = jetclient.JETClient(
                customer="mcore", gitlab_ci_token=os.getenv("RO_API_TOKEN"), env="prod"
            ).workloads.submit(
                workloads=[
                    jetclient.JETWorkloadManifest(
                        **build_prepare_workload(
                            build=build,
                            source_image=source_image,
                            local_path=local_path,
                            time_limit=time_limit,
                        )
                    )
                ],
                config_id=f"mcore/{recipe_parser.resolve_cluster_config(cluster)}",
                custom_config={
                    "launchers": {cluster: cluster_config},
                    "executors": {
                        "jet-ci": {
                            "environments": {
                                cluster: {
                                    "variables": {
                                        "PYTHONUNBUFFERED": "1",
                                        "RO_API_TOKEN": os.getenv("RO_API_TOKEN") or "",
                                    }
                                }
                            }
                        }
                    },
                },
                wait_for_validation=True,
                max_wait_time=60 * 60,
            )
        except (
            jetclient.clients.gitlab.GitlabAPIError,
            jetclient.facades.objects.util.WaitTimeExceeded,
        ) as e:
            logger.error("Faced %s. Waiting and retrying...", str(e))
            n_submission_attempts += 1
            time.sleep(2**n_submission_attempts * 5)
            continue

        if pipeline.get_status() == PipelineStatus.SUBMISSION_FAILED:
            n_submission_attempts += 1
            logger.info("Submission failed, attempt again (%s/3)", str(n_submission_attempts))
            continue

        return pipeline

    logger.error("Failed to submit local SquashFS image prepare workload after 3 attempts")
    sys.exit(1)


def prepare_local_image(
    build: str,
    source_image: str,
    local_path: str,
    cluster: str,
    account: str,
    partition: Optional[str],
    time_limit: int,
) -> bool:
    logger.info("Preparing %s from %s on %s", local_path, source_image, cluster)
    pipeline = submit_prepare_workload(
        build=build,
        source_image=source_image,
        local_path=local_path,
        cluster=cluster,
        account=account,
        partition=partition,
        time_limit=time_limit,
    )

    launch_jet_workload.register_pipeline_terminator(pipeline=pipeline)
    logger.info(
        "Pipeline triggered; inspect it here: https://gitlab-master.nvidia.com/dl/jet/ci/-/pipelines/%s",
        pipeline.jet_id,
    )

    try:
        status = launch_jet_workload.wait_for_pipeline_completion(
            pipeline=pipeline, max_wait_time=time_limit + 60 * 30, interval=30
        )
    except jetclient.facades.objects.util.WaitTimeExceeded:
        logger.error("Pipeline %s exceeded the wall-clock budget; cancelling.", pipeline.jet_id)
        try:
            pipeline.cancel()
        except jetclient.clients.gitlab.GitlabAPIError:
            logger.exception("Failed to cancel pipeline %s", pipeline.jet_id)
        return False

    logger.info("Pipeline terminated; status: %s", status)
    return status == PipelineStatus.SUCCESS


@click.command()
@click.option("--scope", required=True, type=str, help="Test scope")
@click.option("--environment", required=True, type=str, help="LTS or dev features")
@click.option(
    "--test-cases", required=True, type=str, help="Comma-separated list of test_cases, or 'all'"
)
@click.option("--platform", required=True, type=str, help="Platform to select")
@click.option("--cluster", required=True, type=str, help="Cluster to run on")
@click.option("--container-tag", required=True, type=str, help="Container tag to use")
@click.option(
    "--workload-local-image-path",
    required=True,
    type=str,
    help="Local SquashFS image path/template to create for JET basic workloads.",
)
@click.option(
    "--account",
    required=False,
    type=str,
    help="Slurm account to use",
    default="coreai_dlalgo_mcore",
)
@click.option("--partition", required=False, type=str, help="Slurm partition to use", default=None)
@click.option("--tag", required=False, type=str, help="Tag (only relevant for unit tests)")
@click.option(
    "--cadence",
    required=False,
    type=str,
    default=None,
    help="Trigger cadence to filter tests by (pr|nightly|mergegroup).",
)
@click.option(
    "--time-limit",
    required=False,
    default=1800,
    type=int,
    help="Slurm time limit in seconds for the image prepare workload.",
)
def main(
    scope: str,
    environment: str,
    test_cases: str,
    platform: str,
    cluster: str,
    container_tag: str,
    workload_local_image_path: str,
    account: str,
    partition: Optional[str],
    tag: Optional[str] = None,
    cadence: Optional[str] = None,
    time_limit: int = 1800,
):
    prepare_cluster = recipe_parser.resolve_local_image_prepare_cluster(cluster)
    local_image_sources: List[recipe_parser.dotdict] = (
        recipe_parser.resolve_workload_local_image_sources(
            container_tag=container_tag,
            tag=tag,
            environment=environment,
            platform=platform,
            test_cases=test_cases,
            scope=scope,
            workload_local_image_path=workload_local_image_path,
            cadence=cadence or None,
        )
    )

    if not local_image_sources:
        logger.info("No local SquashFS images to prepare.")
        return

    logger.info(
        "Preparing %s local SquashFS image(s) on %s", len(local_image_sources), prepare_cluster
    )
    for image_source in local_image_sources:
        success = prepare_local_image(
            build=image_source.build,
            source_image=image_source.source_image,
            local_path=image_source.local_path,
            cluster=prepare_cluster,
            account=account,
            partition=partition,
            time_limit=time_limit,
        )
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
