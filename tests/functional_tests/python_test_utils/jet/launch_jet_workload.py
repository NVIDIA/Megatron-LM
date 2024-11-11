import os
import pathlib
import re
import signal
import sys
import tempfile
import time
from typing import List, Optional, Tuple

import click
import jetclient
import requests
import yaml
from jetclient.services.dtos.pipeline import PipelineStatus

from tests.functional_tests.python_test_utils.jet import common

BASE_PATH = pathlib.Path(__file__).parent.resolve()


def resolve_cluster_config(cluster: str) -> str:
    if cluster == "dgxh100_eos":
        return "mcore/eos"
    if cluster == "dgxa100_dracooci":
        return "mcore/draco-oci"
    if cluster == "dgxa100_dracooci-ord":
        return "mcore/draco-oci-ord"
    if cluster == "dgxh100_coreweave":
        return "mcore/coreweave"
    raise ValueError(f"Unknown cluster {cluster} provided.")


def register_pipeline_terminator(pipeline: jetclient.JETPipeline):
    def sigterm_handler(_signo, _stack_frame):
        print(f"Trying to terminate pipeline {pipeline.jet_id}")
        pipeline.cancel()
        print(f"Pipeline {pipeline.jet_id} terminated")
        sys.exit(0)

    signal.signal(signal.SIGINT, sigterm_handler)
    signal.signal(signal.SIGTERM, sigterm_handler)


def launch_and_wait_for_completion(
    test_case: str,
    environment: str,
    n_repeat: int,
    time_limit: int,
    container_image: str,
    container_tag: str,
    cluster: str,
    account: str,
    run_name: Optional[str],
    wandb_experiment: Optional[str],
) -> jetclient.JETPipeline:
    pipeline = jetclient.JETClient(
        customer='mcore', gitlab_ci_token=os.getenv("RO_API_TOKEN"), env="prod"
    ).workloads.submit(
        workloads=common.load_workloads(
            test_case=test_case,
            n_repeat=n_repeat,
            time_limit=time_limit,
            container_image=container_image,
            container_tag=container_tag,
            environment=environment,
        ),
        config_id=resolve_cluster_config(cluster),
        custom_config={
            "launchers": {cluster: {"account": account, "ntasks_per_node": 8}},
            "executors": {
                "jet-ci": {
                    "environments": {
                        cluster: {
                            "variables": {
                                "RUN_NAME": run_name or "",
                                "WANDB_API_KEY": os.getenv("WANDB_API_KEY") or "",
                                "WANDB_EXPERIMENT": wandb_experiment or "",
                            }
                        }
                    }
                }
            },
        },
        wait_for_validation=True,
    )

    register_pipeline_terminator(pipeline=pipeline)

    print(
        f"Pipeline triggered; inspect it here: https://gitlab-master.nvidia.com/dl/jet/ci/-/pipelines/{pipeline.jet_id}",
        flush=True,
    )

    pipeline.wait(max_wait_time=60 * 60 * 24 * 7)

    print(f"Pipeline terminated; status: {pipeline.get_status()}")
    return pipeline


def download_job_assets(job: jetclient.JETJob, iteration: int = 0) -> List[str]:
    logs = job.get_logs()
    if not logs:
        return [""]

    assets_base_path = BASE_PATH / ".." / ".." / ".." / ".." / "results" / f"iteration={iteration}"

    for restart_idx, log in enumerate(logs):
        assets = log.get_assets()
        assets_path = assets_base_path / f"restart={restart_idx}"
        assets_path.mkdir(parents=True, exist_ok=True)
        for log_filename in assets.keys():
            with open(assets_path / log_filename, "w") as fh:
                assets[log_filename].download(pathlib.Path(fh.name))


def download_job_logs(job: jetclient.JETJob) -> List[str]:
    logs = job.get_logs()
    if not logs:
        return [""]

    assets = logs[0].get_assets()
    log_filename = [key for key in assets.keys() if key.endswith(".log")][0]

    with tempfile.NamedTemporaryFile() as tmp_file:
        assets[log_filename].download(pathlib.Path(tmp_file.name))
        with open(pathlib.Path(tmp_file.name), "r") as fh:
            return fh.readlines()


def parse_failed_job(logs: List[str]) -> Optional[bool]:
    for log_row in logs[::-1]:
        match = re.search(r"Job finished with status 'FAILED'", log_row)
        if match is not None:
            return True
    return False


def parse_finished_training(logs: List[str]) -> Optional[bool]:
    for log_row in logs[::-1]:
        match = re.search(r"after training is done", log_row)
        if match is not None:
            return True
    return False


@click.command()
@click.option("--model", required=True, type=str, help="Model")
@click.option("--test-case", required=True, type=str, help="Test case")
@click.option(
    "--environment", required=True, type=click.Choice(['dev', 'lts']), help="Pytorch LTS or DEV"
)
@click.option("--n-repeat", required=False, default=1, type=int)
@click.option("--time-limit", required=False, default=1800, type=int)
@click.option(
    "--account",
    required=False,
    type=str,
    help="Slurm account to use",
    default="coreai_dlalgo_mcore",
)
@click.option("--cluster", required=True, type=str, help="Cluster to run on")
@click.option("--container-tag", required=True, type=str, help="Base image of Mcore image")
@click.option("--container-image", required=False, type=str, help="Base image of Mcore image")
@click.option(
    "--run-name", required=False, type=str, help="Run name (only relevant for release tests)"
)
@click.option(
    "--wandb-experiment",
    required=False,
    type=str,
    help="Wandb experiment (only relevant for release tests)",
)
def main(
    model: str,
    test_case: str,
    environment: str,
    n_repeat: int,
    time_limit: int,
    account: str,
    cluster: str,
    container_tag: str,
    container_image: Optional[str] = None,
    run_name: Optional[str] = None,
    wandb_experiment: Optional[str] = None,
):

    with open(
        pathlib.Path(
            BASE_PATH / ".." / ".." / "test_cases" / model / test_case / "model_config.yaml"
        )
    ) as stream:
        try:
            test_case_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    test_type = test_case_dict['TEST_TYPE']

    if test_type == "release" and (run_name is None or wandb_experiment is None):
        print(f"Not all arguments provided ({run_name=}, {wandb_experiment=})")
        sys.exit(1)

    n_attempts = 0
    n_iteration = 0
    while True and n_attempts < 3:
        pipeline = launch_and_wait_for_completion(
            test_case=test_case,
            environment=environment,
            n_repeat=n_repeat,
            time_limit=time_limit,
            container_image=container_image,
            container_tag=container_tag,
            cluster=cluster,
            account=account,
            run_name=run_name,
            wandb_experiment=wandb_experiment,
        )

        main_job = [job for job in pipeline.get_jobs() if job.name.startswith("basic")][0]

        logs = download_job_logs(job=main_job)
        concat_logs = "\n".join(logs)
        print(f"Logs:\n{concat_logs}")

        download_job_assets(job=main_job, iteration=n_iteration)

        if test_type != "release":
            success = pipeline.get_status() == PipelineStatus.SUCCESS

            if (
                "Some NCCL operations have failed or timed out." in concat_logs
                or "uncorrectable ECC error encountered" in concat_logs
                or "illegal memory access" in concat_logs
                or "illegal instruction" in concat_logs
            ):
                print("Detected NCCL failure, attempt restart.")
                n_attempts += 1
                continue

            sys.exit(int(not success))  # invert for exit 0

        if parse_failed_job(logs=logs):
            n_attempts += 1
            continue

        if parse_finished_training(logs=logs):
            success = pipeline.get_status() == PipelineStatus.SUCCESS
            sys.exit(int(not success))  # invert for exit 0
        n_iteration += 1
    sys.exit(1)


if __name__ == "__main__":
    main()
