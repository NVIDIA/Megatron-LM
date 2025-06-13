import json
import logging
import os
import pathlib
import re
import signal
import sys
import time
import zipfile
from typing import Dict, List, Optional

import click
import jetclient
import pandas as pd
import requests
import yaml
from jetclient.facades.objects import log as jet_log
from jetclient.services.dtos.pipeline import PipelineStatus

from tests.test_utils.python_scripts import common

BASE_PATH = pathlib.Path(__file__).parent.resolve()
DASHBOARD_ENDPOINT = os.getenv("DASHBOARD_ENDPOINT")

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)


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
    scope: str,
    container_image: Optional[str],
    container_tag: str,
    cluster: str,
    platform: str,
    account: str,
    record_checkpoints: str,
    partition: Optional[str],
    tag: Optional[str],
    run_name: Optional[str],
    wandb_experiment: Optional[str],
    enable_lightweight_mode: bool,
) -> jetclient.JETPipeline:
    cluster_config = {"account": account}
    if partition is not None:
        cluster_config["partition"] = partition

    n_submission_attempts = 0
    while n_submission_attempts < 3:
        try:
            pipeline = jetclient.JETClient(
                customer="mcore", gitlab_ci_token=os.getenv("RO_API_TOKEN"), env="prod"
            ).workloads.submit(
                workloads=common.load_workloads(
                    test_case=test_case,
                    n_repeat=n_repeat,
                    time_limit=(1200 if enable_lightweight_mode else time_limit),
                    tag=tag,
                    scope=scope,
                    container_image=container_image,
                    container_tag=container_tag,
                    platform=platform,
                    environment=environment,
                    record_checkpoints=record_checkpoints,
                ),
                config_id=f"mcore/{common.resolve_cluster_config(cluster)}",
                custom_config={
                    "launchers": {cluster: cluster_config},
                    "executors": {
                        "jet-ci": {
                            "environments": {
                                cluster: {
                                    "variables": {
                                        "RUN_NAME": run_name or "",
                                        "ENABLE_LIGHTWEIGHT_MODE": str(
                                            enable_lightweight_mode
                                        ).lower(),
                                        "WANDB_API_KEY": os.getenv("WANDB_API_KEY") or "",
                                        "WANDB_EXPERIMENT": wandb_experiment or "",
                                        "RECORD_CHECKPOINTS": str(
                                            record_checkpoints == "true"
                                        ).lower(),
                                        "RO_API_TOKEN": os.getenv("RO_API_TOKEN") or "",
                                        "MCORE_REPO": os.getenv("CI_REPOSITORY_URL") or "",
                                        "MCORE_MR_COMMIT": os.getenv("MCORE_MR_COMMIT") or "",
                                        "MCORE_BACKWARDS_COMMIT": (
                                            os.getenv("MCORE_BACKWARDS_COMMIT") or ""
                                        ),
                                    }
                                }
                            }
                        }
                    },
                    "outputs": {
                        "enabled": True,
                        "artifacts_storages": [common.resolve_artifact_config(cluster)],
                    },
                },
                wait_for_validation=True,
                max_wait_time=(60 * 60),
            )
        except jetclient.clients.gitlab.GitlabAPIError as e:
            logger.error(f"Faced {str(e)}. Waiting and retrying...")
            n_submission_attempts += 1
            time.sleep(2**n_submission_attempts * 5)
            continue

        if pipeline.get_status() == PipelineStatus.SUBMISSION_FAILED:
            n_submission_attempts += 1
            logger.info("Submission failed, attempt again (%s/3)", str(n_submission_attempts))
            continue
        break

    if n_submission_attempts == 3:
        sys.exit(1)

    register_pipeline_terminator(pipeline=pipeline)

    logger.info(
        "Pipeline triggered; inspect it here: https://gitlab-master.nvidia.com/dl/jet/ci/-/pipelines/%s",
        pipeline.jet_id,
    )

    pipeline.wait(max_wait_time=60 * 60 * 24 * 7, interval=60 * 1, retries_on_error=3)

    logger.info(f"Pipeline terminated; status: {pipeline.get_status()}")
    return pipeline


def download_job_assets(logs: List[jet_log.JETLog], iteration: int = 0) -> Optional[pathlib.Path]:
    if not logs:
        logger.info("No logs found for download.")
        return None

    assets_base_path = (
        BASE_PATH / ".." / ".." / ".." / "results" / f"iteration={iteration}"
    ).resolve()

    for restart_idx, log in enumerate(logs):
        assets = log.get_assets()
        assets_path = assets_base_path / f"restart={restart_idx}"
        assets_path.mkdir(parents=True, exist_ok=True)
        for asset in assets:
            (assets_path / asset.source_path).parent.mkdir(parents=True, exist_ok=True)
            with open(assets_path / asset.source_path, "w") as fh:
                dest = pathlib.Path(fh.name)
                logger.info("Downloading log %s to %s", asset.source_path, str(dest))
                asset.download(dest)
    return assets_base_path


def extract_torchrunlogs_to_string(logs_path: pathlib.Path) -> Dict[int, List[str]]:
    logs_dict = {}

    # Iterate through all restart folders
    for restart_dir in logs_path.glob("restart=*"):
        # Find all stdout.log files
        for stdout_file in restart_dir.glob("assets/basic/*/logs/*/*/attempt_0/*/std*.log"):
            # Extract rank from path
            rank = int(stdout_file.parent.name)

            # Read log file
            try:
                with open(stdout_file) as f:
                    log_content = f.readlines()
                    if rank not in logs_dict:
                        logs_dict[rank] = log_content
                    else:
                        logs_dict[rank] += log_content
            except Exception as e:
                logger.error(f"Error reading log file {stdout_file}: {e}")
                continue
    return logs_dict


def extract_main_log_to_string(logs_path: pathlib.Path) -> List[str]:
    logs = []

    # Iterate through all restart folders
    for restart_dir in logs_path.glob("restart=*"):
        # Find all stdout.log files
        for stdout_file in restart_dir.glob(
            "assets/basic/*/jet_assets/output_logs/output_script-0.log"
        ):
            # Read log file
            try:
                with open(stdout_file) as f:
                    log_content = f.readlines()
                    logs += log_content
            except Exception as e:
                logger.error(f"Error reading log file {stdout_file}: {e}")
                continue
    return logs


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


def telemetrics_and_exit(
    success: bool, test_case: str, environment: str, pipeline_id: int, is_integration_test: bool
):
    payload = json.dumps(
        [
            {
                "pipeline_id": pipeline_id,
                "success": success,
                "test_case": test_case,
                "environment": environment,
                "is_integration_test": is_integration_test,
                "duration_seconds": (
                    pd.Timestamp.now(tz="UTC")
                    - pd.Timestamp(os.getenv("CI_JOB_STARTED_AT"), tz="UTC")
                ).total_seconds(),
                "is_merge_request": os.getenv("CI_MERGE_REQUEST_IID") is not None,
                "ci_merge_request_iid": os.getenv("CI_MERGE_REQUEST_IID"),
            }
        ]
    )
    logger.info(payload)

    if DASHBOARD_ENDPOINT is None:
        logger.info("No dashboard endpoint found, skipping telemetrics")
        return

    res = requests.post(
        DASHBOARD_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json", "Accept-Charset": "UTF-8"},
    )

    if not res.ok:
        raise requests.exceptions.HTTPError(
            f"Failed to make POST request. Received response: {res.status_code}"
        )
    sys.exit(int(not success))


@click.command()
@click.option("--model", required=True, type=str, help="Model")
@click.option("--test-case", required=True, type=str, help="Test case")
@click.option(
    "--environment", required=True, type=click.Choice(["dev", "lts"]), help="Pytorch LTS or DEV"
)
@click.option("--n-repeat", required=False, default=1, type=int)
@click.option("--time-limit", required=False, default=1800, type=int)
@click.option("--scope", required=False, default="mr", type=str)
@click.option(
    "--account",
    required=False,
    type=str,
    help="Slurm account to use",
    default="coreai_dlalgo_mcore",
)
@click.option("--partition", required=False, type=str, help="Slurm partition to use", default=None)
@click.option("--cluster", required=True, type=str, help="Cluster to run on")
@click.option("--platform", required=True, type=str, help="Platform to select")
@click.option("--container-tag", required=True, type=str, help="Base image of Mcore image")
@click.option("--container-image", required=False, type=str, help="Base image of Mcore image")
@click.option("--tag", required=False, type=str, help="Tag (only relevant for unit tests)")
@click.option("--record-checkpoints", required=False, type=str, help="Values are 'true' or 'false'")
@click.option(
    "--run-name", required=False, type=str, help="Run name (only relevant for release tests)"
)
@click.option(
    "--wandb-experiment",
    required=False,
    type=str,
    help="Wandb experiment (only relevant for release tests)",
)
@click.option(
    "--enable-lightweight-mode",
    is_flag=True,
    show_default=True,
    required=False,
    type=bool,
    default=False,
    help="Wandb experiment (only relevant for release tests)",
)
def main(
    model: str,
    test_case: str,
    environment: str,
    n_repeat: int,
    time_limit: int,
    scope: str,
    account: str,
    partition: Optional[str],
    cluster: str,
    platform: str,
    container_tag: str,
    record_checkpoints: str,
    tag: Optional[str] = None,
    container_image: Optional[str] = None,
    run_name: Optional[str] = None,
    wandb_experiment: Optional[str] = None,
    enable_lightweight_mode: bool = False,
):
    logging.basicConfig(level=logging.INFO)
    logger.info("Started")

    model_config_path = pathlib.Path(
        BASE_PATH
        / ".."
        / ".."
        / "functional_tests"
        / "test_cases"
        / model
        / test_case
        / "model_config.yaml"
    )

    if model_config_path.exists():
        with open(model_config_path) as stream:
            try:
                test_case_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        test_type = test_case_dict["TEST_TYPE"]
    else:
        test_type = "unit_test"

    logger.info("test_type will be %s", test_type)

    if test_type == "release" and (run_name is None or wandb_experiment is None):
        logger.error(f"Not all arguments provided ({run_name=}, {wandb_experiment=})")
        sys.exit(1)

    n_attempts = 0
    n_nondeterminism_attemps = 0
    n_iteration = 0
    while True and n_attempts < 3 and n_nondeterminism_attemps < 2:
        pipeline = launch_and_wait_for_completion(
            test_case=test_case,
            environment=environment,
            n_repeat=n_repeat,
            time_limit=time_limit,
            scope=scope,
            container_image=container_image,
            container_tag=container_tag,
            cluster=cluster,
            platform=platform,
            account=account,
            partition=partition,
            tag=tag,
            run_name=run_name,
            wandb_experiment=wandb_experiment,
            record_checkpoints=record_checkpoints,
            enable_lightweight_mode=enable_lightweight_mode,
        )

        n_download_attempt = 0
        while n_download_attempt < 3:
            try:
                main_job = [job for job in pipeline.get_jobs() if job.name.startswith("basic")][0]
                jet_log = main_job.get_logs()
                assets_base_path = download_job_assets(logs=jet_log, iteration=n_iteration)
                if assets_base_path is None:
                    no_log = True
                    break
                allranks_logs = extract_torchrunlogs_to_string(logs_path=assets_base_path)
                mainrank_log = extract_main_log_to_string(logs_path=assets_base_path)
                no_log = False
                break
            except (
                requests.exceptions.ConnectionError,
                json.decoder.JSONDecodeError,
                UnicodeDecodeError,
                zipfile.BadZipFile,
            ) as e:
                logger.error(e)
                time.sleep(2 * n_download_attempt * 15)
                n_download_attempt += 1
                no_log = True
            except Exception as e:
                logger.error(e)
                no_log = True
                n_download_attempt += 1
                break
            
            n_download_attempt += 1

        if no_log:
            logger.error("Did not find any logs to download, retry.")
            n_attempts += 1
            continue

        concat_allranks_logs = "\n".join(
            ["\n".join(log_lines) for log_lines in allranks_logs.values()]
        )
        concat_mainrank_log = "\n".join(mainrank_log)
        if concat_allranks_logs.strip() == "":
            logger.error("No logs found. Try again.")
            n_attempts += 1
            continue

        if test_type != "release":
            print(f"Logs:\n{concat_mainrank_log}")

        n_status_attempts = 0
        status = None
        while n_status_attempts < 3:
            try:
                status = pipeline.get_status()
                break
            except jetclient.clients.gitlab.GitlabAPIError:
                logging.info("Could not fetch status, try again")
                time.sleep(2 * n_status_attempts * 15)
                n_status_attempts += 1

        if status is None:
            continue

        success = status == PipelineStatus.SUCCESS
        logger.info("Pipeline terminated with status %s", status.name)

        if test_type == "unit_test":
            if (
                "The server socket has failed to listen on any local network address."
                in concat_allranks_logs
            ):
                logger.error("TCP error, attempt restart.")
                n_attempts += 1
                continue

            sys.exit(int(not success))  # invert for exit 0

        if test_type != "release":
            if success:
                telemetrics_and_exit(
                    success=True,
                    test_case=test_case,
                    environment=environment,
                    pipeline_id=int(os.getenv("PARENT_PIPELINE_ID", 0)),
                    is_integration_test=enable_lightweight_mode,
                )

            if (
                "The server socket has failed to listen on any local network address."
                in concat_allranks_logs
                or "Some NCCL operations have failed or timed out." in concat_allranks_logs
                or "uncorrectable ECC error encountered" in concat_allranks_logs
                or "illegal memory access" in concat_allranks_logs
                or "illegal instruction" in concat_allranks_logs
                or "torch.distributed.DistNetworkError" in concat_allranks_logs
                or "Segmentation fault" in concat_allranks_logs
                or "found NaN in local forward loss calculation" in concat_allranks_logs
                or "For debugging consider passing CUDA_LAUNCH_BLOCKING=1" in concat_allranks_logs
            ):
                logger.error("Detected NCCL failure, attempt restart.")
                n_attempts += 1
                continue

            if "FAILED tests/functional_tests/python_test_utils" in concat_mainrank_log:
                logger.error("Non-determinism, let's try another node.")
                n_nondeterminism_attemps += 1
                continue

            telemetrics_and_exit(
                success=False,
                test_case=test_case,
                environment=environment,
                pipeline_id=int(os.getenv("PARENT_PIPELINE_ID", 0)),
                is_integration_test=enable_lightweight_mode,
            )

        if parse_failed_job(logs=mainrank_log):
            n_attempts += 1
            continue

        if parse_finished_training(logs=mainrank_log):
            sys.exit(int(not success))  # invert for exit 0
        n_iteration += 1

    telemetrics_and_exit(
        success=False,
        test_case=test_case,
        environment=environment,
        pipeline_id=int(os.getenv("PARENT_PIPELINE_ID", 0)),
        is_integration_test=enable_lightweight_mode,
    )


if __name__ == "__main__":
    main()
