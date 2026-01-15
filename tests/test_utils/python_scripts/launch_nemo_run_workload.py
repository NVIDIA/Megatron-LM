# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
import os
import pathlib
import sys
from typing import Optional

import click
import nemo_run as run

from tests.test_utils.python_scripts import recipe_parser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_flaky_failure(concat_allranks_logs: str) -> bool:
    """Assumes that certain keywords hint towards intermittent failures"""

    return (
        "The server socket has failed to listen on any local network address."
        in concat_allranks_logs
        or "Some NCCL operations have failed or timed out." in concat_allranks_logs
        or "uncorrectable ECC error encountered" in concat_allranks_logs
        or "illegal memory access" in concat_allranks_logs
        or "illegal instruction" in concat_allranks_logs
        or "torch.distributed.DistNetworkError" in concat_allranks_logs
        or "Segmentation fault" in concat_allranks_logs
        or "found NaN in" in concat_allranks_logs
        or "For debugging consider passing CUDA_LAUNCH_BLOCKING=1" in concat_allranks_logs
        or "double free or corruption" in concat_allranks_logs
        or "Call to CUDA function failed." in concat_allranks_logs
        or "Connection reset by peer" in concat_allranks_logs
        or "invalid pointer" in concat_allranks_logs
        or "malloc(): unaligned tcache chunk detected" in concat_allranks_logs
        or "zmq.error.ZMQError: Address already in use" in concat_allranks_logs
        or "We couldn't connect to 'https://huggingface.co'" in concat_allranks_logs
        or "Unpack failed: incomplete input" in concat_allranks_logs
        or "unspecified launch failure" in concat_allranks_logs
        or "free(): corrupted unsorted chunks" in concat_allranks_logs
        or "Segfault encountered" in concat_allranks_logs
    )


@click.command()
@click.option("--scope", required=True, type=str, help="Scope of the workload")
@click.option("--model", required=True, type=str, help="Model of the workload")
@click.option("--test-case", required=True, type=str, help="Test case of the workload")
@click.option("--environment", required=True, type=str, help="Environment of the workload")
@click.option("--platform", required=True, type=str, help="Platform of the workload")
@click.option("--container-image", required=True, type=str, help="Container image of the workload")
@click.option("--data-dir", required=False, type=str, help="Data directory of the workload")
@click.option("--tag", required=False, type=str, help="Tag of the workload")
@click.option(
    "--enable-lightweight-mode",
    is_flag=True,
    show_default=True,
    required=False,
    type=bool,
    default=False,
    help="To enable lightweight mode",
)
def main(
    scope,
    model,
    test_case,
    environment,
    platform,
    container_image,
    data_dir: Optional[str] = None,
    tag: Optional[str] = None,
    enable_lightweight_mode: Optional[bool] = False,
):
    workloads = recipe_parser.load_workloads(
        container_image="none",
        scope=scope,
        model=model,
        test_case=test_case,
        environment=environment,
        container_tag="none",
        platform=platform,
        tag=tag,
    )

    workloads = [workload for workload in workloads if workload.type != "build"]

    assert len(workloads) == 1, f"Expected exactly one workload, got {len(workloads)}"

    workload = workloads[0]
    magic_values = dict(workload.spec)
    magic_values["assets_dir"] = "/opt/megatron-lm/assets_dir"
    magic_values["artifacts_dir"] = "/opt/megatron-lm/artifacts_dir"
    magic_values["environment"] = environment
    magic_values["test_case"] = workload.spec["test_case"]
    magic_values["name"] = workload.spec["name"].format(**magic_values)
    workload.spec["script"] = workload.spec["script"].format(**magic_values)

    inline_script = run.Script(inline=workload.spec["script"])

    artifacts = []
    artifacts.append(f"{os.getcwd()}:/opt/megatron-lm")
    if data_dir:
        artifacts.append(f"{pathlib.Path(data_dir)}:/mnt/artifacts")

    executor = run.DockerExecutor(
        container_image=container_image,
        num_gpus=-1,
        runtime="nvidia",
        ipc_mode="host",
        shm_size="30g",
        env_vars={
            "PYTHONUNBUFFERED": "1",
            "OUTPUT_PATH": os.getcwd(),
            "ENABLE_LIGHTWEIGHT_MODE": str(enable_lightweight_mode).lower(),
            "N_REPEAT": "1",
            "CLUSTER": "dgxh100_dgxc",
            "NCCL_DEBUG": "INFO",
        },
        packager=run.Packager(),
        volumes=artifacts,
    )

    n_attempts = 0
    while n_attempts < 3:
        with run.Experiment("mcore-ci-test", executor=executor, log_level="INFO") as exp:
            _ = exp.add([inline_script], tail_logs=False, name="task-1")

            exp.dryrun(log=True)
            exp.run(detach=False, tail_logs=True, sequential=False)

        result_dict = exp.status(return_dict=True)
        _, job_dict = list(result_dict.items())[0]
        succeeded = str(job_dict["status"]) == "SUCCEEDED"

        if succeeded:
            logger.info(f"Job succeeded with status: {job_dict["status"]}")
            sys.exit(0)

        logger.error(f"Job failed with status: {job_dict["status"]}")
        log_file_paths = pathlib.Path(os.getcwd()).glob("assets_dir/logs/*/*/attempt_0/*/std*.log")
        all_ranks_all_logs = []
        for log_file_path in log_file_paths:
            with open(log_file_path, "r") as f:
                all_logs = f.readlines()
            all_ranks_all_logs.extend(all_logs)
        all_ranks_all_logs_string = "\n".join(all_ranks_all_logs)
        if is_flaky_failure(all_ranks_all_logs_string):
            logger.warning("Detected flaky failure, attempt restart.")
            n_attempts += 1
            continue

        sys.exit(1)

    sys.exit(1)


if __name__ == "__main__":
    main()
