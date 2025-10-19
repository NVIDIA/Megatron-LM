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


@click.command()
@click.option("--scope", required=True, type=str, help="Scope of the workload")
@click.option("--model", required=True, type=str, help="Model of the workload")
@click.option("--test-case", required=True, type=str, help="Test case of the workload")
@click.option("--environment", required=True, type=str, help="Environment of the workload")
@click.option("--platform", required=True, type=str, help="Platform of the workload")
@click.option("--container-image", required=True, type=str, help="Container image of the workload")
@click.option("--data-dir", required=False, type=str, help="Data directory of the workload")
@click.option("--tag", required=False, type=str, help="Tag of the workload")
def main(
    scope,
    model,
    test_case,
    environment,
    platform,
    container_image,
    data_dir: Optional[str] = None,
    tag: Optional[str] = None,
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
            "ENABLE_LIGHTWEIGHT_MODE": "true",
            "N_REPEAT": "1",
        },
        packager=run.Packager(),
        volumes=artifacts,
    )
    with run.Experiment("mcore-ci-test", executor=executor, log_level="INFO") as exp:
        _ = exp.add([inline_script], tail_logs=False, name="task-1")

        exp.dryrun(log=True)
        exp.run(detach=False, tail_logs=True, sequential=False)

    result_dict = exp.status(return_dict=True)
    _, job_dict = list(result_dict.items())[0]

    logger.info(f"Job status: {job_dict["status"]}")
    sys.exit(0 if str(job_dict["status"]) == "SUCCEEDED" else 1)


if __name__ == "__main__":
    main()
