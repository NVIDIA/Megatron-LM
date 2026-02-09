import os
import pathlib

import click
import nemo_run as run

from tests.test_utils.python_scripts import common


@click.command()
@click.option("--scope", required=True, type=str, help="Scope of the workload")
@click.option("--model", required=True, type=str, help="Model of the workload")
@click.option("--test-case", required=True, type=str, help="Test case of the workload")
@click.option("--environment", required=True, type=str, help="Environment of the workload")
@click.option("--platform", required=True, type=str, help="Platform of the workload")
def main(scope, model, test_case, environment, platform):
    workloads = common.load_workloads(
        container_image="none",
        scope=scope,
        model=model,
        test_case=test_case,
        environment=environment,
        container_tag="none",
        platform=platform,
    )

    workloads = [workload for workload in workloads if workload.type != "build"]

    print(workloads)
    assert len(workloads) == 1, f"Expected exactly one workload, got {len(workloads)}"

    workload = workloads[0]
    magic_values = dict(workload.spec)
    magic_values["assets_dir"] = "$OUTPUT_PATH"
    magic_values["artifacts_dir"] = "$OUTPUT_PATH"
    magic_values["environment"] = environment
    magic_values["test_case"] = workload.spec["test_case"]
    magic_values["name"] = workload.spec["name"].format(**magic_values)
    workload.spec["script"] = workload.spec["script"].format(**magic_values)

    inline_script = run.Script(inline=workload.spec["script"])

    artifacts = [
        "{host_path}:{mount_path}".format(
            mount_path=mount_path, host_path=str(pathlib.Path("/root") / host_path)
        )
        for mount_path, host_path in workload.spec["artifacts"].items()
    ]
    artifacts.append(f"{os.getcwd()}:/opt/megatron-lm")
    print(artifacts)

    executor = run.DockerExecutor(
        container_image="megatron-core",
        num_gpus=-1,
        runtime="nvidia",
        ipc_mode="host",
        shm_size="30g",
        env_vars={
            "PYTHONUNBUFFERED": "1",
            "OUTPUT_PATH": os.getcwd(),
            "ENABLE_LIGHTWEIGHT_MODE": "true",
        },
        packager=run.Packager(),
        volumes=artifacts,
    )
    with run.Experiment("docker-experiment", executor=executor, log_level="INFO") as exp:
        _ = exp.add([inline_script], tail_logs=False, name="task-1")

        exp.run(detach=False, tail_logs=True, sequential=False)


if __name__ == "__main__":
    main()
