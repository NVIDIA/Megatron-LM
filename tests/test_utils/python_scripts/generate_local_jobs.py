"""Generate launch scripts for local execution.

This script allows to generate pre-filled launch scripts that allow for local execution of Megatron-LM functional tests inside containerized enviroments (i.e. Slurm enroot or Docker).

This script will generate scripts into `$(pwd)/test_cases`.
"""

import pathlib
from typing import Optional

import click
import jetclient
import yaml

from tests.test_utils.python_scripts import common


def load_script(config_path: str) -> str:
    with open(config_path) as stream:
        try:
            return jetclient.JETWorkloadManifest(**yaml.safe_load(stream)).spec.script
        except yaml.YAMLError as exc:
            raise exc


@click.command()
@click.option("--model", required=False, type=str, help="Filters all tests by matching model")
@click.option("--scope", required=False, type=str, help="Filters all tests by matching scope")
@click.option(
    "--test-case", required=False, type=str, help="Returns a single test-case with matching name."
)
@click.option(
    "--environment",
    required=True,
    type=str,
    help="Pass 'lts' for PyTorch 24.01 and 'dev' for a more recent version.",
)
@click.option(
    "--output-path",
    required=True,
    type=str,
    help="Directory where the functional test will write its artifacts to (Tensorboard logs)",
    default="/opt/megatron-lm",
)
@click.option(
    "--enable-lightweight-mode",
    is_flag=True,
    show_default=True,
    required=False,
    type=bool,
    default=False,
    help="Run 2-step smoke tests instead of full training",
)
@click.option(
    "--record-checkpoints",
    is_flag=True,
    show_default=True,
    required=False,
    type=bool,
    default=False,
    help="Save checkpoints, do not run pytest",
)
def main(
    model: Optional[str],
    scope: Optional[str],
    test_case: Optional[str],
    environment: str,
    output_path: str,
    enable_lightweight_mode: bool = False,
    record_checkpoints: bool = False,
):
    workloads = common.load_workloads(
        container_image='none',
        scope=scope,
        model=model,
        test_case=test_case,
        environment=environment,
        container_tag='none',
    )

    for workload in workloads:
        if workload.type == "build":
            continue
        magic_values = dict(workload.spec)
        magic_values["assets_dir"] = "$OUTPUT_PATH"
        magic_values["artifacts_dir"] = "$OUTPUT_PATH"
        magic_values["environment"] = environment
        magic_values["test_case"] = workload.spec.test_case
        magic_values["name"] = workload.spec.name.format(**magic_values)

        file_path = (
            pathlib.Path.cwd()
            / "test_cases"
            / workload.spec.model
            / f"{workload.spec.test_case}.sh"
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write(f"export ENABLE_LIGHTWEIGHT_MODE={str(enable_lightweight_mode).lower()}\n")
            fh.write(f"export RECORD_CHECKPOINTS={str(record_checkpoints).lower()}\n")
            fh.write(
                f'export OUTPUT_PATH={output_path}/runs/$(python3 -c "import uuid; print(uuid.uuid4())")\n'
            )
            fh.write(workload.spec.script.format(**magic_values))


if __name__ == "__main__":
    main()
