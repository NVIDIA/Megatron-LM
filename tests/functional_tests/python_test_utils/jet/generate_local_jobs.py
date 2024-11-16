"""Generate launch scripts for local execution.

This script allows to generate pre-filled launch scripts that allow for local execution of Megatron-LM functional tests inside containerized enviroments (i.e. Slurm enroot or Docker).

This script will generate scripts into `$(pwd)/test_cases`.
"""

import pathlib
from typing import Optional

import click
import jetclient
import yaml

from tests.functional_tests.python_test_utils.jet import common


def load_script(config_path: str) -> str:
    with open(config_path) as stream:
        try:
            jetclient.JETWorkloadManifest(**yaml.safe_load(stream)).spec.script
        except yaml.YAMLError as exc:
            raise exc


@click.command()
@click.option("--model", required=False, type=str, help="Filters all tests by matching model")
@click.option("--scope", required=False, type=str, help="Filters all tests by matching scope")
@click.option(
    "--test-case", required=False, type=str, help="Returns a single test-case with matching name."
)
@click.option(
    "--output-path",
    required=True,
    type=str,
    help="Directory where the functional test will write its artifacts to (Tensorboard logs)",
    default="/opt/megatron-lm",
)
def main(model: Optional[str], scope: Optional[str], test_case: Optional[str], output_path: str):
    workloads = common.load_workloads(
        container_image='none', scope=scope, model=model, test_case=test_case, container_tag='none'
    )

    for workload in workloads:
        if workload.type == "build":
            continue
        magic_values = dict(workload.spec)
        magic_values["assets_dir"] = output_path

        file_path = (
            pathlib.Path.cwd()
            / "test_cases"
            / workload.spec.model
            / f"{workload.spec.test_case}.sh"
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write(workload.spec.script.format(**magic_values))


if __name__ == "__main__":
    main()
