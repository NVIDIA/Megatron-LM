from typing import Optional

import click
import yaml

from tests.functional_tests.python_test_utils.jet import common


@click.command()
@click.option("--scope", required=True, type=str, help="Test scope")
@click.option("--a100-cluster", required=True, type=str, help="A100 Cluster to run on")
@click.option("--h100-cluster", required=True, type=str, help="H100 Cluster to run on")
@click.option("--output-path", required=True, type=str, help="Path to write GitLab job to")
@click.option("--container-image", required=True, type=str, help="Container tag to use")
@click.option("--container-tag", required=True, type=str, help="Container tag to use")
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
    scope: str,
    a100_cluster: str,
    h100_cluster: str,
    output_path: str,
    container_image: str,
    container_tag: str,
    run_name: Optional[str] = None,
    wandb_experiment: Optional[str] = None,
):

    gitlab_pipeline = {"stages": ["functional_tests"], "default": {"interruptible": True}}

    for test_case in common.load_workloads(scope=scope, container_tag=container_tag):
        if test_case.type == "build":
            continue

        if test_case.spec.platforms == "dgx_a100":
            cluster = a100_cluster
        elif test_case.spec.platforms == "dgx_h100":
            cluster = h100_cluster
        else:
            raise ValueError(f"Platform {test_case.spec.platforms} unknown")

        script = [
            "export PYTHONPATH=$(pwd); "
            "python tests/functional_tests/python_test_utils/jet/launch_jet_workload.py",
            f"--model {test_case.spec.model}",
            f"--test-case {test_case.spec.test_case}",
            f"--container-tag {container_tag}",
            f"--cluster {cluster}",
        ]

        if run_name is not None and wandb_experiment is not None:
            script.append(f"--run-name {run_name}")
            script.append(f"--wandb-experiment {wandb_experiment}")

        gitlab_pipeline[test_case.spec.test_case] = {
            "stage": "functional_tests",
            "image": f"{container_image}:{container_tag}",
            "tags": ["mcore-docker-node-jet"],
            "rules": [
                {"if": '$CI_PIPELINE_SOURCE == "parent_pipeline"'},
                {"if": '$CI_MERGE_REQUEST_ID'},
            ],
            "timeout": "7 days",
            "needs": [{"pipeline": '$PARENT_PIPELINE_ID', "job": "jet-generate"}],
            "script": [" ".join(script)],
        }

    with open(output_path, 'w') as outfile:
        yaml.dump(gitlab_pipeline, outfile, default_flow_style=False)


if __name__ == "__main__":
    main()
