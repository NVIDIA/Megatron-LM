import pathlib
from typing import Optional

import click
import yaml

from tests.test_utils.python_scripts import common

BASE_PATH = pathlib.Path(__file__).parent.resolve()


@click.command()
@click.option("--scope", required=True, type=str, help="Test scope")
@click.option("--environment", required=True, type=str, help="LTS or dev features")
@click.option("--n-repeat", required=False, default=1, type=int)
@click.option("--time-limit", required=False, default=1, type=int)
@click.option(
    "--test-cases", required=True, type=str, help="Comma-separated list of test_cases, or 'all'"
)
@click.option("--a100-cluster", required=True, type=str, help="A100 Cluster to run on")
@click.option("--h100-cluster", required=True, type=str, help="H100 Cluster to run on")
@click.option(
    "--a100-partition", required=False, type=str, help="Slurm partition to use", default=None
)
@click.option(
    "--h100-partition", required=False, type=str, help="Slurm partition to use", default=None
)
@click.option("--output-path", required=True, type=str, help="Path to write GitLab job to")
@click.option("--container-image", required=True, type=str, help="LTS Container image to use")
@click.option("--container-tag", required=True, type=str, help="Container tag to use")
@click.option(
    "--dependent-job",
    required=True,
    type=str,
    help="Name of job that created the downstream pipeline",
)
@click.option("--record-checkpoints", required=False, type=str, help="Values are 'true' or 'false'")
@click.option("--tag", required=False, type=str, help="Tag (only relevant for unit tests)")
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
    environment: str,
    n_repeat: int,
    time_limit: int,
    test_cases: str,
    a100_cluster: str,
    h100_cluster: str,
    a100_partition: Optional[str],
    h100_partition: Optional[str],
    output_path: str,
    container_image: str,
    container_tag: str,
    dependent_job: str,
    record_checkpoints: str,
    tag: Optional[str] = None,
    run_name: Optional[str] = None,
    wandb_experiment: Optional[str] = None,
):
    list_of_test_cases = [
        test_case
        for test_case in common.load_workloads(
            scope=scope,
            container_tag=container_tag,
            environment=environment,
            test_cases=test_cases,
            tag=tag,
        )
        if test_case.type != "build"
    ]

    tags = [
        "arch/amd64",
        "env/prod",
        "origin/jet-fleet",
        "owner/jet-core",
        "purpose/jet-client",
        "team/megatron",
    ]

    if not list_of_test_cases:
        gitlab_pipeline = {
            "stages": ["empty-pipeline-placeholder"],
            "default": {"interruptible": True},
            "empty-pipeline-placeholder-job": {
                "stage": "empty-pipeline-placeholder",
                "image": f"{container_image}:{container_tag}",
                "tags": tags,
                "rules": [
                    {"if": '$CI_PIPELINE_SOURCE == "parent_pipeline"'},
                    {"if": '$CI_MERGE_REQUEST_ID'},
                ],
                "timeout": "7 days",
                "needs": [{"pipeline": '$PARENT_PIPELINE_ID', "job": "functional:configure"}],
                "script": ["sleep 1"],
                "artifacts": {"paths": ["results/"], "when": "always"},
            },
        }

    else:
        gitlab_pipeline = {
            "stages": list(set([test_case.spec.model for test_case in list_of_test_cases])),
            "default": {
                "interruptible": True,
                "retry": {"max": 2, "when": "runner_system_failure"},
            },
        }

        for test_case in list_of_test_cases:
            if test_case.spec.platforms == "dgx_a100":
                cluster = a100_cluster
                partition = a100_partition
            elif test_case.spec.platforms == "dgx_h100":
                cluster = h100_cluster
                partition = h100_partition
            else:
                raise ValueError(f"Platform {test_case.spec.platforms} unknown")

            job_tags = list(tags)
            job_tags.append(f"cluster/{common.resolve_cluster_config(cluster)}")

            script = [
                "export PYTHONPATH=$(pwd); "
                "python tests/test_utils/python_scripts/launch_jet_workload.py",
                f"--model {test_case.spec.model}",
                f"--environment {test_case.spec.environment}",
                f"--n-repeat {n_repeat}",
                f"--time-limit {time_limit}",
                f"--scope {scope}",
                f"--test-case '{test_case.spec.test_case}'",
                f"--container-tag {container_tag}",
                f"--cluster {cluster}",
                f"--record-checkpoints {record_checkpoints}",
            ]

            if partition is not None:
                script.append(f"--partition {partition}")

            if tag is not None:
                script.append(f"--tag {tag}")

            if run_name is not None and wandb_experiment is not None:
                script.append(f"--run-name {run_name}")
                test_case.spec.model
                script.append(
                    f"--wandb-experiment {wandb_experiment}-{test_case.spec.model}-{test_case.spec.test_case}"
                )

            gitlab_pipeline[test_case.spec.test_case] = {
                "stage": f"{test_case.spec.model}",
                "image": f"{container_image}:{container_tag}",
                "tags": job_tags,
                "rules": [
                    {"if": '$CI_PIPELINE_SOURCE == "parent_pipeline"'},
                    {"if": '$CI_MERGE_REQUEST_ID'},
                ],
                "timeout": "7 days",
                "needs": [{"pipeline": '$PARENT_PIPELINE_ID', "job": dependent_job}],
                "script": [" ".join(script)],
                "artifacts": {"paths": ["results/"], "when": "always"},
            }

    with open(output_path, 'w') as outfile:
        yaml.dump(gitlab_pipeline, outfile, default_flow_style=False)


if __name__ == "__main__":
    main()
