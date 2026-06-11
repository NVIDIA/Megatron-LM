# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pathlib
from typing import Optional

import click
import yaml

from tests.test_utils.python_scripts import recipe_parser

BASE_PATH = pathlib.Path(__file__).parent.resolve()


@click.command()
@click.option("--scope", required=True, type=str, help="Test scope")
@click.option("--environment", required=True, type=str, help="LTS or dev features")
@click.option("--n-repeat", required=False, default=1, type=int)
@click.option("--time-limit", required=False, default=1, type=int)
@click.option(
    "--test-cases", required=True, type=str, help="Comma-separated list of test_cases, or 'all'"
)
@click.option("--platform", required=True, type=str, help="Platform to select")
@click.option("--cluster", required=True, type=str, help="Cluster to run on")
@click.option("--partition", required=False, type=str, help="Slurm partition to use", default=None)
@click.option("--output-path", required=True, type=str, help="Path to write GitLab job to")
@click.option("--container-image", required=True, type=str, help="LTS Container image to use")
@click.option("--container-tag", required=True, type=str, help="Container tag to use")
@click.option(
    "--workload-local-image-path",
    required=False,
    type=str,
    help=(
        "Local SquashFS image path/template to use for JET basic workloads via "
        "spec.image_source.local_path. Skips the JET build workload."
    ),
)
@click.option(
    "--dependent-job",
    required=True,
    type=str,
    help="Name of job that created the downstream pipeline",
)
@click.option("--record-checkpoints", required=False, type=str, help="Values are 'true' or 'false'")
@click.option("--slurm-account", required=True, type=str, help="Slurm account to use")
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
    "--enable-warmup/--no-enable-warmup",
    required=False,
    is_flag=True,
    default=True,
    type=bool,
    help="Run one job as dependency to others as to warm up cache",
)
@click.option(
    "--cadence",
    required=False,
    type=str,
    default=None,
    help=(
        "Trigger cadence to filter tests by (pr|nightly|mergegroup). "
        "Empty/unset disables the cadence filter."
    ),
)
def main(
    scope: str,
    environment: str,
    n_repeat: int,
    time_limit: int,
    test_cases: str,
    platform: Optional[str],
    cluster: Optional[str],
    partition: Optional[str],
    output_path: str,
    container_image: str,
    container_tag: str,
    workload_local_image_path: Optional[str],
    dependent_job: str,
    record_checkpoints: str,
    slurm_account: str,
    tag: Optional[str] = None,
    run_name: Optional[str] = None,
    wandb_experiment: Optional[str] = None,
    enable_lightweight_mode: bool = False,
    enable_warmup: Optional[bool] = None,
    cadence: Optional[str] = None,
):
    # Treat empty string as "no cadence filter" so callers can wire shell
    # variables in directly without conditional flag emission.
    cadence_arg = cadence or None

    list_of_test_cases = [
        test_case
        for test_case in recipe_parser.load_workloads(
            scope=scope,
            container_tag=container_tag,
            workload_local_image_path=workload_local_image_path,
            environment=environment,
            test_cases=test_cases,
            platform=platform,
            tag=tag,
            cadence=cadence_arg,
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
            "workflow": {
                "rules": [
                    {"if": '$CI_PIPELINE_SOURCE == "parent_pipeline" || $CI_MERGE_REQUEST_ID'},
                    {"when": "never"},
                ]
            },
            "default": {"interruptible": True},
            "empty-pipeline-placeholder-job": {
                "stage": "empty-pipeline-placeholder",
                "image": f"{container_image}:{container_tag}",
                "tags": tags,
                "timeout": "7 days",
                "needs": [{"pipeline": '$PARENT_PIPELINE_ID', "job": dependent_job}],
                "script": ["sleep 1"],
                "artifacts": {"paths": ["results/"], "when": "always"},
                "retry": {
                    "max": 2,
                    "when": [
                        "unknown_failure",
                        "stuck_or_timeout_failure",
                        "runner_system_failure",
                    ],
                },
            },
        }

    else:
        list_of_test_cases = sorted(list_of_test_cases, key=lambda x: x["spec"]["model"])
        prepare_job_name = "prepare_sqsh_image" if workload_local_image_path is not None else None
        model_stages = sorted(
            list(set([test_case["spec"]["model"] for test_case in list_of_test_cases]))
        )
        stages = model_stages
        if prepare_job_name is not None:
            stages = ["prepare_sqsh"] + stages

        gitlab_pipeline = {
            "stages": stages,
            "workflow": {
                "rules": [
                    {
                        "if": '($CI_PIPELINE_SOURCE == "parent_pipeline" || $CI_MERGE_REQUEST_ID) && $CI_COMMIT_BRANCH == "main"',
                        "auto_cancel": {"on_new_commit": "interruptible"},
                    },
                    {"if": '$CI_PIPELINE_SOURCE == "parent_pipeline" || $CI_MERGE_REQUEST_ID'},
                    {"when": "never"},
                ],
                "auto_cancel": {"on_new_commit": "interruptible"},
            },
            "default": {
                "interruptible": True,
                "retry": {"max": 2, "when": "runner_system_failure"},
            },
        }

        if prepare_job_name is not None:
            prepare_cluster = recipe_parser.resolve_local_image_prepare_cluster(cluster)
            prepare_tags = list(tags)
            prepare_tags.append(f"cluster/{recipe_parser.resolve_cluster_config(prepare_cluster)}")
            prepare_script = [
                "export PYTHONPATH=$(pwd); "
                "python tests/test_utils/python_scripts/prepare_jet_sqsh_image.py",
                f"--scope {scope}",
                f"--environment {environment}",
                f"--test-cases '{test_cases}'",
                f"--platform {platform}",
                f"--cluster {cluster}",
                f"--container-tag {container_tag}",
                f"--workload-local-image-path '{workload_local_image_path}'",
                f"--account {slurm_account}",
                "--time-limit 1800",
            ]

            if tag is not None:
                prepare_script.append(f"--tag {tag}")

            if partition is not None and prepare_cluster == cluster:
                prepare_script.append(f"--partition {partition}")

            if cadence_arg is not None:
                prepare_script.append(f"--cadence {cadence_arg}")

            gitlab_pipeline[prepare_job_name] = {
                "stage": "prepare_sqsh",
                "image": f"{container_image}:{container_tag}",
                "tags": prepare_tags,
                "timeout": "7 days",
                "needs": [{"pipeline": '$PARENT_PIPELINE_ID', "job": dependent_job}],
                "script": [" ".join(prepare_script)],
                "artifacts": {"paths": ["results/"], "when": "always"},
                "retry": {
                    "max": 2,
                    "when": [
                        "unknown_failure",
                        "stuck_or_timeout_failure",
                        "runner_system_failure",
                    ],
                },
            }

        warmup_job = ""

        for test_idx, test_case in enumerate(list_of_test_cases):
            job_tags = list(tags)
            job_tags.append(f"cluster/{recipe_parser.resolve_cluster_config(cluster)}")

            script = [
                "export PYTHONPATH=$(pwd); "
                "python tests/test_utils/python_scripts/launch_jet_workload.py",
                f"--model {test_case['spec']['model']}",
                f"--environment {test_case['spec']['environment']}",
                f"--n-repeat {n_repeat}",
                f"--time-limit {time_limit}",
                f"--scope {scope}",
                f"--test-case '{test_case['spec']['test_case']}'",
                f"--container-tag {container_tag}",
                f"--cluster {cluster}",
                f"--platform {platform}",
                f"--record-checkpoints {record_checkpoints}",
                f"--account {slurm_account}",
            ]

            if workload_local_image_path is not None:
                script.append(f"--workload-local-image-path '{workload_local_image_path}'")

            if partition is not None:
                script.append(f"--partition {partition}")

            if tag is not None:
                script.append(f"--tag {tag}")

            if enable_lightweight_mode is True:
                script.append("--enable-lightweight-mode")

            if run_name is not None and wandb_experiment is not None:
                script.append(f"--run-name {run_name}")
                script.append(
                    f"--wandb-experiment {wandb_experiment}-{test_case['spec']['model']}-{test_case['spec']['test_case']}"
                )

            needs = [{"pipeline": '$PARENT_PIPELINE_ID', "job": dependent_job}]

            if prepare_job_name is not None:
                needs.append({"job": prepare_job_name})

            if enable_warmup:
                if test_idx == 0:
                    warmup_job = test_case['spec']['test_case']
                elif warmup_job != "":
                    needs.append({"job": warmup_job})

            gitlab_pipeline[test_case['spec']['test_case']] = {
                "stage": f"{test_case['spec']['model']}",
                "image": f"{container_image}:{container_tag}",
                "tags": job_tags,
                "timeout": "7 days",
                "needs": needs,
                "script": [" ".join(script)],
                "artifacts": {"paths": ["results/"], "when": "always"},
                "allow_failure": test_case["spec"].get("allow_failure", False)
                or test_case["spec"]["model"] == "gpt-nemo",
                "retry": {
                    "max": 2,
                    "when": [
                        "unknown_failure",
                        "stuck_or_timeout_failure",
                        "runner_system_failure",
                    ],
                },
            }

    with open(output_path, 'w') as outfile:
        yaml.dump(gitlab_pipeline, outfile, default_flow_style=False)


if __name__ == "__main__":
    main()
