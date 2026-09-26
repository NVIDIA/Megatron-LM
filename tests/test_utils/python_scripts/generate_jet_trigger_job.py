# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pathlib
import subprocess
from typing import Optional

import click
import yaml

from tests.test_utils.python_scripts import recipe_parser

BASE_PATH = pathlib.Path(__file__).parent.resolve()
TRIAGE_LOG_PATH = "jet_workload.log"
TRIAGE_REPORT_PATH = "error_report.json"
TEST_CASES = pathlib.PurePosixPath("tests/functional_tests/test_cases")
RECIPES = pathlib.PurePosixPath("tests/test_utils/recipes")
GITHUB_SCOPES = ("L0", "L1")
GPUS_PER_RUNNER = {"dgx_h100": 8, "dgx_gb200": 4}


def _git(repo_root: pathlib.Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo_root, text=True)


def _recipe_at_ref(repo_root: pathlib.Path, ref: str, path: str) -> list[recipe_parser.dotdict]:
    manifest = recipe_parser.dotdict(yaml.safe_load(_git(repo_root, "show", f"{ref}:{path}")))
    return recipe_parser.set_build_dependency(
        recipe_parser.flatten_workload(recipe_parser.flatten_products(manifest))
    )


def _workload_key(workload: recipe_parser.dotdict) -> tuple:
    return tuple(
        workload.spec.get(field)
        for field in ("model", "test_case", "scope", "environment", "platforms")
    )


def _changed_workloads(
    repo_root: pathlib.Path, base_ref: str, platform: str, head_ref: str
) -> list[recipe_parser.dotdict]:
    # Identify changes on the PR branch, excluding changes from the target
    # branch that are only present in the merge commit being tested.
    base = _git(repo_root, "merge-base", base_ref, head_ref).strip()
    changes = _git(
        repo_root,
        "diff",
        "--name-status",
        "--no-renames",
        "-z",
        base,
        head_ref,
        "--",
        str(TEST_CASES),
        str(RECIPES),
    ).split("\0")
    changed_files = dict(zip(changes[1::2], changes[0::2]))
    changed_cases = set()
    for filename in changed_files:
        path = pathlib.PurePosixPath(filename)
        if path.is_relative_to(TEST_CASES):
            parts = path.relative_to(TEST_CASES).parts
            if len(parts) >= 3:
                changed_cases.add((parts[0], parts[1]))

    changed = []
    for recipe_path in sorted((repo_root / RECIPES).glob("**/*.yaml")):
        workloads = recipe_parser.load_and_flatten(str(recipe_path))
        relative_path = recipe_path.relative_to(repo_root).as_posix()
        changed_rows = set()
        if changed_files.get(relative_path) not in (None, "D"):
            previous = (
                []
                if changed_files[relative_path] == "A"
                else _recipe_at_ref(repo_root, base, relative_path)
            )
            changed_rows = {
                _workload_key(workload)
                for workload in _recipe_at_ref(repo_root, head_ref, relative_path)
                if workload not in previous
            }

        for workload in workloads:
            spec = workload.spec
            # GitLab-only and disabled scopes must never become GitHub jobs.
            # Keep the same environment/platform availability as the default matrix.
            if (
                workload.type == "build"
                or spec.get("model") == "unit-tests"
                or spec.get("scope") not in GITHUB_SCOPES
                or spec.get("environment") != "dev"
                or spec.get("platforms") != platform
                # The GitHub launcher uses one DockerExecutor, unlike JET's
                # multi-node launch path used by some nightly recipes.
                or spec.get("nodes", 1) != 1
                or spec.get("gpus", GPUS_PER_RUNNER[platform]) > GPUS_PER_RUNNER[platform]
            ):
                continue
            case = (spec["model"], spec["test_case"])
            if not (repo_root / TEST_CASES / case[0] / case[1]).is_dir():
                # Exclude deleted cases. Recipes can launch Python tests directly
                # without using the model_config.yaml training harness.
                continue
            if case in changed_cases or _workload_key(workload) in changed_rows:
                changed.append(workload)
    return changed


def build_test_script(command: str) -> str:
    """Wrap a workload command with non-blocking error extraction."""
    return "\n".join(
        [
            "set +e",
            "set -o pipefail",
            f"{command} 2>&1 | tee {TRIAGE_LOG_PATH}",
            'exit_code=${PIPESTATUS[0]}',
            "set -e",
            (
                f"extract-errors {TRIAGE_LOG_PATH} --output {TRIAGE_REPORT_PATH} "
                '--exit-code "$exit_code" || true'
            ),
            'exit "$exit_code"',
        ]
    )


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
@click.option(
    "--enable-error-extraction/--no-enable-error-extraction",
    default=False,
    help="Extract a structured error report from GitLab child-job output.",
)
@click.option("--base-ref", default=None, help="Target commit used to find PR-only changes")
@click.option("--head-ref", default="HEAD", help="PR branch SHA used to identify its changes")
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
    dependent_job: str,
    record_checkpoints: str,
    slurm_account: str,
    tag: Optional[str] = None,
    run_name: Optional[str] = None,
    wandb_experiment: Optional[str] = None,
    enable_lightweight_mode: bool = False,
    enable_warmup: Optional[bool] = None,
    cadence: Optional[str] = None,
    enable_error_extraction: bool = False,
    base_ref: Optional[str] = None,
    head_ref: str = "HEAD",
) -> None:
    # Treat empty string as "no cadence filter" so callers can wire shell
    # variables in directly without conditional flag emission.
    cadence_arg = cadence or None

    list_of_test_cases = [
        test_case
        for test_case in recipe_parser.load_workloads(
            scope=scope,
            container_tag=container_tag,
            environment=environment,
            test_cases=test_cases,
            platform=platform,
            tag=tag,
            cadence=cadence_arg,
            time_limit=time_limit,
        )
        if test_case.type != "build"
    ]

    changed_cases = {}
    if base_ref and platform:
        selected = {(case.spec["model"], case.spec["test_case"]) for case in list_of_test_cases}
        changed = _changed_workloads(BASE_PATH.parents[2], base_ref, platform, head_ref)
        for case in sorted(changed, key=lambda item: GITHUB_SCOPES.index(item.spec["scope"])):
            key = (case.spec["model"], case.spec["test_case"])
            if key not in selected:
                selected.add(key)
                changed_cases[key] = case.spec["scope"]
                list_of_test_cases.append(case)

    tags = [
        "arch/amd64",
        "env/prod",
        "origin/jet-fleet",
        "owner/jet-core",
        "purpose/jet-client",
        "team/megatron",
    ]

    gitlab_pipeline: dict
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

        gitlab_pipeline = {
            "stages": sorted(
                list(set([test_case["spec"]["model"] for test_case in list_of_test_cases]))
            ),
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

        warmup_job = ""

        for test_idx, test_case in enumerate(list_of_test_cases):
            key = (test_case.spec["model"], test_case.spec["test_case"])
            case_scope = changed_cases.get(key, scope)
            job_tags = list(tags)
            job_tags.append(f"cluster/{recipe_parser.resolve_cluster_config(cluster)}")

            script = [
                "export PYTHONPATH=$(pwd); "
                "python tests/test_utils/python_scripts/launch_jet_workload.py",
                f"--model {test_case['spec']['model']}",
                f"--environment {test_case['spec']['environment']}",
                f"--n-repeat {n_repeat}",
                f"--time-limit {test_case['spec'].get('time_limit', time_limit)}",
                f"--scope {case_scope}",
                f"--test-case '{test_case['spec']['test_case']}'",
                f"--container-tag {container_tag}",
                f"--cluster {cluster}",
                f"--platform {platform}",
                f"--record-checkpoints {record_checkpoints}",
                f"--account {slurm_account}",
            ]

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

            if enable_warmup:
                if test_idx == 0:
                    warmup_job = test_case['spec']['test_case']
                elif warmup_job != "":
                    needs.append({"job": warmup_job})

            test_script = " ".join(script)
            artifact_paths = ["results/"]
            if enable_error_extraction:
                test_script = build_test_script(test_script)
                artifact_paths.extend([TRIAGE_LOG_PATH, TRIAGE_REPORT_PATH])

            gitlab_pipeline[test_case['spec']['test_case']] = {
                "stage": f"{test_case['spec']['model']}",
                "image": f"{container_image}:{container_tag}",
                "tags": job_tags,
                "timeout": "7 days",
                "needs": needs,
                "script": [test_script],
                "artifacts": {"paths": artifact_paths, "when": "always"},
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

            if key in changed_cases:
                # GitHub reads these fields when converting the YAML to its matrix.
                # Added cases keep their own tier and bypass the normal cadence.
                gitlab_pipeline[test_case['spec']['test_case']]["variables"] = {
                    "FUNCTIONAL_TEST_SCOPE": case_scope,
                    "FUNCTIONAL_TEST_CADENCE": "",
                }

    with open(output_path, 'w') as outfile:
        yaml.dump(gitlab_pipeline, outfile, default_flow_style=False)


if __name__ == "__main__":
    main()
