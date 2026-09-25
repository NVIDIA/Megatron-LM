# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Build the GitHub functional matrix, including active tests changed by a PR."""

import json
import pathlib
import subprocess
from typing import Optional

import click
import yaml

from tests.test_utils.python_scripts import recipe_parser

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
TEST_CASES = pathlib.PurePosixPath("tests/functional_tests/test_cases")
RECIPES = pathlib.PurePosixPath("tests/test_utils/recipes")
GITHUB_SCOPES = ("L0", "L1")
GPUS_PER_RUNNER = {"dgx_h100": 8, "dgx_gb200": 4}


def _git(repo_root: pathlib.Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo_root, text=True)


def _changed_workloads(
    repo_root: pathlib.Path, base_ref: str, platform: str
) -> list[recipe_parser.dotdict]:
    # Compare the entire PR with its merge base, including on reruns and when
    # the target branch has advanced since the tested merge commit was made.
    base = _git(repo_root, "merge-base", base_ref, "HEAD").strip()
    changes = _git(
        repo_root,
        "diff",
        "--name-status",
        "--no-renames",
        "-z",
        base,
        "HEAD",
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
        previous = None
        if relative_path in changed_files:
            if changed_files[relative_path] == "A":
                previous = []
            else:
                manifest = recipe_parser.dotdict(
                    yaml.safe_load(_git(repo_root, "show", f"{base}:{relative_path}"))
                )
                previous = recipe_parser.set_build_dependency(
                    recipe_parser.flatten_workload(recipe_parser.flatten_products(manifest))
                )

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
            if case in changed_cases or (previous is not None and workload not in previous):
                changed.append(workload)
    return changed


def generate_matrix(
    scope: str,
    platform: str,
    cadence: Optional[str] = None,
    base_ref: Optional[str] = None,
    repo_root: pathlib.Path = REPO_ROOT,
) -> list[dict[str, str]]:
    """Keep the current suite and append changed functional cases once each.

    Changed cases bypass scope/cadence selection, but only active L0/L1 recipe
    rows on the requested platform are eligible. Existing selections win when
    a case belongs to multiple tiers, preserving their execution settings.
    Without a PR base reference, selection is unchanged.
    """
    workloads = recipe_parser.load_workloads(
        container_tag="latest",
        scope=scope,
        cadence=cadence or None,
        environment="dev",
        platform=platform,
    )
    matrix = {}
    for workload in workloads:
        if workload.type != "build":
            spec = workload.spec
            matrix[(spec["model"], spec["test_case"])] = {
                "model": spec["model"],
                "test_case": spec["test_case"],
                "scope": spec["scope"],
                "cadence": cadence or "",
            }

    if base_ref:
        changed = _changed_workloads(repo_root, base_ref, platform)
        # Prefer the lowest active tier when the same changed test is
        # registered in multiple scopes. Normal-suite rows above always win.
        for workload in sorted(changed, key=lambda item: GITHUB_SCOPES.index(item.spec["scope"])):
            spec = workload.spec
            matrix.setdefault(
                (spec["model"], spec["test_case"]),
                {
                    "model": spec["model"],
                    "test_case": spec["test_case"],
                    "scope": spec["scope"],
                    "cadence": "",
                },
            )

    return [matrix[key] for key in sorted(matrix)]


@click.command()
@click.option("--scope", required=True, help="Existing functional-test scope")
@click.option(
    "--platform",
    required=True,
    type=click.Choice(tuple(GPUS_PER_RUNNER)),
    help="Platform to select",
)
@click.option("--cadence", default=None, help="Existing cadence; empty disables the filter")
@click.option("--base-ref", default=None, help="Pinned PR base SHA; omit outside PR pushes")
def main(scope: str, platform: str, cadence: Optional[str], base_ref: Optional[str]) -> None:
    """Print the functional-test matrix as compact GitHub Actions JSON."""
    click.echo(
        json.dumps(generate_matrix(scope, platform, cadence, base_ref), separators=(",", ":"))
    )


if __name__ == "__main__":
    main()
