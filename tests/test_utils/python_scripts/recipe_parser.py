# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import copy
import itertools
import logging
import pathlib
from typing import List, Optional

import click
import yaml

BASE_PATH = pathlib.Path(__file__).parent.resolve()

logger = logging.getLogger(__name__)

DEFAULT_CADENCE = ["pr", "nightly", "mergegroup"]
ALLOWED_CADENCE_VALUES = set(DEFAULT_CADENCE) | {"weekly"}

# Maps legacy `scope` values (encoded "when" + "which suite" together) onto the
# new vocabulary: an L-tier (cost class) plus, where the legacy name implied a
# trigger, a default cadence. The tier acts purely as a suite/cost label;
# cadence remains the trigger axis.
#
# Only GitHub-side scopes (`mr-github-slim`, `mr-github`) are aliased onto the
# L-tier names. GitLab-only scopes (`mr`, `mr-slim`, `unit-tests`) are
# intentionally left as pass-through so GitLab `--scope mr*` / `--scope
# unit-tests` continue to match recipes verbatim and don't bleed into the
# GitHub L0 / L1 matrix.
LEGACY_SCOPE_ALIASES = {
    # GitHub-only scopes are aliased onto the L-tier vocabulary so the GH CI
    # workflow can filter on `L0` / `L1`. GitLab-only scopes (`mr`, `mr-slim`)
    # are intentionally NOT aliased: they pass through to recipe rows verbatim
    # and remain matchable by GitLab's `--scope mr-slim` / `--scope mr` calls,
    # without bleeding into the GitHub `L0` / `L1` matrix.
    "mr-github-slim": ("L0", None),
    "mr-github": ("L1", None),
    "nightly": ("L2", ["nightly"]),
    "weekly": ("L3", ["weekly"]),
}


def _resolve_scope_alias(scope_value: str) -> str:
    """Resolve a legacy scope value to its L-tier alias (or return it unchanged).

    Applied both to recipe rows when flattening and to the `--scope` filter
    input, so callers can pass either the legacy name (e.g. `nightly`) or the
    new L-tier name (e.g. `L2`) and hit the same recipe rows.
    """
    if scope_value in LEGACY_SCOPE_ALIASES:
        return LEGACY_SCOPE_ALIASES[scope_value][0]
    return scope_value


def _apply_scope_alias(scope_value: str, explicit_cadence: Optional[List[str]]) -> tuple:
    """Resolve a legacy scope value to (new_scope, cadence).

    If the scope value is in the alias map, returns the new tier name. The
    alias's cadence is used only when `explicit_cadence` is None — explicit
    per-product or outer cadence always wins.
    """
    if scope_value not in LEGACY_SCOPE_ALIASES:
        return scope_value, explicit_cadence
    new_scope, alias_cadence = LEGACY_SCOPE_ALIASES[scope_value]
    if explicit_cadence is not None or alias_cadence is None:
        return new_scope, explicit_cadence
    return new_scope, list(alias_cadence)


def _validate_cadence(cadence: List[str], test_case: str) -> None:
    if not isinstance(cadence, list):
        raise ValueError(
            f"cadence for test_case {test_case} must be a list, got {type(cadence).__name__}"
        )
    invalid = [c for c in cadence if c not in ALLOWED_CADENCE_VALUES]
    if invalid:
        raise ValueError(
            f"Invalid cadence value(s) {invalid} for test_case {test_case}. "
            f"Allowed: {sorted(ALLOWED_CADENCE_VALUES)}"
        )


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def resolve_cluster_config(cluster: str) -> str:
    if cluster == "dgxh100_eos":
        return "eos"
    if cluster == "dgxgb200_oci-hsg":
        return "oci-hsg"
    if cluster == "dgxa100_dracooci":
        return "draco-oci-iad"
    if cluster == "dgxa100_dracooci-ord":
        return "draco-oci-ord"
    if cluster == "dgxh100_coreweave":
        return "coreweave"
    if cluster == "ghci":
        return "ghci"
    raise ValueError(f"Unknown cluster {cluster} provided.")


def flatten_products(workload_manifest: dotdict) -> dotdict:
    """Flattens a nested dict of products"""
    flattened_products = []
    products = workload_manifest.products or []

    for product in products:
        if "products" not in product:
            continue

        test_case = product["test_case"][0]
        # Outer-level cadence (next to test_case) acts as a default for every
        # inner products block under this test_case. Inner cadence wins when
        # both are present.
        outer_cadence = product.get("cadence")
        if outer_cadence is not None:
            _validate_cadence(outer_cadence, test_case)

        for param_dict in product["products"]:
            # cadence is a list-valued attribute, not a cartesian dimension.
            # Pull it out of the cartesian product before expansion so a list
            # like ["pr", "nightly"] doesn't multiply the workload count.
            inner_cadence = param_dict.get("cadence")
            if inner_cadence is not None:
                _validate_cadence(inner_cadence, test_case)
            cartesian_keys = [k for k in param_dict.keys() if k != "cadence"]
            cartesian_values = [param_dict[k] for k in cartesian_keys]

            # Resolve effective cadence: inner overrides outer, default loose.
            # `explicit_cadence` is None when neither inner nor outer set it,
            # which is the only case where a legacy scope alias may inject a
            # default cadence (e.g. scope: [nightly] -> cadence: [nightly]).
            explicit_cadence = inner_cadence if inner_cadence is not None else outer_cadence
            default_cadence = (
                list(explicit_cadence) if explicit_cadence is not None else list(DEFAULT_CADENCE)
            )

            param_combinations = itertools.product(*cartesian_values)

            for value_combination in param_combinations:
                # Map parameter names to their values
                flattened = dict(zip(cartesian_keys, value_combination))
                flattened["test_case"] = test_case
                # Apply legacy scope alias per row so that scope: [mr, nightly]
                # produces two rows with the right (tier, cadence) each.
                row_cadence = default_cadence
                if "scope" in flattened:
                    new_scope, aliased_cadence = _apply_scope_alias(
                        flattened["scope"], explicit_cadence
                    )
                    flattened["scope"] = new_scope
                    if aliased_cadence is not None:
                        row_cadence = list(aliased_cadence)
                flattened["cadence"] = row_cadence
                flattened_products.append(flattened)

    workload_manifest.products = flattened_products
    return workload_manifest


def flatten_workload(workload_manifest: dotdict) -> List[dotdict]:
    """Flattens a workload with products into a list of workloads that don't have products."""
    workload_manifest = dict(workload_manifest)
    products = workload_manifest.pop("products")
    workload_manifests = []
    for product in products:
        workload = copy.deepcopy(workload_manifest)
        workload["spec"] = {k: v for k, v in workload["spec"].items() if k not in product.keys()}
        workload["spec"] = dict(**dict(workload["spec"].items()), **product)
        workload_manifests.append(dotdict(**workload))
    return workload_manifests


def set_build_dependency(workload_manifests: List[dotdict]) -> List[dotdict]:
    for workload_manifest in workload_manifests:
        workload_manifest.spec["build"] = workload_manifest.spec["build"].format(
            **dict(workload_manifest.spec)
        )
    return workload_manifests


def load_config(config_path: str) -> dotdict:
    """Loads and parses a yaml file into a JETWorkloadManifest"""
    with open(config_path) as stream:
        try:
            return dotdict(**yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            raise exc


def load_and_flatten(config_path: str) -> List[dotdict]:
    """Wrapper function for doing all the fun at once."""
    return set_build_dependency(
        flatten_workload(flatten_products(load_config(config_path=config_path)))
    )


def filter_by_test_case(workload_manifests: List[dotdict], test_case: str) -> Optional[dotdict]:
    """Returns a workload with matching name. Raises an error if there no or more than a single workload."""
    print(len(workload_manifests))
    workload_manifests = list(
        workload_manifest
        for workload_manifest in workload_manifests
        if workload_manifest["spec"]["test_case"] == test_case
    )
    print(len(workload_manifests))

    for w in workload_manifests:
        print(w["spec"]["test_case"])

    if len(workload_manifests) > 1:
        logger.info("Duplicate test_case found!")
        return None

    if len(workload_manifests) == 0:
        logger.info("No test_case found!")
        return None

    return workload_manifests[0]


def filter_by_scope(workload_manifests: List[dotdict], scope: str) -> List[dotdict]:
    """Returns all workload with matching scope.

    The filter input is run through the same legacy-scope alias as recipe
    rows, so callers passing the legacy name (e.g. `--scope nightly`,
    `--scope mr-github`) match recipes that have already been rewritten to
    the new L-tier vocabulary (e.g. `scope: [L2]`, `scope: [L1]`).
    """
    resolved_scope = _resolve_scope_alias(scope)
    workload_manifests = list(
        workload_manifest
        for workload_manifest in workload_manifests
        if workload_manifest.spec["scope"] == resolved_scope
    )

    if len(workload_manifests) == 0:
        logger.info("No test_case found!")
        return []

    return workload_manifests


def filter_by_cadence(workload_manifests: List[dotdict], cadence: Optional[str]) -> List[dotdict]:
    """Returns workloads whose cadence list includes the requested cadence value.

    A cadence of None disables the filter (used for the label-based bypass path).
    Workloads missing a cadence field default to all triggers (loose default).
    """
    if cadence is None:
        return workload_manifests

    if cadence not in ALLOWED_CADENCE_VALUES:
        raise ValueError(f"Invalid cadence {cadence!r}. Allowed: {sorted(ALLOWED_CADENCE_VALUES)}")

    filtered = list(
        workload_manifest
        for workload_manifest in workload_manifests
        if cadence in workload_manifest.spec.get("cadence", DEFAULT_CADENCE)
    )

    if len(filtered) == 0:
        logger.info("No test_case found for cadence %s!", cadence)
        return []

    return filtered


def filter_by_environment(workload_manifests: List[dotdict], environment: str) -> List[dotdict]:

    workload_manifests_copy = list(
        workload_manifest
        for workload_manifest in workload_manifests.copy()
        if (
            hasattr(dotdict(**workload_manifest["spec"]), "environment")
            and workload_manifest["spec"]["environment"] == environment
        )
    )

    if len(workload_manifests_copy) == 0:
        logger.info("No test_case found!")
        return []

    return workload_manifests_copy


def filter_by_platform(workload_manifests: List[dotdict], platform: str) -> List[dotdict]:
    workload_manifests = list(
        workload_manifest
        for workload_manifest in workload_manifests
        if (
            hasattr(dotdict(**workload_manifest["spec"]), "platforms")
            and workload_manifest.spec["platforms"] == platform
        )
    )

    if len(workload_manifests) == 0:
        logger.info("No test_case found!")
        return []

    return workload_manifests


def filter_by_model(workload_manifests: List[dotdict], model: str) -> List[dotdict]:
    """Returns all workload with matching model."""
    workload_manifests = list(
        workload_manifest
        for workload_manifest in workload_manifests
        if workload_manifest.spec["model"] == model
    )

    if len(workload_manifests) == 0:
        logger.info("No test_case found!")
        return []

    return workload_manifests


def filter_by_tag(workload_manifests: List[dotdict], tag: str) -> List[dotdict]:
    """Returns all workload with matching tag."""
    workload_manifests = list(
        workload_manifest
        for workload_manifest in workload_manifests
        if hasattr(dotdict(**workload_manifest["spec"]), "tag")
        and workload_manifest["spec"]["tag"] == tag
    )

    if len(workload_manifests) == 0:
        logger.info("No test_case found!")
        return []

    return workload_manifests


def filter_by_test_cases(workload_manifests: List[dotdict], test_cases: str) -> List[dotdict]:
    """Returns a workload with matching name. Raises an error if there no or more than a single workload."""
    workload_manifests = list(
        workload_manifest
        for workload_manifest in workload_manifests
        for test_case in test_cases.split(",")
        if workload_manifest["spec"]["test_case"] == test_case
    )

    if len(workload_manifests) == 0:
        logger.info("No test_case found!")
        return []

    return workload_manifests


def load_workloads(
    container_tag: str,
    n_repeat: int = 1,
    time_limit: int = 1800,
    tag: Optional[str] = None,
    environment: Optional[str] = None,
    platform: Optional[str] = None,
    test_cases: str = "all",
    scope: Optional[str] = None,
    model: Optional[str] = None,
    test_case: Optional[str] = None,
    container_image: Optional[str] = None,
    record_checkpoints: Optional[str] = None,
    cadence: Optional[str] = None,
) -> List[dotdict]:
    """Return all workloads from disk that match scope and platform."""
    recipes_dir = BASE_PATH / ".." / "recipes"
    local_dir = BASE_PATH / ".." / "local_recipes"

    workloads: List[dotdict] = []
    build_workloads: List = []
    for file in list(recipes_dir.glob("**/*.yaml")) + list(local_dir.glob("**/*.yaml")):
        workloads += load_and_flatten(config_path=str(file))
        if file.stem.startswith("_build"):
            build_workloads.append(load_config(config_path=str(file)))

    if scope:
        workloads = filter_by_scope(workload_manifests=workloads, scope=scope)

    if workloads and cadence:
        workloads = filter_by_cadence(workload_manifests=workloads, cadence=cadence)

    if workloads and environment:
        workloads = filter_by_environment(workload_manifests=workloads, environment=environment)

    if workloads and model:
        workloads = filter_by_model(workload_manifests=workloads, model=model)

    if workloads and tag:
        workloads = filter_by_tag(workload_manifests=workloads, tag=tag)

    if workloads and platform:
        workloads = filter_by_platform(workload_manifests=workloads, platform=platform)

    if workloads and test_cases != "all":
        workloads = filter_by_test_cases(workload_manifests=workloads, test_cases=test_cases)

    if workloads and test_case:
        workloads = [filter_by_test_case(workload_manifests=workloads, test_case=test_case)]

    if not workloads:
        return []

    for workload in list(workloads):
        for build_workload in build_workloads:
            if (
                workload.spec["build"] == build_workload.spec["name"]
            ) and build_workload not in workloads:
                container_image = container_image or build_workload.spec["source"]["image"]
                build_workload.spec["source"]["image"] = f"{container_image}:{container_tag}"
                workloads.append(build_workload)

        workload.spec["n_repeat"] = n_repeat
        workload.spec["time_limit"] = time_limit
        workload.spec["artifacts"] = {
            key: value.replace(r"{platforms}", workload.spec["platforms"])
            for key, value in (
                workload.spec["artifacts"].items()
                if "artifacts" in workload.spec and workload.spec["artifacts"] is not None
                else {}
            )
        }

        if record_checkpoints == "true":
            workload.outputs = [
                {
                    "type": "artifact",
                    "key": f"unverified/model/mcore-ci/{container_tag}/{{model}}/{{name}}",
                    "subdir": "checkpoints",
                    "name": r"{model}/{name}",
                    "description": r"Checkpoint of {model}/{name}",
                    "pic": {"name": "Mcore CI", "email": "okoenig@nvidia.com"},
                    "labels": {"origin": "ADLR/Megatron-LM"},
                }
            ]
    return workloads


@click.command()
@click.option("--model", required=False, type=str, default=None, help="Model to select")
@click.option("--test-case", required=False, type=str, default=None, help="Test case to select")
def main(model: Optional[str], test_case: Optional[str]):
    workflows = load_workloads(container_tag="main", model=model, test_case=test_case)
    # Save workflows to YAML file
    output_file = "workflows.yaml"
    with open(output_file, "w") as f:
        yaml.dump([dict(workflow) for workflow in workflows], f)


if __name__ == "__main__":
    main()
