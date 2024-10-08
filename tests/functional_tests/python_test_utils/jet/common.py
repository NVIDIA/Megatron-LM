import copy
import itertools
import pathlib
from typing import List, Optional

import jetclient
import yaml

BASE_PATH = pathlib.Path(__file__).parent.resolve()


def flatten_products(
    workload_manifest: jetclient.JETWorkloadManifest,
) -> jetclient.JETWorkloadManifest:
    """Flattens a nested dict of products"""
    workload_manifest.products = [
        dict(zip(inp.keys(), values))
        for inp in workload_manifest.products
        for values in itertools.product(*inp.values())
    ]

    return workload_manifest


def flatten_workload(
    workload_manifest: jetclient.JETWorkloadManifest,
) -> List[jetclient.JETWorkloadManifest]:
    """Flattens a workload with products into a list of workloads that don't have products."""
    workload_manifest = dict(workload_manifest)
    products = workload_manifest.pop("products")
    workload_manifests = []
    for product in products:
        workload = copy.deepcopy(workload_manifest)
        workload['spec'] = {k: v for k, v in workload['spec'] if k not in product.keys()}
        workload['spec'] = dict(**dict(workload['spec']), **product)
        workload_manifests.append(jetclient.JETWorkloadManifest(**workload))
    return workload_manifests


def load_config(config_path: str) -> jetclient.JETWorkloadManifest:
    """Loads and parses a yaml file into a JETWorkloadManifest"""
    with open(config_path) as stream:
        try:
            return jetclient.JETWorkloadManifest(**yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            raise exc


def load_and_flatten(config_path: str) -> List[jetclient.JETWorkloadManifest]:
    """Wrapper function for doing all the fun at once."""
    return flatten_workload(flatten_products(load_config(config_path=config_path)))


def filter_by_test_case(
    workload_manifests: List[jetclient.JETWorkloadManifest], test_case: str
) -> jetclient.JETWorkloadManifest:
    """Returns a workload with matching name. Raises an error if there no or more than a single workload."""
    workload_manifests = list(
        workload_manifest
        for workload_manifest in workload_manifests
        if workload_manifest.spec.test_case == test_case
    )

    if len(workload_manifests) > 1:
        raise ValueError("Duplicate test_case found!")

    if len(workload_manifests) == 0:
        raise ValueError("No test_case found!")

    return workload_manifests[0]


def filter_by_scope(
    workload_manifests: List[jetclient.JETWorkloadManifest], scope: str
) -> List[jetclient.JETWorkloadManifest]:
    """Returns all workload with matching scope."""
    workload_manifests = list(
        workload_manifest
        for workload_manifest in workload_manifests
        if workload_manifest.spec.scope == scope
    )

    if len(workload_manifests) == 0:
        raise ValueError("No test_case found!")

    return workload_manifests


def filter_by_model(
    workload_manifests: List[jetclient.JETWorkloadManifest], model: str
) -> List[jetclient.JETWorkloadManifest]:
    """Returns all workload with matching model."""
    workload_manifests = list(
        workload_manifest
        for workload_manifest in workload_manifests
        if workload_manifest.spec.model == model
    )

    if len(workload_manifests) == 0:
        raise ValueError("No test_case found!")

    return workload_manifests


def load_workloads(
    container_tag: str,
    scope: Optional[str] = None,
    model: Optional[str] = None,
    test_case: Optional[str] = None,
    container_image: Optional[str] = None,
) -> List[jetclient.JETWorkloadManifest]:
    """Return all workloads from disk that match scope and platform."""
    recipes_dir = BASE_PATH / ".." / ".." / "jet_recipes"
    local_dir = BASE_PATH / ".." / ".." / "local_recipes"

    workloads: List[jetclient.JETWorkloadManifest] = []
    build_workloads: List[jetclient.JETClient] = []
    for file in list(recipes_dir.glob("*.yaml")) + list(local_dir.glob("*.yaml")):
        workloads += load_and_flatten(config_path=file)
        if file.stem.startswith("_build"):
            build_workloads.append(load_config(config_path=file))

    if scope:
        workloads = filter_by_scope(workload_manifests=workloads, scope=scope)

    if model:
        workloads = filter_by_model(workload_manifests=workloads, model=model)

    if test_case:
        workloads = [filter_by_test_case(workload_manifests=workloads, test_case=test_case)]

    for workload in list(workloads):
        for build_workload in build_workloads:
            if (
                workload.spec.build == build_workload.spec.name
            ) and build_workload not in workloads:
                container_image = container_image or build_workload.spec.source.image
                build_workload.spec.source.image = f"{container_image}:{container_tag}"
                workloads.append(build_workload)
    return workloads
