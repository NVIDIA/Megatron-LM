# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy

import pytest

jetclient = pytest.importorskip("jetclient")

from tests.test_utils.python_scripts import recipe_parser
from tests.test_utils.python_scripts.launch_jet_workload import build_workload_manifest


@pytest.fixture
def workloads():
    return {
        workload["type"]: workload
        for workload in recipe_parser.load_workloads(
            container_tag="validation",
            scope="gb300-comparison",
            environment="dev",
            platform="dgx_gb200",
            test_cases="deepseek_proxy_mfsdp_v1_ep2_1node",
        )
    }


def test_gb300_comparison_replaces_dataset_mounts_without_mutating_recipe(workloads):
    workload = workloads["basic"]
    workload["launchers"] = {
        "name:dgxgb300_oci-jhb": {"partition": "batch", "mounts": {"/mnt/rp2": "/missing/rp2"}},
        "type:slurm": {"exclusive": True},
    }
    original = copy.deepcopy(workload)

    manifest = build_workload_manifest(workload, "dgxgb300_oci-jhb", "gb300-comparison")

    assert workload == original
    assert manifest.launchers["type:slurm"] == {"exclusive": True}
    launcher = manifest.launchers["name:dgxgb300_oci-jhb"]
    assert launcher["partition"] == "batch"
    assert launcher["mounts"] == {
        "/lustre/fsw/coreai_dlalgo_mcore": (
            "/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore/"
        ),
        "/mnt/artifacts": ("/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore/mcore_ci"),
    }


@pytest.mark.parametrize(
    ("cluster", "scope", "workload_type"),
    [
        ("dgxgb200_oci-hsg", "gb300-comparison", "basic"),
        ("dgxgb300_oci-jhb", "nightly", "basic"),
        ("dgxgb300_oci-jhb", "gb300-comparison", "build"),
    ],
)
def test_other_workloads_keep_their_launcher_configuration(
    workloads, cluster, scope, workload_type
):
    launchers = {"type:slurm": {"mounts": {"/mnt/rp2": "/datasets/rp2"}}}
    workload = workloads[workload_type]
    workload["launchers"] = launchers

    manifest = build_workload_manifest(workload, cluster, scope)

    assert manifest.launchers == launchers
