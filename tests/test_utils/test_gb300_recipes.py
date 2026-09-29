# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from tests.test_utils.python_scripts import recipe_parser


@pytest.fixture
def workloads():
    return recipe_parser.load_workloads(
        container_tag="validation", scope="nightly", environment="dev", platform="dgx_gb300"
    )


@pytest.mark.parametrize("scope", ["nightly", "L2"])
def test_gb300_platform_selects_the_five_nightly_cases(scope):
    workloads = recipe_parser.load_workloads(
        container_tag="validation", scope=scope, environment="dev", platform="dgx_gb300"
    )
    cases = {
        workload["spec"]["test_case"]: workload["spec"]
        for workload in workloads
        if workload["type"] == "basic"
    }
    assert set(cases) == {
        "deepseek_proxy_mfsdp_v1_ep2",
        "deepseek_proxy_mfsdp_v1_ep2_1node",
        "gpt3_mcore_te_tp1_pp2_resume_torch_dist_reshard_2x1x4_te_8experts2parallel_dist_optimizer",
        "gpt3_mcore_te_tp2_pp1_te_8experts2parallel_ddp_average_in_collective",
        "nemotron3_5_lightning_nightly_tp1_pp1_cp1_ep8_dgx_gb200",
    }
    for name, spec in cases.items():
        assert spec["platforms"] == "dgx_gb300"
        assert spec["gpus"] == 4
        assert spec["nodes"] == (1 if name.endswith("_1node") else 2)
    assert cases["nemotron3_5_lightning_nightly_tp1_pp1_cp1_ep8_dgx_gb200"]["time_limit"] == 10800


@pytest.mark.parametrize(("scope", "environment"), [("mr", "dev"), ("nightly", "lts")])
def test_gb300_platform_only_selects_dev_nightly(scope, environment):
    assert not recipe_parser.load_workloads(
        container_tag="validation", scope=scope, environment=environment, platform="dgx_gb300"
    )


def test_gb300_manifests_mount_required_data_only_on_jhb(workloads):
    jetclient = pytest.importorskip("jetclient")
    for workload in workloads:
        manifest = jetclient.JETWorkloadManifest(**workload)
        if workload["type"] == "build":
            assert manifest.launchers == {}
        else:
            assert manifest.launchers == {
                "name:dgxgb300_oci-jhb": {
                    "mounts": {
                        "/lustre/fsw/coreai_dlalgo_mcore": (
                            "/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore/"
                        ),
                        "/mnt/artifacts": (
                            "/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore/mcore_ci"
                        ),
                    }
                }
            }


def test_gb200_recipes_keep_cluster_default_mounts():
    workloads = recipe_parser.load_workloads(
        container_tag="validation", scope="nightly", environment="dev", platform="dgx_gb200"
    )
    assert workloads
    assert all("launchers" not in workload for workload in workloads)
