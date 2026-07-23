# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
import types

import prepare_jet_sqsh_image
import recipe_parser

SMOKE_TEST_CASE = "gpt3_mcore_te_tp1_pp1_te_4experts_groupedGEMM_op_fuser"


def _load_smoke_workloads(**kwargs):
    return recipe_parser.load_workloads(
        container_tag="12345",
        environment="dev",
        platform="dgx_h100",
        scope="L0-smoke",
        test_case=SMOKE_TEST_CASE,
        **kwargs,
    )


def test_load_workloads_keeps_build_dependency_by_default():
    workloads = _load_smoke_workloads()

    basic_workloads = [workload for workload in workloads if workload.type == "basic"]
    build_workloads = [workload for workload in workloads if workload.type == "build"]

    assert len(basic_workloads) == 1
    assert len(build_workloads) == 1
    assert basic_workloads[0].spec["build"] == "mcore-pyt-dev"
    assert build_workloads[0].spec["source"]["image"].endswith(":12345")


def test_load_workloads_uses_local_image_source_without_build_workload():
    workloads = _load_smoke_workloads(
        workload_local_image_path="/lustre/enroot/{build}-{platforms}-{container_tag}.sqsh"
    )

    basic_workloads = [workload for workload in workloads if workload.type == "basic"]

    assert len(basic_workloads) == 1
    assert all(workload.type != "build" for workload in workloads)
    assert "build" not in basic_workloads[0].spec
    assert basic_workloads[0].spec["image_source"] == {
        "local_path": "/lustre/enroot/mcore-pyt-dev-dgx_h100-12345.sqsh"
    }


def test_resolve_workload_local_image_sources_uses_build_source_image():
    sources = recipe_parser.resolve_workload_local_image_sources(
        container_tag="12345",
        environment="dev",
        platform="dgx_h100",
        scope="L0-smoke",
        test_case=SMOKE_TEST_CASE,
        workload_local_image_path="/lustre/enroot/{build}-{platforms}-{container_tag}.sqsh",
    )

    assert len(sources) == 1
    assert sources[0].build == "mcore-pyt-dev"
    assert sources[0].source_image == "gitlab-master.nvidia.com/adlr/megatron-lm/mcore_ci_dev:12345"
    assert sources[0].local_path == "/lustre/enroot/mcore-pyt-dev-dgx_h100-12345.sqsh"


def test_resolve_local_image_prepare_cluster_uses_same_site_cpu_cluster():
    assert recipe_parser.resolve_local_image_prepare_cluster("dgxa100_dracooci") == "cpu_dracooci"
    assert recipe_parser.resolve_local_image_prepare_cluster("dgxh100_coreweave") == "cpu_coreweave"
    assert recipe_parser.resolve_local_image_prepare_cluster("dgxgb200_oci-hsg") == "cpu_oci-hsg"
    assert recipe_parser.resolve_local_image_prepare_cluster("cpu_coreweave") == "cpu_coreweave"


def test_prepare_workload_uses_image_source_without_build():
    workload = prepare_jet_sqsh_image.build_prepare_workload(
        source_image="gitlab-master.nvidia.com/adlr/megatron-lm/mcore_ci_dev:12345",
        local_path="/lustre/enroot/mcore-pyt-dev-dgx_h100-12345.sqsh",
        time_limit=1800,
    )

    spec = workload["spec"]
    assert "build" not in spec
    assert spec["image_source"] == {
        "image_tag": "gitlab-master.nvidia.com/adlr/megatron-lm/mcore_ci_dev:12345"
    }


def test_submit_prepare_workload_passes_raw_image_source_workload(monkeypatch):
    captured = {}

    class FakePipeline:
        jet_id = 123

        def get_status(self):
            return "success"

    class FakeWorkloads:
        def submit(self, **kwargs):
            captured.update(kwargs)
            return FakePipeline()

    class FakeJETClient:
        def __init__(self, **kwargs):
            self.workloads = FakeWorkloads()

    fake_pipeline_module = types.SimpleNamespace(PipelineStatus=types.SimpleNamespace(SUBMISSION_FAILED="failed"))
    fake_jetclient = types.SimpleNamespace(
        JETClient=FakeJETClient,
        clients=types.SimpleNamespace(gitlab=types.SimpleNamespace(GitlabAPIError=Exception)),
        facades=types.SimpleNamespace(
            objects=types.SimpleNamespace(util=types.SimpleNamespace(WaitTimeExceeded=Exception))
        ),
    )
    monkeypatch.setitem(sys.modules, "jetclient", fake_jetclient)
    monkeypatch.setitem(sys.modules, "jetclient.services", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "jetclient.services.dtos", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "jetclient.services.dtos.pipeline", fake_pipeline_module)

    pipeline = prepare_jet_sqsh_image.submit_prepare_workload(
        source_image="gitlab-master.nvidia.com/adlr/megatron-lm/mcore_ci_dev:12345",
        local_path="/lustre/enroot/mcore-pyt-dev-dgx_h100-12345.sqsh",
        cluster="cpu_coreweave",
        account="coreai_dlalgo_ci",
        partition=None,
        time_limit=1800,
    )

    assert pipeline.jet_id == 123
    workload = captured["workloads"][0]
    assert "build" not in workload["spec"]
    assert workload["spec"]["image_source"] == {
        "image_tag": "gitlab-master.nvidia.com/adlr/megatron-lm/mcore_ci_dev:12345"
    }
