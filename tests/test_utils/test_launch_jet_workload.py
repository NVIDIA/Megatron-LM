# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import Mock

import pytest

from tests.test_utils.python_scripts import launch_jet_workload


@pytest.mark.parametrize("cluster", ["dgxh100_coreweave", "dgxgb200_oci-hsg"])
@pytest.mark.parametrize("submission_retry", [False, True])
def test_submission_scopes_checkout_workaround_and_preserves_workload(
    monkeypatch, cluster, submission_retry
):
    pipeline = Mock()
    statuses = [launch_jet_workload.PipelineStatus.SUCCESS]
    if submission_retry:
        statuses.insert(0, launch_jet_workload.PipelineStatus.SUBMISSION_FAILED)
    pipeline.get_status.side_effect = statuses
    client = Mock()
    client.workloads.submit.return_value = pipeline
    monkeypatch.setattr(launch_jet_workload.jetclient, "JETClient", Mock(return_value=client))
    monkeypatch.setattr(
        launch_jet_workload.jetclient, "JETWorkloadManifest", lambda **workload: workload
    )
    workload = {"type": "basic", "spec": {"name": "checkout-test"}}
    load_workloads = Mock(return_value=[workload])
    monkeypatch.setattr(launch_jet_workload.recipe_parser, "load_workloads", load_workloads)
    monkeypatch.setattr(launch_jet_workload, "register_pipeline_terminator", Mock())
    wait = Mock(return_value=launch_jet_workload.PipelineStatus.SUCCESS)
    monkeypatch.setattr(launch_jet_workload, "wait_for_pipeline_completion", wait)

    result = launch_jet_workload.launch_and_wait_for_completion(
        test_case="checkout-test",
        environment="dev",
        n_repeat=5,
        time_limit=3600,
        scope="mr",
        container_image="test-image",
        container_tag="test-revision",
        cluster=cluster,
        platform="dgx_h100" if cluster == "dgxh100_coreweave" else "dgx_gb200",
        account="test-account",
        record_checkpoints="true",
        partition="test-partition",
        tag=None,
        run_name=None,
        wandb_experiment=None,
        enable_lightweight_mode=False,
    )

    assert result is pipeline
    wait.assert_called_once_with(pipeline)
    assert client.workloads.submit.call_count == 1 + submission_retry
    for call in client.workloads.submit.call_args_list:
        assert call.kwargs["workloads"] == [workload]
        config = call.kwargs["custom_config"]
        assert config["launchers"] == {
            cluster: {"account": "test-account", "partition": "test-partition"}
        }
        environments = config["executors"]["jet-ci"]["environments"]
        assert list(environments) == [cluster]
        variables = environments[cluster]["variables"]
        if cluster == "dgxh100_coreweave":
            assert variables["FF_SET_PERMISSIONS_BEFORE_CLEANUP"] == "false"
        else:
            assert "FF_SET_PERMISSIONS_BEFORE_CLEANUP" not in variables
        assert not any(name.startswith("GIT_") for name in variables)
        assert variables["ENABLE_LIGHTWEIGHT_MODE"] == "false"
        assert variables["RECORD_CHECKPOINTS"] == "true"
    for call in load_workloads.call_args_list:
        assert call.kwargs["n_repeat"] == 5
        assert call.kwargs["time_limit"] == 3600
        assert call.kwargs["scope"] == "mr"
        assert call.kwargs["container_tag"] == "test-revision"
