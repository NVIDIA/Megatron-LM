# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Native config construction and migration of legacy training inputs."""

from argparse import ArgumentParser
from copy import deepcopy

import pytest
import torch
import yaml

from megatron.core.distributed import (
    DistributedDataParallelConfig,
    FullyShardedDataParallelConfig,
    TorchFullyShardedDataParallelConfig,
)
from megatron.training.arguments import add_megatron_arguments
from megatron.training.config.instantiate_utils import InstantiationException
from megatron.training.config.yaml_utils import safe_yaml_representers
from megatron.training.training import get_megatron_data_parallel_config


@pytest.mark.parametrize("version", [1, 2])
def test_native_fsdp_config_round_trip(run_config, tmp_path, version):
    config = FullyShardedDataParallelConfig(
        version=version,
        main_grads_dtype=torch.float32,
        grad_comm_dtype=torch.bfloat16,
        data_parallel_sharding_strategy="optim_grads_params",
    )
    run_config.data_parallel = config
    path = str(tmp_path / "run_config.yaml")
    run_config.to_yaml(path)
    restored = type(run_config).from_yaml(path)
    assert restored.data_parallel_config == config
    assert isinstance(restored.data_parallel_config, FullyShardedDataParallelConfig)
    assert "ddp" not in restored.to_dict()
    assert restored.to_dict() == run_config.to_dict()


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("use_fsdp", [False, True])
@pytest.mark.parametrize("serialized_yaml", [False, True])
def test_legacy_ddp_payload_migrates_before_construction(
    run_config, tmp_path, version, use_fsdp, serialized_yaml
):
    payload = run_config.to_dict()
    payload["ddp"] = payload.pop("data_parallel")
    payload["ddp"].update(
        use_megatron_fsdp=use_fsdp,
        use_custom_fsdp=False,
        megatron_fsdp_version=version,
        megatron_fsdp_main_grads_dtype=run_config._convert_value_to_dict(torch.float32),
        data_parallel_sharding_strategy="optim_grads_params",
        expert_data_parallel_sharding_strategy="optim",
        fsdp_double_buffer=True,
    )
    before = deepcopy(payload)
    if serialized_yaml:
        path = tmp_path / "legacy.yaml"
        with safe_yaml_representers():
            path.write_text(yaml.safe_dump(payload))
        restored = type(run_config).from_yaml(str(path))
    else:
        restored = type(run_config).from_dict(payload)
    assert payload == before
    config = restored.data_parallel_config
    if use_fsdp:
        assert isinstance(config, FullyShardedDataParallelConfig)
        assert config.version == version
        assert config.main_grads_dtype == torch.float32
        assert config.data_parallel_sharding_strategy == "optim_grads_params"
        assert config.expert_data_parallel_sharding_strategy == "optim"
        assert config.double_buffer
        assert not hasattr(config, "use_distributed_optimizer")
    else:
        assert isinstance(config, DistributedDataParallelConfig)
        assert not hasattr(config, "use_megatron_fsdp")
        assert not hasattr(config, "data_parallel_sharding_strategy")
    assert "ddp" not in restored.to_dict()


def test_legacy_loader_preserves_unknown_fields_for_strict_validation(run_config):
    payload = run_config.to_dict()
    payload["ddp"] = payload.pop("data_parallel")
    payload["ddp"].update(use_megatron_fsdp=True, misspelled_setting=True)
    with pytest.raises(InstantiationException, match="misspelled_setting"):
        type(run_config).from_dict(payload)


def test_container_rejects_both_serialized_inputs(run_config):
    payload = run_config.to_dict()
    payload["ddp"] = payload["data_parallel"].copy()
    with pytest.raises(ValueError, match="only one of data_parallel and ddp"):
        type(run_config).from_dict(payload)


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("cuda_graph_impl", ["none", "full_iteration"])
def test_cli_builds_native_fsdp_config(version, cuda_graph_impl):
    parser = ArgumentParser()
    add_megatron_arguments(parser)
    args = parser.parse_args(
        [
            "--use-megatron-fsdp",
            "--megatron-fsdp-version",
            str(version),
            "--data-parallel-sharding-strategy",
            "optim_grads_params",
            "--no-megatron-fsdp-padded-shards",
            "--cuda-graph-impl",
            cuda_graph_impl,
        ]
    )
    config = get_megatron_data_parallel_config(args, use_torch_fsdp2=False)
    assert isinstance(config, FullyShardedDataParallelConfig)
    assert config.version == version
    assert config.data_parallel_sharding_strategy == "optim_grads_params"
    assert config.padded_shards is False
    assert config.cuda_graph_mode is (cuda_graph_impl != "none")
    assert config.all_gather_in_start_param_sync is (cuda_graph_impl != "full_iteration")


@pytest.mark.parametrize("torch_fsdp", [False, True])
def test_cli_selects_other_backends(torch_fsdp):
    parser = ArgumentParser()
    add_megatron_arguments(parser)
    config = get_megatron_data_parallel_config(parser.parse_args([]), use_torch_fsdp2=torch_fsdp)
    assert type(config) is (
        TorchFullyShardedDataParallelConfig if torch_fsdp else DistributedDataParallelConfig
    )


def test_cli_rejects_conflicting_backends():
    parser = ArgumentParser()
    add_megatron_arguments(parser)
    args = parser.parse_args(["--use-megatron-fsdp"])
    with pytest.raises(ValueError, match="both Megatron-FSDP and Torch FSDP"):
        get_megatron_data_parallel_config(args, use_torch_fsdp2=True)
