# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Native and legacy data-parallel configuration serialization."""

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig, FullyShardedDataParallelConfig
from megatron.training.config.instantiate_utils import InstantiationException


@pytest.mark.parametrize("version", [1, 2])
def test_native_fsdp_config_round_trip(run_config, version):
    run_config.ddp = None
    config = FullyShardedDataParallelConfig(
        version=version,
        main_grads_dtype=torch.float32,
        grad_comm_dtype=torch.bfloat16,
        data_parallel_sharding_strategy="optim_grads_params",
    )
    run_config.data_parallel = config
    payload = run_config.to_dict()
    restored = type(run_config).from_dict(payload)
    assert restored.data_parallel_config == config
    assert isinstance(restored.data_parallel_config, FullyShardedDataParallelConfig)
    assert restored.ddp is None
    assert restored.to_dict() == payload


def test_legacy_ddp_payload_still_loads(run_config):
    run_config.ddp = DistributedDataParallelConfig(
        use_megatron_fsdp=True,
        megatron_fsdp_version=2,
        data_parallel_sharding_strategy="optim_grads_params",
    )
    payload = run_config.to_dict()
    payload.pop("data_parallel")  # Existing YAML has only the legacy key.
    restored = type(run_config).from_dict(payload)
    assert restored.to_dict()["ddp"] == payload["ddp"]
    assert restored.data_parallel_config.megatron_fsdp_version == 2


def test_container_rejects_both_inputs(run_config):
    payload = run_config.to_dict()
    native = FullyShardedDataParallelConfig()
    payload["data_parallel"] = run_config._convert_value_to_dict(native)
    with pytest.raises(InstantiationException, match="only one of data_parallel and ddp"):
        type(run_config).from_dict(payload)
