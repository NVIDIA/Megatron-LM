# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import inspect

import pytest

from megatron.core.distributed import DistributedDataParallelConfig as MCoreFSDPConfig
from megatron.core.distributed.fsdp.src.megatron_fsdp.distributed_data_parallel_config import (
    DistributedDataParallelConfig as StandaloneFSDPConfig,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import (
    fully_shard,
    fully_shard_model,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.megatron_fsdp import MegatronFSDP


@pytest.mark.parametrize("config_cls", [MCoreFSDPConfig, StandaloneFSDPConfig])
def test_fsdp_persistent_buffer_config_validation(config_cls):
    """Persistent buffer capacity must agree with the pool enable flag."""
    assert config_cls().fsdp_double_buffer is False

    config = config_cls(fsdp_double_buffer=True, fsdp_buffer_count=3)
    assert config.fsdp_buffer_count == 3

    with pytest.raises(ValueError, match="must be at least 2"):
        config_cls(fsdp_double_buffer=True, fsdp_buffer_count=1)

    with pytest.raises(ValueError, match="may only be changed"):
        config_cls(fsdp_double_buffer=False, fsdp_buffer_count=3)


@pytest.mark.parametrize("config_cls", [MCoreFSDPConfig, StandaloneFSDPConfig])
@pytest.mark.parametrize(
    "enabling_option", ["nccl_ub", "megatron_fsdp_max_pool_double_buffer"]
)
def test_fsdp_persistent_buffer_automatic_enablement(config_cls, enabling_option, monkeypatch):
    """Options that require persistent buffers enable the pool before validation."""
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    config = config_cls(**{enabling_option: True, "fsdp_buffer_count": 3})
    assert config.fsdp_double_buffer is True


@pytest.mark.parametrize(
    ("api", "previous_last_parameter"),
    [
        (fully_shard_model, "maxpool_double_buffer"),
        (fully_shard, "maxpool_double_buffer"),
        (MegatronFSDP.__init__, "report_nan_in_param_grad"),
    ],
)
def test_fsdp_buffer_count_preserves_positional_api_compatibility(api, previous_last_parameter):
    """New options must follow all parameters accepted by the previous public API."""
    parameters = inspect.signature(api).parameters
    assert list(parameters).index("fsdp_buffer_count") > list(parameters).index(
        previous_last_parameter
    )
    assert parameters["fsdp_buffer_count"].default == 2
