# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Independent FSDP configuration, explicit conversion, and strict constructor inputs."""

from dataclasses import asdict

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig, FullyShardedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
    FullyShardedDataParallel,
    FullyShardedDataParallelV1,
    FullyShardedDataParallelV2,
)


def test_independent_config_and_read_only_resolution():
    config = FullyShardedDataParallelConfig(
        data_parallel_sharding_strategy="optim_grads_params",
        outer_dp_sharding_strategy="optim",
        max_pool_double_buffer=True,
    )
    original = asdict(config)
    config.validate()
    resolved = config.resolved()
    assert asdict(config) == original
    assert not isinstance(config, DistributedDataParallelConfig)
    assert "use_megatron_fsdp" not in original
    assert "use_distributed_optimizer" not in original
    assert "pad_buckets_for_high_nccl_busbw" not in original
    assert resolved.expert_data_parallel_sharding_strategy == "optim_grads_params"
    assert resolved.expert_outer_dp_sharding_strategy == "optim"
    assert resolved.double_buffer


@pytest.mark.parametrize("version", [1, 2])
def test_legacy_conversion_preserves_precision_and_sharding(version):
    legacy = DistributedDataParallelConfig(
        use_megatron_fsdp=True,
        megatron_fsdp_version=version,
        megatron_fsdp_padded_shards=False,
        data_parallel_sharding_strategy="optim_grads_params",
        expert_data_parallel_sharding_strategy="optim",
        outer_dp_sharding_strategy="optim",
        expert_outer_dp_sharding_strategy="no_shard",
        num_distributed_optimizer_instances=2,
        megatron_fsdp_main_params_dtype=torch.float32,
        megatron_fsdp_main_grads_dtype=torch.float32,
        megatron_fsdp_grad_comm_dtype=torch.bfloat16,
        suggested_communication_unit_size=4096,
    )
    before = asdict(legacy)
    config = FullyShardedDataParallelConfig.from_ddp_config(legacy)
    assert asdict(legacy) == before
    assert config.version == version
    assert not config.padded_shards
    assert config.expert_data_parallel_sharding_strategy == "optim"
    assert config.expert_outer_dp_sharding_strategy == "no_shard"
    assert config.num_distributed_optimizer_instances == 2
    assert config.main_params_dtype == torch.float32
    assert config.main_grads_dtype == torch.float32
    assert config.grad_comm_dtype == torch.bfloat16
    assert config.suggested_communication_unit_size == 4096


def test_already_resolved_legacy_bucket_count():
    legacy = DistributedDataParallelConfig(num_buckets=4)
    legacy.bucket_size = 1024  # Training derives this after DDPConfig construction.
    config = FullyShardedDataParallelConfig.from_ddp_config(legacy)
    assert config.bucket_size == 1024
    assert config.num_buckets is None
    assert legacy.num_buckets == 4


@pytest.mark.parametrize(
    "wrapper", [FullyShardedDataParallel, FullyShardedDataParallelV1, FullyShardedDataParallelV2]
)
def test_wrappers_require_native_config(wrapper):
    legacy = DistributedDataParallelConfig(use_megatron_fsdp=True)
    with pytest.raises(TypeError, match="ddp_config"):
        wrapper(config=None, module=None, ddp_config=legacy)
    with pytest.raises(TypeError, match="FullyShardedDataParallelConfig"):
        wrapper(config=None, module=None, fsdp_config=legacy)


@pytest.mark.parametrize(
    "wrapper,version", [(FullyShardedDataParallelV1, 2), (FullyShardedDataParallelV2, 1)]
)
def test_explicit_wrapper_version_is_checked(wrapper, version):
    with pytest.raises(ValueError, match="requires version"):
        wrapper(
            config=None, module=None, fsdp_config=FullyShardedDataParallelConfig(version=version)
        )


def test_resolution_revalidates_mutable_inputs():
    config = FullyShardedDataParallelConfig()
    config.expert_data_parallel_sharding_strategy = ""
    with pytest.raises(ValueError, match="expert_data_parallel_sharding_strategy"):
        config.resolved()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"version": 3}, "version"),
        ({"data_parallel_sharding_strategy": "invalid"}, "sharding_strategy"),
        ({"expert_outer_dp_sharding_strategy": "invalid"}, "sharding_strategy"),
        ({"num_distributed_optimizer_instances": 0}, "positive"),
        ({"num_buckets": 0}, "greater than 0"),
        ({"num_buckets": 2, "bucket_size": 1024}, "both"),
        ({"version": 2, "manual_registration": True}, "manual_registration"),
        ({"version": 2, "num_buckets": 2}, "num_buckets"),
        ({"version": 2, "nccl_ub": True, "padded_shards": False}, "padded_shards"),
        ({"version": 2, "outer_dp_sharding_strategy": "optim"}, "outer DP axis"),
        (
            {"version": 2, "nccl_ub": True, "disable_symmetric_registration": True},
            "symmetric registration",
        ),
    ],
)
def test_invalid_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FullyShardedDataParallelConfig(**kwargs)
