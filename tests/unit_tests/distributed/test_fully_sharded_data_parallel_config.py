# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Independent FSDP configuration, legacy conversion, and shared compatibility state."""

import copy
import pickle
from dataclasses import asdict

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig, FullyShardedDataParallelConfig
from megatron.core.distributed.fully_sharded_data_parallel_config import resolve_fsdp_config


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
    with pytest.warns(DeprecationWarning, match="fsdp_config"):
        config, view = resolve_fsdp_config(ddp_config=legacy)
    assert asdict(legacy) == before
    assert config.version == version
    assert not config.padded_shards
    assert not view.megatron_fsdp_padded_shards
    assert config.expert_data_parallel_sharding_strategy == "optim"
    assert config.expert_outer_dp_sharding_strategy == "no_shard"
    assert config.num_distributed_optimizer_instances == 2
    assert config.main_params_dtype == torch.float32
    assert config.main_grads_dtype == torch.float32
    assert config.grad_comm_dtype == torch.bfloat16
    assert view.suggested_communication_unit_size == 4096


def test_v1_runtime_updates_share_one_authoritative_config():
    config = FullyShardedDataParallelConfig().resolved()
    view = config.as_legacy_config()
    # V1 enables overlap and double buffering during construction.
    view.overlap_param_gather = True
    view.fsdp_double_buffer = True
    assert config.overlap_param_gather
    assert config.double_buffer
    config.main_grads_dtype = torch.bfloat16
    assert view.megatron_fsdp_main_grads_dtype == torch.bfloat16


@pytest.mark.parametrize("clone", [copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))])
def test_compatibility_view_can_be_copied(clone):
    config = FullyShardedDataParallelConfig(version=2, main_grads_dtype=torch.bfloat16)
    copied = clone(config.as_legacy_config())
    assert copied.fsdp_config == config
    assert copied.fsdp_config is not config
    assert copied.megatron_fsdp_main_grads_dtype == torch.bfloat16


def test_already_resolved_legacy_bucket_count():
    legacy = DistributedDataParallelConfig(num_buckets=4)
    legacy.bucket_size = 1024  # Training derives this after DDPConfig construction.
    config = FullyShardedDataParallelConfig.from_ddp_config(legacy)
    assert config.bucket_size == 1024
    assert config.num_buckets is None
    assert legacy.num_buckets == 4


def test_ambiguous_wrapper_inputs_rejected():
    with pytest.raises(ValueError, match="only one"):
        resolve_fsdp_config(
            fsdp_config=FullyShardedDataParallelConfig(), ddp_config=DistributedDataParallelConfig()
        )


def test_explicit_wrapper_version_preserves_legacy_selection():
    legacy = DistributedDataParallelConfig(use_megatron_fsdp=True)
    with pytest.warns(DeprecationWarning):
        config, view = resolve_fsdp_config(ddp_config=legacy, version=2)
    assert config.version == 2
    assert view.megatron_fsdp_version == 2
    assert legacy.megatron_fsdp_version == 1
    with pytest.raises(ValueError, match="requires version=2"):
        resolve_fsdp_config(fsdp_config=FullyShardedDataParallelConfig(version=1), version=2)


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
