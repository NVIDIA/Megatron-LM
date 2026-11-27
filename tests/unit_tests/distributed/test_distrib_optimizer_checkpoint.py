# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused tests for DistributedOptimizer checkpoint shard coverage."""

from copy import deepcopy
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.dist_checkpointing import ShardedObject
from megatron.core.optimizer.distrib_optimizer import (
    DistributedOptimizer,
    _is_trailing_padding_only_shard,
)
from megatron.core.optimizer.grad_scaler import DynamicGradScaler
from megatron.core.optimizer.optimizer_config import OptimizerConfig


@pytest.mark.parametrize(
    "sharding_type",
    [
        "dp_zero_gather_scatter",
        "dp_reshardable",
        "fully_reshardable",
        "fully_sharded_model_space",
        "fsdp_dtensor",
    ],
)
@pytest.mark.parametrize("with_scaler", [False, True])
@pytest.mark.parametrize("is_loading", [False, True])
def test_stub_checkpoint_common_state(sharding_type, with_scaler, is_loading):
    """A stub preserves its scaler using each checkpoint format's common-state layout."""
    scaler = None
    if with_scaler:
        scaler = DynamicGradScaler(8, 1, 2, 0.5, 2, 2)
        for found_inf in (False, False, True, False):
            scaler.update(found_inf)
    config = OptimizerConfig(fp16=with_scaler, bf16=not with_scaler, use_distributed_optimizer=True)
    group = mock.Mock()
    group.rank.return_value = 1
    model = SimpleNamespace(
        ddp_config=SimpleNamespace(use_megatron_fsdp=sharding_type == "fsdp_dtensor")
    )
    optimizer = DistributedOptimizer(
        None,
        config,
        scaler,
        None,
        model_chunks=[model],
        per_model_buffers=None,
        data_parallel_group=group,
        data_parallel_group_gloo=group,
        data_parallel_group_idx=7,
        distributed_optimizer_instance_id=0,
    )
    expected = deepcopy(optimizer.state_dict())
    state = optimizer.sharded_state_dict(
        {},
        is_loading=is_loading,
        metadata={"distrib_optim_sharding_type": sharding_type, "checkpoint_version": 3.0},
    )
    if not with_scaler:
        assert expected == state == {}
        optimizer.load_state_dict({})
        return

    assert set(state) == {"grad_scaler"}
    if sharding_type not in optimizer.checkpoint_fully_reshardable_formats:
        wrapped = state["grad_scaler"]
        assert isinstance(wrapped, ShardedObject)
        assert wrapped.key == "optimizer.distributed.dp_group_idx_7.grad_scaler"
        assert wrapped.replica_id == (0, 0, 1)
        state = {"grad_scaler": wrapped.data}
    torch.testing.assert_close(state, expected)
    scaler.update(True)
    optimizer.load_state_dict(state)
    torch.testing.assert_close(optimizer.state_dict(), expected)


@pytest.mark.parametrize(
    ('data_parallel_rank', 'local_numel', 'world_numel_unpadded', 'expected'),
    [(0, 1024, 1024, False), (1, 1024, 1024, True), (1, 1024, 1500, False), (3, 1024, 1500, True)],
)
def test_trailing_padding_only_optimizer_shard(
    data_parallel_rank, local_numel, world_numel_unpadded, expected
):
    assert (
        _is_trailing_padding_only_shard(data_parallel_rank, local_numel, world_numel_unpadded)
        is expected
    )


def _make_optimizer_with_empty_local_shard(data_parallel_rank, numel_unpadded):
    dtype = torch.float32
    optimizer = mock.Mock()
    optimizer.data_parallel_group.rank.return_value = data_parallel_rank
    optimizer.data_parallel_group.size.return_value = 4
    optimizer.per_bucket_numel_unpadded = [[numel_unpadded]]
    optimizer.gbuf_ranges = [{dtype: [{"param_map": {}}]}]
    optimizer.buffers = [
        SimpleNamespace(
            buckets=[SimpleNamespace(grad_data=torch.empty(4096), numel_unpadded=numel_unpadded)]
        )
    ]
    state_dict = {"per_bucket_numel_unpadded": [[numel_unpadded]]}
    return optimizer, state_dict


def _make_optimizer_for_empty_shard_save(data_parallel_rank, numel_unpadded):
    optimizer, _ = _make_optimizer_with_empty_local_shard(data_parallel_rank, numel_unpadded)
    optimizer.data_parallel_group_idx = 0
    optimizer.distributed_optimizer_instance_id = 0
    optimizer.get_parameter_state_dp_reshardable.return_value = {
        "per_bucket_numel": [[4096]],
        "per_bucket_numel_unpadded": [[numel_unpadded]],
        0: {torch.float32: [[]]},
    }
    return optimizer


def test_save_skips_trailing_padding_only_shard():
    optimizer = _make_optimizer_for_empty_shard_save(data_parallel_rank=1, numel_unpadded=1024)

    state = DistributedOptimizer.sharded_param_state_dp_reshardable(optimizer, {})

    assert state[0][torch.float32][0] == []


def test_save_rejects_empty_shard_that_intersects_optimizer_state():
    optimizer = _make_optimizer_for_empty_shard_save(data_parallel_rank=1, numel_unpadded=1500)

    with pytest.raises(AssertionError, match="empty bucket intersects unpadded optimizer state"):
        DistributedOptimizer.sharded_param_state_dp_reshardable(optimizer, {})


def test_load_skips_checkpoint_key_for_trailing_padding_only_shard():
    optimizer, state_dict = _make_optimizer_with_empty_local_shard(
        data_parallel_rank=1, numel_unpadded=1024
    )

    DistributedOptimizer.load_parameter_state_from_dp_reshardable(optimizer, state_dict)


def test_load_rejects_empty_shard_that_intersects_optimizer_state():
    optimizer, state_dict = _make_optimizer_with_empty_local_shard(
        data_parallel_rank=1, numel_unpadded=1500
    )

    with pytest.raises(AssertionError, match="empty bucket intersects unpadded optimizer state"):
        DistributedOptimizer.load_parameter_state_from_dp_reshardable(optimizer, state_dict)
