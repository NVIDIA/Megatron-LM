# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused tests for DistributedOptimizer checkpoint shard coverage."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.optimizer.distrib_optimizer import (
    DistributedOptimizer,
    _is_trailing_padding_only_shard,
)


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
