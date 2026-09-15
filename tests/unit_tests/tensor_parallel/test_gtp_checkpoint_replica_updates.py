# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression coverage for outer replica changes on physical optimizer factories."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import ShardedTensorFactory, is_main_replica
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.ssm.utils import _split_tensor_factory
from megatron.core.tensor_parallel.gtp_utils import _fused_projection_optimizer_factory
from megatron.core.transformer.moe.experts import SequentialMLP
from tests.unit_tests.tensor_parallel.test_gtp_checkpoint_optimizer import _optimizer_stub

pytestmark = pytest.mark.internal


@pytest.mark.parametrize(
    'current_replica,companion_replica,expected',
    [
        ((2, 3, 4), (0, 0, 1), (2, 0, 4)),
        (0, 7, 0),
        (7, 0, 7),
        ((2,), (1,), (2,)),
        ((2, 3, 4), 0, (2, 3, 4)),
    ],
)
def test_optimizer_companion_propagates_current_replica_ownership(
    current_replica, companion_replica, expected
):
    weight = torch.nn.Parameter(torch.zeros(4, 3))
    companion = ShardedTensorFactory(
        'weight', weight, lambda *args: [], lambda parts: parts, companion_replica
    )
    factory = replace(
        companion,
        key='model_0.weight',
        data=torch.zeros(8, 3),
        replica_id=current_replica,
        optimizer_factory=companion,
    )
    resolved = factory.for_optimizer()
    assert resolved.replica_id == expected
    assert resolved.key == 'model_0.weight'
    assert resolved.data is weight
    assert companion.replica_id == companion_replica
    assert factory.replica_id == current_replica


@pytest.mark.parametrize('gtp_rank', [0, 1])
def test_sequential_expert_dp_writer_updates_reach_optimizer_companion(gtp_rank):
    """Expert DP0 must write each physical shard after SequentialMLP rewrites replicas."""
    logical = torch.arange(24).float().reshape(8, 3)
    weight = torch.nn.Parameter(logical.chunk(2)[gtp_rank].clone())

    class Expert:
        def sharded_state_dict(self, prefix, offsets, metadata):
            # The caller's dense-DP rank is 1, but this expert has DP rank 0.
            original = ShardedTensor.from_rank_offsets(
                prefix + 'linear_fc1.weight',
                logical,
                *offsets,
                (len(offsets), 0, 1),
                prepend_axis_num=len(offsets),
                replica_id=(0, gtp_rank, 1),
            )
            factory = _split_tensor_factory(original, [4, 4], ['gate', 'up'], 0)
            companion = _fused_projection_optimizer_factory(
                factory, weight, shard_offset=gtp_rank * weight.numel(), replica_id=(0, 0, 1)
            )
            return {prefix + 'linear_fc1.weight': replace(factory, optimizer_factory=companion)}

    experts = SimpleNamespace(
        ep_group=SimpleNamespace(rank=lambda: 1, size=lambda: 2),
        dp_group=SimpleNamespace(rank=lambda: 0),
        num_local_experts=1,
        local_experts=[Expert()],
    )
    model_state = SequentialMLP.sharded_state_dict(
        experts, prefix='model.', metadata={'dp_cp_group': object()}
    )
    factory = next(iter(model_state.values()))
    assert factory.replica_id == (0, gtp_rank, 0)
    optimizer_state = DistributedOptimizer.sharded_param_state_fully_reshardable(
        _optimizer_stub(weight, (0, weight.numel())), model_state, metadata={}
    )
    for state_key, state_factory in optimizer_state[0].items():
        assert state_factory.data.shape == weight.shape
        for shard in state_factory.build()['chunks']:
            shard.validate_metadata_integrity()
            assert shard.replica_id == (0, 0, 0)
            assert is_main_replica(shard.replica_id)
            assert shard.key.startswith(f'optimizer.state.{state_key}.model.experts.')
            assert shard.global_offset[0] == 1
