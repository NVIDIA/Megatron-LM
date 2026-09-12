# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU regressions for source identity in distributed optimizer checkpoints."""

from dataclasses import replace

import pytest
import torch

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import ShardedTensorFactory
from megatron.core.optimizer.distrib_optimizer import (
    DistributedOptimizer,
    _resolve_gtp_sharded_metadata,
)
from tests.unit_tests.tensor_parallel.test_gtp_checkpoint_optimizer import (
    _loaded_tree,
    _optimizer_stub,
)

pytestmark = pytest.mark.internal


def _entry(param):
    dequant = param.detach().to(torch.bfloat16).clone()
    dequant._gtp_dequant_src = param
    return ShardedTensor.from_rank_offsets(
        'model_0.experts.weight',
        dequant,
        (0, 3, 8),
        (1, 1, 2),
        prepend_axis_num=1,
        replica_id=(0, 0, 2),
    )


def test_dequant_backlink_preserves_expert_metadata():
    param = torch.nn.Parameter(torch.arange(12).float().reshape(4, 3))
    param.allreduce = False
    entry = _entry(param)
    assert _resolve_gtp_sharded_metadata(param, {'weight': entry}) is entry
    assert entry.global_offset == (3, 4, 0)
    assert entry.replica_id == (0, 0, 2)


@pytest.mark.parametrize('expert', [False, True])
def test_unmatched_metadata_does_not_guess_by_debug_name(expert):
    param = torch.nn.Parameter(torch.ones(4, 3))
    param.allreduce = not expert
    param._debug_name = 'model_0.experts.weight'
    raw_factory = ShardedTensorFactory(
        param._debug_name, torch.ones(8, 3), lambda *args: [], lambda parts: parts
    )
    assert _resolve_gtp_sharded_metadata(param, {'weight': raw_factory}) is None
    with pytest.raises(ValueError, match='no source-bound metadata'):
        DistributedOptimizer.sharded_param_state_fully_reshardable(
            _optimizer_stub(param, (0, param.numel())), {'weight': raw_factory}, metadata={}
        )


def test_factory_companion_preserves_outer_prefix_and_identity():
    param = torch.nn.Parameter(torch.ones(4, 3))
    companion = ShardedTensorFactory('weight', param, lambda *args: [], lambda parts: parts)
    factory = replace(companion, data=torch.ones(8, 3), optimizer_factory=companion)
    resolved = _resolve_gtp_sharded_metadata(
        param, {'weight': replace(factory, key='model_0.weight')}
    )
    assert resolved.data is param
    assert resolved.key == 'model_0.weight'


@pytest.mark.parametrize('mode', ['fully_reshardable', 'fs_model_space'])
def test_distributed_optimizer_resolves_dequantized_source(mode):
    param = torch.nn.Parameter(torch.arange(12).float().reshape(4, 3))
    entry = _entry(param)
    stub = _optimizer_stub(param, (1, 11))
    result = getattr(DistributedOptimizer, 'sharded_param_state_' + mode)(
        stub, {'weight': entry}, metadata={}
    )
    for key, state in result[0].items():
        assert state.key == f'optimizer.state.{key}.{entry.key}'
        delta = {'param': 0, 'fp32_param': 0, 'exp_avg': 5000, 'exp_avg_sq': 9000}[key]
        expected = param.detach() + delta
        if mode == 'fs_model_space':
            assert isinstance(state, ShardedTensorFactory)
            expected = expected.flatten()[1:11]
            parts = state.build()
            assert [part.global_offset for part in parts['chunks']] == [
                (3, 4, 1),
                (3, 5, 0),
                (3, 7, 0),
            ]
            for part in parts['chunks']:
                assert part.key == state.key
                assert part.global_shape == entry.global_shape
                assert part.dtype == torch.float32
                assert part.replica_id == (0, 0, 0)
                assert part.flattened_range is None
                part.validate_metadata_integrity()
            actual = state.merge_fn(_loaded_tree(parts))
        else:
            assert state.global_shape == entry.global_shape
            assert state.global_offset == entry.global_offset
            assert state.dtype == torch.float32
            actual = state.data
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
