# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU/Gloo regressions for ordinary optimizer fragments in logical model coordinates."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.optimizer import make_sharded_optimizer_fragment
from tests.unit_tests.dist_checkpointing.cpu_test_utils import TorchDistCPUSaveShardedStrategy

pytestmark = pytest.mark.internal


def _merge(factory, parts):
    return factory.merge_fn(
        {
            'chunks': [part.data.clone() for part in parts['chunks']],
            'layout': parts['layout'].unwrap(),
        }
    )


@pytest.mark.parametrize('shape', [(), (7,), (4, 3), (2, 3, 4), (2, 2, 3, 4)])
@pytest.mark.parametrize('prepend_axes', [0, 2])
@pytest.mark.parametrize('fragment', ['full', 'interior', 'empty'])
def test_optimizer_fragments_preserve_coordinates_and_state(shape, prepend_axes, fragment):
    """Independent tensor indexing checks every emitted rectangle and its global position."""
    values = torch.arange(torch.Size(shape).numel(), dtype=torch.float64).reshape(shape) + 5000
    numel = values.numel()
    start, stop = {'full': (0, numel), 'interior': (1, max(1, numel - 1)), 'empty': (numel, numel)}[
        fragment
    ]
    offsets = tuple((axis, axis + 1, axis + 3) for axis in range(prepend_axes))
    if shape:
        offsets += ((prepend_axes, 1, 2),)
    model = ShardedTensor.from_rank_offsets(
        'experts.weight',
        values.float(),
        *offsets,
        prepend_axis_num=prepend_axes,
        replica_id=(0, 0, 3),
    )
    state = torch.nn.Parameter(values.flatten()[start:stop].clone())
    factory = make_sharded_optimizer_fragment(
        model, state, 'optimizer.state.exp_avg', slice(start, stop), replica_id=(0, 0, 0)
    )
    factory = replace(factory, key='model_0.' + factory.key)
    parts = factory.build()
    covered = []
    for part in parts['chunks']:
        part.validate_metadata_integrity()
        assert part.key == factory.key
        assert part.dtype == state.dtype
        assert part.replica_id == (0, 0, 0)
        assert part.global_shape == model.global_shape
        assert part.global_offset[:prepend_axes] == model.global_offset[:prepend_axes]
        assert part.flattened_range is None
        assert not part.data.requires_grad
        local_offset = [
            actual - original
            for actual, original in zip(
                part.global_offset[prepend_axes:], model.global_offset[prepend_axes:]
            )
        ]
        index = tuple(
            slice(offset, offset + length)
            for offset, length in zip(local_offset, part.local_shape[prepend_axes:])
        )
        expected = values[index].reshape(part.local_shape)
        torch.testing.assert_close(part.data, expected, rtol=0, atol=0)
        covered.extend(torch.arange(numel).reshape(shape)[index].reshape(-1).tolist())
    assert covered == list(range(start, stop))
    torch.testing.assert_close(_merge(factory, parts), state, rtol=0, atol=0)
    assert state.requires_grad and state.grad is None


@pytest.mark.parametrize('start,stop', [(0, 12), (6, 12), (9, 12), (12, 12)])
def test_optimizer_fragments_restore_explicit_physical_padding(start, stop):
    model = ShardedTensor.from_rank_offsets('weight', torch.arange(9).reshape(3, 3))
    state = torch.arange(start, stop, dtype=torch.float64) + 100
    factory = make_sharded_optimizer_fragment(
        model, state, 'optimizer.state.exp_avg', slice(start, stop), physical_numel=12
    )
    parts = factory.build()
    expected = state.clone()
    expected[max(0, 9 - start) :] = 0
    torch.testing.assert_close(_merge(factory, parts), expected, rtol=0, atol=0)
    assert sum(part.data.numel() for part in parts['chunks']) == max(0, min(stop, 9) - start)


@pytest.mark.parametrize(
    'fragment', [slice(None, 3), slice(-1, 3), slice(4, 3), slice(0, 13), slice(0, 12, 2)]
)
def test_optimizer_fragments_reject_invalid_ranges(fragment):
    model = ShardedTensor.from_rank_offsets('weight', torch.zeros(4, 3))
    factory = make_sharded_optimizer_fragment(
        model, torch.zeros(3), 'optimizer.state.exp_avg', fragment
    )
    with pytest.raises(ValueError, match='Invalid optimizer fragment'):
        factory.build()


def test_optimizer_fragments_reject_missing_data_and_unvalidated_padding():
    model = ShardedTensor.from_rank_offsets('weight', torch.zeros(4, 3))
    with pytest.raises(ValueError, match='Physical parameter size is smaller'):
        make_sharded_optimizer_fragment(
            model, torch.zeros(3), 'state', slice(0, 3), physical_numel=11
        )
    factory = make_sharded_optimizer_fragment(model, torch.zeros(3), 'state', slice(0, 4))
    with pytest.raises(ValueError, match='data size does not match'):
        factory.build()
    factory = make_sharded_optimizer_fragment(model, torch.zeros(13), 'state', slice(0, 13))
    with pytest.raises(ValueError, match='Invalid optimizer fragment'):
        factory.build()


def test_optimizer_fragments_dcp_reshards_expert_axes(cpu_default_process_group, tmp_path):
    """Save actual DP fragments and load a dense tensor and a differently partitioned target."""
    rank, world_size = dist.get_rank(), dist.get_world_size()
    reference = torch.arange(120, dtype=torch.float64).reshape(2, 3, 5, 4) + 5000
    local_numel = reference[0].numel()
    start, stop = rank * local_numel // world_size, (rank + 1) * local_numel // world_size
    factories = []
    for expert in range(2):
        model = ShardedTensor.from_rank_offsets(
            'weight', reference[expert].float(), (0, expert, 2), prepend_axis_num=1
        )
        factories.append(
            make_sharded_optimizer_fragment(
                model,
                reference[expert].flatten()[start:stop].clone(),
                'optimizer.state.exp_avg',
                slice(start, stop),
            )
        )
    paths = [str(tmp_path / 'checkpoint') if rank == 0 else None]
    dist.broadcast_object_list(paths, src=0)
    checkpoint_dir = Path(paths[0])
    if rank == 0:
        checkpoint_dir.mkdir()
    dist.barrier()
    dist_checkpointing.save(
        {'states': factories}, checkpoint_dir, sharded_strategy=TorchDistCPUSaveShardedStrategy()
    )
    dist.barrier()
    target = ShardedTensor.from_rank_offsets(
        'optimizer.state.exp_avg.weight', torch.zeros_like(reference), replica_id=(0, 0, rank)
    )
    dense = dist_checkpointing.load({'state': target}, checkpoint_dir)
    torch.testing.assert_close(dense['state'], reference, rtol=0, atol=0)
    target_rank = world_size - 1 - rank
    target_start = target_rank * reference.numel() // world_size
    target_stop = (target_rank + 1) * reference.numel() // world_size
    model = ShardedTensor.from_rank_offsets('weight', torch.zeros_like(reference))
    fragment = make_sharded_optimizer_fragment(
        model,
        torch.zeros(target_stop - target_start, dtype=torch.float64),
        'optimizer.state.exp_avg',
        slice(target_start, target_stop),
    )
    loaded = dist_checkpointing.load({'state': fragment}, checkpoint_dir)
    torch.testing.assert_close(
        loaded['state'], reference.flatten()[target_start:target_stop], rtol=0, atol=0
    )
    dist.barrier()
