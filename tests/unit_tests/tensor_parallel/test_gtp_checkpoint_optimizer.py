# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU/Gloo regressions for optimizer state of GTP fused projections.

Run with ``torchrun --nproc-per-node=4`` or any larger even world size.
"""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, ShardedTensorFactory
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer, Range
from megatron.core.ssm.utils import _split_in_proj_factory, _split_tensor_factory
from megatron.core.tensor_parallel.gtp_utils import _gtp_slice_rows_on_load
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory
from tests.unit_tests.dist_checkpointing.cpu_test_utils import TorchDistCPUSaveShardedStrategy

pytestmark = pytest.mark.internal

_KEY = 'decoder.layers.in_proj.weight'
_SECTIONS = [3, 5, 4]
_NAMES = ['query', 'value', 'z']
_WIDTH = 3
_PAD_ROWS = 2


@pytest.fixture(scope='module')
def optimizer_groups(cpu_default_process_group):
    """Keep every process group owned by this module separate from the rest of the suite."""
    world_size = dist.get_world_size()
    assert world_size >= 4 and world_size % 2 == 0, 'Run with at least four even torchrun ranks'
    rank = dist.get_rank()
    rank_sets = [(i,) for i in range(world_size)]
    rank_sets += [tuple(range(i, i + 2)) for i in range(0, world_size, 2)]
    rank_sets += [tuple(range(i, world_size, 2)) for i in range(2)]
    groups = {ranks: dist.new_group(list(ranks), backend='gloo') for ranks in rank_sets}
    yield {
        'gtp': groups[(rank // 2 * 2, rank // 2 * 2 + 1)],
        'dp': groups[tuple(range(rank % 2, world_size, 2))],
        'single': groups[(rank,)],
    }
    for group in reversed(list(groups.values())):
        if group != dist.GroupMember.NON_GROUP_MEMBER:
            dist.destroy_process_group(group)


def _make_factory(groups, *, offsets=()):
    """Supply the semantic model factory that the shared GTP load helper wraps."""
    logical = torch.cat(
        [
            torch.arange(rows * _WIDTH).reshape(rows, _WIDTH) + index * 1000
            for index, rows in enumerate(_SECTIONS)
        ]
    ).float()
    gtp_rank = dist.get_rank(groups['gtp'])
    dp_rank = dist.get_rank(groups['dp'])
    padded = torch.nn.functional.pad(logical, (0, 0, 0, _PAD_ROWS), value=-12345)
    weight = torch.nn.Parameter(padded.chunk(2)[gtp_rank].clone())
    weight.group = groups['gtp']
    weight.gtp_remat_size = 2
    weight.pad_length = _PAD_ROWS
    original = ShardedTensor.from_rank_offsets(
        _KEY,
        logical,
        *offsets,
        (len(offsets), 0, 1),
        prepend_axis_num=len(offsets),
        replica_id=(0, gtp_rank, dp_rank),
    )
    factory = _gtp_slice_rows_on_load(
        _split_tensor_factory(original, _SECTIONS, _NAMES, split_dim=0), weight
    )
    expected = torch.nn.functional.pad(logical, (0, 0, 0, _PAD_ROWS)).chunk(2)[gtp_rank]
    return factory, weight, logical, expected


def _loaded_tree(tree, saved=None):
    """Perform load replacement independently from the factory's merge implementation."""
    if isinstance(tree, ShardedTensor):
        tree.validate_metadata_integrity()
        assert tree.flattened_range is None
        assert not tree.data.requires_grad
        if saved is None:
            return tree.data.clone()
        index = tuple(
            slice(start, start + size)
            for start, size in zip(
                tree.global_offset, (1,) * tree.prepend_axis_num + tree.local_shape
            )
        )
        return saved[tree.key][index].reshape(tree.local_shape).clone()
    if isinstance(tree, LocalNonpersistentObject):
        return tree.unwrap()
    if isinstance(tree, dict):
        return {key: _loaded_tree(value, saved) for key, value in tree.items()}
    if isinstance(tree, list):
        return [_loaded_tree(value, saved) for value in tree]
    if isinstance(tree, tuple):
        return tuple(_loaded_tree(value, saved) for value in tree)
    return tree


def _leaves(tree):
    if isinstance(tree, ShardedTensor):
        yield tree
    elif isinstance(tree, dict):
        for value in tree.values():
            yield from _leaves(value)
    elif isinstance(tree, (list, tuple)):
        for value in tree:
            yield from _leaves(value)


@pytest.mark.parametrize('offsets', [(), ((0, 0, 1),)], ids=['no_prepend', 'layer_axis'])
def test_gtp_optimizer_factory_preserves_source_and_keys(optimizer_groups, offsets):
    factory, weight, logical, expected = _make_factory(optimizer_groups, offsets=offsets)
    companion = factory.optimizer_factory
    assert isinstance(companion, ShardedTensorFactory)
    assert companion.data is weight
    assert companion.replica_id == (0, 0, dist.get_rank(optimizer_groups['dp']))
    assert factory.data is not weight
    state = replace(companion, key=f'optimizer.state.exp_avg.{_KEY}', data=weight.double())
    with mock.patch.object(
        dist, 'all_gather_into_tensor', side_effect=AssertionError('collective')
    ):
        parts = state.build()
        merged = state.merge_fn(_loaded_tree(parts))
    assert merged.dtype == torch.float64
    torch.testing.assert_close(merged, expected.double(), rtol=0, atol=0)
    expected_keys = {f'optimizer.state.exp_avg.{_KEY}.{name}' for name in _NAMES}
    for part in _leaves(parts):
        assert part.key in expected_keys
        assert part.global_shape[-1] == _WIDTH
        assert part.replica_id == companion.replica_id
    assert weight.requires_grad and weight.grad is None


def test_gtp_optimizer_companion_preserves_prefix(optimizer_groups):
    """The explicit optimizer view recovers identity and propagates outer key rewrites."""
    factory, weight, _, expected = _make_factory(optimizer_groups)
    factory = replace(factory, key='model_0.' + _KEY)
    companion = factory.for_optimizer()
    assert companion.data is weight
    assert companion.key == factory.key
    assert companion.for_optimizer() is companion
    moment = weight.detach().double() + 17
    expected = expected.double() + 17
    if dist.get_rank(optimizer_groups['gtp']) == 1:
        expected[-_PAD_ROWS:] = 0
    state = replace(companion, key='optimizer.state.exp_avg.' + companion.key, data=moment)
    parts = state.build()
    assert all(part.key.startswith(state.key + '.') for part in _leaves(parts))
    torch.testing.assert_close(state.merge_fn(_loaded_tree(parts)), expected, rtol=0, atol=0)


@pytest.mark.parametrize('start,stop', [(0, 21), (1, 20), (8, 17), (18, 21), (21, 21)])
def test_gtp_optimizer_factory_handles_flat_ranges(optimizer_groups, start, stop):
    """Cover partial columns, semantic boundaries, padding-only and empty local ranges."""
    factory, weight, _, expected = _make_factory(optimizer_groups)
    state = replace(
        factory.optimizer_factory,
        key=f'optimizer.state.exp_avg.{_KEY}',
        data=weight.flatten()[start:stop].double(),
        flattened_range=slice(start, stop),
        replica_id=(0, 0, 0),
    )
    with mock.patch.object(
        dist, 'all_gather_into_tensor', side_effect=AssertionError('collective')
    ):
        parts = state.build()
        merged = state.merge_fn(_loaded_tree(parts))
    torch.testing.assert_close(merged, expected.flatten()[start:stop].double(), rtol=0, atol=0)
    assert merged.shape == (stop - start,)
    assert all(part.replica_id == (0, 0, 0) for part in _leaves(parts))


@pytest.mark.parametrize('singleton', [False, True], ids=['shared_key', 'separate_keys'])
@pytest.mark.parametrize('flat_range', [None, slice(1, 20)], ids=['full_state', 'flat_fragment'])
def test_gtp_swiglu_optimizer_layout(optimizer_groups, singleton, flat_range):
    """The same helper preserves SwiGLU's shared-key offsets and separate _w/_v keys."""
    _, weight, logical, expected = _make_factory(optimizer_groups)
    original = ShardedTensor.from_rank_offsets(
        'linear_fc1.weight',
        logical,
        replica_id=(
            0,
            dist.get_rank(optimizer_groups['gtp']),
            dist.get_rank(optimizer_groups['dp']),
        ),
    )
    factory = apply_swiglu_sharded_factory(
        original,
        (),
        singleton_local_shards=singleton,
        tp_group=optimizer_groups['single'],
        dp_group=optimizer_groups['dp'],
    )
    factory = _gtp_slice_rows_on_load(factory, weight)
    moment = weight.detach().double() + 25
    expected = expected.double() + 25
    if dist.get_rank(optimizer_groups['gtp']) == 1:
        expected[-_PAD_ROWS:] = 0
    if flat_range is not None:
        moment, expected = moment.flatten()[flat_range], expected.flatten()[flat_range]
    key = 'optimizer.state.exp_avg.linear_fc1.weight'
    state = replace(factory.optimizer_factory, key=key, data=moment, flattened_range=flat_range)
    canonical = logical.double() + 25
    if singleton:
        reference = {key + '_w': canonical[:6], key + '_v': canonical[6:]}
    else:
        reference = {key: canonical}
    with mock.patch.object(
        dist, 'all_gather_into_tensor', side_effect=AssertionError('collective')
    ):
        parts = state.build()
        for part in _leaves(parts):
            rows, columns = part.global_offset
            height, width = part.local_shape
            torch.testing.assert_close(
                part.data,
                reference[part.key][rows : rows + height, columns : columns + width],
                rtol=0,
                atol=0,
            )
        merged = state.merge_fn(_loaded_tree(parts, reference))
    torch.testing.assert_close(merged, expected, rtol=0, atol=0)


def _optimizer_stub(weight, local_range):
    start, stop = local_range
    full_states = {
        'param': weight.detach().flatten().clone(),
        'exp_avg': weight.detach().flatten().clone() + 5000,
        'exp_avg_sq': weight.detach().flatten().clone() + 9000,
    }

    def get_states(param):
        assert param is weight
        return {key: value[start:stop].clone() for key, value in full_states.items()}

    n = weight.numel()
    return SimpleNamespace(
        gbuf_ranges=[{torch.float32: [{'param_map': {weight: {'param': Range(start, stop)}}}]}],
        buffers=[SimpleNamespace(param_index_map={weight: (0, n, 0)}, buckets=[None])],
        distributed_optimizer_instance_id=0,
        _get_main_param_and_optimizer_states=get_states,
        get_parameter_state_dp_zero=lambda **kwargs: {0: {torch.float32: full_states}},
    )


@pytest.mark.parametrize('mode', ['fully_reshardable', 'fs_model_space'])
def test_gtp_distributed_optimizer_uses_companion(optimizer_groups, mode):
    factory, weight, _, expected = _make_factory(optimizer_groups)
    n = weight.numel()
    dp_rank = dist.get_rank(optimizer_groups['dp'])
    dp_size = dist.get_world_size(optimizer_groups['dp'])
    local_range = (dp_rank * n // dp_size, (dp_rank + 1) * n // dp_size)
    stub = _optimizer_stub(weight, local_range)
    method = getattr(DistributedOptimizer, f'sharded_param_state_{mode}')
    with mock.patch.object(
        dist, 'all_gather_into_tensor', side_effect=AssertionError('collective')
    ):
        result = method(stub, {_KEY: replace(factory, key='model_0.' + _KEY)}, metadata={})
        for state_key, state in result[0].items():
            assert state.key == f'optimizer.state.{state_key}.model_0.{_KEY}'
            merged = state.merge_fn(_loaded_tree(state.build()))
            delta = {'param': 0, 'fp32_param': 0, 'exp_avg': 5000, 'exp_avg_sq': 9000}[state_key]
            target = expected.clone()
            # Padding is structural zero even when the source optimizer stores nonzero values.
            valid_rows = sum(_SECTIONS) - dist.get_rank(optimizer_groups['gtp']) * weight.shape[0]
            target[: max(0, min(valid_rows, target.shape[0]))] += delta
            if mode == 'fs_model_space':
                target = target.flatten()[slice(*local_range)]
            torch.testing.assert_close(merged, target, rtol=0, atol=0)


@pytest.mark.parametrize('is_loading', [False, True], ids=['save', 'load'])
def test_gtp_optimizer_memory_efficient_owner_selection(optimizer_groups, is_loading):
    """DP nonowners may return early without stranding other GTP peers in a collective."""
    factory, weight, _, expected = _make_factory(optimizer_groups)
    stub = _optimizer_stub(weight, (0, weight.numel()))
    dp_rank = dist.get_rank(optimizer_groups['dp'])
    all_states = stub.get_parameter_state_dp_zero()
    stub.get_parameter_state_dp_zero = mock.Mock(
        return_value=all_states if dp_rank == 0 or is_loading else None
    )
    with mock.patch.object(
        dist, 'all_gather_into_tensor', side_effect=AssertionError('collective')
    ):
        result = DistributedOptimizer.sharded_param_state_fully_reshardable(
            stub,
            {_KEY: factory},
            is_loading=is_loading,
            metadata={'distrib_optim_fully_reshardable_mem_efficient': True},
        )
        if dp_rank == 0 or is_loading:
            state = result[0]['param']
            torch.testing.assert_close(
                state.merge_fn(_loaded_tree(state.build())), expected, rtol=0, atol=0
            )
        else:
            assert result is None
    stub.get_parameter_state_dp_zero.assert_called_once_with(
        use_gloo_comm=True, empty_data=is_loading, return_on_all_ranks=is_loading
    )


@pytest.mark.parametrize('offsets', [(), ((0, 0, 1),)], ids=['no_prepend', 'layer_axis'])
@pytest.mark.parametrize('flat_target', [False, True], ids=['full_target', 'dp_flat_target'])
def test_gtp_optimizer_dcp_loads_without_gtp(optimizer_groups, tmp_path, offsets, flat_target):
    """Real DCP validates rectangular DP fragments and retiles into non-GTP optimizer state."""
    factory, weight, logical, _ = _make_factory(optimizer_groups, offsets=offsets)
    dp_rank = dist.get_rank(optimizer_groups['dp'])
    dp_size = dist.get_world_size(optimizer_groups['dp'])
    start, stop = dp_rank * weight.numel() // dp_size, (dp_rank + 1) * weight.numel() // dp_size
    source = replace(
        factory.optimizer_factory,
        key=f'optimizer.state.exp_avg.{_KEY}',
        data=weight.detach().flatten()[start:stop].clone(),
        flattened_range=slice(start, stop),
        replica_id=(0, 0, 0),
    )
    path_list = [str(tmp_path / 'checkpoint') if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(path_list, src=0)
    checkpoint_dir = path_list[0]
    if dist.get_rank() == 0:
        Path(checkpoint_dir).mkdir()
    dist.barrier()
    dist_checkpointing.save(
        {'moment': source}, checkpoint_dir, sharded_strategy=TorchDistCPUSaveShardedStrategy()
    )
    dist.barrier()
    target_weight = torch.nn.Parameter(torch.zeros_like(logical))
    original = ShardedTensor.from_rank_offsets(
        f'optimizer.state.exp_avg.{_KEY}',
        target_weight,
        *offsets,
        prepend_axis_num=len(offsets),
        replica_id=(0, 0, dist.get_rank()),
    )
    target = _split_in_proj_factory(
        original,
        _SECTIONS,
        _NAMES,
        weight=target_weight,
        tp_group=optimizer_groups['single'],
        dp_cp_group=dist.group.WORLD,
        sharded_offsets=offsets,
    )
    expected = logical
    if flat_target:
        rank, size = dist.get_rank(), dist.get_world_size()
        target_start = rank * logical.numel() // size
        target_stop = (rank + 1) * logical.numel() // size
        target = replace(
            target.optimizer_factory,
            data=target_weight.detach().flatten()[target_start:target_stop],
            flattened_range=slice(target_start, target_stop),
            replica_id=(0, 0, 0),
        )
        expected = logical.flatten()[target_start:target_stop]
    loaded = dist_checkpointing.load({'moment': target}, checkpoint_dir)
    torch.testing.assert_close(loaded['moment'], expected, rtol=0, atol=0)
    dist.barrier()
