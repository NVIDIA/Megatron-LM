# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GTP checkpoint regressions for the shared in_proj factory; run on 4 or 8 ranks."""

from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.dict_utils import dict_list_map_outplace, nested_values
from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, is_main_replica
from megatron.core.ssm import utils as ssm_utils

pytestmark = pytest.mark.internal

_KEY = 'decoder.in_proj.weight'
_SECTIONS = [4, 8]
_NAMES = ['first', 'second']
_OFFSETS = ((0, 2, 5),)


@pytest.fixture(scope='module')
def projection_groups(cpu_default_process_group):
    """Build GTP/DP grids plus a TP2 grid to cover semantic-section offsets."""
    rank, world_size = dist.get_rank(), dist.get_world_size()
    assert world_size in (4, 8), 'Run with torchrun --nproc-per-node=4 or 8'
    rank_sets = []
    for size in (1, 2, 4):
        rank_sets.extend(tuple(range(start, start + size)) for start in range(0, world_size, size))
        rank_sets.extend(tuple(range(start, world_size, size)) for start in range(size))
    rank_sets.extend((base + i, base + i + 2) for base in range(0, world_size, 4) for i in range(2))
    groups = {
        ranks: dist.new_group(list(ranks), backend='gloo') for ranks in dict.fromkeys(rank_sets)
    }
    topologies = {
        size: (
            groups[(rank,)],
            groups[tuple(range(rank // size * size, (rank // size + 1) * size))],
            groups[tuple(range(rank % size, world_size, size))],
        )
        for size in (1, 2, 4)
    }
    base = rank // 4 * 4
    topologies['tp2'] = (
        groups[(base + rank % 2, base + rank % 2 + 2)],
        topologies[2][1],
        groups[tuple(range(rank % 4, world_size, 4))],
    )
    yield topologies
    for group in reversed(list(groups.values())):
        if group != dist.GroupMember.NON_GROUP_MEMBER:
            dist.destroy_process_group(group)


def _factory(
    data, topology, padding_rows, sections=_SECTIONS, sharded_offsets=_OFFSETS, dp_cp_group=None
):
    tp, gtp, dp = topology
    gtp_rank, gtp_size = dist.get_rank(gtp), dist.get_world_size(gtp)
    local = F.pad(data, (0, 0, 0, padding_rows), value=-1).chunk(gtp_size)[gtp_rank].clone()
    weight = torch.nn.Parameter(local)
    weight.group, weight.gtp_remat_size = gtp, gtp_size
    weight.gtp_replica_group = dp
    weight.pad_length, weight._unsharded_shape = padding_rows, tuple(data.shape)
    original = ShardedTensor.from_rank_offsets(
        _KEY,
        weight,
        *sharded_offsets,
        (len(sharded_offsets), gtp_rank, gtp_size),
        prepend_axis_num=len(sharded_offsets),
        replica_id=(0, 0, dist.get_rank(dp)),
    )
    # Only parameter detection is mocked; gather, splitting and load-side slicing are real.
    api = SimpleNamespace(HAVE_GTP=True, is_gtp_param=lambda p: p is weight and gtp_size > 1)
    with mock.patch.object(ssm_utils, 'gtp_api', api):
        factory = ssm_utils._split_in_proj_factory(
            original,
            sections,
            _NAMES,
            weight=weight,
            tp_group=tp,
            dp_cp_group=dp if dp_cp_group is None else dp_cp_group,
            sharded_offsets=sharded_offsets,
        )
    return factory, weight


@pytest.mark.parametrize('padding_rows', [0, 4], ids=['no_padding', 'with_padding'])
def test_in_proj_factory_gtp_roundtrip_and_reshard(projection_groups, padding_rows):
    """Save GTP2 across a section boundary, then restore GTP2, GTP4 and GTP1 shards."""
    data = torch.arange(36, dtype=torch.bfloat16).reshape(12, 3)
    factory, weight = _factory(data, projection_groups[2], padding_rows)
    assert weight.shape[0] not in (4, 12), 'GTP boundary must cut through a section'
    torch.testing.assert_close(factory.data, data, rtol=0, atol=0)
    assert not factory.data.requires_grad
    parts = factory.build()
    assert len(parts) == 2
    _, gtp, dp = projection_groups[2]
    for part, name, expected in zip(parts, _NAMES, data.split(_SECTIONS)):
        assert part.key == f'{_KEY}.{name}'
        assert part.local_shape == tuple(expected.shape)
        assert part.global_shape == (5, *expected.shape)
        assert part.global_offset == (2, 0, 0)
        assert part.replica_id == (0, dist.get_rank(gtp), dist.get_rank(dp))
        torch.testing.assert_close(part.data, expected, rtol=0, atol=0)

    # Each logical chunk is replicated over GTP and DP but must have exactly one writer.
    replicas = [None] * dist.get_world_size()
    dist.all_gather_object(replicas, parts[0].replica_id)
    assert len(set(replicas)) == dist.get_world_size()
    assert sum(is_main_replica(replica) for replica in replicas) == 1

    for size in (2, 4, 1):
        pad = padding_rows if size > 1 else 0
        target = factory
        if size != 2:
            target, _ = _factory(torch.zeros_like(data), projection_groups[size], pad)
        # Other GTP degrees start from zeros and must restore values from the saved sections.
        restored = target.merge_fn([part.data for part in parts])
        rank = dist.get_rank(projection_groups[size][1])
        expected = F.pad(data, (0, 0, 0, pad)).chunk(size)[rank]
        torch.testing.assert_close(restored, expected, rtol=0, atol=0)


def test_in_proj_factory_gtp_rejects_padding_as_data(projection_groups):
    """Requested sections cannot count physical padding as logical projection rows."""
    with pytest.raises(ValueError, match='Split sections must cover the whole dimension size'):
        _factory(torch.ones(12, 3), projection_groups[2], padding_rows=4, sections=[4, 12])


def test_in_proj_optimizer_factory_without_gtp(projection_groups):
    """An ordinary parameter's optimizer companion keeps the parameter identity and replica."""
    data = torch.arange(36, dtype=torch.float32).reshape(12, 3)
    factory, weight = _factory(data, projection_groups[1], padding_rows=0)
    assert factory.data is weight
    assert factory.optimizer_factory.data is weight
    assert factory.optimizer_factory.replica_id == factory.replica_id


@pytest.mark.parametrize(('tp_size', 'inclusive_dp'), [(1, False), (2, False), (1, True)])
@pytest.mark.parametrize('flat_dp', [False, True], ids=['full_parameter', 'flat_dp_fragment'])
def test_in_proj_optimizer_factory_gtp_offsets(projection_groups, flat_dp, tp_size, inclusive_dp):
    """Physical GTP/DP optimizer slices tile each semantic section exactly once and restore."""
    data = torch.arange(36 * tp_size, dtype=torch.float32).reshape(12 * tp_size, 3)
    topology = projection_groups[2 if tp_size == 1 else 'tp2']
    tp, gtp, dp = topology
    sections = data.split([size * tp_size for size in _SECTIONS])
    logical = torch.cat([section.chunk(tp_size)[dist.get_rank(tp)] for section in sections])
    padding_rows = 4
    # Training metadata includes GTP in DP; those ranks own distinct optimizer shards.
    factory, weight = _factory(
        logical,
        topology,
        padding_rows,
        sharded_offsets=(),
        dp_cp_group=dist.group.WORLD if inclusive_dp else None,
    )
    companion = factory.for_optimizer()
    # The optimizer values and dtype differ from the model checkpoint representation.
    # Nonzero source padding must disappear when materializing the semantic sections.
    optimizer_data = companion.data.detach().double() * 3 + 0.125
    prefix = f'optimizer.state.exp_avg.{_KEY}'
    state = replace(companion, key=prefix, data=optimizer_data)
    if flat_dp:
        dp_rank, dp_size = dist.get_rank(dp), dist.get_world_size(dp)
        start = dp_rank * optimizer_data.numel() // dp_size
        stop = (dp_rank + 1) * optimizer_data.numel() // dp_size
        state = replace(
            state,
            data=optimizer_data.flatten()[start:stop],
            flattened_range=slice(start, stop),
            replica_id=(0, 0, 0),
        )
    with mock.patch.object(
        dist, 'all_gather_into_tensor', side_effect=AssertionError('collective')
    ):
        tree = state.build()
    parts = [part for part in nested_values(tree) if isinstance(part, ShardedTensor)]
    payloads = [None] * dist.get_world_size()
    dist.all_gather_object(payloads, parts)
    expected_sections = {
        f'{prefix}.{name}': section.double() * 3 + 0.125 for name, section in zip(_NAMES, sections)
    }
    saved = {key: torch.empty_like(full) for key, full in expected_sections.items()}
    coverage = {
        key: torch.zeros(full.shape, dtype=torch.int32) for key, full in expected_sections.items()
    }

    def selection(part):
        return tuple(
            slice(offset, offset + size)
            for offset, size in zip(part.global_offset, part.local_shape)
        )

    for rank_parts in payloads:
        for part in rank_parts:
            part.validate_metadata_integrity()
            assert part.dtype == torch.float64
            assert not part.data.requires_grad
            assert part.prepend_axis_num == 0
            assert part.flattened_range is None
            assert part.global_shape == expected_sections[part.key].shape
            if is_main_replica(part.replica_id):
                index = selection(part)
                saved[part.key][index].copy_(part.data)
                coverage[part.key][index] += 1
    for key, expected in expected_sections.items():
        assert torch.all(coverage[key] == 1), (key, coverage[key])
        torch.testing.assert_close(saved[key], expected, rtol=0, atol=0)

    def load_leaf(leaf):
        if isinstance(leaf, ShardedTensor):
            return saved[leaf.key][selection(leaf)].clone()
        assert isinstance(leaf, LocalNonpersistentObject)
        return leaf.unwrap()

    loaded = dict_list_map_outplace(load_leaf, tree)
    expected = F.pad(logical.double() * 3 + 0.125, (0, 0, 0, padding_rows)).chunk(2)[
        dist.get_rank(gtp)
    ]
    if flat_dp:
        expected = expected.flatten()[start:stop]
    with mock.patch.object(
        dist, 'all_gather_into_tensor', side_effect=AssertionError('collective')
    ):
        restored = state.merge_fn(loaded)
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)
