# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GTP checkpoint regressions for the shared in_proj factory; run on 4 or 8 ranks."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import is_main_replica
from megatron.core.ssm import utils as ssm_utils

pytestmark = pytest.mark.internal

_KEY = 'decoder.in_proj.weight'
_SECTIONS = [4, 8]
_NAMES = ['first', 'second']


@pytest.fixture(scope='module')
def projection_groups(cpu_default_process_group):
    """Keep TP fixed at one and redistribute GTP ranks into DP when changing GTP size."""
    rank, world_size = dist.get_rank(), dist.get_world_size()
    assert world_size in (4, 8), 'Run with torchrun --nproc-per-node=4 or 8'
    rank_sets = []
    for size in (1, 2, 4):
        rank_sets.extend(tuple(range(start, start + size)) for start in range(0, world_size, size))
        rank_sets.extend(tuple(range(start, world_size, size)) for start in range(size))
    groups = {
        ranks: dist.new_group(list(ranks), backend='gloo') for ranks in dict.fromkeys(rank_sets)
    }
    yield {
        size: (
            groups[(rank,)],
            groups[tuple(range(rank // size * size, (rank // size + 1) * size))],
            groups[tuple(range(rank % size, world_size, size))],
        )
        for size in (1, 2, 4)
    }
    for group in reversed(list(groups.values())):
        if group != dist.GroupMember.NON_GROUP_MEMBER:
            dist.destroy_process_group(group)


def _factory(data, topology, padding_rows, sections=_SECTIONS):
    tp, gtp, dp = topology
    gtp_rank, gtp_size = dist.get_rank(gtp), dist.get_world_size(gtp)
    local = F.pad(data, (0, 0, 0, padding_rows), value=-1).chunk(gtp_size)[gtp_rank].clone()
    weight = torch.nn.Parameter(local)
    weight.group, weight.gtp_remat_size = gtp, gtp_size
    weight.pad_length, weight._unsharded_shape = padding_rows, tuple(data.shape)
    original = ShardedTensor.from_rank_offsets(
        _KEY,
        weight,
        (0, 2, 5),
        (1, gtp_rank, gtp_size),
        prepend_axis_num=1,
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
            dp_cp_group=dp,
            sharded_offsets=((0, 2, 5),),
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
