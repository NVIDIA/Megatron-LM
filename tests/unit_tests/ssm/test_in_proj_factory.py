# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU checkpoint-layout regressions; run distributed cases with torchrun on 4 or 8 ranks."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import is_main_replica
from megatron.core.dist_checkpointing.validation import (
    determine_global_metadata,
    validate_sharding_integrity,
)
from megatron.core.ssm import utils as ssm_utils

pytestmark = pytest.mark.internal

# GDP uses two householder steps, whose V/K/b blocks must remain separate checkpoint keys.
_LAYOUTS = [
    pytest.param([4, 4, 6, 6, 2, 2], ['query', 'key', 'value', 'z', 'beta', 'alpha'], id='gdn'),
    pytest.param(
        [6, 6, 6, 2, 2, 2, 2, 2, 2], ['z', 'V0', 'V1', 'K0', 'K1', 'Q', 'b0', 'b1', 'a'], id='gdp'
    ),
    pytest.param([6, 6, 4, 4, 2], ['z', 'x', 'B', 'C', 'dt'], id='mamba'),
]
_OFFSETS = ((0, 2, 5),)
_KEY = 'decoder.layers.in_proj.weight'


@pytest.mark.parametrize('sections,names', _LAYOUTS)
@pytest.mark.parametrize('is_bias', [False, True], ids=['weight', 'bias'])
@pytest.mark.parametrize(
    'replica_id', [0, (0, 3, 7)], ids=['integer_replica_id', 'tuple_replica_id']
)
@pytest.mark.parametrize(
    'have_gtp,is_gtp,remat_size',
    [(False, True, 2), (True, False, 2), (True, False, 1)],
    ids=['gtp_unavailable', 'ordinary_parameter', 'remat_one'],
)
def test_in_proj_factory_without_gtp(
    sections, names, is_bias, replica_id, have_gtp, is_gtp, remat_size
):
    """Unsharded weights and replicated 1-D biases keep their TP layout and replica metadata."""
    shape = (sum(sections),) if is_bias else (sum(sections), 3)
    data = torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)
    weight = torch.nn.Parameter(data)
    weight.gtp_remat_size = remat_size
    original = ShardedTensor.from_rank_offsets(
        _KEY, data, *_OFFSETS, (1, 1, 2), prepend_axis_num=1, replica_id=replica_id
    )
    # Replace this consumer's API reference without changing the shared GTP module.
    with (
        mock.patch.object(
            ssm_utils, 'gtp_api', SimpleNamespace(HAVE_GTP=have_gtp, is_gtp_param=lambda _: is_gtp)
        ),
        mock.patch.object(ssm_utils, '_gtp_gather_rows_for_save') as gather,
        mock.patch.object(ssm_utils, '_gtp_slice_rows_on_load') as slice_rows,
    ):
        factory = ssm_utils._split_in_proj_factory(
            original, sections, names, weight=weight, tp_group=None, dp_cp_group=None
        )
    gather.assert_not_called()
    slice_rows.assert_not_called()
    assert factory.data is data
    assert factory.replica_id == replica_id
    parts = factory.build()
    expected_parts = data.split(sections)
    for part, expected, section, name in zip(parts, expected_parts, sections, names):
        assert part.key == f'{_KEY}.{name}'
        assert part.global_shape == (5, section * 2) + shape[1:]
        assert part.global_offset == (2, section) + (0,) * (len(shape) - 1)
        assert part.replica_id == replica_id
        torch.testing.assert_close(part.data, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        factory.merge_fn([part.data for part in parts]), data, rtol=0, atol=0
    )


def test_in_proj_factory_rejects_incomplete_sections():
    data = torch.ones(8, 3)
    original = ShardedTensor.from_rank_offsets(_KEY, data, (0, 0, 1))
    with pytest.raises(ValueError, match='Split sections must cover the whole dimension size'):
        ssm_utils._split_in_proj_factory(
            original, [2, 4], ['x', 'z'], weight=data, tp_group=None, dp_cp_group=None
        )


@pytest.fixture(scope='module')
def projection_groups(cpu_default_process_group):
    """Use real CPU collectives with independent TP, GTP, and DP axes."""
    world_size = dist.get_world_size()
    assert world_size >= 4 and world_size % 4 == 0, 'Run with torchrun --nproc-per-node=4 or 8'
    rank = dist.get_rank()
    rank_sets = [(i,) for i in range(world_size)]
    for base in range(0, world_size, 4):
        rank_sets.extend(
            [
                (base, base + 1),
                (base + 2, base + 3),
                (base, base + 2),
                (base + 1, base + 3),
                tuple(range(base, base + 4)),
            ]
        )
    rank_sets.extend(tuple(range(i, world_size, 4)) for i in range(4))
    # Removing GTP folds its ranks into DP, while preserving the two TP coordinates.
    target_dp_ranks = [
        tuple(rank for rank in range(world_size) if rank % 4 // 2 == tp_rank)
        for tp_rank in range(2)
    ]
    rank_sets.extend(target_dp_ranks)
    groups = {}
    for ranks in dict.fromkeys(rank_sets):
        groups[ranks] = dist.new_group(list(ranks), backend='gloo')
    base = rank // 4 * 4
    local_rank = rank % 4
    yield {
        'tp': groups[(base + local_rank % 2, base + local_rank % 2 + 2)],
        'gtp': groups[tuple(range(base + local_rank // 2 * 2, base + local_rank // 2 * 2 + 2))],
        'dp': groups[tuple(range(local_rank, world_size, 4))],
        'dp_without_gtp': groups[target_dp_ranks[local_rank // 2]],
        'single': groups[(rank,)],
        'four': groups[tuple(range(base, base + 4))],
    }
    for group in reversed(list(groups.values())):
        if group != dist.GroupMember.NON_GROUP_MEMBER:
            dist.destroy_process_group(group)


def _global_sections(sections, dtype=torch.float32):
    return [
        (torch.arange(section * 2 * 3, dtype=torch.float32).reshape(section * 2, 3) + i * 1000).to(
            dtype
        )
        for i, section in enumerate(sections)
    ]


def _factory_for_topology(
    global_parts, names, tp_group, gtp_group, dp_group, padding_rows, sharded_offsets=_OFFSETS
):
    tp_rank, tp_size = dist.get_rank(tp_group), dist.get_world_size(tp_group)
    gtp_rank, gtp_size = dist.get_rank(gtp_group), dist.get_world_size(gtp_group)
    sections = [part.shape[0] // tp_size for part in global_parts]
    logical = torch.cat(
        [part.narrow(0, tp_rank * size, size) for part, size in zip(global_parts, sections)]
    )
    assert (logical.shape[0] + padding_rows) % gtp_size == 0
    padded = torch.nn.functional.pad(logical, (0, 0, 0, padding_rows), value=-12345)
    local = padded.chunk(gtp_size)[gtp_rank].clone()
    weight = torch.nn.Parameter(local)
    weight.group = gtp_group
    weight.gtp_remat_size = gtp_size
    weight.pad_length = padding_rows
    weight._unsharded_shape = tuple(logical.shape)
    original = ShardedTensor.from_rank_offsets(
        _KEY,
        weight,
        *sharded_offsets,
        (len(sharded_offsets), tp_rank * gtp_size + gtp_rank, tp_size * gtp_size),
        prepend_axis_num=len(sharded_offsets),
        replica_id=(0, 0, dist.get_rank(dp_group)),
    )
    # Real module constructors leave GTP size-one weights as ordinary parameters.
    # Only detection is mocked; collectives, metadata and tensors are real.
    with mock.patch.object(
        ssm_utils,
        'gtp_api',
        SimpleNamespace(HAVE_GTP=True, is_gtp_param=lambda x: x is weight and gtp_size > 1),
    ):
        factory = ssm_utils._split_in_proj_factory(
            original,
            sections,
            names,
            weight=weight,
            tp_group=tp_group,
            dp_cp_group=dp_group,
            sharded_offsets=sharded_offsets,
        )
    assert weight.requires_grad
    assert weight.grad is None
    expected = torch.nn.functional.pad(logical, (0, 0, 0, padding_rows)).chunk(gtp_size)[gtp_rank]
    return factory, logical, expected


@pytest.mark.parametrize('sections,names', _LAYOUTS)
@pytest.mark.parametrize('padding_rows', [0, 8], ids=['no_padding', 'with_padding'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16], ids=['fp32', 'bf16'])
def test_in_proj_factory_gtp_boundary_and_reshard(
    projection_groups, sections, names, padding_rows, dtype
):
    """Checkpoint sections preserve semantic rows across TP2/GTP2 -> TP1/GTP4 or TP2/GTP1."""
    groups = projection_groups
    tp_rank = dist.get_rank(groups['tp'])
    gtp_rank = dist.get_rank(groups['gtp'])
    dp_rank = dist.get_rank(groups['dp'])
    global_parts = _global_sections(sections, dtype)
    factory, logical, expected = _factory_for_topology(
        global_parts, names, groups['tp'], groups['gtp'], groups['dp'], padding_rows
    )
    # Every fixture deliberately places the GTP shard edge inside a semantic section.
    shard_rows = (sum(sections) + padding_rows) // 2
    assert shard_rows not in torch.tensor(sections).cumsum(0).tolist()
    torch.testing.assert_close(factory.data, logical, rtol=0, atol=0)
    assert not factory.data.requires_grad
    assert factory.data.grad_fn is None
    parts = factory.build()
    assert len(parts) == len(sections)
    for part, full, section, name in zip(parts, global_parts, sections, names):
        assert part.key == f'{_KEY}.{name}'
        assert part.local_shape == (section, 3)
        assert part.global_shape == (5, section * 2, 3)
        assert part.global_offset == (2, tp_rank * section, 0)
        assert part.axis_fragmentations == (5, 2, 1)
        assert part.replica_id == (0, gtp_rank, dp_rank)
        assert not part.allow_shape_mismatch
        assert not part.data.requires_grad
        assert part.data.grad_fn is None
        torch.testing.assert_close(part.data, full[tp_rank * section : (tp_rank + 1) * section])
    torch.testing.assert_close(
        factory.merge_fn([part.data for part in parts]), expected, rtol=0, atol=0
    )

    # Materialize a checkpoint by key and offset, admitting exactly one writer for each row.
    payloads = [None] * dist.get_world_size()
    dist.all_gather_object(payloads, parts)
    saved = {part.key: torch.empty_like(full) for part, full in zip(parts, global_parts)}
    coverage = {key: torch.zeros(value.shape[0], dtype=torch.int32) for key, value in saved.items()}
    for rank_parts in payloads:
        for part in rank_parts:
            if is_main_replica(part.replica_id):
                start, rows = part.global_offset[1], part.local_shape[0]
                saved[part.key][start : start + rows].copy_(part.data)
                coverage[part.key][start : start + rows] += 1
    assert all(torch.all(count == 1) for count in coverage.values())
    for name, full in zip(names, global_parts):
        torch.testing.assert_close(saved[f'{_KEY}.{name}'], full, rtol=0, atol=0)

    # Load using a different GTP degree and TP degree, including a GTP-disabled destination.
    for tp_group, gtp_group, dp_group in [
        (groups['single'], groups['four'], groups['dp']),
        (groups['tp'], groups['single'], groups['dp_without_gtp']),
    ]:
        target_rows = sum(full.shape[0] // dist.get_world_size(tp_group) for full in global_parts)
        target_gtp = dist.get_world_size(gtp_group)
        target_pad = (-target_rows) % target_gtp
        target, _, target_expected = _factory_for_topology(
            global_parts, names, tp_group, gtp_group, dp_group, target_pad
        )
        loaded_parts = [
            saved[part.key].narrow(0, part.global_offset[1], part.local_shape[0]).clone()
            for part in target.build()
        ]
        torch.testing.assert_close(target.merge_fn(loaded_parts), target_expected, rtol=0, atol=0)


@pytest.mark.parametrize('sections,names', _LAYOUTS)
def test_in_proj_factory_gtp_sharding_integrity(projection_groups, sections, names):
    """DCP sees complete, nonoverlapping source and destination checkpoint metadata."""
    groups = projection_groups
    global_parts = _global_sections(sections)
    for tp_group, gtp_group, dp_group in [
        (groups['tp'], groups['gtp'], groups['dp']),
        (groups['single'], groups['four'], groups['dp']),
        (groups['tp'], groups['single'], groups['dp_without_gtp']),
    ]:
        target_rows = sum(full.shape[0] // dist.get_world_size(tp_group) for full in global_parts)
        padding_rows = (-target_rows) % dist.get_world_size(gtp_group)
        # The offset regression above represents only layer 2 of 5. Use a complete
        # tensor here so DCP can validate both tensor coverage and replica ownership.
        factory, _, _ = _factory_for_topology(
            global_parts, names, tp_group, gtp_group, dp_group, padding_rows, sharded_offsets=()
        )
        _, metadata = determine_global_metadata({'in_proj': factory.build()})
        validate_sharding_integrity(metadata)


@pytest.mark.parametrize('section_delta', [-2, 2], ids=['too_few_rows', 'too_many_rows'])
def test_in_proj_factory_gtp_rejects_incorrect_sections(projection_groups, section_delta):
    """The section list cannot truncate real rows or reinterpret alignment padding as data."""
    groups = projection_groups
    logical = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    padded = torch.nn.functional.pad(logical, (0, 0, 0, 4))
    gtp_rank = dist.get_rank(groups['gtp'])
    weight = torch.nn.Parameter(padded.chunk(2)[gtp_rank].clone())
    weight.group = groups['gtp']
    weight.gtp_remat_size = 2
    weight.pad_length = 4
    weight._unsharded_shape = tuple(logical.shape)
    original = ShardedTensor.from_rank_offsets(_KEY, weight.data, (0, gtp_rank, 2))
    with (
        mock.patch.object(
            ssm_utils, 'gtp_api', SimpleNamespace(HAVE_GTP=True, is_gtp_param=lambda _: True)
        ),
        pytest.raises(ValueError, match='Split sections must cover the whole dimension size'),
    ):
        ssm_utils._split_in_proj_factory(
            original,
            [4, 4 + section_delta],
            ['x', 'z'],
            weight=weight,
            tp_group=groups['tp'],
            dp_cp_group=groups['dp'],
        )
