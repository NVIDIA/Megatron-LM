# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Chunk-metadata tests for the experimental Megatron-FSDP path."""

import math

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.uneven_dtensor import (
    chunk_metadata_by_fqn,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    gather_and_compute_chunk_metadata,
)

# Collectives the analytic chunk metadata must not need. Megatron-FSDP's generic helper recovers
# each shard's offset with one all_gather_object per DTensor; this path derives it from the layout.
_COLLECTIVE_NAMES = ("all_gather", "all_gather_object", "all_gather_into_tensor", "all_reduce")


class _PackedBlock(nn.Module):
    """Three parameters whose flat packing leaves a padding gap that a smaller one fills.

    Row sizes 4, 2, and 6 give a chunk size of 12, so ``regular`` (16 elements) pads to 24 and
    ``fragment`` (4 elements) is placed in the 8-element gap it leaves. Over two ranks the buffer
    is 48 elements, so the rank boundary at 24 lands after ``fragment`` ends: rank 1 owns none of
    ``regular`` or ``fragment`` and all of ``tail``, which no canonical ``Shard(0)`` split of any
    of the three parameters describes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.regular = nn.Parameter(torch.randn(4, 4))
        self.fragment = nn.Parameter(torch.randn(2, 2))
        self.tail = nn.Parameter(torch.randn(2, 6))


class _PackedModel(nn.Module):
    """A gap-filling parameter group and a Linear whose weight and bias pack unevenly."""

    def __init__(self) -> None:
        super().__init__()
        self.block = _PackedBlock()
        self.linear = nn.Linear(8, 16)


def _build_sharded(mesh: DeviceMesh, device: torch.device) -> nn.Module:
    model = _PackedModel().to(device=device)
    placements = Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])
    fully_shard(model.block, mesh=mesh, placements=placements)
    fully_shard(model.linear, mesh=mesh, placements=placements)
    return model


def _even_shard_rows(rows: int, world_size: int, rank: int) -> int:
    """Rows a canonical ``Shard(0)`` split would give this rank."""
    shard_rows = math.ceil(rows / world_size)
    return min(max(rows - rank * shard_rows, 0), shard_rows)


def test_chunk_metadata_matches_gathered_metadata(distributed_setup) -> None:
    """The analytic chunk metadata equals what gathering every shard's shape computes.

    Megatron-FSDP's generic helper recovers offsets from an all-gather of local shard shapes.
    Deriving them from the layout instead must describe exactly the same chunks, otherwise this
    path would write checkpoints that differ from the ones the stable path writes.
    """
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = _build_sharded(mesh, device)

    metadata_by_fqn = chunk_metadata_by_fqn(model)
    parameters = dict(model.named_parameters())
    assert metadata_by_fqn.keys() == parameters.keys()

    uneven = False
    for fqn, parameter in parameters.items():
        chunk = metadata_by_fqn[fqn]
        expected = gather_and_compute_chunk_metadata(parameter)
        assert tuple(chunk.offsets) == tuple(expected.offsets), f"{fqn} chunk offsets"
        assert tuple(chunk.sizes) == tuple(expected.sizes), f"{fqn} chunk sizes"
        uneven = uneven or chunk.sizes[0] != _even_shard_rows(
            parameter.shape[0], distributed_setup.world_size, distributed_setup.rank
        )

    # Chunk metadata only matters for shards a canonical Shard(0) split would misplace, so this
    # rank must hold at least one such shard for the comparison above to mean anything. On a
    # single rank nothing can be misplaced: the one shard is the whole tensor.
    if distributed_setup.world_size > 1:
        assert uneven, "No parameter sharded unevenly, so the metadata was not exercised."


def test_chunk_metadata_issues_no_collectives(
    distributed_setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Computing the chunk metadata is rank-local."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = _build_sharded(mesh, device)

    def _fail(*args, **kwargs):
        raise AssertionError("Computing chunk metadata must not issue a collective.")

    for name in _COLLECTIVE_NAMES:
        monkeypatch.setattr(dist, name, _fail)
    metadata_by_fqn = chunk_metadata_by_fqn(model)

    assert metadata_by_fqn.keys() == dict(model.named_parameters()).keys()


def test_chunk_metadata_covers_every_parameter_once(distributed_setup) -> None:
    """Each parameter's chunks tile its global shape exactly once."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = _build_sharded(mesh, device)

    metadata_by_fqn = chunk_metadata_by_fqn(model)
    all_metadata = [None] * distributed_setup.world_size
    dist.all_gather_object(all_metadata, metadata_by_fqn)

    for fqn, parameter in model.named_parameters():
        chunks = sorted(
            (metadata[fqn] for metadata in all_metadata), key=lambda chunk: chunk.offsets[0]
        )
        covered = 0
        for chunk in chunks:
            if chunk.sizes[0] == 0:
                continue
            assert chunk.offsets[0] == covered, f"{fqn} chunks are not contiguous"
            assert tuple(chunk.sizes[1:]) == tuple(parameter.shape[1:]), f"{fqn} chunk shape"
            covered += chunk.sizes[0]
        assert covered == parameter.shape[0], f"{fqn} chunks do not cover the global tensor"
