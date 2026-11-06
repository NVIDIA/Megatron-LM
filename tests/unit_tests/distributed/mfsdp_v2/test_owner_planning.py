# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed tests for owner assignment with real device meshes."""

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import Replicate

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import DBuffer
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.layout import GlobalLayout
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.owner_planning import (
    assign_owner_work,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import RowAtomic


def _layout(
    shapes: list[tuple[int, ...]], offsets: list[int], size: int, dp_size: int
) -> GlobalLayout:
    """Use explicit offsets so expected shard geometry is independent of packing."""
    return GlobalLayout(
        tensor_shapes=tuple(torch.Size(shape) for shape in shapes),
        tensor_to_offset=tuple(offsets),
        size=size,
        rank_to_offset=tuple(rank * (size // dp_size) for rank in range(dp_size)),
    )


@pytest.mark.parametrize(
    "shapes, offsets, size, ranks, expected",
    [
        # Equally expensive boundary tensors balance across their holders.
        ([(6, 4), (6, 4)], [0, 24], 48, [0, 1, 2], {0: 0, 1: 1}),
        # Preserve the original tensor index after filtering.
        ([(1,)] * 7 + [(8, 8)], list(range(7)) + [8], 80, [1, 3], {7: 1}),
        # Only ranks with nonempty shards can own a tensor.
        ([(6, 4)], [20], 64, [0, 1, 2, 3], {0: 1}),
        # Assign expensive boundary work before cheap work, after accounting for local work.
        ([(6, 2), (6, 4), (2, 2)], [12, 24, 0], 64, [0, 1, 2, 3], {0: 0, 1: 1, 2: 0}),
        # A fully local tensor stays with its holder.
        ([(4, 2)], [8], 16, [1, 3], {0: 3}),
        # The local tensor's cost biases the boundary tensor toward the other holder.
        ([(2, 2), (8, 8)], [0, 8], 80, [1, 3], {0: 1, 1: 3}),
        # Local work must count first even when the boundary tensor appears first.
        ([(8, 8), (2, 2)], [8, 0], 80, [1, 3], {0: 3, 1: 1}),
        # The eligibility rule excludes vectors.
        ([(4,)], [0], 8, [1, 3], {}),
        # A 3D tensor's cost uses its full shape, so it is assigned before the matrix.
        ([(1, 1), (3, 3), (2, 3, 4)], [0, 10, 20], 48, [0, 1, 2], {0: 0, 1: 0, 2: 1}),
        # Equal-cost boundary tensors follow tensor index.
        ([(1, 1), (2, 2), (2, 2)], [0, 14, 30], 48, [0, 1, 2], {0: 0, 1: 1, 2: 2}),
    ],
)
def test_assign_owner_work(shapes, offsets, size, ranks, expected, distributed_setup):
    """Balance eligible tensors on the specified mesh."""
    if distributed_setup.world_size <= max(ranks):
        pytest.skip(f"Mesh {ranks} requires at least {max(ranks) + 1} ranks.")
    mesh = DeviceMesh(distributed_setup.device.type, ranks)
    if mesh.get_coordinate() is None:
        assert assign_owner_work([]) == []  # Planning is collective over all ranks.
        pytest.skip("Rank is outside the test mesh.")
    layout = _layout(shapes, offsets, size, mesh.size())
    buffer = DBuffer(mesh, [RowAtomic()], layout, torch.float32, distributed_setup.device)
    assert assign_owner_work([buffer]) == [expected]


def test_assign_owner_work_requires_a_nonempty_shard(distributed_setup):
    """Every rank raises, including ranks that do not hold the buffer."""
    if distributed_setup.world_size < 2:
        pytest.skip("Needs at least two ranks.")
    mesh = DeviceMesh(distributed_setup.device.type, [0, 1])
    buffers = []
    if mesh.get_coordinate() is not None:
        layout = _layout([(0, 2)], [0], 4 * mesh.size(), mesh.size())
        buffers.append(
            DBuffer(mesh, [RowAtomic()], layout, torch.float32, distributed_setup.device)
        )
    with pytest.raises(RuntimeError, match="No eligible owner for tensor 0"):
        assign_owner_work(buffers)


def test_assign_owner_work_uses_buffer_placements(distributed_setup):
    """Replicated ranks hold the full buffer instead of all-RowAtomic fragments."""
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size, 1))
    layout = _layout([(2, 2)], [4 * (mesh.size() - 1)], 4 * mesh.size(), mesh.size())
    buffer = DBuffer(
        mesh, [Replicate(), RowAtomic()], layout, torch.float32, distributed_setup.device
    )
    # Every rank holds the tensor and plans for its own replica; a hard-coded
    # all-RowAtomic layout picks the last rank.
    assert assign_owner_work([buffer]) == [{0: mesh.get_rank()}]


@pytest.mark.parametrize("ranks", [[0, 1], [1, 3]])
@pytest.mark.parametrize(
    "first_shape, first_size, second_shape, second_size, expected",
    [
        # Forced work in a later group must count before an earlier group's boundary work.
        ((8, 8), 64, (8, 8), 128, [{0: 1}, {0: 0}]),
        # Expensive boundary work in a later group must be assigned before cheap work.
        ((8, 1), 8, (8, 8), 64, [{0: 1}, {0: 0}]),
        # Equal-cost boundary work follows group order.
        ((8, 8), 64, (8, 8), 64, [{0: 0}, {0: 1}]),
    ],
)
def test_assign_owner_work_across_groups(
    ranks, first_shape, first_size, second_shape, second_size, expected, distributed_setup
):
    """Balance all groups together, preserving global owner ranks on a subgroup mesh."""
    if distributed_setup.world_size <= max(ranks):
        pytest.skip(f"Mesh {ranks} requires at least {max(ranks) + 1} ranks.")
    mesh = DeviceMesh(distributed_setup.device.type, ranks)
    if mesh.get_coordinate() is None:
        # Join both collective planning calls below.
        assert assign_owner_work([]) == assign_owner_work([]) == []
        pytest.skip("Rank is outside the test mesh.")
    buffers = [
        DBuffer(
            mesh,
            [RowAtomic()],
            _layout([shape], [0], size, mesh.size()),
            torch.float32,
            distributed_setup.device,
        )
        for shape, size in [(first_shape, first_size), (second_shape, second_size)]
    ]
    assert assign_owner_work([buffers[0]]) == [{0: ranks[0]}]
    assert assign_owner_work(iter(buffers)) == [
        {index: ranks[owner] for index, owner in owners.items()} for owners in expected
    ]


def test_assign_owner_work_empty_groups(distributed_setup):
    """Buffers with no eligible tensors keep their result slot."""
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    buffer = DBuffer.empty(
        mesh, [Replicate()], [(4,), (8, 8)], torch.float32, distributed_setup.device
    )
    empty_buffer = DBuffer.empty(mesh, [Replicate()], [], torch.float32, distributed_setup.device)
    vector_buffer = DBuffer.empty(
        mesh, [Replicate()], [(4,)], torch.float32, distributed_setup.device
    )
    assert assign_owner_work([]) == []
    assert assign_owner_work([empty_buffer]) == [{}]
    # Each replicated rank owns the tensors it plans for.
    owners = assign_owner_work([empty_buffer, buffer, vector_buffer])
    assert owners == [{}, {1: mesh.get_rank()}, {}]


def test_assign_owner_work_balances_across_meshes(distributed_setup):
    """Work on another mesh counts toward a shared rank's load."""
    if distributed_setup.world_size < 2:
        pytest.skip("Needs at least two ranks.")
    device = distributed_setup.device
    buffers = [
        DBuffer.empty(mesh, [RowAtomic()], [(8, 8)], torch.float32, device)
        for mesh in (DeviceMesh(device.type, [0, 1]), DeviceMesh(device.type, [0]))
        if mesh.get_coordinate() is not None
    ]
    # Rank 0's matrix on the single-rank mesh moves the pair's shared matrix to rank 1.
    assert assign_owner_work(buffers) == [{0: 1}, {0: 0}][: len(buffers)]
