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
    ns_cost_fn,
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
    "shapes, offsets, size, ranks, indices, expected",
    [
        # Equally expensive boundary tensors balance across their holders.
        ([(6, 4), (6, 4)], [0, 24], 48, [0, 1, 2], [0, 1], {0: 0, 1: 1}),
        # Preserve the original tensor index after filtering.
        ([(1,)] * 7 + [(8, 8)], list(range(7)) + [8], 80, [1, 3], [7], {7: 1}),
        # Only ranks with nonempty shards can own a tensor.
        ([(6, 4)], [20], 64, [0, 1, 2, 3], [0], {0: 1}),
        # Assign expensive boundary work before cheap work, after accounting for local work.
        ([(6, 2), (6, 4), (2, 2)], [12, 24, 0], 64, [0, 1, 2, 3], [0, 1, 2], {0: 0, 1: 1, 2: 0}),
        # A fully local tensor stays with its holder.
        ([(4, 2)], [8], 16, [1, 3], [0], {0: 3}),
        # The local tensor's cost biases the boundary tensor toward the other holder.
        ([(2, 2), (8, 8)], [0, 8], 80, [1, 3], [0, 1], {0: 1, 1: 3}),
        # Local work must count first even when the boundary tensor appears first.
        ([(2, 2), (8, 8)], [0, 8], 80, [1, 3], [1, 0], {0: 1, 1: 3}),
        ([(4,)], [0], 8, [1, 3], [], {}),
    ],
)
def test_assign_owner_work(shapes, offsets, size, ranks, indices, expected, distributed_setup):
    """Balance tensors on the specified mesh using a single-pass iterable of indices."""
    if distributed_setup.world_size <= max(ranks):
        pytest.skip(f"Mesh {ranks} requires at least {max(ranks) + 1} ranks.")
    mesh = DeviceMesh(distributed_setup.device.type, ranks)
    if mesh.get_coordinate() is None:
        pytest.skip("Rank is outside the test mesh.")
    layout = _layout(shapes, offsets, size, mesh.size())
    buffer = DBuffer(mesh, [RowAtomic()], layout, torch.float32, distributed_setup.device)
    assert assign_owner_work([(buffer, iter(indices))]) == [expected]


def test_assign_owner_work_requires_a_nonempty_shard(distributed_setup):
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    buffer = DBuffer(
        mesh,
        [RowAtomic()],
        _layout([(0, 2)], [0], 4 * mesh.size(), mesh.size()),
        torch.float32,
        distributed_setup.device,
    )
    with pytest.raises(RuntimeError, match="No eligible owner for tensor 0"):
        assign_owner_work([(buffer, [0])])


def test_ns_cost_uses_full_shape():
    assert ns_cost_fn(5)(torch.Size((2, 3, 4))) == 24 * (2 * 5 + 1)


def test_assign_owner_work_uses_buffer_placements(distributed_setup):
    """Replicated ranks hold the full buffer instead of all-RowAtomic fragments."""
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size, 1))
    layout = _layout([(2, 2)], [4 * (mesh.size() - 1)], 4 * mesh.size(), mesh.size())
    buffer = DBuffer(
        mesh, [Replicate(), RowAtomic()], layout, torch.float32, distributed_setup.device
    )
    # Every rank holds the tensor; a hard-coded all-RowAtomic layout picks the last rank.
    assert assign_owner_work([(buffer, [0])]) == [{0: 0}]


def test_assign_owner_work_custom_selection_and_cost(distributed_setup):
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    buffer = DBuffer.empty(
        mesh, [RowAtomic()], [(6, 3), (4,), (4, 2)], torch.float32, distributed_setup.device
    )
    seen_shapes = []

    def cost(shape):
        seen_shapes.append(shape)
        return shape.numel()

    indices = (i for i, shape in enumerate(buffer.layout.tensor_shapes) if len(shape) >= 2)
    (tensor_to_owner,) = assign_owner_work([(buffer, indices)], cost_fn=cost)
    assert set(tensor_to_owner) == {0, 2}
    assert seen_shapes == [torch.Size((6, 3)), torch.Size((4, 2))]


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
    assert assign_owner_work([(buffers[0], [0])]) == [{0: ranks[0]}]
    assert assign_owner_work((buffer, iter([0])) for buffer in buffers) == [
        {index: ranks[owner] for index, owner in owners.items()} for owners in expected
    ]


def test_assign_owner_work_empty_groups(distributed_setup):
    """Empty groups keep their result slot, and selected tensors keep their buffer index."""
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    buffer = DBuffer.empty(
        mesh, [Replicate()], [(4,), (8, 8)], torch.float32, distributed_setup.device
    )
    assert assign_owner_work([]) == []
    assert assign_owner_work([(buffer, [])]) == [{}]
    assert assign_owner_work([(buffer, []), (buffer, [1]), (buffer, [])]) == [{}, {1: 0}, {}]


def test_assign_owner_work_tie_breaks_by_tensor_index(distributed_setup):
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    buffer = DBuffer.empty(
        mesh, [Replicate()], [(8, 8), (8, 8)], torch.float32, distributed_setup.device
    )
    assert assign_owner_work([(buffer, [1, 0])]) == [{0: 0, 1: 1}]


@pytest.mark.parametrize("other_ranks", [[0], [0, 2]])
def test_assign_owner_work_rejects_mixed_meshes(other_ranks, distributed_setup):
    """Reject both different mesh sizes and different rank sets of the same size."""
    if distributed_setup.world_size <= max(1, *other_ranks):
        pytest.skip(f"Test meshes require at least {max(1, *other_ranks) + 1} ranks.")
    meshes = [DeviceMesh(distributed_setup.device.type, ranks) for ranks in [[0, 1], other_ranks]]
    if any(mesh.get_coordinate() is None for mesh in meshes):
        pytest.skip("Rank is outside one of the test meshes.")
    buffers = [
        DBuffer.empty(mesh, [RowAtomic()], [(8, 8)], torch.float32, distributed_setup.device)
        for mesh in meshes
    ]
    with pytest.raises(ValueError, match="same DP mesh"):
        assign_owner_work([(buffer, [0]) for buffer in buffers])
