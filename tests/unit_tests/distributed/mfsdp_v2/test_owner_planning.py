# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed tests for owner assignment with real device meshes."""

import pytest
import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.layout import GlobalLayout
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.owner_planning import (
    GroupOwnerLayout,
    assign_owner_work,
    ns_cost_fn,
)


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
    assert assign_owner_work(layout, mesh, iter(indices)) == expected


def test_assign_owner_work_requires_a_nonempty_shard(distributed_setup):
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    with pytest.raises(RuntimeError, match="No eligible owner for tensor 0"):
        assign_owner_work(_layout([(0, 2)], [0], 4 * mesh.size(), mesh.size()), mesh, [0])


def test_ns_cost_uses_full_shape():
    assert ns_cost_fn(5)(torch.Size((2, 3, 4))) == 24 * (2 * 5 + 1)


def test_group_owner_layout_reuses_dbuffer_layout(distributed_setup):
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    module = nn.ParameterList(
        nn.Parameter(torch.zeros(shape, device=distributed_setup.device))
        for shape in [(6, 3), (4,), (4, 2)]
    )
    with fully_shard_context(device=distributed_setup.device):
        fully_shard(
            module,
            mesh=mesh,
            placements=Placements(
                dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
            ),
        )
    (group,) = module.parameter_groups
    layout = group.main_weight.layout
    owner_layout = GroupOwnerLayout.from_group(group)
    assert owner_layout.mesh is group.mesh
    assert owner_layout.layout is layout
    assert set(owner_layout.tensor_to_owner) == {0, 2}
    assert owner_layout.tensor_to_owner == assign_owner_work(layout, group.mesh, [0, 2])


def test_group_owner_layout_custom_eligibility_and_cost(distributed_setup):
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    module = nn.ParameterList(
        nn.Parameter(torch.zeros(shape, device=distributed_setup.device))
        for shape in [(6, 3), (4, 2), (2, 2)]
    )
    with fully_shard_context(device=distributed_setup.device):
        fully_shard(
            module,
            mesh=mesh,
            placements=Placements(
                dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
            ),
        )
    (group,) = module.parameter_groups
    seen_shapes = []

    def cost(shape):
        seen_shapes.append(shape)
        return shape.numel()

    owner_layout = GroupOwnerLayout.from_group(
        group, cost_fn=cost, eligible_fn=lambda p: p.numel() >= 8
    )
    assert owner_layout.layout is group.main_weight.layout
    assert set(owner_layout.tensor_to_owner) == {0, 1}
    assert seen_shapes == [torch.Size((6, 3)), torch.Size((4, 2))]
