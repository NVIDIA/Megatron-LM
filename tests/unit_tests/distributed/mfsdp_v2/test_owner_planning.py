# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed tests for owner assignment and packing with real device meshes."""

import pytest
import torch
import torch.distributed as dist
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
    OwnerGatherPlan,
    OwnerScatterPlan,
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
    plan = GroupOwnerLayout.from_group(group)
    assert plan.mesh is group.mesh
    assert plan.layout is layout
    assert set(plan.tensor_to_owner) == {0, 2}
    assert plan.tensor_to_owner == assign_owner_work(layout, group.mesh, [0, 2])


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

    plan = GroupOwnerLayout.from_group(group, cost_fn=cost, eligible_fn=lambda p: p.numel() >= 8)
    assert plan.layout is group.main_weight.layout
    assert set(plan.tensor_to_owner) == {0, 1}
    assert seen_shapes == [torch.Size((6, 3)), torch.Size((4, 2))]


def _exchange_buffers(
    plan: OwnerGatherPlan | OwnerScatterPlan, device: torch.device
) -> dict[int, torch.Tensor]:
    """Exchange packed buffers using the plan's global peer ranks and receive sizes."""
    received = {
        src: torch.empty(numel, dtype=torch.float32, device=device)
        for src, numel in plan.recv_sizes.items()
    }
    ops = [dist.P2POp(dist.irecv, buffer, src) for src, buffer in received.items()]
    ops.extend(dist.P2POp(dist.isend, buffer, dest) for dest, buffer in plan.send_buffers.items())
    if ops:
        for request in dist.batch_isend_irecv(ops):
            request.wait()
    return received


@pytest.mark.parametrize("ranks", [(0, 1, 2), (4, 1, 7)])
def test_pack_gather_scatter_round_trip(ranks, distributed_setup):
    """Reconstruct full tensors and scatter changed results using global peer ranks.

    The hand-written shard slices check geometry independently of the range helpers.
    The last tensor is fully local, and the buffer includes trailing padding.
    """
    if distributed_setup.world_size <= max(ranks):
        pytest.skip(f"Mesh {ranks} requires at least {max(ranks) + 1} ranks.")
    device = distributed_setup.device
    mesh = DeviceMesh(device.type, ranks)
    # Initialize WORLD on every rank before a subset uses it for batched P2P.
    dist.barrier(device_ids=[device.index] if device.type == "cuda" else None)
    if mesh.get_coordinate() is None:
        pytest.skip("Rank is outside the test mesh.")
    rank = mesh.get_rank()
    index = ranks.index(rank)
    layout = _layout([(6, 3), (4, 2), (2, 2)], [0, 18, 26], 36, 3)
    fulls = {
        i: torch.arange(shape.numel(), dtype=torch.float32, device=device).view(shape) + 100 * i
        for i, shape in enumerate(layout.tensor_shapes)
    }
    slices = [
        {0: slice(0, 12)},
        {0: slice(12, 18), 1: slice(0, 6)},
        {1: slice(6, 8), 2: slice(0, 4)},
    ]
    local_slices = slices[index]
    tensor_to_owner = assign_owner_work(layout, mesh, range(3))
    assert tensor_to_owner == {i: ranks[i] for i in range(3)}
    # Deliberately reverse dictionary order; packing must follow tensor indices.
    plan = GroupOwnerLayout(mesh, layout, dict(reversed(list(tensor_to_owner.items()))))
    shards = {i: fulls[i].flatten()[part].clone() for i, part in local_slices.items()}
    gather = OwnerGatherPlan.pack(plan, shards)
    assert gather.recv_sizes == [{ranks[1]: 6}, {ranks[2]: 2}, {}][index]
    gathered = _exchange_buffers(gather, device)
    full = gather.reconstruct_full(index, gathered)
    torch.testing.assert_close(full, fulls[index].flatten(), atol=0, rtol=0)

    scatter = OwnerScatterPlan.pack(plan, {index: (full + 1).view(fulls[index].shape)})
    scattered = _exchange_buffers(scatter, device)
    results = scatter.unpack(scattered)
    assert set(results) == {i for i in local_slices if tensor_to_owner[i] != rank}
    for i, result in results.items():
        expected = fulls[i].flatten()[local_slices[i]] + 1
        torch.testing.assert_close(result, expected, atol=0, rtol=0)


def test_pack_preserves_buffer_order_on_2d_mesh(distributed_setup):
    if distributed_setup.world_size < 4:
        pytest.skip("The 2D mesh requires at least 4 ranks.")
    device = distributed_setup.device
    mesh = DeviceMesh(device.type, [[3, 1], [0, 2]])
    # Initialize WORLD on every rank before a subset uses it for batched P2P.
    dist.barrier(device_ids=[device.index] if device.type == "cuda" else None)
    if mesh.get_coordinate() is None:
        pytest.skip("Rank is outside the test mesh.")
    layout = _layout([(8, 4)], [0], 32, 4)
    full = torch.arange(32, dtype=torch.float32, device=device)
    # Reverse-axis sharding orders the chunks as ranks 3, 0, 1, 2.
    offsets = {3: 0, 0: 8, 1: 16, 2: 24}
    rank = mesh.get_rank()
    offset = offsets[rank]
    tensor_to_owner = assign_owner_work(layout, mesh, [0])
    assert tensor_to_owner == {0: 3}
    plan = GroupOwnerLayout(mesh, layout, tensor_to_owner)
    gather = OwnerGatherPlan.pack(plan, {0: full[offset : offset + 8]})
    received = _exchange_buffers(gather, device)
    if rank == 3:
        torch.testing.assert_close(gather.reconstruct_full(0, received), full)

    scatter = OwnerScatterPlan.pack(plan, {0: full + 1} if rank == 3 else {})
    received = _exchange_buffers(scatter, device)
    result = scatter.unpack(received)
    if rank == 3:
        assert result == {}
    else:
        torch.testing.assert_close(result[0], full[offset : offset + 8] + 1)


def test_pack_with_no_eligible_params(distributed_setup):
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    module = nn.ParameterList([nn.Parameter(torch.zeros(16, device=distributed_setup.device))])
    with fully_shard_context(device=distributed_setup.device):
        fully_shard(
            module,
            mesh=mesh,
            placements=Placements(
                dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
            ),
        )
    (group,) = module.parameter_groups
    plan = GroupOwnerLayout.from_group(group)
    assert plan.layout is group.main_weight.layout
    assert plan.tensor_to_owner == {}
    gather = OwnerGatherPlan.pack(plan, {})
    assert gather.send_buffers == {}
    assert gather.recv_sizes == {}
    assert gather.own_shards == {}
    assert gather.recv_offsets == {}
    scatter = OwnerScatterPlan.pack(plan, {})
    assert scatter.send_buffers == {}
    assert scatter.recv_sizes == {}
    assert scatter.recv_offsets == {}
    assert scatter.unpack({}) == {}
