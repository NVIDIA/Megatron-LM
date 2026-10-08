# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests for owner assignment and packing, with P2P simulated in-process."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.layout import GlobalLayout
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.owner_planning import (
    GroupOwnerLayout,
    OwnerGatherPlan,
    OwnerScatterPlan,
    assign_owner_work,
    ns_cost_fn,
)


def _mock_mesh(ranks, this_rank=None):
    """Supply mesh metadata without initializing a process group for packing tests."""
    rank_tensor = torch.tensor(ranks)
    return SimpleNamespace(
        mesh=rank_tensor,
        ndim=rank_tensor.ndim,
        size=lambda axis=None: rank_tensor.numel() if axis is None else rank_tensor.size(axis),
        get_rank=lambda: ranks[0] if this_rank is None else this_rank,
    )


def _layout(shapes, offsets, size, dp_size):
    return GlobalLayout(
        tensor_shapes=tuple(torch.Size(shape) for shape in shapes),
        tensor_to_offset=tuple(offsets),
        size=size,
        rank_to_offset=tuple(rank * (size // dp_size) for rank in range(dp_size)),
    )


def _mock_group(layout, mesh):
    """Supply only the group fields used by the owner-layout factory."""
    return SimpleNamespace(
        mesh=mesh,
        main_weight=SimpleNamespace(layout=layout),
        fsdp_parameters=tuple(
            SimpleNamespace(sharded=nn.Parameter(torch.empty(shape)))
            for shape in layout.tensor_shapes
        ),
    )


@pytest.mark.parametrize(
    'shapes, offsets, size, ranks, indices, expected',
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
        ([(4,)], [0], 8, [1, 3], [], {}),
    ],
)
def test_assign_owner_work(shapes, offsets, size, ranks, indices, expected):
    assert (
        assign_owner_work(_layout(shapes, offsets, size, len(ranks)), _mock_mesh(ranks), indices)
        == expected
    )


def test_assign_owner_work_requires_a_nonempty_shard():
    with pytest.raises(RuntimeError, match='No eligible owner for tensor 0'):
        assign_owner_work(_layout([(0, 2)], [0], 8, 2), _mock_mesh([1, 3]), [0])


def test_ns_cost_uses_full_shape():
    assert ns_cost_fn(5)(torch.Size((2, 3, 4))) == 24 * (2 * 5 + 1)


def test_group_owner_layout_reuses_dbuffer_layout():
    layout = _layout([(6, 3), (4,), (4, 2)], [0, 18, 22], 36, 3)
    group = _mock_group(layout, _mock_mesh([1, 3, 5]))
    plan = GroupOwnerLayout.from_group(group)
    assert plan.mesh is group.mesh
    assert plan.layout is layout
    assert set(plan.owners) == {0, 2}
    assert plan.owners == assign_owner_work(layout, group.mesh, [0, 2])


def test_group_owner_layout_custom_eligibility_and_cost():
    layout = _layout([(6, 3), (4, 2), (2, 2)], [0, 18, 26], 36, 3)
    group = _mock_group(layout, _mock_mesh([4, 1, 7]))
    seen_shapes = []

    def cost(shape):
        seen_shapes.append(shape)
        return shape.numel()

    plan = GroupOwnerLayout.from_group(group, cost_fn=cost, eligible_fn=lambda p: p.numel() >= 8)
    assert plan.layout is layout
    assert set(plan.owners) == {0, 1}
    assert seen_shapes == [torch.Size((6, 3)), torch.Size((4, 2))]


def _simulate_p2p(plans):
    """Deliver each packed peer buffer to its destination's receive dictionary."""
    received = {rank: {} for rank in plans}
    for src, plan in plans.items():
        for dest, buffer in plan.send_buffers.items():
            received[dest][src] = buffer.clone()
    for rank, plan in plans.items():
        assert {src: buffer.numel() for src, buffer in received[rank].items()} == plan.recv_sizes
    return received


@pytest.mark.parametrize('ranks', [(0, 1, 2), (4, 1, 7)])
def test_pack_gather_scatter_round_trip(ranks):
    """Reconstruct full tensors and scatter changed results using global peer ranks.

    The hand-written shard slices check geometry independently of the range helpers.
    The last tensor is fully local, and the buffer includes trailing padding.
    """
    layout = _layout([(6, 3), (4, 2), (2, 2)], [0, 18, 26], 36, 3)
    fulls = {
        i: torch.arange(shape.numel(), dtype=torch.float32).view(shape) + 100 * i
        for i, shape in enumerate(layout.tensor_shapes)
    }
    slices = [
        {0: slice(0, 12)},
        {0: slice(12, 18), 1: slice(0, 6)},
        {1: slice(6, 8), 2: slice(0, 4)},
    ]
    owner_plans = {}
    gather_plans = {}
    for rank, local_slices in zip(ranks, slices):
        mesh = _mock_mesh(ranks, rank)
        owners = assign_owner_work(layout, mesh, range(3))
        assert owners == {i: ranks[i] for i in range(3)}
        # Deliberately reverse dictionary order; packing must follow tensor indices.
        plan = GroupOwnerLayout(mesh, layout, dict(reversed(list(owners.items()))))
        owner_plans[rank] = plan
        shards = {i: fulls[i].flatten()[part].clone() for i, part in local_slices.items()}
        gather_plans[rank] = OwnerGatherPlan.pack(plan, shards)

    gathered = _simulate_p2p(gather_plans)
    assert gather_plans[ranks[0]].recv_sizes == {ranks[1]: 6}
    assert gather_plans[ranks[2]].recv_sizes == {}
    scatter_plans = {}
    for i, rank in enumerate(ranks):
        full = gather_plans[rank].reconstruct_full(i, gathered[rank])
        torch.testing.assert_close(full, fulls[i].flatten(), atol=0, rtol=0)
        scatter_plans[rank] = OwnerScatterPlan.pack(
            owner_plans[rank], {i: (full + 1).view(fulls[i].shape)}
        )

    scattered = _simulate_p2p(scatter_plans)
    for rank, local_slices in zip(ranks, slices):
        results = scatter_plans[rank].unpack(scattered[rank])
        assert set(results) == {i for i in local_slices if owner_plans[rank].owners[i] != rank}
        for i, result in results.items():
            expected = fulls[i].flatten()[local_slices[i]] + 1
            torch.testing.assert_close(result, expected, atol=0, rtol=0)


def test_pack_preserves_buffer_order_on_2d_mesh():
    layout = _layout([(8, 4)], [0], 32, 4)
    full = torch.arange(32, dtype=torch.float32)
    # Reverse-axis sharding orders the chunks as ranks 3, 0, 1, 2.
    offsets = {3: 0, 0: 8, 1: 16, 2: 24}
    owner_plans = {}
    gather_plans = {}
    for rank, offset in offsets.items():
        mesh = _mock_mesh([[3, 1], [0, 2]], rank)
        owners = assign_owner_work(layout, mesh, [0])
        assert owners == {0: 3}
        plan = GroupOwnerLayout(mesh, layout, owners)
        owner_plans[rank] = plan
        gather_plans[rank] = OwnerGatherPlan.pack(plan, {0: full[offset : offset + 8]})
    received = _simulate_p2p(gather_plans)
    torch.testing.assert_close(gather_plans[3].reconstruct_full(0, received[3]), full)

    scatter_plans = {
        rank: OwnerScatterPlan.pack(plan, {0: full + 1} if rank == 3 else {})
        for rank, plan in owner_plans.items()
    }
    received = _simulate_p2p(scatter_plans)
    for rank, plan in scatter_plans.items():
        result = plan.unpack(received[rank])
        if rank == 3:
            assert result == {}
        else:
            offset = offsets[rank]
            torch.testing.assert_close(result[0], full[offset : offset + 8] + 1)


def test_pack_with_no_eligible_params():
    group = _mock_group(_layout([(16,)], [0], 16, 2), _mock_mesh([1, 3]))
    plan = GroupOwnerLayout.from_group(group)
    assert plan.layout is group.main_weight.layout
    assert plan.owners == {}
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
