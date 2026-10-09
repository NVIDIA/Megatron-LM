# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Pure CPU tests for the parameter layout and owner-compute packing logic.

These tests exercise functions without a process group or any `torch.distributed` dependency. P2P
communication is simulated in-process by `_simulate_p2p`.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.layout import GlobalLayout
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.owner_planning import (
    GroupOwnerLayout,
    OwnerGatherPlan,
    OwnerScatterPlan,
    ParameterLayout,
    assign_owner_work,
    ns_cost_fn,
)


def _mock_mesh(dp_size: int, this_rank: int):
    """Mock a `DeviceMesh` providing just the DP size and this rank's index."""
    return SimpleNamespace(size=lambda: dp_size, get_local_rank=lambda: this_rank)


def _mock_group(shapes, dp_size, this_rank=0, mesh=None):
    """Mock a `FsdpParameterGroup` whose DBuffer uses the RowAtomic layout built for `shapes`.

    Creates `nn.Parameter`s for each shape so the default `eligible_fn` (`param.ndim >= 2`) can
    filter on them. When `mesh` is given, the group uses that mesh; pass one shared mesh to mock
    several groups on the same DP mesh.
    """
    layout = GlobalLayout.build_for_row_atomic(shapes, dp_size)
    params = tuple(nn.Parameter(torch.zeros(s)) for s in shapes)
    fsdp_parameters = tuple(SimpleNamespace(sharded=p) for p in params)
    if mesh is None:
        mesh = _mock_mesh(dp_size, this_rank)
    return SimpleNamespace(
        mesh=mesh,
        main_weight=SimpleNamespace(layout=layout),
        fsdp_parameters=fsdp_parameters,
        sharded_parameters=params,
    )


# ---------------------------------------------------------------------------
# Layout construction
# ---------------------------------------------------------------------------


def test_from_group_even_split():
    """A parameter exactly divisible by dp_size splits evenly across ranks."""
    group = _mock_group([(8, 4)], dp_size=2)
    layouts = ParameterLayout.from_group(group)
    assert list(layouts) == [0]
    layout = layouts[0]
    assert layout.full_shape == torch.Size((8, 4))
    assert layout.full_numel() == 32
    assert layout.flat_counts == (16, 16)
    assert layout.rank_numel(0) == 16
    assert layout.rank_numel(1) == 16
    assert layout.rank_offset(1) == 16


def test_from_group_boundary_param_split_across_ranks():
    """A parameter landing across a rank boundary splits at the flat element level."""
    # 6×3 = 18 elements; each rank's flat shard is 9 elements. rank0 owns flat [0, 9), rank1 owns
    # [9, 18). Tensor occupies [0, 18) fully.
    group = _mock_group([(6, 3)], dp_size=2)
    layout = ParameterLayout.from_group(group)[0]
    assert layout.flat_counts == (9, 9)
    assert layout.is_boundary()


def test_from_group_fully_local_params():
    """A parameter fully contained in one rank's flat shard has exactly one holder."""
    # Two 4×3 tensors (12 elements each) at offsets 0 and 12. rank0 shard = [0, 12),
    # rank1 = [12, 24). Each tensor lies entirely in one rank.
    layouts = ParameterLayout.from_group(_mock_group([(4, 3), (4, 3)], dp_size=2))

    assert layouts[0].flat_counts == (12, 0)
    assert not layouts[0].is_boundary()
    assert layouts[0].owner_candidates() == (0,)

    # A leading empty rank: rank0 holds nothing, so rank1's flat offset within the tensor is 0.
    assert layouts[1].flat_counts == (0, 12)
    assert not layouts[1].is_boundary()
    assert layouts[1].owner_candidates() == (1,)
    assert layouts[1].rank_offset(1) == 0


def test_from_group_keys_are_tensor_indices():
    """Keys are the params' tensor indices; non-eligible params leave gaps."""
    # 2D weight (tensor 0), 1D bias (tensor 1), 2D weight (tensor 2).
    # Only the ≥2D params are eligible (default `eligible_fn`); the 1D bias is
    # excluded.
    group = _mock_group([(8, 4), (16,), (4, 4)], dp_size=2)
    layouts = ParameterLayout.from_group(group)
    assert list(layouts) == [0, 2]
    assert layouts[0].full_shape == torch.Size((8, 4))
    assert layouts[2].full_shape == torch.Size((4, 4))


def test_from_group_per_rank_data_not_uniform():
    """RowAtomic sharding with uniform buffer size but non-uniform per-rank tensor data.

    `GlobalLayout.build` pads the total size to a multiple of `chunk_size * dp_size` so every rank's
    flat buffer is the same size. However, the actual tensor data per rank is not necessarily
    uniform.

    This test verifies `from_group` correctly computes the non-uniform per-rank element counts from
    a uniform `rank_flat_shard_size`.
    """
    # 5×4 = 20 elements, dp_size = 3.
    # GlobalLayout.build pads to 24 (next multiple of chunk_size * dp_size = 4 * 3 = 12), so
    # rank_flat_shard_size = 24 // 3 = 8 (uniform buffer per rank).
    # But the tensor occupies only 20 of 24 elements:
    #   rank 0: [0, 8)   -> 8 elements
    #   rank 1: [8, 16)  -> 8 elements
    #   rank 2: [16, 24) -> 4 elements (4 elements of padding)
    group = _mock_group([(5, 4)], dp_size=3)
    layout = ParameterLayout.from_group(group)[0]
    assert layout.flat_counts == (8, 8, 4)
    assert layout.rank_numel(0) == 8
    assert layout.rank_numel(1) == 8
    assert layout.rank_numel(2) == 4
    assert layout.rank_offset(2) == 16
    assert layout.is_boundary()


def test_from_group_is_shape_agnostic():
    """The flat math ignores shape: 3D and 1D params split by elements."""
    # A 3D conv-like weight (tensor 0) and a 1D vector (tensor 1), each 24 elements, placed on
    # opposite ranks. With the default `eligible_fn`, only the ≥2D param participates; with the
    # override, both do.
    group = _mock_group([(2, 3, 4), (24,)], dp_size=2)
    layouts = ParameterLayout.from_group(group)
    assert list(layouts) == [0]
    assert layouts[0].flat_counts == (24, 0)

    layouts = ParameterLayout.from_group(group, eligible_fn=lambda _param: True)
    assert list(layouts) == [0, 1]
    assert layouts[0].flat_counts == (24, 0)
    assert layouts[1].flat_counts == (0, 24)


def test_parameter_layout_validates_counts_sum():
    """`flat_counts` must sum to the parameter's numel."""
    with pytest.raises(ValueError, match="flat_counts sum"):
        ParameterLayout(torch.Size((4, 2)), (4, 0))


# ---------------------------------------------------------------------------
# Owner assignment balancing
# ---------------------------------------------------------------------------


def test_assign_owner_work_balances_by_cost():
    """Owners are balanced so the cheapest eligible rank takes each parameter."""
    # Two boundary params, all four ranks eligible for each.
    layout0 = ParameterLayout(torch.Size((8, 8)), (16, 16, 16, 16))  # cost 64 * 41
    layout1 = ParameterLayout(torch.Size((4, 4)), (4, 4, 4, 4))  # cost 16 * 21
    # Greedy min running cost: first param -> rank0 (cost 2624), second -> rank1 (cost 336).
    owners = assign_owner_work([{0: layout0, 1: layout1}], ns_cost_fn(5))
    assert owners == [{0: 0, 1: 1}]


def test_assign_owner_work_keys_pass_through():
    """Keys pass through unchanged. Arbitrary tensor indices, not positions."""
    layout = ParameterLayout(torch.Size((8, 8)), (16, 16, 16, 16))
    (owners,) = assign_owner_work([{7: layout}], ns_cost_fn(5))
    assert owners == {7: 0}


def test_assign_owner_work_only_eligible_ranks_can_own():
    """A rank with an empty shard can never be the owner."""
    # Only ranks 0 and 2 hold elements for both params.
    layout0 = ParameterLayout(torch.Size((8, 8)), (32, 0, 32, 0))
    layout1 = ParameterLayout(torch.Size((8, 8)), (32, 0, 32, 0))
    (owners,) = assign_owner_work([{0: layout0, 1: layout1}], ns_cost_fn(5))
    assert all(owner in (0, 2) for owner in owners.values())
    # Two equal-cost params split across the two eligible ranks.
    assert owners[0] != owners[1]


def test_assign_owner_work_lpt_sorts_by_descending_cost():
    """Params are assigned in descending cost order, not input order.

    With two boundary params (cheap first, expensive second in input order), LPT processes the
    expensive one first.
    """
    # Cheap param listed FIRST in input order.
    layout_cheap = ParameterLayout(torch.Size((8, 1)), (2, 2, 2, 2))
    # Expensive param listed SECOND in input order.
    layout_expensive = ParameterLayout(torch.Size((8, 8)), (16, 16, 16, 16))
    (owners,) = assign_owner_work([{0: layout_cheap, 1: layout_expensive}], ns_cost_fn(5))
    # LPT: expensive (tensor 1) → rank0 first, then cheap (tensor 0) → rank1.
    assert owners == {0: 1, 1: 0}


def test_assign_owner_work_non_boundary_gets_sole_holder():
    """Non-boundary params are assigned their sole holder, not skipped."""
    # All 8 elements on rank 0; rank 1 holds nothing.
    layout = ParameterLayout(torch.Size((4, 2)), (8, 0))
    (owners,) = assign_owner_work([{3: layout}], ns_cost_fn(3))
    assert owners == {3: 0}


def test_assign_owner_work_non_boundary_cost_counts_toward_balance():
    """Forced non-boundary work biases the greedy balancer away from that rank."""
    non_boundary = ParameterLayout(torch.Size((2, 2)), (4, 0))
    boundary = ParameterLayout(torch.Size((8, 8)), (32, 32))
    (owners,) = assign_owner_work([{0: non_boundary, 1: boundary}], ns_cost_fn(5))
    assert owners == {0: 0, 1: 1}


# ---------------------------------------------------------------------------
# Cross-group owner assignment balancing
# ---------------------------------------------------------------------------


def test_assign_owner_work_balances_across_groups():
    """Joint balancing: load forced onto a rank by one group steers other groups' boundary params
    away from that rank.
    """
    # Group 0's non-boundary param is rank 0's alone (cost 64 * 41 = 2624).
    forced = ParameterLayout(torch.Size((8, 8)), (64, 0))
    # Group 1's boundary param is eligible on both ranks (same cost 2624).
    boundary = ParameterLayout(torch.Size((8, 8)), (32, 32))
    # Alone, all running costs are 0 and the lowest eligible rank wins.
    (owners,) = assign_owner_work([{0: boundary}], ns_cost_fn(5))
    assert owners == {0: 0}
    # Jointly, rank 0 already carries group 0's forced cost, so rank 1 takes the boundary param.
    owners_per_group = assign_owner_work([{0: forced}, {0: boundary}], ns_cost_fn(5))
    assert owners_per_group == [{0: 0}, {0: 1}]


def test_assign_owner_work_lpt_across_groups():
    """Boundary params are longest-processing-time-sorted across groups: an expensive param in a
    later group is assigned before a cheap param in an earlier one.
    """
    cheap = ParameterLayout(torch.Size((8, 1)), (2, 2, 2, 2))  # cost 8 * 6 = 48
    expensive = ParameterLayout(torch.Size((8, 8)), (16, 16, 16, 16))  # cost 64 * 41 = 2624
    owners_per_group = assign_owner_work([{0: cheap}, {0: expensive}], ns_cost_fn(5))
    # LPT: expensive (group 1) → rank 0 first, then cheap (group 0) → rank 1.
    assert owners_per_group == [{0: 1}, {0: 0}]


def test_assign_owner_work_tie_break_across_groups():
    """Equal-cost boundary params are assigned in (group, tensor index) order."""
    layout = ParameterLayout(torch.Size((8, 8)), (32, 32))  # cost 2624 each
    owners_per_group = assign_owner_work([{0: layout}, {0: layout}], ns_cost_fn(5))
    assert owners_per_group == [{0: 0}, {0: 1}]
    # Same tie-break within one group: tensor 0 comes before tensor 1.
    (owners,) = assign_owner_work([{0: layout, 1: layout}], ns_cost_fn(5))
    assert owners == {0: 0, 1: 1}


def test_assign_owner_work_empty_groups():
    """No layouts yield no assignments; empty groups keep their (empty) slot."""
    assert assign_owner_work([]) == []
    assert assign_owner_work([{}]) == [{}]
    layout = ParameterLayout(torch.Size((4, 2)), (4, 4))  # boundary, eligible on ranks 0 and 1
    assert assign_owner_work([{}, {5: layout}, {}], ns_cost_fn(5)) == [{}, {5: 0}, {}]


def test_assign_owner_work_rejects_mixed_dp_sizes():
    """Groups on different DP mesh sizes cannot be balanced jointly."""
    dp2 = ParameterLayout(torch.Size((8, 8)), (32, 32))
    dp3 = ParameterLayout(torch.Size((8, 8)), (22, 21, 21))
    with pytest.raises(ValueError, match="share one DP mesh size"):
        assign_owner_work([{0: dp2}, {0: dp3}], ns_cost_fn(5))


# ---------------------------------------------------------------------------
# Group owner layout
# ---------------------------------------------------------------------------


def _round_trip_group(this_rank=0):
    """A 3-param group (DP size 3): two boundary params and one non-boundary.

    `GlobalLayout` size 36 → each rank's flat shard is 12 elements:
    rank 0 [0, 12), rank 1 [12, 24), rank 2 [24, 36).
      - tensor 0 (6, 3) at offset 0: elements (12, 6, 0) – boundary.
      - tensor 1 (4, 2) at offset 18: elements (0, 6, 2) – boundary.
      - tensor 2 (2, 2) at offset 26: elements (0, 0, 4) – non-boundary, holder rank 2.
    """
    return _mock_group([(6, 3), (4, 2), (2, 2)], dp_size=3, this_rank=this_rank)


def _per_rank_owner_layouts():
    """Build the round-trip group's `GroupOwnerLayout` once per rank (DP size 3)."""
    return [
        GroupOwnerLayout.from_groups([_round_trip_group(this_rank=rank)], cost_fn=ns_cost_fn(5))[0]
        for rank in range(3)
    ]


def test_group_owner_layout_from_groups_composes_the_steps():
    """For one group, `from_groups` bundles the mesh, the layouts, and the balanced owners."""
    group = _round_trip_group()
    (owner_layout,) = GroupOwnerLayout.from_groups([group], cost_fn=ns_cost_fn(5))
    assert owner_layout.mesh is group.mesh
    # Composition equivalence: the bundle is exactly the two steps composed.
    assert owner_layout.layouts == ParameterLayout.from_group(group)
    assert owner_layout.owners == assign_owner_work([owner_layout.layouts], ns_cost_fn(5))[0]


def test_group_owner_layout_from_groups_respects_eligible_fn():
    """`eligible_fn` filters participation; layouts and owners cover exactly those."""
    group = _round_trip_group()
    (owner_layout,) = GroupOwnerLayout.from_groups(
        [group], cost_fn=ns_cost_fn(5), eligible_fn=lambda param: param.numel() >= 8
    )
    assert list(owner_layout.layouts) == [0, 1]
    assert set(owner_layout.owners) == {0, 1}


def test_group_owner_layout_from_groups_balances_across_groups():
    """`from_groups` balances jointly: group 0's forced load steers group 1's boundary owner."""
    mesh = _mock_mesh(2, 0)
    # Group 0's (4, 3) param lands entirely on rank 0's flat shard (cost 12 * 16 = 192); its 1D
    # param absorbs the rest of the buffer but is excluded by the default `eligible_fn`.
    group0 = _mock_group([(4, 3), (18,)], dp_size=2, mesh=mesh)
    # Group 1's (8, 8) param spans both ranks (cost 64 * 41 = 2624).
    group1 = _mock_group([(8, 8)], dp_size=2, mesh=mesh)

    owner_layouts = GroupOwnerLayout.from_groups([group0, group1], cost_fn=ns_cost_fn(5))

    assert len(owner_layouts) == 2
    assert owner_layouts[0].mesh is mesh
    assert owner_layouts[1].mesh is mesh
    assert owner_layouts[0].layouts == ParameterLayout.from_group(group0)
    assert owner_layouts[1].layouts == ParameterLayout.from_group(group1)
    # Alone, group 1's boundary param would go to rank 0 (all running costs are equal).
    (g1_owner_layout,) = GroupOwnerLayout.from_groups([group1], cost_fn=ns_cost_fn(5))
    assert g1_owner_layout.owners == {0: 0}
    # Jointly, rank 0 already carries group 0's non-boundary cost, so rank 1 takes it.
    assert owner_layouts[0].owners == {0: 0}
    assert owner_layouts[1].owners == {0: 1}


def test_group_owner_layout_from_groups_rejects_mixed_meshes():
    """Groups on distinct same-size meshes are rejected; the `dp_size` check cannot catch them."""
    # Both groups are on a DP mesh of size 2, but distinct ones: owner ranks would live in
    # different rank spaces, silently mixing the joint balancing.
    group0 = _mock_group([(2, 2)], dp_size=2)
    group1 = _mock_group([(8, 8)], dp_size=2)
    with pytest.raises(ValueError, match="same DP mesh"):
        GroupOwnerLayout.from_groups([group0, group1], cost_fn=ns_cost_fn(5))


def test_group_owner_layout_from_groups_with_no_groups():
    """No groups yields no owner layouts."""
    assert GroupOwnerLayout.from_groups([]) == []


# ---------------------------------------------------------------------------
# Pack / reconstruct round trip (simulated P2P)
# ---------------------------------------------------------------------------


def _simulate_p2p(
    per_rank_send_buffers: list[dict[int, torch.Tensor]], dp_size: int
) -> list[dict[int, torch.Tensor]]:
    """Deliver per-owner send buffers to their owners (CPU sim of batch_isend_irecv).

    Returns, per rank, the dict of `{src_rank: received_buffer}` it receives.
    """
    per_rank_recv: list[dict[int, torch.Tensor]] = [dict() for _ in range(dp_size)]
    for src in range(dp_size):
        for dst, buf in per_rank_send_buffers[src].items():
            if buf.numel() == 0:
                continue
            per_rank_recv[dst][src] = buf.clone()
    return per_rank_recv


def test_pack_and_reconstruct_round_trip():
    """Gathered + reconstructed flat tensors match the full tensors on every owner.

    Includes a non-boundary param (tensor 2): its owner reconstructs from its own shard alone, with
    no P2P traffic.
    """
    torch.manual_seed(0)
    dp_size = 3
    per_rank_owner_layout = _per_rank_owner_layouts()
    # `this_rank` only enters via the mesh: every rank's owner layout agrees on layouts and owners.
    # Least processing time over the boundary params:
    #   tensor 0 → rank 0, tensor 1 → rank 1, tensor 2 → rank 2.
    assert [owner_layout.mesh.get_local_rank() for owner_layout in per_rank_owner_layout] == [
        0,
        1,
        2,
    ]
    for owner_layout in per_rank_owner_layout:
        assert owner_layout.owners == {0: 0, 1: 1, 2: 2}

    fulls = {
        0: torch.arange(18, dtype=torch.float32),
        1: torch.arange(8, dtype=torch.float32) + 100,
        2: torch.arange(4, dtype=torch.float32) + 200,
    }

    per_rank_send = []
    per_rank_gather = []
    for rank, owner_layout in enumerate(per_rank_owner_layout):
        local_shards = {}
        for tensor_index, layout in owner_layout.layouts.items():
            offset = layout.rank_offset(rank)
            numel = layout.rank_numel(rank)
            if numel > 0:
                local_shards[tensor_index] = fulls[tensor_index][offset : offset + numel].clone()
        gather = OwnerGatherPlan.pack(owner_layout, local_shards)
        per_rank_send.append(gather.send_buffers)
        per_rank_gather.append(gather)

    recv = _simulate_p2p(per_rank_send, dp_size)

    # Rank 0 owns tensor 0 and receives only rank 1's 6-element shard.
    assert per_rank_gather[0].recv_sizes == {1: 6}
    # Rank 2 owns the non-boundary tensor 2 and receives nothing.
    assert per_rank_gather[2].recv_sizes == {}
    # Each owner reconstructs the full flat tensor from its own + received shard.
    for tensor_index, owner in per_rank_owner_layout[0].owners.items():
        full = per_rank_gather[owner].reconstruct_full(tensor_index, recv[owner])
        assert full.ndim == 1
        torch.testing.assert_close(full, fulls[tensor_index], atol=0, rtol=0)


def test_pack_and_unpack_result_round_trip():
    """Scattered flat result shards match the owner's full result sliced per rank.

    The owner passes 2D matrices; `pack` flattens them as views, and `unpack` returns flat shards.
    """
    torch.manual_seed(1)
    dp_size = 3
    per_rank_owner_layout = _per_rank_owner_layouts()

    # Results with arbitrary shapes.
    full_results = {
        0: (torch.arange(18, dtype=torch.float32) + 1.0).reshape(6, 3),
        1: (torch.arange(8, dtype=torch.float32) + 2.0).reshape(4, 2),
        2: (torch.arange(4, dtype=torch.float32) + 3.0).reshape(2, 2),
    }
    per_rank_send = []
    per_rank_scatter = []
    for rank, owner_layout in enumerate(per_rank_owner_layout):
        owned_results = {
            tensor_index: result
            for tensor_index, result in full_results.items()
            if owner_layout.owners[tensor_index] == rank
        }
        scatter = OwnerScatterPlan.pack(owner_layout, owned_results)
        per_rank_send.append(scatter.send_buffers)
        per_rank_scatter.append(scatter)

    recv = _simulate_p2p(per_rank_send, dp_size)

    for rank in range(dp_size):
        received = per_rank_scatter[rank].unpack(recv[rank])
        for tensor_index, shard in received.items():
            # Only params this rank holds elements of but does NOT own are received.
            assert per_rank_owner_layout[rank].owners[tensor_index] != rank
            layout = per_rank_owner_layout[rank].layouts[tensor_index]
            offset = layout.rank_offset(rank)
            numel = layout.rank_numel(rank)
            expected = full_results[tensor_index].flatten()[offset : offset + numel]
            assert shard.ndim == 1
            torch.testing.assert_close(shard, expected, atol=0, rtol=0)
    # Rank 0 receives nothing: it only holds elements of tensor 0, which it owns.
    assert per_rank_scatter[0].unpack(recv[0]) == {}


def test_pack_with_no_eligible_params():
    """A group with no eligible params packs to an empty owner layout."""
    group = _mock_group([(16,)], dp_size=2)  # 1D bias only.
    (owner_layout,) = GroupOwnerLayout.from_groups([group], cost_fn=ns_cost_fn(5))
    assert owner_layout.layouts == {}
    assert owner_layout.owners == {}

    gather = OwnerGatherPlan.pack(owner_layout, {})
    assert gather.send_buffers == {}
    assert gather.recv_sizes == {}
    assert gather.own_shards == {}
    assert gather.recv_offsets == {}

    scatter = OwnerScatterPlan.pack(owner_layout, {})
    assert scatter.send_buffers == {}
    assert scatter.recv_sizes == {}
    assert scatter.recv_offsets == {}
    assert scatter.unpack({}) == {}
