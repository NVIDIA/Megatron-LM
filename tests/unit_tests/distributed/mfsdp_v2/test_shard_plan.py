# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Pure CPU tests for the shard-planning and owner-compute packing logic.

These tests exercise `ShardPlan`, `ShardPlan.from_flat_layout`, `assign_owner_work`,
`pack_owner_work`, `OwnerGatherPlan.reconstruct_full`, `pack_update_shards`, and
`OwnerScatterPlan.unpack` without a process group or any `torch.distributed` dependency. P2P
communication is simulated in-process by `_simulate_p2p`.
"""

import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.shard_plan import (
    ShardPlan,
    assign_owner_work,
    pack_owner_work,
    pack_update_shards,
)

# ---------------------------------------------------------------------------
# Shard-plan math
# ---------------------------------------------------------------------------


def test_compute_shard_plan_even_split():
    """A matrix exactly divisible by world_size splits evenly across ranks."""
    plan = ShardPlan.from_flat_layout(
        torch.Size((8, 4)), tensor_flat_offset=0, rank_flat_shard_size=16, world_size=2
    )
    assert plan.full_shape == torch.Size((8, 4))
    assert plan.row_size == 4
    assert plan.rank_rows == ((0, 4), (4, 4))
    assert plan.shard_numel(0) == 16
    assert plan.shard_numel(1) == 16


def test_compute_shard_plan_boundary_param_split_across_ranks():
    """A small matrix landing across a rank boundary is split unevenly."""
    # 6 rows, 3 cols; each rank's flat shard is 9 elements (= 3 rows). rank0 owns flat [0,9), rank1
    # owns [9,18). Tensor occupies [0,18) fully.
    plan = ShardPlan.from_flat_layout(
        torch.Size((6, 3)), tensor_flat_offset=0, rank_flat_shard_size=9, world_size=2
    )
    assert plan.rank_rows == ((0, 3), (3, 3))
    assert plan.is_boundary()


def test_compute_shard_plan_fully_local_param_on_one_rank():
    """A matrix fully contained in one rank's flat shard is fully local."""
    # 4 rows, 2 cols = 8 elements. rank0 shard = [0,12), rank1 = [12,24). Tensor at offset 0 with 8
    # elements fits entirely in rank0.
    plan = ShardPlan.from_flat_layout(
        torch.Size((4, 2)), tensor_flat_offset=0, rank_flat_shard_size=12, world_size=2
    )
    assert plan.rank_rows == ((0, 4), (0, 0))
    assert not plan.is_boundary()
    assert plan.owner_candidates() == (0,)


def test_compute_shard_plan_empty_rank_has_zero_rows():
    """A rank whose flat shard does not overlap the tensor owns zero rows."""
    # Tensor offset 12 (entirely in rank1). rank0 gets (0,0).
    plan = ShardPlan.from_flat_layout(
        torch.Size((4, 3)), tensor_flat_offset=12, rank_flat_shard_size=12, world_size=2
    )
    assert plan.rank_rows == ((0, 0), (0, 4))
    assert plan.is_boundary() is False
    assert plan.owner_candidates() == (1,)


# ---------------------------------------------------------------------------
# Owner assignment balancing
# ---------------------------------------------------------------------------


def test_assign_owner_work_balances_by_cost():
    """Owners are balanced so the cheapest eligible rank takes each parameter."""
    # Two boundary params, all four ranks eligible for each.
    plan0 = ShardPlan(torch.Size((8, 8)), ((0, 4), (0, 4), (0, 4), (0, 4)), 8)  # cost 64*41
    plan1 = ShardPlan(torch.Size((4, 4)), ((0, 2), (0, 2), (0, 2), (0, 2)), 4)  # cost 16*21
    owners = assign_owner_work([plan0, plan1], num_ns_steps=5)
    # Greedy min running cost: first param -> rank0 (cost 2624), second -> rank1 (cost 336).
    assert owners == {0: 0, 1: 1}


def test_assign_owner_work_only_eligible_ranks_can_own():
    """A rank with an empty shard can never be the owner."""
    # Only ranks 0 and 2 have shards for both params.
    plan0 = ShardPlan(torch.Size((8, 8)), ((0, 4), (0, 0), (0, 4), (0, 0)), 8)
    plan1 = ShardPlan(torch.Size((8, 8)), ((0, 4), (0, 0), (0, 4), (0, 0)), 8)
    owners = assign_owner_work([plan0, plan1], num_ns_steps=5)
    assert all(owners[i] in (0, 2) for i in owners)
    # Two equal-cost params split across the two eligible ranks.
    assert owners[0] != owners[1]


def test_assign_owner_work_fully_local_owner_is_single_candidate():
    """A fully local parameter is assigned to its single owning rank."""
    plan = ShardPlan(torch.Size((4, 2)), ((0, 4), (0, 0)), 2)
    owners = assign_owner_work([plan], num_ns_steps=3)
    assert owners == {0: 0}


# ---------------------------------------------------------------------------
# Pack / reconstruct round trip (simulated P2P)
# ---------------------------------------------------------------------------


def _simulate_p2p(
    per_rank_send_buffers: list[dict[int, torch.Tensor]], world_size: int
) -> list[dict[int, torch.Tensor]]:
    """Deliver per-owner send buffers to their owners (CPU sim of batch_isend_irecv).

    Returns, per rank, the dict of `{src_rank: received_buffer}` it receives.
    """
    per_rank_recv: list[dict[int, torch.Tensor]] = [dict() for _ in range(world_size)]
    for src in range(world_size):
        for dst, buf in per_rank_send_buffers[src].items():
            if buf.numel() == 0:
                continue
            per_rank_recv[dst][src] = buf.clone()
    return per_rank_recv


def test_pack_and_reconstruct_round_trip():
    """Gathered + reconstructed pre-NS matches the concatenation of local shards."""
    torch.manual_seed(0)
    world_size = 2
    # Two params, both boundary, shapes (6,3) and (4,2). Owners: param0->rank0, param1->rank1.
    plan0 = ShardPlan.from_flat_layout(torch.Size((6, 3)), 0, 9, world_size)
    plan1 = ShardPlan.from_flat_layout(torch.Size((4, 2)), 0, 4, world_size)
    plans = [plan0, plan1]
    owners = {0: 0, 1: 1}

    # Build per-rank local shards as distinct values so reconstruction is checkable.
    full_p0 = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    full_p1 = torch.arange(8, dtype=torch.float32).reshape(4, 2) + 100
    per_rank_local = []
    for r in range(world_size):
        rs0, rc0 = plan0.rank_rows[r]
        rs1, rc1 = plan1.rank_rows[r]
        per_rank_local.append([full_p0[rs0 : rs0 + rc0].clone(), full_p1[rs1 : rs1 + rc1].clone()])

    per_rank_send = []
    per_rank_gather = []
    for r in range(world_size):
        gather = pack_owner_work(
            plans,
            owners,
            per_rank_local[r],
            world_size,
            r,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        per_rank_send.append(gather.send_buffers)
        per_rank_gather.append(gather)

    recv = _simulate_p2p(per_rank_send, world_size)

    # Rank0 owns param0; reconstruct and compare to full_p0.
    full0 = per_rank_gather[0].reconstruct_full(0, plan0, recv[0], owner_rank=0)
    torch.testing.assert_close(full0, full_p0, atol=0, rtol=0)
    # Rank1 owns param1; reconstruct and compare to full_p1.
    full1 = per_rank_gather[1].reconstruct_full(1, plan1, recv[1], owner_rank=1)
    torch.testing.assert_close(full1, full_p1, atol=0, rtol=0)


def test_pack_and_unpack_update_round_trip():
    """Scattered update shards match the owner's full update sliced per rank."""
    torch.manual_seed(1)
    world_size = 2
    plan0 = ShardPlan.from_flat_layout(torch.Size((6, 3)), 0, 9, world_size)
    plan1 = ShardPlan.from_flat_layout(torch.Size((4, 2)), 0, 4, world_size)
    plans = [plan0, plan1]
    owners = {0: 0, 1: 1}

    full_update0 = torch.arange(18, dtype=torch.float32).reshape(6, 3) + 1.0
    full_update1 = torch.arange(8, dtype=torch.float32).reshape(4, 2) + 2.0
    full_updates_by_rank = {0: {0: full_update0}, 1: {1: full_update1}}

    per_rank_send = []
    per_rank_scatter = []
    for r in range(world_size):
        scatter = pack_update_shards(
            full_updates_by_rank[r],
            plans,
            owners,
            world_size,
            r,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        per_rank_send.append(scatter.send_buffers)
        per_rank_scatter.append(scatter)

    recv = _simulate_p2p(per_rank_send, world_size)

    for r in range(world_size):
        received = per_rank_scatter[r].unpack(recv[r])
        # Rank r receives the update shard for the param it does NOT own.
        other = 1 - r
        plan = plans[other]
        rs, rc = plan.rank_rows[r]
        expected = full_updates_by_rank[owners[other]][other][rs : rs + rc]
        torch.testing.assert_close(received[other], expected, atol=0, rtol=0)
