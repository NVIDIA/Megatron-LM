# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Pure CPU tests for the shard-planning and owner-compute packing logic.

These tests exercise `ParameterLayout.from_group`, `assign_owner_work`, `OwnerGatherPlan.pack`,
`OwnerGatherPlan.reconstruct_full`, `OwnerScatterPlan.pack`, and `OwnerScatterPlan.unpack` without a
process group or any `torch.distributed` dependency. P2P communication is simulated in-process by
`_simulate_p2p`.
"""

from collections.abc import Callable
from types import SimpleNamespace

import torch
import torch.nn as nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.layout import (
    GlobalLayout,
    non_leading_numel,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.shard_plan import (
    OwnerGatherPlan,
    OwnerScatterPlan,
    ParameterLayout,
    assign_owner_work,
)


def _ns_cost(num_ns_steps: int) -> Callable[[ParameterLayout], int]:
    """Cost function matching the Newton-Schulz orthogonalization estimate.

    `numel * (min(rows, cols) * num_steps + 1)` – used by the tests to exercise the greedy balancer
    with a non-trivial cost.
    """

    def cost_fn(layout: ParameterLayout) -> int:
        shape = layout.full_shape
        short_dim = min(shape[0], non_leading_numel(shape))
        return layout.full_numel() * (short_dim * num_ns_steps + 1)

    return cost_fn


def _mock_mesh(dp_size: int, this_rank: int):
    """Mock a `DeviceMesh` providing just the DP size and this rank's index."""
    return SimpleNamespace(size=lambda: dp_size, get_local_rank=lambda: this_rank)


def _mock_group(shapes, offsets, size, dp_size):
    """Mock a `FsdpParameterGroup` with the given DBuffer layout.

    Creates `nn.Parameter`s for each shape so the default `eligible_fn` (`param.ndim >= 2`) can
    filter on them.
    """
    layout = GlobalLayout(
        tensor_shapes=tuple(torch.Size(s) for s in shapes),
        tensor_to_offset=tuple(offsets),
        size=size,
    )
    params = tuple(nn.Parameter(torch.zeros(s)) for s in shapes)
    fsdp_parameters = tuple(SimpleNamespace(sharded=p) for p in params)
    return SimpleNamespace(
        mesh=_mock_mesh(dp_size, 0),
        main_weight=SimpleNamespace(layout=layout),
        fsdp_parameters=fsdp_parameters,
        sharded_parameters=params,
    )


# ---------------------------------------------------------------------------
# Layout construction
# ---------------------------------------------------------------------------


def test_from_group_even_split():
    """A matrix exactly divisible by dp_size splits evenly across ranks."""
    group = _mock_group([(8, 4)], [0], 32, dp_size=2)
    layouts = ParameterLayout.from_group(group)
    assert list(layouts) == [0]
    layout = layouts[0]
    assert layout.full_shape == torch.Size((8, 4))
    assert layout.row_size == 4
    assert layout.row_counts == (4, 4)
    assert layout.shard_numel(0) == 16
    assert layout.shard_numel(1) == 16


def test_from_group_boundary_param_split_across_ranks():
    """A small matrix landing across a rank boundary is split unevenly."""
    # 6 rows, 3 cols; each rank's flat shard is 9 elements (= 3 rows). rank0 owns flat [0,9), rank1
    # owns [9,18). Tensor occupies [0,18) fully.
    group = _mock_group([(6, 3)], [0], 18, dp_size=2)
    layout = ParameterLayout.from_group(group)[0]
    assert layout.row_counts == (3, 3)
    assert layout.is_boundary()


def test_from_group_fully_local_param_on_one_rank():
    """A matrix fully contained in one rank's flat shard is fully local."""
    # 4 rows, 2 cols = 8 elements. rank0 shard = [0,12), rank1 = [12,24). Tensor at offset 0 with 8
    # elements fits entirely in rank0.
    group = _mock_group([(4, 2)], [0], 24, dp_size=2)
    layout = ParameterLayout.from_group(group)[0]
    assert layout.row_counts == (4, 0)
    assert not layout.is_boundary()
    assert layout.owner_candidates() == (0,)


def test_from_group_empty_rank_has_zero_rows():
    """A rank whose flat shard does not overlap the tensor owns zero rows."""
    # Tensor at offset 12 (entirely in rank1). rank0 gets (0,0).
    group = _mock_group([(4, 3)], [12], 24, dp_size=2)
    layout = ParameterLayout.from_group(group)[0]
    assert layout.row_counts == (0, 4)
    assert layout.is_boundary() is False
    assert layout.owner_candidates() == (1,)


def test_from_group_keys_are_tensor_indices():
    """Keys are the params' tensor indices; non-eligible params leave gaps."""
    # 2D weight (tensor 0), 1D bias (tensor 1), 2D weight (tensor 2).
    # Only the ≥2D params are eligible (default `eligible_fn`); the 1D bias is
    # excluded.
    group = _mock_group([(8, 4), (16,), (4, 4)], [0, 32, 48], 64, dp_size=2)
    layouts = ParameterLayout.from_group(group)
    assert list(layouts) == [0, 2]
    assert layouts[0].full_shape == torch.Size((8, 4))
    assert layouts[2].full_shape == torch.Size((4, 4))


def test_from_group_per_rank_data_not_uniform():
    """Flat sharding with uniform buffer size but non-uniform per-rank tensor data.

    `GlobalLayout.build` pads the total size to a multiple of `chunk_size * dp_size` so every rank's
    flat buffer is the same size. However, the actual tensor data per rank is not necessarily
    uniform.

    This test verifies `from_group` correctly computes the non-uniform per-rank row counts from a
    uniform `rank_flat_shard_size`.
    """
    # 5 rows, 4 cols = 20 elements. dp_size = 3.
    # GlobalLayout.build pads to 24 (next multiple of chunk_size * dp_size = 4 * 3 = 12), so
    # rank_flat_shard_size = 24 // 3 = 8 (uniform buffer per rank).
    # But the tensor occupies only 20 of 24 elements:
    #   rank 0: [0, 8)   -> 8 elements -> 2 rows
    #   rank 1: [8, 16)  -> 8 elements -> 2 rows
    #   rank 2: [16, 24) -> 4 elements -> 1 row (4 elements of padding)
    group = _mock_group([(5, 4)], [0], 24, dp_size=3)
    layout = ParameterLayout.from_group(group)[0]
    assert layout.row_counts == (2, 2, 1)
    assert layout.shard_numel(0) == 8
    assert layout.shard_numel(1) == 8
    assert layout.shard_numel(2) == 4
    assert layout.rank_row_count(0) == 2
    assert layout.rank_row_count(1) == 2
    assert layout.rank_row_count(2) == 1
    assert layout.is_boundary()


# ---------------------------------------------------------------------------
# Owner assignment balancing
# ---------------------------------------------------------------------------


def test_assign_owner_work_balances_by_cost():
    """Owners are balanced so the cheapest eligible rank takes each parameter."""
    # Two boundary params, all four ranks eligible for each.
    layout0 = ParameterLayout(torch.Size((8, 8)), (4, 4, 4, 4), 8)  # cost 64*41
    layout1 = ParameterLayout(torch.Size((4, 4)), (2, 2, 2, 2), 4)  # cost 16*21
    # Greedy min running cost: first param -> rank0 (cost 2624), second -> rank1 (cost 336).
    owners = assign_owner_work({0: layout0, 1: layout1}, _ns_cost(5))
    assert owners == {0: 0, 1: 1}


def test_assign_owner_work_keys_pass_through():
    """Keys pass through unchanged. Arbitrary tensor indices, not positions."""
    layout = ParameterLayout(torch.Size((8, 8)), (4, 4, 4, 4), 8)
    owners = assign_owner_work({7: layout}, _ns_cost(5))
    assert owners == {7: 0}


def test_assign_owner_work_only_eligible_ranks_can_own():
    """A rank with an empty shard can never be the owner."""
    # Only ranks 0 and 2 have shards for both params.
    layout0 = ParameterLayout(torch.Size((8, 8)), (4, 0, 4, 0), 8)
    layout1 = ParameterLayout(torch.Size((8, 8)), (4, 0, 4, 0), 8)
    owners = assign_owner_work({0: layout0, 1: layout1}, _ns_cost(5))
    assert all(owner in (0, 2) for owner in owners.values())
    # Two equal-cost params split across the two eligible ranks.
    assert owners[0] != owners[1]


def test_assign_owner_work_lpt_sorts_by_descending_cost():
    """Params are assigned in descending cost order, not input order.

    With two boundary params (cheap first, expensive second in input order), LPT processes the
    expensive one first.
    """
    # Cheap param listed FIRST in input order.
    layout_cheap = ParameterLayout(torch.Size((8, 1)), (1, 1, 1, 1), 1)
    # Expensive param listed SECOND in input order.
    layout_expensive = ParameterLayout(torch.Size((8, 8)), (2, 2, 2, 2), 8)
    owners = assign_owner_work({0: layout_cheap, 1: layout_expensive}, _ns_cost(5))
    # LPT: expensive (tensor 1) → rank0 first, then cheap (tensor 0) → rank1.
    assert owners == {0: 1, 1: 0}


def test_assign_owner_work_non_boundary_gets_sole_holder():
    """Non-boundary params are assigned their sole holder, not skipped."""
    # All 4 rows on rank 0; rank 1 holds nothing.
    layout = ParameterLayout(torch.Size((4, 2)), (4, 0), 2)
    owners = assign_owner_work({3: layout}, _ns_cost(3))
    assert owners == {3: 0}


def test_assign_owner_work_non_boundary_cost_counts_toward_balance():
    """Forced non-boundary work biases the greedy balancer away from that rank."""
    non_boundary = ParameterLayout(torch.Size((2, 2)), (2, 0), 2)
    boundary = ParameterLayout(torch.Size((8, 8)), (4, 4), 8)
    owners = assign_owner_work({0: non_boundary, 1: boundary}, _ns_cost(5))
    assert owners == {0: 0, 1: 1}


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


def _round_trip_group():
    """A 3-param group (dp_size 3): two boundary params and one non-boundary.

    `GlobalLayout` size 36 → each rank's flat shard is 12 elements:
    rank 0 [0,12), rank 1 [12,24), rank 2 [24,36).
      - tensor 0 (6,3) at offset 0: rows (4, 2, 0) – boundary.
      - tensor 1 (4,2) at offset 18: rows (0, 3, 1) – boundary.
      - tensor 2 (2,2) at offset 26: rows (0, 0, 2) – non-boundary, holder rank 2.
    """
    return _mock_group([(6, 3), (4, 2), (2, 2)], [0, 18, 26], 36, dp_size=3)


def test_pack_and_reconstruct_round_trip():
    """Gathered + reconstructed shards match the full tensors on every owner.

    Includes a non-boundary param (tensor 2): its owner reconstructs from its own
    shard alone, with no P2P traffic.
    """
    torch.manual_seed(0)
    dp_size = 3
    layouts = ParameterLayout.from_group(_round_trip_group())
    owners = assign_owner_work(layouts, _ns_cost(5))
    # Least processing time over the boundary params:
    #   tensor 0 → rank 0, tensor 1 → rank 1, tensor 2 → rank 2.
    assert owners == {0: 0, 1: 1, 2: 2}

    fulls = {
        0: torch.arange(18, dtype=torch.float32).reshape(6, 3),
        1: torch.arange(8, dtype=torch.float32).reshape(4, 2) + 100,
        2: torch.arange(4, dtype=torch.float32).reshape(2, 2) + 200,
    }

    per_rank_send = []
    per_rank_gather = []
    for rank in range(dp_size):
        local_shards = {}
        for tensor_index, layout in layouts.items():
            row_start = layout.rank_row_start(rank)
            row_count = layout.rank_row_count(rank)
            if row_count > 0:
                local_shards[tensor_index] = fulls[tensor_index][
                    row_start : row_start + row_count
                ].clone()
        gather = OwnerGatherPlan.pack(layouts, owners, local_shards, _mock_mesh(dp_size, rank))
        per_rank_send.append(gather.send_buffers)
        per_rank_gather.append(gather)

    recv = _simulate_p2p(per_rank_send, dp_size)

    # Rank 0 owns tensor 0 and receives only rank 1's 2-row shard (6 elements).
    assert per_rank_gather[0].recv_sizes == {1: 6}
    # Rank 2 owns the non-boundary tensor 2 and receives nothing.
    assert per_rank_gather[2].recv_sizes == {}
    # Each owner reconstructs the full tensor from its own + received shards.
    for tensor_index, owner in owners.items():
        full = per_rank_gather[owner].reconstruct_full(tensor_index, recv[owner])
        torch.testing.assert_close(full, fulls[tensor_index], atol=0, rtol=0)


def test_pack_and_unpack_result_round_trip():
    """Scattered result shards match the owner's full result sliced per rank."""
    torch.manual_seed(1)
    dp_size = 3
    layouts = ParameterLayout.from_group(_round_trip_group())
    owners = assign_owner_work(layouts, _ns_cost(5))

    full_results = {
        0: torch.arange(18, dtype=torch.float32).reshape(6, 3) + 1.0,
        1: torch.arange(8, dtype=torch.float32).reshape(4, 2) + 2.0,
        2: torch.arange(4, dtype=torch.float32).reshape(2, 2) + 3.0,
    }
    per_rank_send = []
    per_rank_scatter = []
    for rank in range(dp_size):
        owned_results = {
            tensor_index: result
            for tensor_index, result in full_results.items()
            if owners[tensor_index] == rank
        }
        scatter = OwnerScatterPlan.pack(layouts, owners, owned_results, _mock_mesh(dp_size, rank))
        per_rank_send.append(scatter.send_buffers)
        per_rank_scatter.append(scatter)

    recv = _simulate_p2p(per_rank_send, dp_size)

    for rank in range(dp_size):
        received = per_rank_scatter[rank].unpack(recv[rank])
        for tensor_index, shard in received.items():
            # Only params this rank holds rows of but does NOT own are received.
            assert owners[tensor_index] != rank
            layout = layouts[tensor_index]
            row_start = layout.rank_row_start(rank)
            row_count = layout.rank_row_count(rank)
            expected = full_results[tensor_index][row_start : row_start + row_count]
            torch.testing.assert_close(shard, expected, atol=0, rtol=0)
    # Rank 0 receives nothing: it only holds rows of tensor 0, which it owns.
    assert per_rank_scatter[0].unpack(recv[0]) == {}


def test_pack_with_no_eligible_params():
    """An empty layout dict (no eligible params) packs to an empty plan."""
    mesh = _mock_mesh(2, 0)
    gather = OwnerGatherPlan.pack({}, {}, {}, mesh)
    assert gather.send_buffers == {}
    assert gather.recv_sizes == {}
    assert gather.own_shards == {}
    assert gather.recv_offsets == {}

    scatter = OwnerScatterPlan.pack({}, {}, {}, mesh)
    assert scatter.send_buffers == {}
    assert scatter.recv_sizes == {}
    assert scatter.recv_offsets == {}
    assert scatter.unpack({}) == {}
