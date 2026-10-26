# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from megatron.core import tensor_parallel
from megatron.core.transformer.moe.eplb_moe_scheduler import EPLBLoadPlanner, build_eplb_placement
from megatron.core.transformer.moe.moe_scheduler import (
    MoEPlacementResult,
    MoEScheduler,
    SchedulerContext,
)

pytestmark = pytest.mark.launch_on_gb200


def _context(ep_rank: int = 0, *, ep_group=None) -> SchedulerContext:
    return SchedulerContext(
        layer_number=1,
        num_logical_experts=4,
        num_local_experts=2,
        local_expert_indices=(2 * ep_rank, 2 * ep_rank + 1),
        ep_size=2,
        ep_rank=ep_rank,
        router_topk=1,
        training=True,
        pg_collection=SimpleNamespace(ep=ep_group),
    )


def _routes(logical_experts: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.tensor(logical_experts, dtype=torch.int64).unsqueeze(1)
    routing_map = torch.nn.functional.one_hot(indices.squeeze(1), num_classes=4).bool()
    probs = routing_map.to(torch.float32).requires_grad_()
    return probs, routing_map


def test_eplb_builds_greedy_replicas_with_fixed_home_slots():
    physical_to_logical_map, logical_to_physical_map, replica_counts = build_eplb_placement(
        torch.tensor([12, 4, 2, 2]), num_redundant_experts=2, ep_size=2
    )

    assert physical_to_logical_map.tolist() == [0, 1, 0, 2, 3, 0]
    assert logical_to_physical_map.tolist() == [[0, 2, 5], [1, -1, -1], [3, -1, -1], [4, -1, -1]]
    assert replica_counts.tolist() == [3, 1, 1, 1]


def test_eplb_planner_round_robins_routes_and_preserves_probability_gradients(monkeypatch):
    counts_from_ep_rank = torch.tensor([[4, 0, 0, 0], [8, 4, 2, 2]])
    monkeypatch.setattr(
        tensor_parallel,
        "gather_from_sequence_parallel_region",
        lambda local_counts, group: counts_from_ep_rank.reshape(-1),
    )
    probs, routing_map = _routes([0, 0, 0, 0])
    planner = EPLBLoadPlanner(num_redundant_experts=2)
    context = _context(ep_group=object())

    home_placement, physical_to_logical_map, placement_result = planner.update_placement(
        probs, routing_map, context
    )
    physical_routing_map, physical_probs = planner.reroute(
        probs, routing_map, placement_result, context
    )

    assert physical_to_logical_map.tolist() == [[0], [0]]
    assert physical_routing_map.nonzero(as_tuple=False).tolist() == [[0, 0], [1, 2], [2, 5], [3, 0]]
    torch.testing.assert_close(physical_probs.sum(dim=1), torch.ones(4))

    physical_probs.sum().backward()
    torch.testing.assert_close(probs.grad, routing_map.to(torch.float32))


def test_eplb_uses_prior_ep_rank_counts_as_round_robin_offset(monkeypatch):
    counts_from_ep_rank = torch.tensor([[4, 0, 0, 0], [1, 0, 0, 0]])
    monkeypatch.setattr(
        tensor_parallel,
        "gather_from_sequence_parallel_region",
        lambda local_counts, group: counts_from_ep_rank.reshape(-1),
    )
    probs, routing_map = _routes([0])
    planner = EPLBLoadPlanner(num_redundant_experts=2)
    context = _context(ep_rank=1, ep_group=object())

    _, _, placement_result = planner.update_placement(probs, routing_map, context)
    physical_routing_map, _ = planner.reroute(probs, routing_map, placement_result, context)

    assert placement_result.local_expert_offsets.tolist() == [4, 0, 0, 0]
    assert physical_routing_map.nonzero(as_tuple=False).tolist() == [[0, 2]]


def test_eplb_requires_ep_group_and_its_own_placement_result():
    probs, routing_map = _routes([0])
    planner = EPLBLoadPlanner(num_redundant_experts=2)
    context = _context()

    with pytest.raises(ValueError, match="pg_collection.ep"):
        planner.update_placement(probs, routing_map, context)

    with pytest.raises(TypeError, match="EPLBPlacementResult"):
        planner.reroute(probs, routing_map, MoEPlacementResult(), context)


def test_scheduler_factory_builds_eplb_planner():
    scheduler = MoEScheduler.from_config(
        SimpleNamespace(
            moe_scheduler_planner_type="eplb",
            moe_scheduler_expert_dispatcher_type="replica_peer_tma",
            moe_scheduler_num_idle_experts=2,
            num_moe_experts=4,
            expert_model_parallel_size=2,
        ),
        SimpleNamespace(ep=object()),
    )

    assert isinstance(scheduler.planner, EPLBLoadPlanner)
