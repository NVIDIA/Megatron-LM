# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import pytest
import torch

from megatron.core.transformer.moe.moe_scheduler import (
    ExpertDispatch,
    MoELoadPlanner,
    MoEPlacementResult,
    MoEScheduler,
    SchedulerContext,
)

pytestmark = pytest.mark.launch_on_gb200


def _route_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.zeros(3, 4)
    routing_map = torch.zeros(3, 4, dtype=torch.bool)
    topk_ids = torch.tensor([[0, 1], [1, 2], [3, 0]])
    topk_probs = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.8, 0.2]])
    routing_map.scatter_(1, topk_ids, True)
    probs.scatter_(1, topk_ids, topk_probs)
    return probs, routing_map, routing_map.sum(dim=0)


def _context() -> SchedulerContext:
    return SchedulerContext(
        layer_number=1,
        num_logical_experts=4,
        num_local_experts=2,
        local_expert_indices=(0, 1),
        ep_size=2,
        ep_rank=0,
        router_topk=2,
        training=True,
    )


def _dense_to_topk(
    routing_map: torch.Tensor, probs: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_ids = torch.empty(routing_map.size(0), topk, dtype=torch.long)
    topk_probs = torch.empty(routing_map.size(0), topk, dtype=probs.dtype)
    for token_idx in range(routing_map.size(0)):
        expert_ids = torch.nonzero(routing_map[token_idx], as_tuple=False).flatten()
        topk_ids[token_idx] = expert_ids
        topk_probs[token_idx] = probs[token_idx, expert_ids]
    return topk_ids, topk_probs


def _physical_token_reroute(
    probs: torch.Tensor, routing_map: torch.Tensor, context: SchedulerContext
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_ids, topk_probs = _dense_to_topk(routing_map, probs, context.router_topk)
    physical_ids = topk_ids.clone()
    physical_ids[0, 0] = 2
    physical_ids[2, 0] = 5
    physical_routing_map = torch.zeros(routing_map.size(0), 6, dtype=torch.bool)
    physical_probs = torch.zeros(routing_map.size(0), 6, dtype=probs.dtype)
    physical_routing_map.scatter_(1, physical_ids, True)
    physical_probs.scatter_(1, physical_ids, topk_probs)
    return physical_probs, physical_routing_map


class _TestPlacementResult(MoEPlacementResult):
    pass


class _EchoStylePlanner(MoELoadPlanner):
    planner_name = "echo-test"

    def __init__(self, events: list[str] | None = None) -> None:
        super().__init__()
        self.events = events

    def update_placement(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        *,
        tokens_per_expert: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, MoEPlacementResult]:
        del probs, routing_map, context, tokens_per_expert
        if self.events is not None:
            self.events.append("update_placement")
        return None, torch.tensor([[0], [3]]), _TestPlacementResult()

    def reroute(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        placement_result: MoEPlacementResult,
        context: SchedulerContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(placement_result, _TestPlacementResult)
        if self.events is not None:
            self.events.append("reroute")
        physical_probs, physical_routing_map = _physical_token_reroute(probs, routing_map, context)
        return physical_routing_map, physical_probs


class _InvalidPlacementPlanner(_EchoStylePlanner):
    def update_placement(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        *,
        tokens_per_expert: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, MoEPlacementResult]:
        del probs, routing_map, context, tokens_per_expert
        return None, torch.arange(6), _TestPlacementResult()


class _InvalidReroutePlanner(_EchoStylePlanner):
    def reroute(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        placement_result: MoEPlacementResult,
        context: SchedulerContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del probs, routing_map, placement_result, context
        return torch.zeros(3, 6, dtype=torch.bool), torch.zeros(3, 5)


class _RecordingDispatch(ExpertDispatch):
    dispatcher_name = "recording-test"

    def __init__(self, events: list[str] | None = None) -> None:
        super().__init__()
        self.events = events
        self.dispatched_physical_to_logical_map = None
        self.materialized_experts = None
        self.finalized = False
        self.supports_called = False

    def dispatch(
        self,
        experts: torch.nn.Module,
        physical_to_logical_map: torch.Tensor,
        context: SchedulerContext,
    ) -> None:
        del context
        if self.events is not None:
            self.events.append("dispatch")
        self.dispatched_physical_to_logical_map = physical_to_logical_map
        self.materialized_experts = experts

    def supports(self, physical_to_logical_map: torch.Tensor, context: SchedulerContext) -> bool:
        self.supports_called = True
        return super().supports(physical_to_logical_map, context)

    def finalize(self, context: SchedulerContext) -> None:
        del context
        self.finalized = True


class _RejectingDispatch(ExpertDispatch):
    dispatcher_name = "rejecting-test"

    def supports(self, physical_to_logical_map: torch.Tensor, context: SchedulerContext) -> bool:
        del physical_to_logical_map, context
        return False

    def dispatch(
        self,
        experts: torch.nn.Module,
        physical_to_logical_map: torch.Tensor,
        context: SchedulerContext,
    ) -> None:
        del experts, physical_to_logical_map, context


def test_scheduler_passes_physical_layout_to_matching_dispatcher():
    probs, routing_map, tokens_per_expert = _route_inputs()
    context = _context()
    events = []
    dispatch = _RecordingDispatch(events)
    scheduler = MoEScheduler(planner=_EchoStylePlanner(events), expert_dispatch=dispatch)
    experts = torch.nn.Identity()

    output_probs, output_routing_map = scheduler.schedule(
        probs, routing_map, experts, context, tokens_per_expert=tokens_per_expert
    )

    assert dispatch.dispatched_physical_to_logical_map.tolist() == [[0], [3]]
    assert dispatch.materialized_experts is experts
    assert output_probs.shape == output_routing_map.shape
    assert events == ["update_placement", "dispatch", "reroute"]

    scheduler.finalize(context)

    assert dispatch.finalized


def test_scheduler_rejects_dispatcher_that_does_not_support_planner_output():
    scheduler = MoEScheduler(planner=_EchoStylePlanner(), expert_dispatch=_RejectingDispatch())

    with pytest.raises(ValueError, match="does not support planner output"):
        probs, routing_map, tokens_per_expert = _route_inputs()
        scheduler.schedule(
            probs, routing_map, torch.nn.Identity(), _context(), tokens_per_expert=tokens_per_expert
        )


def test_scheduler_validates_split_planner_outputs():
    probs, routing_map, tokens_per_expert = _route_inputs()
    context = _context()
    experts = torch.nn.Identity()

    with pytest.raises(ValueError, match="replica"):
        MoEScheduler(
            planner=_InvalidPlacementPlanner(), expert_dispatch=_RecordingDispatch()
        ).schedule(probs, routing_map, experts, context, tokens_per_expert=tokens_per_expert)

    with pytest.raises(ValueError, match="same shape"):
        MoEScheduler(
            planner=_InvalidReroutePlanner(), expert_dispatch=_RecordingDispatch()
        ).schedule(probs, routing_map, experts, context, tokens_per_expert=tokens_per_expert)


def test_scheduler_stages_home_only_and_commits_after_exchange():
    from megatron.core.transformer.moe.replica_weight_transport import HomeExpertPlacement

    events = []

    class Planner(MoELoadPlanner):
        def update_placement(self, probs, routing_map, context, **kwargs):
            return (
                HomeExpertPlacement(torch.tensor([[2, 3], [0, 1]]), 1),
                None,
                _TestPlacementResult(),
            )

        def reroute(self, probs, routing_map, placement_result, context):
            return routing_map, probs

        def step(self, completed_version=None):
            assert completed_version == 1
            events.append("commit")

    class Home(ExpertDispatch):
        def dispatch(self, experts, expert_placement, context):
            events.append("stage")

        def step(self):
            events.append("exchange")
            return 1

    replica = _RecordingDispatch()
    scheduler = MoEScheduler(Planner(), replica, Home())
    probs, routes, _ = _route_inputs()
    actual_probs, actual_routes = scheduler.schedule(probs, routes, torch.nn.Identity(), _context())
    assert events == ["stage"]
    assert replica.dispatched_physical_to_logical_map is None
    assert actual_probs is probs and actual_routes is routes
    scheduler.step()
    assert events == ["stage", "exchange", "commit"]
