# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe.moe_scheduler import SchedulerContext
from megatron.core.transformer.moe.moonep_moe_scheduler import (
    MoonEPLoadPlanner,
    _physical_to_logical_map_from_experts_to_copy,
    extract_semantic_routes,
)
from megatron.core.transformer.moe.replica_expert_dispatch import ReplicaExpertDispatch
from tests.unit_tests.transformer.moe.moonep_reference import (
    ReplicaPlannerWorkspace,
    plan_replica_routes,
)
from tests.unit_tests.transformer.moe.moonep_reference_triton import HAVE_TRITON

pytestmark = pytest.mark.launch_on_gb200


def _route_inputs(
    topk_ids: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    topk_probs = torch.full(topk_ids.shape, 1.0 / topk_ids.size(1), dtype=torch.float32)
    probs = torch.zeros(topk_ids.size(0), num_experts)
    routing_map = torch.zeros(topk_ids.size(0), num_experts, dtype=torch.bool)
    routing_map.scatter_(1, topk_ids, True)
    probs.scatter_(1, topk_ids, topk_probs)
    tokens_per_expert = torch.bincount(topk_ids.reshape(-1), minlength=num_experts)
    return probs, routing_map, tokens_per_expert


def _context(ep_size: int, ep_rank: int, topk: int = 1, num_experts: int = 4) -> SchedulerContext:
    experts_per_rank = num_experts // ep_size
    local_start = ep_rank * experts_per_rank
    return SchedulerContext(
        layer_number=1,
        num_logical_experts=num_experts,
        num_local_experts=experts_per_rank,
        local_expert_indices=tuple(range(local_start, local_start + experts_per_rank)),
        ep_size=ep_size,
        ep_rank=ep_rank,
        router_topk=topk,
        training=True,
    )


def test_moonep_planner_emits_2x_identity_layout_for_single_ep():
    topk_ids = torch.tensor([[0], [1], [2], [3]])
    probs, routing_map, tokens_per_expert = _route_inputs(topk_ids, num_experts=4)
    context = _context(ep_size=1, ep_rank=0)
    planner = MoonEPLoadPlanner(num_redundant_experts=4)
    home_placement, physical_to_logical_map, placement_result = planner.update_placement(
        probs, routing_map, context, tokens_per_expert=tokens_per_expert
    )
    physical_routing_map, physical_probs = planner.reroute(
        probs, routing_map, placement_result, context
    )

    assert physical_to_logical_map.tolist() == [[-1, -1, -1, -1]]
    assert physical_routing_map.shape == (4, 8)
    assert physical_routing_map[:, :4].sum().item() == 4
    assert physical_routing_map[:, 4:].sum().item() == 0
    assert torch.equal(physical_probs.sum(dim=1), probs.sum(dim=1))


def test_moonep_planner_rejects_partial_replica_slots():
    topk_ids = torch.tensor([[0], [0], [1], [1]])
    probs, routing_map, tokens_per_expert = _route_inputs(topk_ids, num_experts=4)

    with pytest.raises(ValueError, match="one replica slot per local home expert"):
        MoonEPLoadPlanner(num_redundant_experts=1).update_placement(
            probs, routing_map, _context(ep_size=2, ep_rank=0), tokens_per_expert=tokens_per_expert
        )


@pytest.mark.parametrize("padding", [0, -1, True, 1.5])
def test_moonep_planner_rejects_invalid_token_padding(padding):
    with pytest.raises(ValueError, match="token_padding"):
        MoonEPLoadPlanner(num_redundant_experts=4, token_padding=padding)


def test_moonep_planner_requires_ep_group_for_multi_ep():
    topk_ids = torch.tensor([[0], [0], [1], [1]])
    probs, routing_map, tokens_per_expert = _route_inputs(topk_ids, num_experts=4)

    with pytest.raises(ValueError, match="pg_collection.ep"):
        MoonEPLoadPlanner(num_redundant_experts=2).update_placement(
            probs, routing_map, _context(ep_size=2, ep_rank=0), tokens_per_expert=tokens_per_expert
        )


def test_moonep_layout_is_accepted_by_unified_replica_dispatch():
    topk_ids = torch.tensor([[0], [1], [2], [3]])
    probs, routing_map, tokens_per_expert = _route_inputs(topk_ids, num_experts=4)
    context = _context(ep_size=1, ep_rank=0)
    _, physical_to_logical_map, _ = MoonEPLoadPlanner(num_redundant_experts=4).update_placement(
        probs, routing_map, context, tokens_per_expert=tokens_per_expert
    )

    class _Group:
        def size(self):
            return 1

        def rank(self):
            return 0

    dispatcher = ReplicaExpertDispatch(
        config=SimpleNamespace(
            num_moe_experts=4,
            expert_model_parallel_size=1,
            moe_scheduler_num_idle_experts=4,
            moe_scheduler_expert_dispatcher_type="replica_nccl",
        ),
        pg_collection=SimpleNamespace(ep=_Group()),
    )

    assert dispatcher.supports(physical_to_logical_map, context)


def test_moonep_experts_to_copy_builds_rank_major_physical_layout():
    context = _context(ep_size=2, ep_rank=1)
    experts_to_copy = torch.tensor([[-1, -1], [0, 1]], dtype=torch.int32)

    physical_to_logical_map = _physical_to_logical_map_from_experts_to_copy(
        experts_to_copy, context
    )

    assert physical_to_logical_map.tolist() == [0, 1, -1, -1, 2, 3, 0, 1]


def test_moonep_planner_rejects_removed_count_matrix_adapter():
    with pytest.raises(NotImplementedError, match="plan_with_count_matrix"):
        MoonEPLoadPlanner(num_redundant_experts=2).plan_with_count_matrix()


@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_moonep_physical_layout_graph_replays_changed_replica_mask():
    context = _context(ep_size=2, ep_rank=1)
    replicas = torch.full((2, 2), -1, dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _physical_to_logical_map_from_experts_to_copy(replicas, context)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        layout = _physical_to_logical_map_from_experts_to_copy(replicas, context)

    for values, expected in [
        ([[-1, -1], [0, 1]], [0, 1, -1, -1, 2, 3, 0, 1]),
        ([[3, 2], [-1, 4]], [0, 1, 3, 2, 2, 3, -1, -1]),
        ([[-7, -1], [-1, -1]], [0, 1, -1, -1, 2, 3, -1, -1]),
    ]:
        replicas.copy_(torch.tensor(values, dtype=torch.int32, device="cuda"))
        graph.replay()
        assert layout.tolist() == expected


def test_moonep_extracts_compact_routes_from_common_dense_ir():
    topk_ids = torch.tensor([[3, 0], [2, 1]])
    probs, routing_map, _ = _route_inputs(topk_ids, num_experts=4)

    topk_probs, compact_ids = extract_semantic_routes(routing_map, probs, router_topk=2)

    assert compact_ids.tolist() == [[0, 3], [1, 2]]
    torch.testing.assert_close(topk_probs, torch.full((2, 2), 0.5))


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_TRITON,
    reason="The fused #6892 planner requires CUDA and Triton.",
)
def test_moonep_fused_planner_process_local_smoke():
    device = torch.device("cuda", torch.cuda.current_device())
    workspace = ReplicaPlannerWorkspace.local(4, 2, device, rank=0)
    workspace.gathered_counts.copy_(
        torch.tensor([[4, 0, 0, 0], [0, 0, 4, 0]], dtype=torch.int32, device=device)
    )
    routes = torch.zeros((4, 1), dtype=torch.int64, device=device)
    probs = torch.ones((4, 1), dtype=torch.float32, device=device, requires_grad=True)

    plan, runtime_probs = plan_replica_routes(routes, probs, workspace, exchange=False)
    runtime_probs.sum().backward()
    torch.cuda.synchronize(device)

    assert plan.virtual_experts.dtype == torch.int16
    assert plan.virtual_experts.long().tolist() == [[0], [0], [0], [0]]
    assert plan.experts_to_copy.tolist() == [[-1, -1], [-1, -1]]
    torch.testing.assert_close(probs.grad, torch.ones_like(probs))
