# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MoEScheduler adapter around the upstream MoonEP cooperative planner.

The planner implementation and symmetric-memory protocol have one production
owner in ops/virtual_expert_triton.py. Compact route probabilities retain their
original autograd graph; the old split implementation is a test-only reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from megatron.core.transformer.moe.moe_scheduler import (
    MoELoadPlanner,
    MoEPlacementResult,
    SchedulerContext,
)
from megatron.core.transformer.moe.moe_scheduler_plan import MoonEPPlanner
from megatron.core.utils import nvtx_decorator

MAX_REPLICA_EP_RANKS = 64
_planner_workspaces = {}


def get_planner_workspace(*, num_experts, device, group):
    """Share upstream workspace per explicit EP group, device and expert layout."""
    from megatron.core.transformer.moe.ops.virtual_expert_triton import (
        VirtualExpertPlannerWorkspace,
    )

    key = (num_experts, device.index, group.group_name)
    if key not in _planner_workspaces:
        _planner_workspaces[key] = VirtualExpertPlannerWorkspace(
            num_experts=num_experts, device=device, group=group
        )
    return _planner_workspaces[key]


def finalize_moonep_planner_workspaces():
    """Release upstream symmetric windows before process group teardown."""
    for workspace in _planner_workspaces.values():
        workspace.destroy()
    _planner_workspaces.clear()


def extract_semantic_routes(
    routing_map: torch.Tensor, probs: torch.Tensor, router_topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover compact top-k ids/probabilities from the scheduler's dense router IR."""
    if routing_map.dim() != 2 or probs.shape != routing_map.shape:
        raise ValueError("MoonEP requires matching 2D routing_map and probs tensors.")
    if routing_map.dtype != torch.bool:
        raise ValueError(f"routing_map must be bool, got {routing_map.dtype}.")
    if router_topk <= 0 or router_topk > routing_map.size(1):
        raise ValueError(f"router_topk must be in [1, {routing_map.size(1)}], got {router_topk}.")
    if routing_map.device != probs.device:
        raise ValueError("routing_map and probs must be on the same device.")
    if routing_map.is_cuda:
        torch._assert_async(
            torch.all(routing_map.sum(dim=1) == router_topk),
            "Every MoonEP token must have exactly router_topk routes.",
        )
    expert_ids = torch.arange(
        routing_map.size(1), dtype=torch.long, device=routing_map.device
    ).expand_as(routing_map)
    sentinel = torch.full_like(expert_ids, routing_map.size(1))
    topk_indices = torch.topk(
        torch.where(routing_map, expert_ids, sentinel),
        k=router_topk,
        dim=1,
        largest=False,
        sorted=True,
    ).values
    return torch.gather(probs, 1, topk_indices), topk_indices


def _physical_to_logical_map_from_experts_to_copy(
    experts_to_copy: torch.Tensor, context: SchedulerContext
) -> torch.Tensor:
    """Build rank-major physical layout from PR #7450 ``experts_to_copy``."""
    if experts_to_copy.shape != (context.ep_size, context.num_logical_experts // context.ep_size):
        raise ValueError(
            "experts_to_copy shape does not match SchedulerContext, got "
            f"{tuple(experts_to_copy.shape)}."
        )

    device = experts_to_copy.device
    num_local_home_experts = context.num_logical_experts // context.ep_size
    logical_ids = torch.arange(
        context.num_logical_experts, dtype=torch.long, device=device
    ).reshape(context.ep_size, num_local_home_experts)
    replica_logical_ids = experts_to_copy.to(dtype=torch.long)
    valid = (replica_logical_ids >= 0) & (replica_logical_ids < context.num_logical_experts)
    # Keep every replica slot in the output, including inactive ones. Boolean
    # indexing produces a dynamic-size tensor and synchronizes with the host,
    # which is illegal during CUDA graph capture. Fixed-shape selection also
    # lets graph replay handle a different active-slot mask each microbatch.
    return torch.cat((logical_ids, torch.where(valid, replica_logical_ids, -1)), dim=1).reshape(-1)


@dataclass(frozen=True, slots=True)
class MoonEPPlacementResult(MoEPlacementResult):
    """Explicit state consumed by the MoonEP token-reroute phase."""

    topk_indices: Optional[torch.Tensor] = None
    virtual_experts: Optional[torch.Tensor] = None
    max_dispatch_routes_per_rank: Optional[int] = None
    max_dispatch_padded_tokens_per_rank: Optional[int] = None


def padded_rank_capacity(num_routes: int, num_experts: int, ep_size: int, alignment: int) -> int:
    """PR #7450 bound on balanced, globally rounded expert blocks (including padding)."""
    return MoonEPPlanner.rank_capacity(num_routes, num_experts, ep_size, alignment)


class MoonEPLoadPlanner(MoELoadPlanner):
    """PR #7450 MoonEP-style L2 planner.

    ``num_redundant_experts`` is the number of replica slots per EP rank.  The
    PR planner requires this to equal the number of native experts per rank,
    giving each rank ``[home experts][replica experts]`` physical slots.
    """

    planner_name = "moon_ep"

    def __init__(
        self, num_redundant_experts: Optional[int] = None, *, token_padding: int = 1
    ) -> None:
        super().__init__()
        if num_redundant_experts is not None and num_redundant_experts < 0:
            raise ValueError("num_redundant_experts must be non-negative when provided.")
        if (
            not isinstance(token_padding, int)
            or isinstance(token_padding, bool)
            or token_padding < 1
        ):
            raise ValueError("MoonEP token_padding must be a positive integer.")
        self.num_redundant_experts = num_redundant_experts
        self.token_padding = token_padding

    def _resolve_num_redundant_experts(self, context: SchedulerContext) -> int:
        if self.num_redundant_experts is not None:
            return self.num_redundant_experts
        return context.num_logical_experts // context.ep_size

    def _validate_inputs(
        self, probs: torch.Tensor, routing_map: torch.Tensor, context: SchedulerContext
    ) -> None:
        if routing_map.dim() != 2:
            raise ValueError(f"routing_map must be 2D, got shape {tuple(routing_map.shape)}.")
        if probs.shape != routing_map.shape:
            raise ValueError(
                f"probs and routing_map must have the same shape, got "
                f"{tuple(probs.shape)} and {tuple(routing_map.shape)}."
            )
        if routing_map.dtype in (torch.int32, torch.int64):
            if routing_map.size(1) != context.router_topk:
                raise ValueError("Compact MoonEP routes must have router_topk columns.")
            return
        if routing_map.size(1) != context.num_logical_experts:
            raise ValueError(
                "routing_map logical expert dimension does not match SchedulerContext, "
                f"got {routing_map.size(1)} and {context.num_logical_experts}."
            )
        if routing_map.dtype != torch.bool:
            raise ValueError(f"routing_map must be bool, got {routing_map.dtype}.")

    def _validate_context(self, context: SchedulerContext) -> None:
        if context.num_logical_experts % context.ep_size != 0:
            raise ValueError("MoonEPLoadPlanner requires num_logical_experts divisible by ep_size.")
        if context.ep_size > MAX_REPLICA_EP_RANKS:
            raise ValueError(
                f"MoonEPLoadPlanner supports at most {MAX_REPLICA_EP_RANKS} EP ranks, "
                f"got {context.ep_size}."
            )
        num_local_home_experts = context.num_logical_experts // context.ep_size
        num_redundant_experts = self._resolve_num_redundant_experts(context)
        if num_redundant_experts not in (0, num_local_home_experts):
            raise ValueError(
                "PR #7450 MoonEPLoadPlanner requires one replica slot per local home expert; "
                f"got num_redundant_experts={num_redundant_experts}, "
                f"num_local_home_experts={num_local_home_experts}."
            )

    @staticmethod
    def _identity_placement(
        routing_map: torch.Tensor, context: SchedulerContext
    ) -> tuple[None, torch.Tensor, MoonEPPlacementResult]:
        return (
            None,
            torch.empty((context.ep_size, 0), dtype=torch.long, device=routing_map.device),
            MoonEPPlacementResult(),
        )

    def _single_rank_placement(
        self, routing_map: torch.Tensor, context: SchedulerContext
    ) -> tuple[None, torch.Tensor, MoonEPPlacementResult]:
        num_redundant_experts = self._resolve_num_redundant_experts(context)
        num_physical_experts = context.num_logical_experts + num_redundant_experts
        physical_to_logical_map = torch.full(
            (num_physical_experts,), -1, dtype=torch.long, device=routing_map.device
        )
        physical_to_logical_map[: context.num_logical_experts] = torch.arange(
            context.num_logical_experts, dtype=torch.long, device=routing_map.device
        )
        return (
            None,
            physical_to_logical_map.reshape(context.ep_size, -1)[
                :, context.num_local_experts :
            ].contiguous(),
            MoonEPPlacementResult(),
        )

    @staticmethod
    def _single_rank_reroute(
        probs: torch.Tensor, routing_map: torch.Tensor, context: SchedulerContext
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if routing_map.dtype != torch.bool:
            return routing_map.to(torch.int16), probs
        num_physical_experts = 2 * context.num_logical_experts
        physical_routing_map = torch.zeros(
            routing_map.size(0), num_physical_experts, dtype=torch.bool, device=routing_map.device
        )
        physical_probs = torch.zeros(
            probs.size(0), num_physical_experts, dtype=probs.dtype, device=probs.device
        )
        physical_routing_map[:, : context.num_logical_experts] = routing_map
        physical_probs[:, : context.num_logical_experts] = probs
        return physical_routing_map, physical_probs

    def _get_ep_group(self, context: SchedulerContext) -> dist.ProcessGroup:
        ep_group = getattr(context.pg_collection, "ep", None)
        if ep_group is None:
            raise ValueError(
                "MoonEPLoadPlanner requires SchedulerContext.pg_collection.ep when ep_size > 1."
            )
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("MoonEPLoadPlanner requires initialized torch.distributed.")
        if dist.get_world_size(group=ep_group) != context.ep_size:
            raise ValueError(
                "SchedulerContext.ep_size does not match the EP process group size, got "
                f"{context.ep_size} and {dist.get_world_size(group=ep_group)}."
            )
        if dist.get_rank(group=ep_group) != context.ep_rank:
            raise ValueError(
                "SchedulerContext.ep_rank does not match the EP process group rank, got "
                f"{context.ep_rank} and {dist.get_rank(group=ep_group)}."
            )
        return ep_group

    @nvtx_decorator(message="virtual_expert_placement")
    def update_placement(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        *,
        tokens_per_expert: Optional[torch.Tensor] = None,
    ) -> tuple[None, torch.Tensor, MoonEPPlacementResult]:
        """Return PR #7450's physical layout and explicit reroute state."""
        del tokens_per_expert
        self._validate_inputs(probs, routing_map, context)
        self._validate_context(context)
        if self._resolve_num_redundant_experts(context) == 0:
            return self._identity_placement(routing_map, context)
        if context.ep_size == 1:
            return self._single_rank_placement(routing_map, context)

        ep_group = self._get_ep_group(context)
        if not routing_map.is_cuda or not probs.is_cuda:
            raise RuntimeError("MoonEPLoadPlanner requires CUDA tensors when ep_size > 1.")
        if routing_map.dtype == torch.bool:
            _, topk_indices = extract_semantic_routes(routing_map, probs, context.router_topk)
        else:
            topk_indices = routing_map
        workspace = get_planner_workspace(
            device=routing_map.device, num_experts=context.num_logical_experts, group=ep_group
        )
        plan = MoonEPPlanner.plan(topk_indices, workspace, self.token_padding)
        experts_to_copy = plan.experts_to_copy
        capacity = padded_rank_capacity(
            topk_indices.numel(), context.num_logical_experts, context.ep_size, self.token_padding
        )
        physical_to_logical_map = _physical_to_logical_map_from_experts_to_copy(
            experts_to_copy, context
        )
        return (
            None,
            physical_to_logical_map.reshape(context.ep_size, -1)[
                :, context.num_local_experts :
            ].contiguous(),
            # Block placement bounds padded GEMM rows, not equal real routes.
            # The same bound also bounds real routes, without double padding.
            # Identity/single-rank paths intentionally retain the dynamic path.
            MoonEPPlacementResult(
                topk_indices=topk_indices,
                virtual_experts=plan.virtual_experts,
                max_dispatch_routes_per_rank=capacity,
                max_dispatch_padded_tokens_per_rank=capacity,
            ),
        )

    @nvtx_decorator(message="virtual_expert_reroute")
    def reroute(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        placement_result: MoEPlacementResult,
        context: SchedulerContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map tokens to the placement returned by ``update_placement``."""
        if not isinstance(placement_result, MoonEPPlacementResult):
            raise TypeError(
                "MoonEPLoadPlanner.reroute requires the MoonEPPlacementResult returned by "
                "MoonEPLoadPlanner.update_placement."
            )
        self._validate_inputs(probs, routing_map, context)
        self._validate_context(context)
        if self._resolve_num_redundant_experts(context) == 0:
            return routing_map, probs
        if context.ep_size == 1:
            return self._single_rank_reroute(probs, routing_map, context)
        if placement_result.topk_indices is None or placement_result.virtual_experts is None:
            raise ValueError("MoonEP distributed reroute requires a retained placement.")
        # The upstream planner already remapped IDs. Probabilities and their gradients
        # stay in top-k order; no dense physical-expert matrix is needed by HybridEP.
        if routing_map.dtype != torch.bool:
            return placement_result.virtual_experts, probs
        topk_probs = torch.gather(probs, 1, placement_result.topk_indices)
        ids = placement_result.virtual_experts.long()
        physical_probs = probs.new_zeros((probs.size(0), 2 * context.num_logical_experts))
        physical_probs = physical_probs.scatter(1, ids, topk_probs)
        physical_routing_map = torch.zeros_like(physical_probs, dtype=torch.bool)
        physical_routing_map.scatter_(1, ids, True)
        return physical_routing_map, physical_probs

    def plan_with_count_matrix(self, *args, **kwargs) -> None:
        """Reject the removed Python count-matrix adapter."""
        del args, kwargs
        raise NotImplementedError(
            "PR #7450 MoonEPLoadPlanner plans from routing_map/probs and the EP process group; "
            "plan_with_count_matrix is not supported."
        )
