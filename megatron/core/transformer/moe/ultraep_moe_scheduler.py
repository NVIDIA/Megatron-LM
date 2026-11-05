# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""UltraEP quota planner over fixed homes.

Only UltraEP's placement and token reroute are used. Replica weights and gradients move through
the configured replica transport, which consumes the ``[EP, S]`` source table like any planner's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from megatron.core.transformer.moe.moe_scheduler import (
    MoELoadPlanner,
    MoEPlacementResult,
    SchedulerContext,
)

# UltraEP placement depends only on the routing map of the current call, so every layer of an EP
# group plans through one planner-only Manager and its single slot. ``schedule()`` reroutes right
# after planning, and backward uses a snapshot, so a later layer may overwrite the slot.
_managers: dict[int, tuple[Any, tuple[int, int]]] = {}
_SLOT = 0


def _get_manager(group: dist.ProcessGroup, num_local_experts: int, num_local_replicas: int):
    sizes = (num_local_experts, num_local_replicas)
    entry = _managers.get(id(group))
    if entry is not None:
        if entry[1] != sizes:
            raise ValueError("UltraEP planner layers of one EP group must share expert counts.")
        return entry[0]
    try:
        from ultra_ep import Manager
    except ImportError as error:
        raise ImportError("moe_scheduler_planner_type='ultra_ep' requires UltraEP.") from error
    # The Manager always allocates communication buffers; this one never moves weights, so use
    # minimal 16-byte expert shards. Creation and destruction are collective over ``group``.
    manager = Manager(
        group=group,
        num_layers=1,
        num_local_master_experts=num_local_experts,
        num_local_redundant_experts=num_local_replicas,
        expert_fc1_numel=8,
        expert_fc2_numel=8,
        is_train=False,
        explicitly_destroy=True,
    )
    _managers[id(group)] = (manager, sizes)
    return manager


def finalize_ultraep_planner_managers() -> None:
    """Collectively release planner Managers before EP process-group teardown."""
    for manager, _ in _managers.values():
        with torch.cuda.device(manager.device):
            torch.cuda.synchronize()
            manager.destroy()
    _managers.clear()


@dataclass(frozen=True)
class UltraEPPlacementResult(MoEPlacementResult):
    """The Manager to reroute with and this forward's physical-to-logical map."""

    manager: Any
    physical_to_logical_map: torch.Tensor


class _SnapshotReroute(torch.autograd.Function):
    """Reroute with UltraEP's quota kernel; backward uses only this forward's saved tensors.

    The physical routes and the placement snapshot fully specify the probability permutation,
    so backward stays correct after later layers reuse the Manager slot. UltraEP sends each
    (token, logical expert) route to exactly one physical instance, so backward scatters into
    unique positions and needs no accumulation; unrouted entries go to a discarded column.
    """

    @staticmethod
    def forward(ctx, probs, routing_map, placement):
        """Reroute through the Manager slot and save this forward's placement snapshot."""
        physical_probs, physical_routes = placement.manager.reroute(_SLOT, probs, routing_map)
        ctx.save_for_backward(placement.physical_to_logical_map, physical_routes)
        ctx.num_logical_experts = probs.size(1)
        ctx.mark_non_differentiable(physical_routes)
        return physical_probs, physical_routes

    @staticmethod
    def backward(ctx, physical_grad, _route_grad):
        """Map physical probability gradients back to their logical experts."""
        if physical_grad is None:
            return None, None, None
        p2l, routes = ctx.saved_tensors
        num_logical = ctx.num_logical_experts
        valid = routes & (p2l >= 0).unsqueeze(0)
        index = torch.where(valid, p2l.long().unsqueeze(0), num_logical)
        logical_grad = physical_grad.new_zeros((physical_grad.size(0), num_logical + 1))
        logical_grad.scatter_(1, index, physical_grad)
        return logical_grad[:, :num_logical], None, None


class UltraEPLoadPlanner(MoELoadPlanner):
    """Replicate hot experts with UltraEP's quota placement; homes stay fixed."""

    planner_name = "ultra_ep"

    def __init__(self, num_redundant_experts: int, group: dist.ProcessGroup) -> None:
        super().__init__()
        self.num_redundant_experts = num_redundant_experts
        self.group = group
        self._pending: UltraEPPlacementResult | None = None

    def update_placement(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        *,
        tokens_per_expert: torch.Tensor | None = None,
    ) -> tuple[None, torch.Tensor, UltraEPPlacementResult]:
        del tokens_per_expert
        if self._pending is not None:
            raise RuntimeError("UltraEP placement must be rerouted before planning again.")
        if not probs.is_cuda or torch.cuda.is_current_stream_capturing():
            raise ValueError("UltraEP planner requires CUDA execution outside graph capture.")
        if (
            probs.dtype != torch.float32
            or routing_map.dtype != torch.bool
            or probs.shape != routing_map.shape
            or probs.size(1) != context.num_logical_experts
        ):
            raise ValueError("UltraEP planner requires FP32 probs and a dense bool routing map.")
        if (
            dist.get_world_size(self.group) != context.ep_size
            or self.num_redundant_experts % context.ep_size
        ):
            raise ValueError("UltraEP planner requires its EP group and whole replica slots.")
        manager = _get_manager(
            self.group, context.num_local_experts, self.num_redundant_experts // context.ep_size
        )
        p2l, _, _ = manager.update_placement(_SLOT, routing_map.contiguous())
        # update_placement returns views of the slot; own the map before the next layer plans.
        placement = UltraEPPlacementResult(manager, p2l.clone())
        self._pending = placement
        replica_sources = placement.physical_to_logical_map.view(context.ep_size, -1)
        return None, replica_sources[:, context.num_local_experts :].contiguous(), placement

    def reroute(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        placement_result: MoEPlacementResult,
        context: SchedulerContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del context
        if placement_result is not self._pending:
            raise ValueError("UltraEP reroute must consume the matching pending placement.")
        self._pending = None
        physical_probs, physical_routes = _SnapshotReroute.apply(
            probs, routing_map, placement_result
        )
        return physical_routes, physical_probs
