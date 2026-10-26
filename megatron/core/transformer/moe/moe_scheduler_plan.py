# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compact MoonEP placement and per-forward execution state."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(slots=True)
class ExpertPlacement:
    """Compact physical routes and the semantic source table; no execution state."""

    virtual_experts: torch.Tensor
    experts_to_copy: torch.Tensor


@dataclass(slots=True)
class VirtualExpertPlan(ExpertPlacement):
    """One forward's routes and in-flight state, retained until its backward finishes.

    ``virtual_experts`` holds int16 [tokens, topk] runtime ids; ``experts_to_copy`` holds
    int32 [EP, local_experts] semantic ids (-1 for unused slots). Repeated MTP forwards
    retain separate plans through their layer-output hooks. ``ready`` joins the planner stream."""

    # The plan's weight push was started and not yet waited for (one wait per push).
    push_in_flight: bool = False
    # The FC layers whose reduction was launched in the backward.
    started: set = field(default_factory=set)
    ready: torch.cuda.Event | None = None


def plan_virtual_expert_routes(
    top_indices: torch.Tensor, workspace, alignment: int = 1
) -> VirtualExpertPlan:
    """Plan deterministic virtual-expert placement for one EP group and map this rank's routes.

    ``top_indices`` are the router's ``[num_tokens, topk]`` expert ids; every rank must route the
    same number of tokens. The histograms are the only cross-rank input and the planner kernel
    exchanges them itself, so every rank computes the same placement.
    ``alignment`` rounds global expert counts once and balances blocks; only real routes are
    remapped. It is a per-call specialization, never a property of the shared workspace.
    """
    if top_indices.dim() != 2 or top_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Virtual-expert planner takes int32/int64 [num_tokens, topk] expert ids.")
    if not isinstance(alignment, int) or alignment < 1:
        raise ValueError("Virtual-expert alignment must be a positive integer.")
    from megatron.core.transformer.moe.ops.virtual_expert_triton import (
        launch_virtual_expert_planner,
    )

    return VirtualExpertPlan(
        *launch_virtual_expert_planner(top_indices.contiguous(), workspace, alignment)
    )


class MoonEPPlanner:
    """Default compact planner policy shared by both scheduler entry points."""

    @staticmethod
    def plan(top_indices, workspace, alignment=1):
        """Use the accepted upstream cooperative kernel without changing route order."""
        return plan_virtual_expert_routes(top_indices, workspace, alignment)

    @staticmethod
    def rank_capacity(num_routes, num_experts, ep_size, alignment):
        """Bound all globally rounded expert blocks, including quantization padding."""
        if num_routes < 0 or min(num_experts, ep_size, alignment) < 1:
            raise ValueError("MoonEP capacity requires nonnegative routes and positive dimensions.")
        total_routes = ep_size * num_routes
        nonempty = min(num_experts, total_routes)
        max_blocks = (total_routes + nonempty * (alignment - 1)) // alignment
        return alignment * ((max_blocks + ep_size - 1) // ep_size)
