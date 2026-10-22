# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Echo load planner for MoEScheduler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from megatron.core import tensor_parallel
from megatron.core.transformer.moe.moe_scheduler import (
    MoELoadPlanner,
    MoEPlacementResult,
    SchedulerContext,
)

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAVE_TRITON = False


@dataclass(frozen=True)
class EchoAssignment:
    """Intermediate Echo assignment state, aligned with Echo PR planner outputs."""

    count_tokens_from_home_expert_to_echo: torch.Tensor
    expert_offloading_map: torch.Tensor
    count_tokens_offloaded_from_ep_rank_to_echo: torch.Tensor
    count_tokens_offloaded_from_ep_rank_from_home_expert: torch.Tensor
    count_tokens_per_expert_after_offload: torch.Tensor
    count_spillover_per_home_expert: torch.Tensor
    capacity_echo_per_ep_rank: torch.Tensor
    assignment_backend: str


@dataclass(frozen=True)
class EchoPlacementResult(MoEPlacementResult):
    """Minimal Echo assignment state needed to reroute one rank's tokens."""

    expert_offloading_map: torch.Tensor
    local_echo_counts: torch.Tensor
    local_home_counts: torch.Tensor


_COMPILED_ECHO_PLACEMENT: Optional[Any] = None


if HAVE_TRITON:

    @triton.jit
    def _reroute_tokens_w_permute_map_kernel(
        indices_token_sorted_ptr,
        idx_expert_for_offload_ptr,
        count_tokens_to_route_ptr,
        offset_cumulative_ptr,
        map_rerouted_ptr,
        map_permute_ptr,
        num_tokens: tl.constexpr,
        num_experts: tl.constexpr,
        num_offloading_experts: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        idx_offload_expert = tl.program_id(0)

        idx_source_expert = tl.load(idx_expert_for_offload_ptr + idx_offload_expert).to(tl.int64)
        if idx_source_expert < 0:
            return
        count_tokens_to_route = tl.load(count_tokens_to_route_ptr + idx_offload_expert).to(tl.int64)
        offset = tl.load(offset_cumulative_ptr + idx_offload_expert).to(tl.int64)

        indices_token_position = tl.arange(0, BLOCK_SIZE)
        mask_valid = indices_token_position < count_tokens_to_route

        offset_base = idx_source_expert * num_tokens + offset
        indices_token = tl.load(
            indices_token_sorted_ptr + offset_base + indices_token_position,
            mask=mask_valid,
            other=0,
        )

        num_total_experts = num_experts + num_offloading_experts
        idx_flat = indices_token * num_total_experts + idx_source_expert
        tl.store(map_rerouted_ptr + idx_flat, False, mask=mask_valid)

        idx_permute = indices_token * num_offloading_experts + idx_offload_expert
        tl.store(map_permute_ptr + idx_permute, idx_source_expert, mask=mask_valid)

        idx_offload_col = num_experts + idx_offload_expert
        idx_flat_rerouted = indices_token * num_total_experts + idx_offload_col
        tl.store(map_rerouted_ptr + idx_flat_rerouted, True, mask=mask_valid)

else:
    _reroute_tokens_w_permute_map_kernel = None


def reroute_tokens_triton(
    routing_map: torch.Tensor,
    probs: torch.Tensor,
    count_tokens_offloading_from_expert: torch.Tensor,
    count_tokens_offloading_to_echo: torch.Tensor,
    expert_offloading_map: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Echo PR's Triton token reroute kernel before rank-major postprocessing."""
    if not HAVE_TRITON or _reroute_tokens_w_permute_map_kernel is None:
        raise RuntimeError("Triton is required for Echo token reroute.")
    if not routing_map.is_cuda or not probs.is_cuda:
        raise RuntimeError("Echo token reroute requires CUDA tensors.")

    del count_tokens_offloading_from_expert
    device = routing_map.device
    num_tokens, num_logical_experts = routing_map.shape
    num_echo_experts = count_tokens_offloading_to_echo.numel()

    logical_routing_map = torch.zeros(
        num_tokens, num_logical_experts + num_echo_experts, dtype=torch.bool, device=device
    )
    logical_routing_map[:, :num_logical_experts] = routing_map.clone()
    if num_tokens == 0 or num_echo_experts == 0:
        logical_probs = torch.zeros(
            num_tokens,
            num_logical_experts + num_echo_experts,
            dtype=probs.dtype,
            device=probs.device,
        )
        logical_probs[:, :num_logical_experts] = probs.clone()
        return logical_routing_map, logical_probs

    count_tokens_offloading_to_echo = count_tokens_offloading_to_echo.to(torch.int64)
    expert_offloading_map = expert_offloading_map.to(device=device, dtype=torch.bool)
    count_tokens_from_home_to_echo = expert_offloading_map.to(
        torch.int64
    ) * count_tokens_offloading_to_echo.unsqueeze(0)
    offset_cumulative = torch.cumsum(count_tokens_from_home_to_echo, dim=1)
    offset_cumulative = offset_cumulative - count_tokens_from_home_to_echo

    has_routing_from_expert = expert_offloading_map.any(dim=0)
    indices_expert = torch.argmax(expert_offloading_map.float(), dim=0)
    idx_expert_for_offload = torch.where(has_routing_from_expert, indices_expert, -1)
    indices_offload_expert = torch.arange(num_echo_experts, device=device)
    safe_indices_expert = torch.clamp(idx_expert_for_offload, 0, num_logical_experts - 1)
    offset_all = offset_cumulative[safe_indices_expert, indices_offload_expert]
    offset_cumulative = torch.where(idx_expert_for_offload >= 0, offset_all, 0)

    indices_token_sorted = routing_map.argsort(dim=0, descending=True).T.contiguous()
    map_permute = torch.zeros((num_tokens, num_echo_experts), dtype=torch.int64, device=device)
    block_size = triton.next_power_of_2(num_tokens)
    _reroute_tokens_w_permute_map_kernel[(num_echo_experts,)](
        indices_token_sorted,
        idx_expert_for_offload,
        count_tokens_offloading_to_echo,
        offset_cumulative,
        logical_routing_map,
        map_permute,
        num_tokens,
        num_logical_experts,
        num_echo_experts,
        block_size,
    )

    probs_gathered = torch.gather(probs, 1, map_permute)
    logical_probs = torch.cat([probs, probs_gathered], dim=1)
    logical_probs = logical_probs * logical_routing_map
    return logical_routing_map, logical_probs


def one_shot_greedy_assignment(
    count_tokens_per_chunk: torch.Tensor, capacity_per_bucket: torch.Tensor
) -> torch.Tensor:
    """Assign chunk token counts to bucket capacities using interval overlap."""
    chunk_ends = torch.cumsum(count_tokens_per_chunk, dim=0)
    bucket_ends = torch.cumsum(capacity_per_bucket, dim=0)
    chunk_starts = chunk_ends - count_tokens_per_chunk
    bucket_starts = bucket_ends - capacity_per_bucket

    overlap_starts = torch.maximum(chunk_starts.unsqueeze(1), bucket_starts.unsqueeze(0))
    overlap_ends = torch.minimum(chunk_ends.unsqueeze(1), bucket_ends.unsqueeze(0))
    return (overlap_ends - overlap_starts).clamp(min=0)


def _logical_to_home_physical_ids(
    logical_expert_ids: torch.Tensor, home_experts_per_rank: int, local_physical_experts: int
) -> torch.Tensor:
    rank_ids = torch.div(logical_expert_ids, home_experts_per_rank, rounding_mode="floor")
    local_ids = logical_expert_ids.remainder(home_experts_per_rank)
    return rank_ids * local_physical_experts + local_ids


def _echo_to_physical_ids(
    echo_expert_ids: torch.Tensor, home_experts_per_rank: int, echo_experts_per_rank: int
) -> torch.Tensor:
    rank_ids = torch.div(echo_expert_ids, echo_experts_per_rank, rounding_mode="floor")
    local_echo_ids = echo_expert_ids.remainder(echo_experts_per_rank)
    local_physical_experts = home_experts_per_rank + echo_experts_per_rank
    return rank_ids * local_physical_experts + home_experts_per_rank + local_echo_ids


def _physical_to_rank_and_slot(
    physical_expert_ids: torch.Tensor, local_physical_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    rank_ids = torch.div(physical_expert_ids, local_physical_experts, rounding_mode="floor")
    local_slots = physical_expert_ids.remainder(local_physical_experts)
    return rank_ids, local_slots


def _postprocess_to_rank_major(
    routing_map: torch.Tensor,
    probs: torch.Tensor,
    num_logical_experts: int,
    num_echo_experts: int,
    ep_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    home_experts_per_rank = num_logical_experts // ep_size
    echo_experts_per_rank = num_echo_experts // ep_size
    home_routing_map = routing_map[:, :num_logical_experts].reshape(
        -1, ep_size, home_experts_per_rank
    )
    echo_routing_map = routing_map[:, num_logical_experts:].reshape(
        -1, ep_size, echo_experts_per_rank
    )
    rank_major_routing_map = torch.cat([home_routing_map, echo_routing_map], dim=-1).reshape(
        -1, num_logical_experts + num_echo_experts
    )

    home_probs = probs[:, :num_logical_experts].reshape(-1, ep_size, home_experts_per_rank)
    echo_probs = probs[:, num_logical_experts:].reshape(-1, ep_size, echo_experts_per_rank)
    rank_major_probs = torch.cat([home_probs, echo_probs], dim=-1).reshape(
        -1, num_logical_experts + num_echo_experts
    )
    return rank_major_routing_map, rank_major_probs


def _physical_to_logical_map_from_echo_offloading_map(
    expert_offloading_map: torch.Tensor,
    num_logical_experts: int,
    num_echo_experts: int,
    ep_size: int,
) -> torch.Tensor:
    home_experts_per_rank = num_logical_experts // ep_size
    echo_experts_per_rank = num_echo_experts // ep_size
    num_physical_experts = num_logical_experts + num_echo_experts
    local_physical_experts = home_experts_per_rank + echo_experts_per_rank
    device = expert_offloading_map.device

    physical_to_logical = torch.full((num_physical_experts,), -1, dtype=torch.long, device=device)
    logical_expert_ids = torch.arange(num_logical_experts, dtype=torch.long, device=device)
    home_physical_ids = _logical_to_home_physical_ids(
        logical_expert_ids, home_experts_per_rank, local_physical_experts
    )
    physical_to_logical[home_physical_ids] = logical_expert_ids

    if num_echo_experts == 0:
        return physical_to_logical

    echo_expert_ids = torch.arange(num_echo_experts, dtype=torch.long, device=device)
    echo_physical_ids = _echo_to_physical_ids(
        echo_expert_ids, home_experts_per_rank, echo_experts_per_rank
    )
    has_echo_source = expert_offloading_map.any(dim=0)
    echo_source_ids = expert_offloading_map.to(dtype=torch.long).argmax(dim=0)
    echo_source_ids = torch.where(
        has_echo_source, echo_source_ids, torch.full_like(echo_source_ids, -1)
    )
    physical_to_logical[echo_physical_ids] = echo_source_ids
    return physical_to_logical


def _echo_compute_intermediate(
    counts_from_ep_rank: torch.Tensor, ep_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    home_experts_per_rank = counts_from_ep_rank.size(1) // ep_size
    total_tokens_per_expert = counts_from_ep_rank.sum(dim=0)
    tokens_per_ep_rank = total_tokens_per_expert.view(ep_size, home_experts_per_rank).sum(dim=1)
    avg_tokens_per_ep_rank = tokens_per_ep_rank.sum() // ep_size
    capacity_echo_per_ep_rank = torch.relu(avg_tokens_per_ep_rank - tokens_per_ep_rank)

    local_counts = total_tokens_per_expert.view(ep_size, home_experts_per_rank)
    sorted_local_counts, sorted_local_indices = local_counts.sort(dim=1)
    spillover_cumsum = (sorted_local_counts.cumsum(dim=1) - avg_tokens_per_ep_rank).clamp(min=0)
    spillover_sorted = torch.cat(
        [spillover_cumsum[:, :1], torch.diff(spillover_cumsum, dim=1)], dim=1
    )
    spillover_per_expert = torch.scatter(
        torch.empty_like(spillover_sorted), 1, sorted_local_indices, spillover_sorted
    ).reshape(-1)
    return spillover_per_expert, capacity_echo_per_ep_rank, avg_tokens_per_ep_rank


def _echo_compute_one_shot_assignment(
    counts_from_ep_rank: torch.Tensor, ep_size: int, num_echo_experts: int
) -> torch.Tensor:
    echo_experts_per_rank = num_echo_experts // ep_size
    spillover, capacity, _ = _echo_compute_intermediate(counts_from_ep_rank, ep_size)
    if echo_experts_per_rank == 0:
        return torch.zeros(
            counts_from_ep_rank.size(1),
            0,
            dtype=counts_from_ep_rank.dtype,
            device=counts_from_ep_rank.device,
        )

    spillover_sorted, spillover_order = torch.sort(spillover, descending=True)
    capacity_sorted, capacity_order = torch.sort(capacity, descending=True)
    sorted_assignment = one_shot_greedy_assignment(spillover_sorted, capacity_sorted)
    num_selected = min(echo_experts_per_rank, sorted_assignment.size(0))
    selected_counts, selected_rows = torch.topk(sorted_assignment, k=num_selected, dim=0)

    assignment = torch.zeros(
        counts_from_ep_rank.size(1),
        num_echo_experts,
        dtype=sorted_assignment.dtype,
        device=counts_from_ep_rank.device,
    )
    if num_selected == 0:
        return assignment

    expert_ids = spillover_order[selected_rows.transpose(0, 1).reshape(-1)]
    sorted_rank_ids = torch.arange(ep_size, device=counts_from_ep_rank.device)
    sorted_rank_ids = sorted_rank_ids.repeat_interleave(num_selected)
    echo_local_ids = torch.arange(num_selected, device=counts_from_ep_rank.device).repeat(ep_size)
    rank_ids = capacity_order[sorted_rank_ids]
    echo_ids = rank_ids * echo_experts_per_rank + echo_local_ids
    assignment[expert_ids, echo_ids] = selected_counts.transpose(0, 1).reshape(-1)
    return assignment


def _echo_breadth_first_allocation(
    counts_from_ep_rank: torch.Tensor, home_to_echo_counts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    counts_float = counts_from_ep_rank.float()
    home_to_echo_float = home_to_echo_counts.float()
    if home_to_echo_float.size(1) == 0:
        return (
            torch.zeros(
                counts_from_ep_rank.size(0), 0, dtype=torch.int64, device=counts_from_ep_rank.device
            ),
            counts_from_ep_rank.clone(),
            home_to_echo_counts.clone(),
        )

    echo_source_expert_ids = home_to_echo_float.argmax(dim=0)
    active_echo_mask = (home_to_echo_float > 0).sum(dim=0) > 0
    capacity = home_to_echo_float[
        echo_source_expert_ids, torch.arange(home_to_echo_float.size(1), device=counts_float.device)
    ]
    source_rank_counts = counts_float[:, echo_source_expert_ids]
    denominator = source_rank_counts.sum(dim=0, keepdim=True)
    proportional = torch.where(
        denominator > 0, source_rank_counts / denominator, torch.zeros_like(source_rank_counts)
    )
    first_pass = (torch.floor(proportional * capacity).to(torch.int64)) * active_echo_mask

    offloaded_from_home = torch.zeros_like(counts_from_ep_rank, dtype=torch.int64)
    source_ids = echo_source_expert_ids.unsqueeze(0).expand(counts_from_ep_rank.size(0), -1)
    offloaded_from_home.scatter_add_(1, source_ids, first_pass)
    counts_after_offload = counts_from_ep_rank - offloaded_from_home
    remaining_capacity = home_to_echo_counts.clone()
    remaining_capacity[
        echo_source_expert_ids,
        torch.arange(home_to_echo_counts.size(1), device=counts_float.device),
    ] -= first_pass.sum(dim=0)
    return first_pass, counts_after_offload, remaining_capacity


def _echo_depth_first_allocation(
    counts_from_ep_rank: torch.Tensor, remaining_capacity: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    num_ep_ranks, num_logical_experts = counts_from_ep_rank.shape
    if remaining_capacity.size(1) == 0:
        return (
            torch.zeros(num_ep_ranks, 0, dtype=torch.int64, device=counts_from_ep_rank.device),
            counts_from_ep_rank.clone(),
        )

    token_ends = torch.cumsum(counts_from_ep_rank, dim=0)
    token_starts = token_ends - counts_from_ep_rank
    capacity_ends = torch.cumsum(remaining_capacity, dim=1)
    capacity_starts = capacity_ends - remaining_capacity

    overlap_starts = torch.maximum(token_starts.unsqueeze(2), capacity_starts.unsqueeze(0))
    overlap_ends = torch.minimum(token_ends.unsqueeze(2), capacity_ends.unsqueeze(0))
    overlaps = (overlap_ends - overlap_starts).clamp(min=0)
    second_pass = overlaps.sum(dim=1).to(torch.int64)

    echo_source_expert_ids = remaining_capacity.argmax(dim=0)
    source_ids = echo_source_expert_ids.unsqueeze(0).expand(num_ep_ranks, -1)
    counts_after_offload = counts_from_ep_rank.scatter_add(1, source_ids, -second_pass)
    return second_pass, counts_after_offload


def _echo_update_placement_impl(
    counts_from_ep_rank: torch.Tensor, ep_rank: int, ep_size: int, num_echo_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the physical layout and local state needed by the reroute phase."""
    home_to_echo_counts = _echo_compute_one_shot_assignment(
        counts_from_ep_rank, ep_size, num_echo_experts
    )

    first_pass, counts_after_first, remaining_capacity = _echo_breadth_first_allocation(
        counts_from_ep_rank, home_to_echo_counts
    )
    second_pass, counts_after = _echo_depth_first_allocation(counts_after_first, remaining_capacity)
    count_to_echo = first_pass + second_pass
    count_from_home = counts_from_ep_rank - counts_after
    expert_offloading_map = home_to_echo_counts > 0
    physical_to_logical_map = _physical_to_logical_map_from_echo_offloading_map(
        expert_offloading_map, counts_from_ep_rank.size(1), num_echo_experts, ep_size
    )
    return (
        physical_to_logical_map,
        expert_offloading_map,
        count_to_echo[ep_rank],
        count_from_home[ep_rank],
    )


def _get_compiled_echo_placement() -> Any:
    global _COMPILED_ECHO_PLACEMENT
    if _COMPILED_ECHO_PLACEMENT is None:
        compile_fn = getattr(torch, "compile", None)
        if callable(compile_fn):
            _COMPILED_ECHO_PLACEMENT = compile_fn(_echo_update_placement_impl)
        else:
            _COMPILED_ECHO_PLACEMENT = _echo_update_placement_impl
    return _COMPILED_ECHO_PLACEMENT


def _dense_to_topk(
    routing_map: torch.Tensor, probs: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a dense multihot routing map into per-token expert ids and probs."""
    physical_expert_ids = torch.empty(
        routing_map.size(0), topk, dtype=torch.long, device=routing_map.device
    )
    assignment_probs = torch.empty(
        routing_map.size(0), topk, dtype=probs.dtype, device=probs.device
    )
    for token_idx in range(routing_map.size(0)):
        expert_ids = torch.nonzero(routing_map[token_idx], as_tuple=False).flatten()
        if expert_ids.numel() != topk:
            raise ValueError(
                f"Expected every token to route to {topk} experts after Echo reroute, "
                f"got {expert_ids.numel()} for token {token_idx}."
            )
        physical_expert_ids[token_idx] = expert_ids
        assignment_probs[token_idx] = probs[token_idx, expert_ids]
    return physical_expert_ids, assignment_probs


class EchoLoadPlanner(MoELoadPlanner):
    """Echo planner that separates expert placement from token rerouting.

    The planner mirrors Echo PR #2368 at the algorithm boundary: assignment is
    computed first, then token flow is split with breadth-first and depth-first
    allocation before the local rank's route map is rewritten.  CUDA inputs use
    the Triton reroute kernel and one-shot greedy assignment.
    """

    planner_name = "echo"

    def __init__(
        self,
        num_echo_experts: int,
        *,
        enable_random_offloading: bool = False,
        random_seed: int = 42,
        use_triton: bool = True,
        require_triton: bool = False,
    ) -> None:
        super().__init__()
        if num_echo_experts < 0:
            raise ValueError("num_echo_experts must be non-negative.")
        self.num_echo_experts = num_echo_experts
        self.enable_random_offloading = enable_random_offloading
        self.random_seed = random_seed
        self.use_triton = use_triton
        self.require_triton = require_triton

    def _can_use_triton(self, *tensors: torch.Tensor) -> bool:
        return self.use_triton and HAVE_TRITON and all(tensor.is_cuda for tensor in tensors)

    def _require_triton_if_configured(self, operation: str, *tensors: torch.Tensor) -> None:
        if self.require_triton and not self._can_use_triton(*tensors):
            raise RuntimeError(
                f"EchoLoadPlanner requires Triton {operation}, but Triton is unavailable "
                "or the planner inputs are not CUDA tensors."
            )

    def _validate_context(self, context: SchedulerContext) -> None:
        if context.num_logical_experts % context.ep_size != 0:
            raise ValueError(
                "EchoLoadPlanner requires num_logical_experts to be divisible by ep_size."
            )
        if self.num_echo_experts % context.ep_size != 0:
            raise ValueError(
                "EchoLoadPlanner requires num_echo_experts to be divisible by ep_size."
            )

    def _get_count_matrix(
        self,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        *,
        tokens_per_expert: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        local_counts = (
            tokens_per_expert if tokens_per_expert is not None else routing_map.sum(dim=0)
        ).to(dtype=torch.int64)
        if local_counts.numel() != context.num_logical_experts:
            raise ValueError(
                "Expected local token counts to match the logical expert dimension, "
                f"got {local_counts.numel()} and {context.num_logical_experts}."
            )
        if context.ep_size == 1:
            return local_counts.unsqueeze(0)

        ep_group = getattr(context.pg_collection, "ep", None)
        if ep_group is None:
            raise ValueError(
                "EchoLoadPlanner requires SchedulerContext.pg_collection.ep when ep_size > 1."
            )
        counts = tensor_parallel.gather_from_sequence_parallel_region(
            local_counts, group=ep_group
        ).reshape(context.ep_size, context.num_logical_experts)
        return counts.to(dtype=torch.int64)

    def _compute_intermediate(
        self, counts_from_ep_rank: torch.Tensor, context: SchedulerContext
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        home_experts_per_rank = context.num_logical_experts // context.ep_size
        total_tokens_per_expert = counts_from_ep_rank.sum(dim=0)
        tokens_per_ep_rank = total_tokens_per_expert.view(
            context.ep_size, home_experts_per_rank
        ).sum(dim=1)
        avg_tokens_per_ep_rank = tokens_per_ep_rank.sum() // context.ep_size
        capacity_echo_per_ep_rank = torch.relu(avg_tokens_per_ep_rank - tokens_per_ep_rank)

        local_counts = total_tokens_per_expert.view(context.ep_size, home_experts_per_rank)
        sorted_local_counts, sorted_local_indices = local_counts.sort(dim=1)
        spillover_cumsum = (sorted_local_counts.cumsum(dim=1) - avg_tokens_per_ep_rank).clamp(min=0)
        spillover_sorted = torch.cat(
            [spillover_cumsum[:, :1], torch.diff(spillover_cumsum, dim=1)], dim=1
        )
        spillover_per_expert = torch.scatter(
            torch.empty_like(spillover_sorted), 1, sorted_local_indices, spillover_sorted
        ).reshape(-1)
        return spillover_per_expert, capacity_echo_per_ep_rank, avg_tokens_per_ep_rank

    def _compute_one_shot_assignment(
        self, counts_from_ep_rank: torch.Tensor, context: SchedulerContext
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        echo_experts_per_rank = self.num_echo_experts // context.ep_size
        spillover, capacity, _ = self._compute_intermediate(counts_from_ep_rank, context)
        if echo_experts_per_rank == 0:
            return (
                torch.zeros(
                    context.num_logical_experts,
                    0,
                    dtype=counts_from_ep_rank.dtype,
                    device=counts_from_ep_rank.device,
                ),
                spillover,
                capacity,
                "torch",
            )

        spillover_sorted, spillover_order = torch.sort(spillover, descending=True)
        capacity_sorted, capacity_order = torch.sort(capacity, descending=True)
        sorted_assignment = one_shot_greedy_assignment(spillover_sorted, capacity_sorted)
        num_selected = min(echo_experts_per_rank, sorted_assignment.size(0))
        selected_counts, selected_rows = torch.topk(sorted_assignment, k=num_selected, dim=0)

        assignment = torch.zeros(
            context.num_logical_experts,
            self.num_echo_experts,
            dtype=sorted_assignment.dtype,
            device=counts_from_ep_rank.device,
        )
        if num_selected == 0:
            return assignment, spillover, capacity, "torch"

        expert_ids = spillover_order[selected_rows.transpose(0, 1).reshape(-1)]
        sorted_rank_ids = torch.arange(context.ep_size, device=counts_from_ep_rank.device)
        sorted_rank_ids = sorted_rank_ids.repeat_interleave(num_selected)
        echo_local_ids = torch.arange(num_selected, device=counts_from_ep_rank.device).repeat(
            context.ep_size
        )
        rank_ids = capacity_order[sorted_rank_ids]
        echo_ids = rank_ids * echo_experts_per_rank + echo_local_ids
        assignment[expert_ids, echo_ids] = selected_counts.transpose(0, 1).reshape(-1)
        return assignment, spillover, capacity, "torch"

    def _compute_random_assignment(
        self, routing_map: torch.Tensor, context: SchedulerContext
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        generator = torch.Generator(device=routing_map.device)
        generator.manual_seed(self.random_seed)
        source_expert_ids = torch.randint(
            0,
            context.num_logical_experts,
            (self.num_echo_experts,),
            device=routing_map.device,
            generator=generator,
        )
        expert_offloading_map = torch.zeros(
            context.num_logical_experts,
            self.num_echo_experts,
            dtype=torch.bool,
            device=routing_map.device,
        )
        expert_offloading_map[
            source_expert_ids, torch.arange(self.num_echo_experts, device=routing_map.device)
        ] = True

        assignment = torch.zeros_like(expert_offloading_map, dtype=torch.int64)
        for expert_id in range(context.num_logical_experts):
            token_count = int(routing_map[:, expert_id].sum().item())
            if token_count == 0:
                continue
            echo_ids = torch.nonzero(expert_offloading_map[expert_id], as_tuple=False).flatten()
            if echo_ids.numel() == 0:
                continue
            offload_ratio = torch.rand(1, device=routing_map.device, generator=generator)
            num_to_offload = int(token_count * (offload_ratio.item() * 0.2 + 0.2))
            if num_to_offload > 0:
                assignment[expert_id, echo_ids[0]] = num_to_offload

        spillover = assignment.sum(dim=1)
        capacity = assignment.reshape(context.num_logical_experts, context.ep_size, -1).sum(
            dim=(0, 2)
        )
        return assignment, spillover, capacity, "random"

    @staticmethod
    def _breadth_first_allocation(
        counts_from_ep_rank: torch.Tensor, home_to_echo_counts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        counts_float = counts_from_ep_rank.float()
        home_to_echo_float = home_to_echo_counts.float()
        if home_to_echo_float.size(1) == 0:
            return (
                torch.zeros(
                    counts_from_ep_rank.size(0),
                    0,
                    dtype=torch.int64,
                    device=counts_from_ep_rank.device,
                ),
                counts_from_ep_rank.clone(),
                home_to_echo_counts.clone(),
            )

        echo_source_expert_ids = home_to_echo_float.argmax(dim=0)
        active_echo_mask = (home_to_echo_float > 0).sum(dim=0) > 0
        capacity = home_to_echo_float[
            echo_source_expert_ids,
            torch.arange(home_to_echo_float.size(1), device=counts_float.device),
        ]
        source_rank_counts = counts_float[:, echo_source_expert_ids]
        denominator = source_rank_counts.sum(dim=0, keepdim=True)
        proportional = torch.where(
            denominator > 0, source_rank_counts / denominator, torch.zeros_like(source_rank_counts)
        )
        first_pass = (torch.floor(proportional * capacity).to(torch.int64)) * active_echo_mask

        offloaded_from_home = torch.zeros_like(counts_from_ep_rank, dtype=torch.int64)
        source_ids = echo_source_expert_ids.unsqueeze(0).expand(counts_from_ep_rank.size(0), -1)
        offloaded_from_home.scatter_add_(1, source_ids, first_pass)
        counts_after_offload = counts_from_ep_rank - offloaded_from_home
        remaining_capacity = home_to_echo_counts.clone()
        remaining_capacity[
            echo_source_expert_ids,
            torch.arange(home_to_echo_counts.size(1), device=counts_float.device),
        ] -= first_pass.sum(dim=0)
        return first_pass, counts_after_offload, remaining_capacity

    @staticmethod
    def _depth_first_allocation(
        counts_from_ep_rank: torch.Tensor, remaining_capacity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_ep_ranks, num_logical_experts = counts_from_ep_rank.shape
        if remaining_capacity.size(1) == 0:
            return (
                torch.zeros(num_ep_ranks, 0, dtype=torch.int64, device=counts_from_ep_rank.device),
                counts_from_ep_rank.clone(),
            )
        if remaining_capacity.size(0) != num_logical_experts:
            raise ValueError("remaining_capacity expert dimension does not match token counts.")

        token_ends = torch.cumsum(counts_from_ep_rank, dim=0)
        token_starts = token_ends - counts_from_ep_rank
        capacity_ends = torch.cumsum(remaining_capacity, dim=1)
        capacity_starts = capacity_ends - remaining_capacity

        overlap_starts = torch.maximum(token_starts.unsqueeze(2), capacity_starts.unsqueeze(0))
        overlap_ends = torch.minimum(token_ends.unsqueeze(2), capacity_ends.unsqueeze(0))
        overlaps = (overlap_ends - overlap_starts).clamp(min=0)
        second_pass = overlaps.sum(dim=1).to(torch.int64)

        echo_source_expert_ids = remaining_capacity.argmax(dim=0)
        source_ids = echo_source_expert_ids.unsqueeze(0).expand(num_ep_ranks, -1)
        counts_after_offload = counts_from_ep_rank.scatter_add(1, source_ids, -second_pass)
        return second_pass, counts_after_offload

    def _compute_assignment(
        self,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        *,
        tokens_per_expert: Optional[torch.Tensor] = None,
    ) -> EchoAssignment:
        counts_from_ep_rank = self._get_count_matrix(
            routing_map, context, tokens_per_expert=tokens_per_expert
        )
        if self.enable_random_offloading:
            home_to_echo_counts, spillover, capacity, assignment_backend = (
                self._compute_random_assignment(routing_map, context)
            )
            count_to_echo = torch.zeros(
                context.ep_size, self.num_echo_experts, dtype=torch.int64, device=routing_map.device
            )
            count_from_home = torch.zeros_like(counts_from_ep_rank)
            count_to_echo[context.ep_rank] = home_to_echo_counts.sum(dim=0)
            count_from_home[context.ep_rank] = home_to_echo_counts.sum(dim=1)
            counts_after = counts_from_ep_rank - count_from_home
        else:
            home_to_echo_counts, spillover, capacity, assignment_backend = (
                self._compute_one_shot_assignment(counts_from_ep_rank, context)
            )

            first_pass, counts_after_first, remaining_capacity = self._breadth_first_allocation(
                counts_from_ep_rank, home_to_echo_counts
            )
            second_pass, counts_after = self._depth_first_allocation(
                counts_after_first, remaining_capacity
            )
            count_to_echo = first_pass + second_pass
            count_from_home = counts_from_ep_rank - counts_after

        return EchoAssignment(
            count_tokens_from_home_expert_to_echo=home_to_echo_counts,
            expert_offloading_map=home_to_echo_counts > 0,
            count_tokens_offloaded_from_ep_rank_to_echo=count_to_echo,
            count_tokens_offloaded_from_ep_rank_from_home_expert=count_from_home,
            count_tokens_per_expert_after_offload=counts_after,
            count_spillover_per_home_expert=spillover,
            capacity_echo_per_ep_rank=capacity,
            assignment_backend=assignment_backend,
        )

    def _build_physical_to_logical_map(
        self, assignment: EchoAssignment, context: SchedulerContext
    ) -> torch.Tensor:
        return _physical_to_logical_map_from_echo_offloading_map(
            assignment.expert_offloading_map,
            context.num_logical_experts,
            self.num_echo_experts,
            context.ep_size,
        )

    def _build_token_reroute(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        placement_result: EchoPlacementResult,
        context: SchedulerContext,
    ) -> dict[str, Any]:
        num_logical_experts = context.num_logical_experts
        local_echo_counts = placement_result.local_echo_counts.to(dtype=torch.int64)
        local_home_counts = placement_result.local_home_counts.to(dtype=torch.int64)

        if self._can_use_triton(
            routing_map, probs, local_echo_counts, placement_result.expert_offloading_map
        ):
            logical_routing_map, logical_probs = reroute_tokens_triton(
                routing_map,
                probs,
                local_home_counts,
                local_echo_counts,
                placement_result.expert_offloading_map,
            )
            reroute_backend = "triton"
        else:
            self._require_triton_if_configured(
                "token reroute",
                routing_map,
                probs,
                local_echo_counts,
                placement_result.expert_offloading_map,
            )
            logical_routing_map = torch.zeros(
                routing_map.size(0),
                num_logical_experts + self.num_echo_experts,
                dtype=torch.bool,
                device=routing_map.device,
            )
            logical_probs = torch.zeros(
                routing_map.size(0),
                num_logical_experts + self.num_echo_experts,
                dtype=probs.dtype,
                device=probs.device,
            )
            logical_routing_map[:, :num_logical_experts] = routing_map.clone()
            logical_probs[:, :num_logical_experts] = probs.clone()

            count_tokens_from_home_to_echo = placement_result.expert_offloading_map.to(
                dtype=torch.int64
            ) * local_echo_counts.unsqueeze(0)
            offset_starts = torch.cumsum(count_tokens_from_home_to_echo, dim=1)
            offset_starts = offset_starts - count_tokens_from_home_to_echo
            sorted_token_indices = routing_map.argsort(dim=0, descending=True).T.contiguous()

            for echo_expert_id in range(self.num_echo_experts):
                num_to_offload = int(local_echo_counts[echo_expert_id].item())
                if num_to_offload <= 0:
                    continue
                source_mask = placement_result.expert_offloading_map[:, echo_expert_id]
                if not bool(source_mask.any()):
                    continue
                source_expert_id = int(torch.argmax(source_mask.to(dtype=torch.int64)).item())
                offset = int(offset_starts[source_expert_id, echo_expert_id].item())
                selected_tokens = sorted_token_indices[
                    source_expert_id, offset : offset + num_to_offload
                ]
                echo_col = num_logical_experts + echo_expert_id
                logical_routing_map[selected_tokens, source_expert_id] = False
                logical_routing_map[selected_tokens, echo_col] = True
                logical_probs[selected_tokens, echo_col] = logical_probs[
                    selected_tokens, source_expert_id
                ]
                logical_probs[selected_tokens, source_expert_id] = 0
            reroute_backend = "torch"

        rerouting_map, rerouted_probs = _postprocess_to_rank_major(
            logical_routing_map,
            logical_probs,
            num_logical_experts,
            self.num_echo_experts,
            context.ep_size,
        )
        return {
            "routing_map": rerouting_map,
            "probs": rerouted_probs,
            "reroute_backend": reroute_backend,
        }

    def update_placement(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        *,
        tokens_per_expert: Optional[torch.Tensor] = None,
    ) -> tuple[None, torch.Tensor, EchoPlacementResult]:
        """Return the Echo physical layout and its explicit assignment state."""
        if routing_map.size(1) != context.num_logical_experts:
            raise ValueError(
                "routing_map logical expert dimension does not match SchedulerContext, "
                f"got {routing_map.size(1)} and {context.num_logical_experts}."
            )
        self._validate_context(context)

        counts_from_ep_rank = self._get_count_matrix(
            routing_map, context, tokens_per_expert=tokens_per_expert
        )
        if not self.enable_random_offloading and self._can_use_triton(
            routing_map, probs, counts_from_ep_rank
        ):
            (
                physical_to_logical_map,
                expert_offloading_map,
                local_echo_counts,
                local_home_counts,
            ) = _get_compiled_echo_placement()(
                counts_from_ep_rank, context.ep_rank, context.ep_size, self.num_echo_experts
            )
            return (
                None,
                physical_to_logical_map.reshape(context.ep_size, -1)[
                    :, context.num_local_experts :
                ].contiguous(),
                EchoPlacementResult(
                    expert_offloading_map=expert_offloading_map,
                    local_echo_counts=local_echo_counts,
                    local_home_counts=local_home_counts,
                ),
            )

        assignment = self._compute_assignment(
            routing_map, context, tokens_per_expert=counts_from_ep_rank[context.ep_rank]
        )
        physical_to_logical_map = self._build_physical_to_logical_map(assignment, context)
        return (
            None,
            physical_to_logical_map.reshape(context.ep_size, -1)[
                :, context.num_local_experts :
            ].contiguous(),
            EchoPlacementResult(
                expert_offloading_map=assignment.expert_offloading_map,
                local_echo_counts=assignment.count_tokens_offloaded_from_ep_rank_to_echo[
                    context.ep_rank
                ],
                local_home_counts=assignment.count_tokens_offloaded_from_ep_rank_from_home_expert[
                    context.ep_rank
                ],
            ),
        )

    def reroute(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        placement_result: MoEPlacementResult,
        context: SchedulerContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reroute tokens using the assignment returned by ``update_placement``."""
        if not isinstance(placement_result, EchoPlacementResult):
            raise TypeError(
                "EchoLoadPlanner.reroute requires the EchoPlacementResult returned by "
                "EchoLoadPlanner.update_placement."
            )
        token_reroute = self._build_token_reroute(probs, routing_map, placement_result, context)
        return token_reroute["routing_map"], token_reroute["probs"]
