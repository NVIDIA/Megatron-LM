# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MoonEP planner adapter for the backend-neutral MoEScheduler contract.

The planner core tracks NVIDIA/Megatron-LM PR #7450 at commit 242c8dbc6. Its
placement kernel histograms compact routes, exchanges histograms through NCCL
symmetric memory, and places replicas. A separate reroute kernel writes
HybridEP inputs while replica-weight dispatch is in flight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Optional

import torch
import torch.distributed as dist

from megatron.core.transformer.moe.moe_scheduler import (
    MoELoadPlanner,
    MoEPlacementResult,
    SchedulerContext,
)
from megatron.core.transformer.moe.moonep_replica_triton import (
    MAX_REPLICA_EP_RANKS,
    PLANNER_PROGRAMS,
    launch_virtual_expert_placement,
    launch_virtual_expert_reroute,
)
from megatron.core.utils import nvtx_decorator


@dataclass(slots=True)
class ReplicaPlan:
    """``virtual_experts``: int16 ``[num_tokens, router_topk]`` runtime ids (HybridEP's dense
    top-k routing); ``experts_to_copy``: int32 ``[ep_size, num_local_experts]`` semantic ids per
    replica slot, ``-1`` if unused.

    Plain data only. The backward hooks capture the plan inside autograd contexts, so a plan
    holding a differentiable tensor (the runtime probabilities) would close a reference cycle
    through the graph that Python's collector cannot see, leaking every layer's graph.
    """

    virtual_experts: torch.Tensor
    experts_to_copy: torch.Tensor


def _scratch_layout(num_experts: int, ep_size: int) -> tuple[dict, int]:
    """Fields of the planner's int32 scratch arena as ``name -> (offset, shape)`` plus its total
    size; mirrors ``_planner_fields`` in the Triton module."""
    block_ep = 1 << (ep_size - 1).bit_length()
    # Each flag word gets its own 128-byte line (the kernel's _FLAG_STRIDE).
    fields = (
        ("placement_grid_sync", (1,)),
        ("_pad0", (31,)),
        ("grid_sync", (1,)),
        ("_pad1", (31,)),
        ("sequence", (1,)),  # exchange launch counter the flags carry
        ("_pad2", (31,)),
        ("balance", (ep_size,)),  # native load minus rank capacity
        ("allocation", (num_experts, ep_size)),  # routes of each expert per destination
        ("destination_boundaries", (num_experts, block_ep)),  # segment ends, local ordinals
        ("virtual_expert_slots", (num_experts, ep_size)),  # slot holding an expert on a rank
        ("program_histogram", (PLANNER_PROGRAMS, num_experts)),
        ("running_counts", (PLANNER_PROGRAMS, num_experts)),
        ("tokens_per_expert", (num_experts,)),  # this rank's histogram
    )
    layout, offset = {}, 0
    for name, shape in fields:
        layout[name] = (offset, shape)
        offset += math.prod(shape)
    return layout, offset


@dataclass(slots=True)
class ReplicaPlannerWorkspace:
    """Planner scratch for one expert layout and EP group; every plan overwrites it.

    ``gathered_counts`` is this rank's NCCL symmetric-memory window: the planner kernel
    publishes the local histogram into every peer's window and reads the peers' rows from its
    own, so planning needs no collective. ``scratch`` is one int32 arena (see
    :func:`_scratch_layout`); :meth:`field` views its parts.
    """

    num_experts: int
    ep_size: int
    rank: int
    gathered_counts: torch.Tensor  # [ep_size, num_experts] window
    histogram_handle: object  # symmetric-memory handle of gathered_counts
    scratch: torch.Tensor
    placement_kernels: dict = dataclass_field(default_factory=dict)
    reroute_kernels: dict = dataclass_field(default_factory=dict)

    @property
    def num_local_experts(self) -> int:
        """Return the number of home experts owned by each EP rank."""
        return self.num_experts // self.ep_size

    def field(self, name: str) -> torch.Tensor:
        """View one field of the scratch arena."""
        offset, shape = _scratch_layout(self.num_experts, self.ep_size)[0][name]
        return self.scratch[offset : offset + math.prod(shape)].view(shape)

    def destroy(self) -> None:
        """Drop the symmetric window while its process group is still alive."""
        if self.histogram_handle is not None:
            torch.cuda.synchronize(self.gathered_counts.device)
        self.histogram_handle = self.gathered_counts = None
        self.placement_kernels.clear()
        self.reroute_kernels.clear()

    @classmethod
    def allocate(cls, *, num_experts: int, device: torch.device, group: dist.ProcessGroup):
        """Allocate the scratch for one expert layout on ``group``, with the symmetric window."""
        import torch.distributed._symmetric_memory as symm_mem

        ep_size = dist.get_world_size(group=group)
        # The window needs the group's communicator (created by a first collective).
        dist.all_reduce(torch.zeros(1, device=device), group=group)
        if symm_mem.get_backend(device) != "NCCL":
            symm_mem.set_backend("NCCL")
        window = symm_mem.empty(ep_size * num_experts, dtype=torch.int32, device=device)
        handle = symm_mem.rendezvous(window, group)
        if handle.signal_pad_size < ep_size * 4 * 4:
            raise RuntimeError(
                "MoonEP replica planner needs one signal word per EP rank; the symmetric "
                f"memory signal pad holds {handle.signal_pad_size} bytes for {ep_size} ranks."
            )
        return cls.local(
            num_experts,
            ep_size,
            device,
            rank=dist.get_rank(group=group),
            gathered_counts=window.view(ep_size, num_experts),
            histogram_handle=handle,
        )

    @classmethod
    def local(
        cls, num_experts, ep_size, device, *, rank=0, gathered_counts=None, histogram_handle=None
    ):
        """The scratch without a window (process-local tests): ``gathered_counts`` is plain
        memory the caller fills with every rank's histogram."""
        if gathered_counts is None:
            gathered_counts = torch.empty((ep_size, num_experts), dtype=torch.int32, device=device)
        _, size = _scratch_layout(num_experts, ep_size)
        return cls(
            num_experts=num_experts,
            ep_size=ep_size,
            rank=rank,
            gathered_counts=gathered_counts,
            histogram_handle=histogram_handle,
            scratch=torch.zeros(size, dtype=torch.int32, device=device),
        )


_planner_workspaces: dict = {}


def get_planner_workspace(*, num_experts: int, device: torch.device, group: dist.ProcessGroup):
    """Return the process-wide planner scratch for one expert layout and process group;
    planning is stream-ordered, so every layer of a device shares it."""
    key = (num_experts, device.index, getattr(group, "group_name", id(group)))
    workspace = _planner_workspaces.get(key)
    if workspace is None:
        workspace = _planner_workspaces[key] = ReplicaPlannerWorkspace.allocate(
            num_experts=num_experts, device=device, group=group
        )
    return workspace


def finalize_moonep_planner_workspaces() -> None:
    """Release planner symmetric windows before EP process-group teardown."""
    for workspace in _planner_workspaces.values():
        workspace.destroy()
    _planner_workspaces.clear()


class _RerouteRoutes(torch.autograd.Function):
    """One reroute launch. The dense runtime probabilities it writes carry the gradient back to
    the router's ``[num_tokens, topk]`` probabilities through a gather at the runtime ids."""

    @staticmethod
    def forward(ctx, probs, top_indices, workspace):
        """Materialize physical routes and save their IDs for probability backward."""
        virtual_experts, runtime_probs = launch_virtual_expert_reroute(
            top_indices, probs, workspace
        )
        ctx.save_for_backward(virtual_experts)
        ctx.probs_dtype = probs.dtype
        ctx.mark_non_differentiable(virtual_experts)
        # Autograd would otherwise zero-fill gradients for the non-differentiable output.
        ctx.set_materialize_grads(False)
        return virtual_experts, runtime_probs

    @staticmethod
    def backward(ctx, grad_virtual_experts, grad_runtime_probs):
        """Gather physical probability gradients back into logical top-k order."""
        if grad_runtime_probs is None:
            return None, None, None
        (virtual_experts,) = ctx.saved_tensors
        grad_probs = grad_runtime_probs.gather(1, virtual_experts.long()).to(ctx.probs_dtype)
        return grad_probs, None, None


def update_replica_placement(
    top_indices: torch.Tensor,
    workspace: ReplicaPlannerWorkspace,
    *,
    exchange: bool = True,
    alignment: int = 1,
) -> torch.Tensor:
    """Compute replica placement without mapping individual token routes."""
    if top_indices.dim() != 2 or top_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("MoonEP replica placement requires 2D integer top_indices.")
    return launch_virtual_expert_placement(
        top_indices.contiguous(), workspace, exchange=exchange, alignment=alignment
    )


def reroute_replica_routes(
    top_indices: torch.Tensor, probs: torch.Tensor, workspace: ReplicaPlannerWorkspace
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map routes to a placement while preserving gradients to compact probabilities."""
    if top_indices.shape != probs.shape or top_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("MoonEP replica reroute takes matching [num_tokens, topk] ids and probs.")
    top_indices, probs = top_indices.contiguous(), probs.contiguous()
    return _RerouteRoutes.apply(probs, top_indices, workspace)


def plan_replica_routes(
    top_indices: torch.Tensor,
    probs: torch.Tensor,
    workspace: ReplicaPlannerWorkspace,
    *,
    exchange: bool = True,
    alignment: int = 1,
) -> tuple[ReplicaPlan, torch.Tensor]:
    """Plan deterministic replica placement for one EP group and map this rank's routes.

    ``top_indices`` / ``probs`` are the router's ``[num_tokens, topk]`` expert ids and
    probabilities; every rank must route the same number of tokens. The histograms are the only
    cross-rank input and the planner kernel exchanges them itself, so every rank computes the
    same placement. ``exchange=False`` (process-local tests) plans from a pre-filled window.
    Returns the plan and the dense float32 ``[num_tokens, 2 * num_experts]`` runtime
    probabilities HybridEP consumes, which carry the gradient back to ``probs``.
    """
    if top_indices.shape != probs.shape or top_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("MoonEP replica planner takes matching [num_tokens, topk] ids and probs.")
    top_indices, probs = top_indices.contiguous(), probs.contiguous()
    experts_to_copy = update_replica_placement(
        top_indices, workspace, exchange=exchange, alignment=alignment
    )
    virtual_experts, runtime_probs = reroute_replica_routes(top_indices, probs, workspace)
    return ReplicaPlan(virtual_experts, experts_to_copy), runtime_probs


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
    workspace: Optional[ReplicaPlannerWorkspace] = None
    max_dispatch_routes_per_rank: Optional[int] = None
    max_dispatch_padded_tokens_per_rank: Optional[int] = None


def padded_rank_capacity(num_routes: int, num_experts: int, ep_size: int, alignment: int) -> int:
    """PR #7450 bound on balanced, globally rounded expert blocks (including padding)."""
    if num_routes < 0 or min(num_experts, ep_size, alignment) < 1:
        raise ValueError("MoonEP capacity requires nonnegative routes and positive dimensions.")
    total_routes = ep_size * num_routes
    nonempty = min(num_experts, total_routes)
    max_blocks = (total_routes + nonempty * (alignment - 1)) // alignment
    return alignment * ((max_blocks + ep_size - 1) // ep_size)


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
        _, topk_indices = extract_semantic_routes(routing_map, probs, context.router_topk)
        workspace = get_planner_workspace(
            device=routing_map.device, num_experts=context.num_logical_experts, group=ep_group
        )
        experts_to_copy = update_replica_placement(
            topk_indices, workspace, alignment=self.token_padding
        )
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
                workspace=workspace,
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
        if placement_result.topk_indices is None or placement_result.workspace is None:
            raise ValueError("MoonEP distributed reroute requires placement workspace state.")

        topk_probs = torch.gather(probs, 1, placement_result.topk_indices)
        virtual_experts, physical_probs = reroute_replica_routes(
            placement_result.topk_indices, topk_probs, placement_result.workspace
        )
        physical_routing_map = torch.zeros_like(physical_probs, dtype=torch.bool)
        physical_routing_map.scatter_(1, virtual_experts.long(), True)
        return physical_routing_map, physical_probs

    def plan_with_count_matrix(self, *args, **kwargs) -> None:
        """Reject the removed Python count-matrix adapter."""
        del args, kwargs
        raise NotImplementedError(
            "PR #7450 MoonEPLoadPlanner plans from routing_map/probs and the EP process group; "
            "plan_with_count_matrix is not supported."
        )
