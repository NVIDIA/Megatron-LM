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
from megatron.core.utils import nvtx_decorator
from tests.unit_tests.transformer.moe.moonep_reference_triton import (
    MAX_REPLICA_EP_RANKS,
    PLANNER_PROGRAMS,
    launch_virtual_expert_placement,
    launch_virtual_expert_reroute,
)


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
