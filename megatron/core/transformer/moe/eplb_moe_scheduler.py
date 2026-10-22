# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""EPLB-style planner for the backend-neutral MoEScheduler contract.

The placement policy follows the two central ideas from DeepSeek's EPLB:

* greedily replicate the expert with the largest per-instance load; and
* place replica instances with longest-processing-time-first (LPT) packing.

By default homes remain fixed. With a home update interval, capacity-constrained
LPT periodically proposes a home permutation. Active logical ownership changes
only after state migration. Every forward distributes logical routes across the
committed homes and transient replicas using a global EP token ordinal.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

import torch

from megatron.core import tensor_parallel
from megatron.core.transformer.moe.moe_scheduler import (
    MoELoadPlanner,
    MoEPlacementResult,
    SchedulerContext,
)
from megatron.core.transformer.moe.replica_weight_transport import HomeExpertPlacement


@dataclass(frozen=True, slots=True)
class EPLBPlacementResult(MoEPlacementResult):
    """Explicit EPLB state consumed by the token-reroute phase."""

    logical_to_physical_map: torch.Tensor
    replica_counts: torch.Tensor
    local_expert_offsets: torch.Tensor
    num_physical_experts: int
    ep_size: int
    ep_rank: int


@dataclass(frozen=True, slots=True)
class EPLBHomeOnlyPlacementResult(MoEPlacementResult):
    """An immutable physical-to-logical column permutation, with no token ordinals."""

    physical_to_logical_map: torch.Tensor
    ep_size: int
    ep_rank: int


def _replicate_experts(
    expert_loads: torch.Tensor, num_redundant_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose logical experts to replicate with EPLB's greedy policy.

    At each step, the next replica belongs to the expert with the largest
    ``load / current_replica_count``. PyTorch's first-index tie breaking makes
    the result deterministic on every EP rank.
    """
    if expert_loads.dim() != 1:
        raise ValueError(f"expert_loads must be 1D, got shape {tuple(expert_loads.shape)}.")
    if expert_loads.numel() == 0:
        raise ValueError("expert_loads must contain at least one logical expert.")
    if num_redundant_experts < 0:
        raise ValueError("num_redundant_experts must be non-negative.")

    num_logical_experts = expert_loads.numel()
    replica_counts = torch.ones(num_logical_experts, dtype=torch.int64, device=expert_loads.device)
    replica_experts = torch.empty(
        num_redundant_experts, dtype=torch.int64, device=expert_loads.device
    )
    one = torch.ones(1, dtype=torch.int64, device=expert_loads.device)
    loads = expert_loads.to(dtype=torch.float32)
    for replica_index in range(num_redundant_experts):
        logical_expert = torch.argmax(loads / replica_counts)
        replica_experts[replica_index : replica_index + 1].copy_(logical_expert.reshape(1))
        replica_counts.scatter_add_(0, logical_expert.reshape(1), one)
    return replica_experts, replica_counts


def _pack_replicas_with_fixed_homes(
    expert_loads: torch.Tensor,
    replica_experts: torch.Tensor,
    replica_counts: torch.Tensor,
    ep_size: int,
    home_layout: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """LPT-pack replicas while retaining each logical expert's home rank."""
    num_logical_experts = expert_loads.numel()
    num_redundant_experts = replica_experts.numel()
    if ep_size <= 0:
        raise ValueError("EPLB ep_size must be positive.")
    if num_logical_experts % ep_size != 0:
        raise ValueError("EPLB requires logical experts to be divisible by ep_size.")
    if num_redundant_experts % ep_size != 0:
        raise ValueError("EPLB requires redundant experts to be divisible by ep_size.")

    num_local_home_experts = num_logical_experts // ep_size
    num_local_replica_slots = num_redundant_experts // ep_size
    num_local_physical_experts = num_local_home_experts + num_local_replica_slots
    num_physical_experts = num_logical_experts + num_redundant_experts
    device = expert_loads.device

    physical_to_logical_map = torch.full(
        (num_physical_experts,), -1, dtype=torch.int64, device=device
    )
    rank_layout = physical_to_logical_map.view(ep_size, num_local_physical_experts)
    rank_layout[:, :num_local_home_experts] = torch.arange(
        num_logical_experts, dtype=torch.int64, device=device
    ).view(ep_size, num_local_home_experts)
    if home_layout is not None:
        rank_layout[:, :num_local_home_experts].copy_(home_layout)
    if num_redundant_experts == 0:
        return physical_to_logical_map

    per_instance_load = expert_loads.to(torch.float32) / replica_counts
    rank_loads = per_instance_load[rank_layout[:, :num_local_home_experts]].sum(dim=1)
    replica_loads = per_instance_load.gather(0, replica_experts)
    placement_order = torch.argsort(replica_loads, descending=True, stable=True)
    slots_used = torch.zeros(ep_size, dtype=torch.int64, device=device)
    one = torch.ones(1, dtype=torch.int64, device=device)

    for placement_index in range(num_redundant_experts):
        replica_index = placement_order[placement_index]
        logical_expert = replica_experts.gather(0, replica_index.reshape(1))
        load = replica_loads.gather(0, replica_index.reshape(1))
        has_capacity = slots_used < num_local_replica_slots
        destination_rank = torch.argmin(
            torch.where(has_capacity, rank_loads, torch.full_like(rank_loads, torch.inf))
        )
        destination_slot = slots_used.gather(0, destination_rank.reshape(1))
        physical_expert = (
            destination_rank * num_local_physical_experts
            + num_local_home_experts
            + destination_slot
        )
        physical_to_logical_map.scatter_(0, physical_expert, logical_expert)
        slots_used.scatter_add_(0, destination_rank.reshape(1), one)
        rank_loads.scatter_add_(0, destination_rank.reshape(1), load)

    return physical_to_logical_map


def _logical_to_physical_map(
    physical_to_logical_map: torch.Tensor, num_logical_experts: int, num_redundant_experts: int
) -> torch.Tensor:
    """Invert a physical map and retain physical-instance order per expert."""
    num_physical_experts = physical_to_logical_map.numel()
    max_replica_count = num_redundant_experts + 1
    same_expert = physical_to_logical_map[:, None] == physical_to_logical_map[None, :]
    replica_rank = torch.tril(same_expert).sum(dim=1, dtype=torch.int64) - 1
    flat_indices = physical_to_logical_map * max_replica_count + replica_rank
    physical_experts = torch.arange(
        num_physical_experts, dtype=torch.int64, device=physical_to_logical_map.device
    )
    logical_to_physical_map = torch.full(
        (num_logical_experts * max_replica_count,),
        -1,
        dtype=torch.int64,
        device=physical_to_logical_map.device,
    )
    logical_to_physical_map.scatter_(0, flat_indices, physical_experts)
    return logical_to_physical_map.view(num_logical_experts, max_replica_count)


def build_eplb_placement(
    expert_loads: torch.Tensor, num_redundant_experts: int, ep_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build fixed-home EPLB physical/logical maps from global expert loads."""
    replica_experts, replica_counts = _replicate_experts(expert_loads, num_redundant_experts)
    physical_to_logical_map = _pack_replicas_with_fixed_homes(
        expert_loads, replica_experts, replica_counts, ep_size
    )
    logical_to_physical_map = _logical_to_physical_map(
        physical_to_logical_map, expert_loads.numel(), num_redundant_experts
    )
    return physical_to_logical_map, logical_to_physical_map, replica_counts


def build_eplb_home_layout(expert_loads, num_redundant_experts, ep_size):
    """Capacity-constrained LPT: exactly H unique homes on each rank.

    Use per-instance costs after greedy replication. Unlike unconstrained EPLB,
    this always supplies one canonical training owner for every logical expert.
    Host planning occurs only at a periodic migration boundary.
    """
    _, counts = _replicate_experts(expert_loads, num_redundant_experts)
    costs = (expert_loads.float() / counts).cpu().tolist()
    h = len(costs) // ep_size
    if h == 0 or len(costs) % ep_size:
        raise ValueError("Home layout requires a positive uniform home capacity.")
    ranks = [[] for _ in range(ep_size)]
    loads = [0.0] * ep_size
    for expert in sorted(range(len(costs)), key=lambda i: (-costs[i], i)):
        rank = min((r for r in range(ep_size) if len(ranks[r]) < h), key=lambda r: (loads[r], r))
        ranks[rank].append(expert)
        loads[rank] += costs[expert]
    return torch.tensor(ranks, dtype=torch.int64, device=expert_loads.device)


class EPLBLoadPlanner(MoELoadPlanner):
    """EPLB planner with transient replicas and optional deferred home permutations."""

    planner_name = "eplb"

    def __init__(
        self,
        num_redundant_experts: int,
        home_update_interval: int = 0,
        *,
        num_logical_experts: Optional[int] = None,
        pg_collection=None,
    ) -> None:
        super().__init__()
        if num_redundant_experts < 0:
            raise ValueError("num_redundant_experts must be non-negative.")
        self.num_redundant_experts = int(num_redundant_experts)
        if home_update_interval < 0:
            raise ValueError("home_update_interval must be non-negative.")
        self.home_update_interval = home_update_interval
        self.num_logical_experts = num_logical_experts
        self.pg_collection = pg_collection
        self._restore_pending = False
        self.register_buffer(
            "active_home_layout", torch.empty(0, dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "active_replica_layout", torch.empty(0, dtype=torch.int64), persistent=False
        )
        self.register_buffer("_load_history", torch.empty(0), persistent=False)
        self._pending_replica_layout = None
        self._pending_home_layout = None
        self._pending_version = None
        self._version = 0
        self._steps = 0
        self._last_proposal_step = -1
        self._loading_logical_checkpoint = False
        self._home_route_cache = None

    def begin_logical_checkpoint_load(self) -> None:
        """Load ordinary logical weights into a fresh identity-owned training run.

        Only the checkpoint loader may select this mode, after inspecting source
        metadata. Missing planner state in an ordinary resume remains an error.
        The first update_placement initializes homes; no migration runs on load.
        """
        if self._steps or self.active_home_layout.numel() or self._pending_version is not None:
            raise RuntimeError("Logical checkpoint initialization requires a fresh EPLB planner.")
        self._loading_logical_checkpoint = True

    def finish_logical_checkpoint_load(self) -> None:
        """Restore required planner serialization after all model-load retries."""
        self._loading_logical_checkpoint = False

    def logical_export_placement(self) -> HomeExpertPlacement | None:
        """Invert only committed ownership; pending plans do not own any weights."""
        if not self.active_home_layout.numel():
            return None
        sources = torch.argsort(self.active_home_layout.flatten()).reshape_as(
            self.active_home_layout
        )
        return HomeExpertPlacement(sources, 1)

    def step(self, completed_version: int | None = None) -> None:
        """Publish new logical ownership only after the dispatcher completed exchange."""
        if completed_version is not None:
            if completed_version != self._pending_version:
                raise RuntimeError("EPLB received a stale or unknown home completion.")
            self.active_home_layout.copy_(
                self._pending_home_layout.to(self.active_home_layout.device)
            )
            self._home_route_cache = None
            self._pending_home_layout = None
            self._pending_version = None
        elif self._pending_version is not None:
            raise RuntimeError("EPLB cannot commit a pending layout without home completion.")
        if self._pending_replica_layout is not None:
            self.active_replica_layout.copy_(
                self._pending_replica_layout.to(self.active_replica_layout.device)
            )
            self._pending_replica_layout = None
        self._steps += 1

    def _checkpoint_topology(self):
        groups = self.pg_collection
        sizes = {
            name: getattr(groups, name).size()
            for name in ("ep", "tp", "dp_cp")
            if getattr(groups, name, None) is not None
        }
        return dict(
            sizes,
            num_logical_experts=self.num_logical_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

    def _checkpoint_rank(self):
        # History is rank-local for zero replicas, EP-reduced otherwise.
        # dp_cp distinguishes EP owners as well as their data replicas.
        return (
            "_".join(
                f"{name}{getattr(self.pg_collection, name).rank()}"
                for name in ("tp", "dp_cp")
                if getattr(self.pg_collection, name, None) is not None
            )
            or "local"
        )

    def get_extra_state(self):
        """Snapshot logical ownership and deferred plans without aliasing live buffers."""

        def snapshot(value):
            return None if value is None else value.detach().cpu().clone()

        return {
            "schema_version": 1,
            "topology": self._checkpoint_topology(),
            "home_update_interval": self.home_update_interval,
            "home_layout": snapshot(self.active_home_layout),
            "replica_layout": snapshot(self.active_replica_layout),
            "steps": self._steps,
            "version": self._version,
            "last_proposal_step": self._last_proposal_step,
            "load_history": snapshot(self._load_history),
            "load_history_scope": "local" if self.num_redundant_experts == 0 else "ep",
            "pending_home_layout": snapshot(self._pending_home_layout),
            "pending_replica_layout": snapshot(self._pending_replica_layout),
            "pending_version": self._pending_version,
        }

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination.pop(prefix + "_extra_state", None)
        if self.home_update_interval and not self._loading_logical_checkpoint:
            # FSDP deliberately discards TE _extra_state. Planner state is a
            # separate field, and opaque bytes allow DCP to load variable-sized
            # layouts into a freshly constructed (empty) planner.
            stream = io.BytesIO()
            torch.save(self.get_extra_state(), stream)
            destination[prefix + "_planner_state"] = {self._checkpoint_rank(): stream.getvalue()}

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Keep rank-local planning history in both MCore and native DCP formats."""
        from megatron.core.dist_checkpointing.mapping import ShardedObject

        state = self.state_dict()
        if not state:
            return {}
        offsets = list(sorted(sharded_offsets))
        if [axis for axis, _, _ in offsets] != list(range(len(offsets))):
            raise ValueError("Planner checkpoint offsets must be contiguous.")
        shape = tuple(size for _, _, size in offsets)
        offset = tuple(rank for _, rank, _ in offsets)
        for name in ("tp", "dp_cp"):
            group = getattr(self.pg_collection, name, None)
            shape += (1 if group is None else group.size(),)
            offset += (0 if group is None else group.rank(),)
        key = prefix + "_planner_state"
        return {key: ShardedObject(key, state["_planner_state"], shape, offset, replica_id=0)}

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        key = prefix + "_planner_state"
        extra_key = prefix + "_extra_state"
        if self._loading_logical_checkpoint:
            if key in state_dict or extra_key in state_dict:
                raise ValueError(
                    "Logical checkpoint initialization cannot discard saved planner state."
                )
            return
        if self.home_update_interval:
            if key not in state_dict:
                # Generic training loaders sometimes retry with strict=False.
                # Missing ownership can silently train the wrong logical model.
                raise RuntimeError(f"Missing required MoEScheduler checkpoint state: {key}")
            packed = state_dict.pop(key)
            rank = self._checkpoint_rank()
            if rank not in packed:
                raise ValueError("MoEScheduler checkpoint has a different rank topology.")
            state_dict[extra_key] = torch.load(
                io.BytesIO(packed[rank]), map_location="cpu", weights_only=True
            )
        elif key in state_dict:
            raise ValueError("A home-placement checkpoint requires periodic EPLB to be enabled.")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        if not self.home_update_interval and extra_key in missing_keys:
            missing_keys.remove(extra_key)

    def set_extra_state(self, state):
        """Restore before forward; replay a deferred physical plan on its first use."""
        if state.get("schema_version") != 1:
            raise ValueError("Unsupported EPLB checkpoint schema version.")
        if state["topology"] != self._checkpoint_topology():
            raise ValueError(
                "EPLB checkpoint requires the same EP/home/replica and TP/DP topology."
            )
        if state["home_update_interval"] != self.home_update_interval:
            raise ValueError("EPLB checkpoint requires the same home update interval.")
        home = state["home_layout"]
        replica = state["replica_layout"]
        if home.numel():
            n = home.numel()
            ep = state["topology"].get("ep", home.shape[0])
            if self.num_logical_experts is not None and n != self.num_logical_experts:
                raise ValueError("Invalid EPLB checkpoint expert count.")
            if (
                home.dtype != torch.int64
                or home.ndim != 2
                or home.shape != (ep, n // ep)
                or not torch.equal(home.flatten().sort().values, torch.arange(n))
            ):
                raise ValueError("Invalid EPLB checkpoint home permutation.")
            if (
                replica.ndim != 2
                or replica.shape[0] != home.shape[0]
                or (
                    replica.numel() != self.num_redundant_experts
                    or torch.any(replica < 0)
                    or torch.any(replica >= n)
                )
            ):
                raise ValueError("Invalid EPLB checkpoint replica layout.")
            if state["load_history"].shape != (n,):
                raise ValueError("Invalid EPLB checkpoint load history.")
            pending_home = state["pending_home_layout"]
            pending_replica = state["pending_replica_layout"]
            if pending_home is not None and (
                pending_home.shape != home.shape
                or pending_home.dtype != home.dtype
                or not torch.equal(pending_home.flatten().sort().values, torch.arange(n))
            ):
                raise ValueError("Invalid EPLB checkpoint deferred home permutation.")
            if pending_replica is not None and (
                pending_replica.shape != replica.shape
                or pending_replica.dtype != replica.dtype
                or torch.any(pending_replica < 0)
                or torch.any(pending_replica >= n)
            ):
                raise ValueError("Invalid EPLB checkpoint deferred replica layout.")
        if (state["pending_version"] is None) != (state["pending_home_layout"] is None):
            raise ValueError("Invalid EPLB checkpoint deferred home version.")
        if state["pending_version"] is not None and (
            state["pending_version"] != state["version"] or state["pending_replica_layout"] is None
        ):
            raise ValueError("Invalid EPLB checkpoint deferred home plan.")
        device = self.active_home_layout.device
        self.active_home_layout = home.to(device).clone()
        self.active_replica_layout = replica.to(device).clone()
        self._load_history = state["load_history"].to(device).clone()
        scope = state.get("load_history_scope", "ep")
        expected_scope = "local" if self.num_redundant_experts == 0 else "ep"
        if scope == "ep" and expected_scope == "local":
            # Legacy zero-replica checkpoints stored EP-summed histories on every
            # rank. Distribute that history once; future observations stay local.
            self._load_history.div_(
                state["topology"].get("ep", home.shape[0] if home.numel() else 1)
            )
        elif scope != expected_scope:
            raise ValueError("Unsupported EPLB load history scope.")
        self._home_route_cache = None
        self._steps = state["steps"]
        self._version = state["version"]
        self._last_proposal_step = state["last_proposal_step"]
        self._pending_version = state["pending_version"]
        self._pending_home_layout = state["pending_home_layout"]
        self._pending_replica_layout = state["pending_replica_layout"]
        self._restore_pending = self._pending_version is not None

    def _validate_inputs(
        self, probs: torch.Tensor, routing_map: torch.Tensor, context: SchedulerContext
    ) -> None:
        if routing_map.dim() != 2:
            raise ValueError(f"routing_map must be 2D, got shape {tuple(routing_map.shape)}.")
        if probs.shape != routing_map.shape:
            raise ValueError(
                "probs and routing_map must have the same shape, got "
                f"{tuple(probs.shape)} and {tuple(routing_map.shape)}."
            )
        if routing_map.size(1) != context.num_logical_experts:
            raise ValueError(
                "routing_map logical expert dimension does not match SchedulerContext, "
                f"got {routing_map.size(1)} and {context.num_logical_experts}."
            )
        if routing_map.dtype != torch.bool:
            raise ValueError(f"routing_map must be bool, got {routing_map.dtype}.")
        if probs.device != routing_map.device:
            raise ValueError("probs and routing_map must be on the same device.")

    def _validate_context(self, context: SchedulerContext) -> None:
        if context.num_logical_experts % context.ep_size != 0:
            raise ValueError("EPLB requires num_logical_experts divisible by ep_size.")
        if self.num_redundant_experts % context.ep_size != 0:
            raise ValueError("EPLB requires num_redundant_experts divisible by ep_size.")
        num_local_home_experts = context.num_logical_experts // context.ep_size
        if context.num_local_experts != num_local_home_experts:
            raise ValueError(
                "EPLB SchedulerContext.num_local_experts does not match the even EP layout."
            )
        local_start = context.ep_rank * num_local_home_experts
        expected_local_experts = tuple(range(local_start, local_start + num_local_home_experts))
        if context.local_expert_indices != expected_local_experts:
            raise ValueError("EPLB requires contiguous rank-major logical home experts.")

    def _get_count_matrix(
        self,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        *,
        tokens_per_expert: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        local_counts = (
            tokens_per_expert if tokens_per_expert is not None else routing_map.sum(dim=0)
        )
        if local_counts.dim() != 1 or local_counts.numel() != context.num_logical_experts:
            raise ValueError(
                "Expected local token counts to be a 1D logical-expert vector, "
                f"got shape {tuple(local_counts.shape)}."
            )
        local_counts = local_counts.to(device=routing_map.device, dtype=torch.int64)
        if context.ep_size == 1:
            return local_counts.unsqueeze(0)

        ep_group = getattr(context.pg_collection, "ep", None)
        if ep_group is None:
            raise ValueError("EPLBLoadPlanner requires SchedulerContext.pg_collection.ep.")
        return tensor_parallel.gather_from_sequence_parallel_region(
            local_counts, group=ep_group
        ).reshape(context.ep_size, context.num_logical_experts)

    @torch.no_grad()
    def update_placement(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        *,
        tokens_per_expert: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[HomeExpertPlacement], Optional[torch.Tensor], MoEPlacementResult]:
        """Compute a common placement from current global EP token counts."""
        self._validate_inputs(probs, routing_map, context)
        self._validate_context(context)
        home_only = self.num_redundant_experts == 0
        if home_only:
            loads = routing_map.sum(dim=0) if tokens_per_expert is None else tokens_per_expert
            if loads.shape != (context.num_logical_experts,):
                raise ValueError("Expected a 1D local token-count vector for every logical expert.")
            loads = loads.to(device=routing_map.device, dtype=torch.int64)
        else:
            counts_from_ep_rank = self._get_count_matrix(
                routing_map, context, tokens_per_expert=tokens_per_expert
            )
            loads = counts_from_ep_rank.sum(dim=0)
        h = context.num_local_experts
        if not self.active_home_layout.numel():
            self.active_home_layout = torch.arange(
                context.num_logical_experts, device=loads.device
            ).reshape(context.ep_size, h)
            self._load_history = torch.zeros_like(loads, dtype=torch.float64)
        if tuple(self.active_home_layout.shape) != (context.ep_size, h):
            raise ValueError("Restored EPLB physical checkpoint has a different EP/home topology.")
        self.active_home_layout = self.active_home_layout.to(loads.device)
        self._load_history = self._load_history.to(loads.device)
        if self.home_update_interval:
            if self.active_replica_layout.ndim != 2:
                initial, initial_counts = _replicate_experts(loads, self.num_redundant_experts)
                layout = _pack_replicas_with_fixed_homes(
                    loads, initial, initial_counts, context.ep_size, self.active_home_layout
                )
                self.active_replica_layout = layout.reshape(context.ep_size, -1)[:, h:].clone()
            self.active_replica_layout = self.active_replica_layout.to(loads.device)
            if tuple(self.active_replica_layout.shape) != (
                context.ep_size,
                self.num_redundant_experts // context.ep_size,
            ):
                raise ValueError("Restored EPLB checkpoint has a different replica slot topology.")
        home_placement = None
        if self._restore_pending:
            logical_to_home = torch.argsort(self.active_home_layout.flatten())
            candidate = self._pending_home_layout.to(loads.device)
            home_placement = HomeExpertPlacement(logical_to_home[candidate], self._pending_version)
            self._restore_pending = False
        if (
            context.training
            and self.home_update_interval
            and self._steps > 0
            and self._steps % self.home_update_interval == 0
            and self._last_proposal_step != self._steps
            and self._pending_version is None
        ):
            historical_loads = self._load_history.clone()
            if home_only and context.ep_size > 1:
                ep = getattr(context.pg_collection, "ep", None)
                if ep is None:
                    raise ValueError("EPLBLoadPlanner requires SchedulerContext.pg_collection.ep.")
                torch.distributed.all_reduce(historical_loads, group=ep)
            # Data replicas must choose identical logical ownership before their
            # sharded optimizer states can move between corresponding EP ranks.
            edp = getattr(context.pg_collection, "expt_dp", None)
            if edp is not None:
                torch.distributed.all_reduce(historical_loads, group=edp)
            candidate = build_eplb_home_layout(
                historical_loads, self.num_redundant_experts, context.ep_size
            )
            candidate_replicas, candidate_counts = _replicate_experts(
                historical_loads, self.num_redundant_experts
            )
            candidate_layout = _pack_replicas_with_fixed_homes(
                historical_loads, candidate_replicas, candidate_counts, context.ep_size, candidate
            )
            self._pending_replica_layout = candidate_layout.reshape(context.ep_size, -1)[
                :, h:
            ].clone()
            self._last_proposal_step = self._steps
            self._load_history.zero_()
            if not torch.equal(candidate, self.active_home_layout):
                self._version += 1
                self._pending_version = self._version
                self._pending_home_layout = candidate
                logical_to_home = torch.argsort(self.active_home_layout.flatten())
                home_placement = HomeExpertPlacement(logical_to_home[candidate], self._version)
        if context.training:
            self._load_history.add_(loads)
        if home_only:
            # Keep the result alive across outstanding forwards. A committed
            # exchange invalidates this snapshot; a pending proposal never does.
            cache = self._home_route_cache
            if cache is None or cache.device != loads.device:
                cache = self.active_home_layout.flatten().clone()
                self._home_route_cache = cache
            return (
                home_placement,
                None,
                EPLBHomeOnlyPlacementResult(cache, context.ep_size, context.ep_rank),
            )
        # Every microbatch copies CURRENT home sources, even while an exchange is pending.
        if self.home_update_interval:
            physical_to_logical_map = torch.cat(
                (self.active_home_layout, self.active_replica_layout), dim=1
            ).flatten()
            replica_counts = torch.bincount(
                physical_to_logical_map, minlength=context.num_logical_experts
            )
        else:
            replicas, replica_counts = _replicate_experts(loads, self.num_redundant_experts)
            physical_to_logical_map = _pack_replicas_with_fixed_homes(
                loads, replicas, replica_counts, context.ep_size, self.active_home_layout
            )
        logical_to_physical_map = _logical_to_physical_map(
            physical_to_logical_map, context.num_logical_experts, self.num_redundant_experts
        )
        logical_to_home = torch.argsort(self.active_home_layout.flatten())
        replica_layout = physical_to_logical_map.reshape(context.ep_size, -1)[:, h:]
        # No replica dispatch or runtime is needed for a pure home permutation.
        replica_placement = (
            logical_to_home[replica_layout].to(torch.int32).contiguous()
            if self.num_redundant_experts
            else None
        )
        local_expert_offsets = counts_from_ep_rank[: context.ep_rank].sum(dim=0)
        return (
            home_placement,
            replica_placement,
            EPLBPlacementResult(
                logical_to_physical_map=logical_to_physical_map,
                replica_counts=replica_counts,
                local_expert_offsets=local_expert_offsets,
                num_physical_experts=physical_to_logical_map.numel(),
                ep_size=context.ep_size,
                ep_rank=context.ep_rank,
            ),
        )

    def reroute(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        placement_result: MoEPlacementResult,
        context: SchedulerContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Round-robin logical routes across their EPLB physical instances."""
        if isinstance(placement_result, EPLBHomeOnlyPlacementResult):
            self._validate_inputs(probs, routing_map, context)
            self._validate_context(context)
            if self.num_redundant_experts or (
                placement_result.ep_size,
                placement_result.ep_rank,
            ) != (context.ep_size, context.ep_rank):
                raise ValueError("Home-only placement belongs to a different EPLB context.")
            columns = placement_result.physical_to_logical_map
            selected_probs = probs.masked_fill(~routing_map, 0)
            return routing_map.index_select(1, columns), selected_probs.index_select(1, columns)
        if not isinstance(placement_result, EPLBPlacementResult):
            raise TypeError(
                "EPLBLoadPlanner.reroute requires the EPLBPlacementResult returned by "
                "EPLBLoadPlanner.update_placement."
            )
        self._validate_inputs(probs, routing_map, context)
        self._validate_context(context)
        if (placement_result.ep_size, placement_result.ep_rank) != (
            context.ep_size,
            context.ep_rank,
        ):
            raise ValueError("EPLB placement result belongs to a different EP context.")

        num_tokens, num_logical_experts = routing_map.shape
        expected_shape = (num_logical_experts, self.num_redundant_experts + 1)
        if tuple(placement_result.logical_to_physical_map.shape) != expected_shape:
            raise ValueError(
                "EPLB logical_to_physical_map has the wrong shape, expected " f"{expected_shape}."
            )
        if placement_result.replica_counts.shape != (num_logical_experts,):
            raise ValueError("EPLB replica_counts has the wrong shape.")
        if placement_result.local_expert_offsets.shape != (num_logical_experts,):
            raise ValueError("EPLB local_expert_offsets has the wrong shape.")

        local_ordinals = routing_map.to(torch.int64).cumsum(dim=0) - 1
        global_ordinals = local_ordinals + placement_result.local_expert_offsets.unsqueeze(0)
        replica_ranks = torch.remainder(
            global_ordinals, placement_result.replica_counts.unsqueeze(0)
        )
        physical_experts = placement_result.logical_to_physical_map.gather(
            1, replica_ranks.transpose(0, 1)
        ).transpose(0, 1)

        output_shape = (num_tokens, placement_result.num_physical_experts)
        physical_routing_map = torch.zeros(
            output_shape, dtype=torch.bool, device=routing_map.device
        ).scatter(1, physical_experts, routing_map)
        selected_probs = torch.where(routing_map, probs, torch.zeros_like(probs))
        physical_probs = probs.new_zeros(output_shape).scatter(1, physical_experts, selected_probs)
        return physical_routing_map, physical_probs


__all__ = ["EPLBLoadPlanner", "EPLBPlacementResult", "build_eplb_placement"]
