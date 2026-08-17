# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Owner-compute orthogonalized optimizer for Megatron-FSDP v2.

This module implements the Muon-style orthogonalizing optimizer step on top of
M-FSDPv2's all-`Flat` placements (`parameter=Flat, gradient=Flat,
optimizer=Flat`). It reuses the `emerging_optimizers` Newton-Schulz kernels
through `OrthogonalizedOptimizer`.

Algorithm (per optimizer step, for 2D "matrix" parameters that cross FSDP rank
boundaries), matching an owner-compute + scatter design:

    1. Compute local orthogonalization-input ("pre-NS") shards: weight decay,
       momentum update, and (optional) Nesterov combination, all on each rank's
       local gradient shard.
    2. P2P-send the pre-NS shards to their owner ranks. Owners are balanced
       across ranks by an orthogonalization compute-cost heuristic so no single
       rank serializes all Newton-Schulz work. The owner's own shard is kept
       locally (no self-send).
    3. On each owner, reconstruct the full pre-NS matrix and run Newton-Schulz
       orthogonalization to produce the full update.
    4. P2P-send the update shards from owners back to their destination ranks.
    5. Each rank applies its local update shard to its local weight shard.

Fully local parameters (owned by a single rank) skip the communication and run
Newton-Schulz locally; their compute overlaps the boundary P2P. Non-2D
parameters fall back to a plain momentum-SGD step (no orthogonalization).

Communication is asynchronous and issued on a dedicated owner-comm stream using
a dedicated (duplicate) owner-comm process group, so owner P2P ordering is
independent of FSDP's forward/backward collectives. The synchronous waiting
(`_wait_for_dist_buffer`) is deferred as late as possible so local Newton-Schulz
work overlaps owner gathers/scatters.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import logging
import warnings
from collections import defaultdict
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from typing import Any, cast, overload, override

import torch
import torch.distributed as dist

try:
    from emerging_optimizers import utils as eo_utils
    from emerging_optimizers.orthogonalized_optimizers import OrthogonalizedOptimizer
    from emerging_optimizers.orthogonalized_optimizers.muon import Muon

    HAVE_EMERGING_OPTIMIZERS = True
except (ModuleNotFoundError, ImportError):
    eo_utils = None
    OrthogonalizedOptimizer = object
    Muon = object
    HAVE_EMERGING_OPTIMIZERS = False
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import ParamsT

from .layout import non_leading_numel
from .parameter_group import FsdpParameterGroup, get_containing_parameter_group

logger = logging.getLogger(__name__)


def _require_emerging_optimizers() -> None:
    if not HAVE_EMERGING_OPTIMIZERS:
        raise ModuleNotFoundError(
            "Emerging-Optimizers is required for orthogonalized optimizer support. "
            "Please install the necessary dependencies with "
            "`pip install 'megatron_fsdp[emerging-optimizers]'`."
        )


@dataclasses.dataclass(frozen=True)
class ShardPlan:
    """How a single 2D parameter's full matrix is split across the DP group.

    M-FSDPv2's all-`Flat` layout shards dim-0 rows contiguously in rank order,
    so rank `r` owns the contiguous global row range
    `[start, start + count)` where `start`/`count` come from the flat
    DBuffer layout. A rank with `count == 0` holds no shard of this parameter.
    """

    full_shape: torch.Size
    rank_rows: tuple[tuple[int, int], ...]
    row_size: int

    def __post_init__(self) -> None:
        if len(self.full_shape) != 2:
            raise ValueError(f"ShardPlan requires a 2D full_shape, got {self.full_shape}.")
        if len(self.rank_rows) == 0:
            raise ValueError("ShardPlan requires at least one rank.")
        if self.row_size != non_leading_numel(self.full_shape):
            raise ValueError(
                f"ShardPlan row_size {self.row_size} != full_shape row size "
                f"{non_leading_numel(self.full_shape)}."
            )

    @property
    def world_size(self) -> int:
        """Number of ranks in the DP group for this parameter."""
        return len(self.rank_rows)

    def rank_row_count(self, rank: int) -> int:
        """Return the number of rows owned by `rank`."""
        return self.rank_rows[rank][1]

    def shard_numel(self, rank: int) -> int:
        """Return the number of elements in `rank`'s shard."""
        return self.rank_row_count(rank) * self.row_size

    def owner_candidates(self) -> tuple[int, ...]:
        """Return the ranks that hold a non-empty shard of this parameter."""
        return tuple(r for r, (_, count) in enumerate(self.rank_rows) if count > 0)

    def is_boundary(self) -> bool:
        """True if more than one rank owns a non-empty shard of this parameter."""
        return sum(1 for _, count in self.rank_rows if count > 0) > 1

    def full_numel(self) -> int:
        """Return the total number of elements in the full (unsharded) parameter."""
        return self.full_shape.numel()


def compute_shard_plan(
    full_shape: torch.Size, tensor_flat_offset: int, rank_flat_shard_size: int, world_size: int
) -> ShardPlan:
    """Compute the per-rank row ranges for one 2D parameter in a flat DBuffer.

    Args:
        full_shape: Global `(rows, cols)` shape of the parameter.
        tensor_flat_offset: Flat-element offset of this parameter inside the
            DBuffer's global layout.
        rank_flat_shard_size: Flat elements each DP rank owns (uniform for the
            even all-`Flat` layout: `layout.size // world_size`).
        world_size: DP group size.

    Returns:
        The `ShardPlan` describing which rows each rank owns.
    """
    if len(full_shape) != 2:
        raise ValueError(f"compute_shard_plan requires a 2D shape, got {full_shape}.")
    row_size = non_leading_numel(full_shape)
    if row_size <= 0:
        raise ValueError(f"compute_shard_plan requires non-empty rows, got shape {full_shape}.")
    tensor_end = tensor_flat_offset + full_shape.numel()

    rank_rows: list[tuple[int, int]] = []
    for rank in range(world_size):
        rank_start = rank * rank_flat_shard_size
        rank_end = rank_start + rank_flat_shard_size
        overlap_start = max(tensor_flat_offset, rank_start)
        overlap_end = min(tensor_end, rank_end)
        if overlap_start >= overlap_end:
            rank_rows.append((0, 0))
            continue
        if (overlap_start - tensor_flat_offset) % row_size != 0:
            raise RuntimeError(
                f"Flat shard boundary is not row-aligned for shape {full_shape}: "
                f"overlap_start={overlap_start}, tensor_flat_offset={tensor_flat_offset}."
            )
        overlap_numel = overlap_end - overlap_start
        if overlap_numel % row_size != 0:
            raise RuntimeError(
                f"Flat shard overlap is not row-aligned for shape {full_shape}: "
                f"overlap_numel={overlap_numel}, row_size={row_size}."
            )
        row_start = (overlap_start - tensor_flat_offset) // row_size
        row_count = overlap_numel // row_size
        rank_rows.append((row_start, row_count))
    return ShardPlan(
        full_shape=torch.Size(full_shape), rank_rows=tuple(rank_rows), row_size=row_size
    )


def assign_owner_work(plans: Sequence[ShardPlan], num_ns_steps: int) -> dict[int, int]:
    """Assign one owner rank to each boundary parameter, balanced by NS cost.

    Only ranks that own a non-empty shard of a parameter are eligible owners.
    Assignment greedily gives each parameter to its eligible rank with the
    smallest running cost total, where a parameter's cost is the Newton-Schulz
    compute estimate `numel * (min(rows, cols) * num_steps + 1)`.

    Args:
        plans: Shard plans indexed by their position in the input sequence.
        num_ns_steps: Newton-Schulz iteration count used in the cost estimate.

    Returns:
        Mapping from parameter index (in `plans`) to owner rank.
    """
    assignments: dict[int, int] = {}
    running: dict[int, float] = {r: 0.0 for r in range(plans[0].world_size)} if plans else {}
    for param_index, plan in enumerate(plans):
        candidates = plan.owner_candidates()
        if not candidates:
            raise RuntimeError(
                f"No eligible owner for parameter {param_index} with shape {plan.full_shape}; "
                "no rank owns a shard."
            )
        if not plan.is_boundary():
            # Fully local: the single owning rank is the owner with no communication.
            assignments[param_index] = candidates[0]
            continue
        rows, cols = plan.full_shape
        short_dim = min(rows, cols)
        cost = float(plan.full_numel() * (short_dim * num_ns_steps + 1))
        owner = min(candidates, key=lambda r: (running[r], r))
        assignments[param_index] = owner
        running[owner] += cost
    return assignments


@dataclasses.dataclass
class OwnerGatherPlan:
    """Metadata and send buffers for the owner-gather P2P step of one chunk.

    The owner keeps its own shard locally (no self-send), so it only receives
    from the other shard-holding ranks and reconstructs each owned matrix by
    concatenating shards in rank order (own shard at the owner's rank rows).

    Attributes:
        send_buffers: Per-destination-owner flat send buffer (this rank's
            pre-NS shards for that owner's params, in param order). Only owners
            other than this rank appear.
        recv_sizes: Per-source-rank element count this rank (as an owner)
            receives. Only sources other than this rank appear.
        own_shards: This rank's local shard per owned parameter (used directly
            in reconstruction, not communicated).
        recv_offsets: Per `(param_index, src_rank)` of `(offset, numel,
            row_count)` describing where this param's shard lands inside the
            recv buffer received from `src_rank`.
    """

    send_buffers: dict[int, torch.Tensor]
    recv_sizes: dict[int, int]
    own_shards: dict[int, torch.Tensor]
    recv_offsets: dict[tuple[int, int], tuple[int, int, int]]


def pack_owner_work(
    plans: Sequence[ShardPlan],
    owners: dict[int, int],
    local_shards: Sequence[torch.Tensor],
    world_size: int,
    this_rank: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> OwnerGatherPlan:
    """Pack this rank's pre-NS shards into per-owner P2P send buffers.

    Args:
        plans: Shard plans in parameter order.
        owners: Mapping from parameter index to owner rank.
        local_shards: This rank's local pre-NS shard per parameter.
        world_size: DP group size.
        this_rank: This rank's DP index.
        device: Device for the send buffers (the pre-NS device).
        dtype: Dtype for the send buffers (the pre-NS dtype).

    Returns:
        The `OwnerGatherPlan` for this rank.
    """
    send_sizes: dict[int, int] = {o: 0 for o in range(world_size) if o != this_rank}
    recv_sizes: dict[int, int] = {s: 0 for s in range(world_size) if s != this_rank}
    for param_index, plan in enumerate(plans):
        owner = owners[param_index]
        if owner != this_rank:
            send_sizes[owner] += plan.shard_numel(this_rank)
        else:
            for src in range(world_size):
                if src == this_rank:
                    continue
                recv_sizes[src] += plan.shard_numel(src)

    send_buffers: dict[int, torch.Tensor] = {}
    for owner, size in send_sizes.items():
        send_buffers[owner] = torch.empty(size, dtype=dtype, device=device)

    # Fill each owner's send buffer in param order.
    cursors: dict[int, int] = {o: 0 for o in send_buffers}
    own_shards: dict[int, torch.Tensor] = {}
    for param_index, (plan, shard) in enumerate(zip(plans, local_shards)):
        owner = owners[param_index]
        if owner == this_rank:
            own_shards[param_index] = shard
            continue
        numel = plan.shard_numel(this_rank)
        if numel == 0:
            continue
        buf = send_buffers[owner]
        buf[cursors[owner] : cursors[owner] + numel].copy_(shard.reshape(-1))
        cursors[owner] += numel

    # Per (owned param, src) recv offset within the recv buffer from src.
    recv_offsets: dict[tuple[int, int], tuple[int, int, int]] = {}
    owned_indices = [i for i in range(len(plans)) if owners[i] == this_rank]
    for src in range(world_size):
        if src == this_rank:
            continue
        offset = 0
        for param_index in owned_indices:
            plan = plans[param_index]
            numel = plan.shard_numel(src)
            recv_offsets[(param_index, src)] = (offset, numel, plan.rank_row_count(src))
            offset += numel

    return OwnerGatherPlan(
        send_buffers=send_buffers,
        recv_sizes=recv_sizes,
        own_shards=own_shards,
        recv_offsets=recv_offsets,
    )


def reconstruct_orthogonalization_input(
    param_index: int,
    plan: ShardPlan,
    gather_plan: OwnerGatherPlan,
    recv_buffers: dict[int, torch.Tensor],
    owner_rank: int,
) -> torch.Tensor:
    """Reconstruct the full pre-NS matrix for one owned parameter.

    Concatenates shards in rank order: the owner's own local shard at its rank
    rows and each source's received shard at that source's rank rows.

    Args:
        param_index: Index of the parameter within the chunk.
        plan: Shard plan for this parameter.
        gather_plan: This rank's owner-gather plan (own_shards, recv_offsets).
        recv_buffers: Per-source-rank received buffer (only sources that sent).
        owner_rank: The owner rank (== the rank running reconstruction).

    Returns:
        The full `(rows, cols)` pre-NS matrix.
    """
    world_size = plan.world_size
    shards: list[torch.Tensor] = []
    for src in range(world_size):
        row_count = plan.rank_row_count(src)
        if src == owner_rank:
            shards.append(gather_plan.own_shards[param_index])
        elif row_count == 0:
            continue
        else:
            offset, numel, _ = gather_plan.recv_offsets[(param_index, src)]
            buf = recv_buffers[src]
            shards.append(buf[offset : offset + numel].view(row_count, plan.row_size))
    if len(shards) == 1:
        return shards[0].contiguous()
    return torch.cat(shards, dim=0)


@dataclasses.dataclass
class OwnerScatterPlan:
    """Metadata and send buffers for the owner-scatter P2P step of one chunk.

    The owner keeps its own update shard (applied directly), so it only sends to
    the other shard-holding ranks.

    Attributes:
        send_buffers: Per-destination-rank flat send buffer (this owner's update
            shards for the params it owns, in param order). Only destinations
            other than this rank appear.
        recv_sizes: Per-owner-rank element count this rank (as a destination)
            receives. Only owners other than this rank appear.
        recv_offsets: Per `(param_index, owner_rank)` of `(offset, numel,
            row_count)` describing where this param's update shard lands inside
            the recv buffer received from `owner_rank`.
    """

    send_buffers: dict[int, torch.Tensor]
    recv_sizes: dict[int, int]
    recv_offsets: dict[tuple[int, int], tuple[int, int, int]]


def pack_update_shards(
    full_updates: dict[int, torch.Tensor],
    plans: Sequence[ShardPlan],
    owners: dict[int, int],
    world_size: int,
    this_rank: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> OwnerScatterPlan:
    """Pack this owner rank's full updates into per-destination P2P send buffers.

    Args:
        full_updates: Full update matrix per owned parameter index.
        plans: Shard plans in parameter order.
        owners: Mapping from parameter index to owner rank.
        world_size: DP group size.
        this_rank: This rank's DP index.
        device: Device for the send buffers (the update device).
        dtype: Dtype for the send buffers (the update dtype).

    Returns:
        The `OwnerScatterPlan` for this rank.
    """
    owned_indices = [i for i in range(len(plans)) if owners[i] == this_rank]
    send_sizes: dict[int, int] = {d: 0 for d in range(world_size) if d != this_rank}
    recv_sizes: dict[int, int] = {o: 0 for o in range(world_size) if o != this_rank}
    for param_index in owned_indices:
        plan = plans[param_index]
        for dest in range(world_size):
            if dest == this_rank:
                continue
            send_sizes[dest] += plan.shard_numel(dest)
    for param_index, plan in enumerate(plans):
        owner = owners[param_index]
        if owner == this_rank:
            continue
        recv_sizes[owner] += plan.shard_numel(this_rank)

    send_buffers: dict[int, torch.Tensor] = {}
    for dest, size in send_sizes.items():
        send_buffers[dest] = torch.empty(size, dtype=dtype, device=device)

    cursors: dict[int, int] = {d: 0 for d in send_buffers}
    for dest in range(world_size):
        if dest == this_rank:
            continue
        for param_index in owned_indices:
            plan = plans[param_index]
            row_start, row_count = plan.rank_rows[dest]
            numel = row_count * plan.row_size
            if numel == 0:
                continue
            full = full_updates[param_index]
            buf = send_buffers[dest]
            buf[cursors[dest] : cursors[dest] + numel].copy_(
                full[row_start : row_start + row_count].reshape(-1)
            )
            cursors[dest] += numel

    recv_offsets: dict[tuple[int, int], tuple[int, int, int]] = {}
    for owner in range(world_size):
        if owner == this_rank:
            continue
        offset = 0
        for param_index, plan in enumerate(plans):
            if owners[param_index] != owner:
                continue
            numel = plan.shard_numel(this_rank)
            recv_offsets[(param_index, owner)] = (offset, numel, plan.rank_row_count(this_rank))
            offset += numel

    return OwnerScatterPlan(
        send_buffers=send_buffers, recv_sizes=recv_sizes, recv_offsets=recv_offsets
    )


def unpack_update_shards(
    scatter_plan: OwnerScatterPlan, recv_buffers: dict[int, torch.Tensor]
) -> dict[int, torch.Tensor]:
    """Extract this rank's local update shards from the per-owner recv buffers.

    Args:
        scatter_plan: This rank's owner-scatter plan (recv_offsets).
        recv_buffers: Per-owner-rank received buffer (only owners that sent).

    Returns:
        Mapping from parameter index to the local update shard `(row_count, cols)`,
        for parameters this rank does NOT own.
    """
    updates: dict[int, torch.Tensor] = {}
    for (param_index, owner), (offset, numel, row_count) in scatter_plan.recv_offsets.items():
        if numel == 0:
            continue
        buf = recv_buffers[owner]
        row_size = numel // row_count if row_count else 1
        updates[param_index] = buf[offset : offset + numel].view(row_count, row_size)
    return updates


class FsdpOrthogonalizedOptimizer(torch.optim.Optimizer):
    """Owner-compute orthogonalized optimizer for all-`Flat` M-FSDPv2 parameters.

    Subclasses `torch.optim.Optimizer` directly so it is a drop-in `torch.optim.Optimizer`
    for the training loop and checkpointer. It composes an
    `OrthogonalizedOptimizer` (held as `self._inner`) only for the Newton-Schulz
    orthogonalization kernel (`orthogonalize`, `scaled_orthogonalize_fn`),
    weight-decay application (`_apply_weight_decay_inplace`), and the pre/post
    weight-update hooks.

    All inner-optimizer arguments (`lr`, `momentum`, `weight_decay`, `nesterov`,
    `weight_decay_method`, `fp32_matmul_prec`, `scaled_orthogonalize_fn`, ...) are
    forwarded to the inner via `*args`/`**kwargs` and are not redeclared on this
    wrapper, so this optimizer cannot diverge from the inner. The inner's
    `defaults`, `param_groups`, and `state` are grabbed automatically and become this
    optimizer's (they are the same objects). The `step` override replaces the inner's
    per-parameter all-gather path with the owner-compute + scatter P2P algorithm.

    Args:
        params: Iterable of parameters or param-group dicts to optimize.
        *args, **kwargs: Forwarded to the inner `OrthogonalizedOptimizer`
            (e.g. `lr`, `momentum`, `weight_decay`, `nesterov`,
            `weight_decay_method`, `fp32_matmul_prec`, `scaled_orthogonalize_fn`).
        dp_mesh: Device mesh of the FSDP data-parallel group. The optimizer shards
            Newton-Schulz work across the ranks of this mesh via P2P.
        use_owner_comm_stream: Whether to use a separate communication stream for
            owner-based peer-to-peer communications. Useful to disable for testing
            reasons, giving a synchronous algorithm. Defaults to True.
        num_ns_steps: Newton-Schulz iteration count, also used by the owner
            load-balancing cost heuristic. If None, defaults to 1.
    """

    # Shared across optimizer instances so a process that constructs several
    # optimizers over the same DP group does not call `new_group` more than
    # once per group (each `new_group` is a collective that allocates NCCL
    # resources, which a multi-rank-on-one-GPU dev box exhausts quickly).
    _shared_owner_group_cache: dict[tuple[int, ...], dist.ProcessGroup] = {}
    _shared_owner_group_initialized: set[tuple[int, ...]] = set()

    def __init__(
        self,
        params: ParamsT,
        inner_optimizer: OrthogonalizedOptimizer,
        dp_mesh: DeviceMesh,
        use_owner_comm_stream: bool = True,
        num_ns_steps: int | None = None,
    ) -> None:
        _require_emerging_optimizers()

        if num_ns_steps is None:
            num_ns_steps = 1
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")

        self.dp_mesh = dp_mesh
        self._num_ns_steps: int = num_ns_steps
        # Owner P2P runs on a dedicated owner-comm stream so it overlaps local
        # Newton-Schulz on the default stream. Multi-rank-on-one-GPU dev boxes
        # cannot reliably run NCCL P2P on a separate stream, so this flag lets
        # tests (the only synchronous-use case) fall back to the default stream.
        self.use_owner_comm_stream: bool = use_owner_comm_stream
        self._shard_plans: dict[int, ShardPlan] = {}
        self._owners: dict[int, int] = {}
        self._owner_comm_stream_cache: dict[torch.device, torch.cuda.Stream] = {}

        # Disable properties while initializing this instance. We'd either have a missing attribute
        # or would reset the inner optimizer's attributes.
        with self._without_property_methods():
            super().__init__(params, {})
        self._inner = inner_optimizer

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        """Delegate `param_groups` to the inner optimizer."""
        return self._inner.param_groups

    @param_groups.setter
    def param_groups(self, value: list[dict[str, Any]]) -> None:
        """Set `param_groups` on the inner optimizer."""
        self._inner.param_groups = value

    @property
    def defaults(self) -> dict[str, Any]:
        """Delegate `defaults` to the inner optimizer."""
        return self._inner.defaults

    @defaults.setter
    def defaults(self, value: dict[str, Any]) -> None:
        """Set `defaults` on the inner optimizer."""
        self._inner.defaults = value

    @property
    def state(self) -> defaultdict[torch.Tensor, Any]:
        """Delegate `state` to the inner optimizer."""
        return self._inner.state

    @state.setter
    def state(self, value: defaultdict[torch.Tensor, Any]) -> None:
        """Set `state` on the inner optimizer."""
        self._inner.state = value

    def _all_params(self):
        """Flatten this optimizer's params from its (now-materialized) groups."""
        return [p for group in self.param_groups for p in group["params"]]

    def _init_group(self, group: dict, skip_non_grad_params: bool = True) -> None:
        """Performs lazy momentum-state initialization, delegated to the inner optimizer."""
        self._inner._init_group(group, skip_non_grad_params=skip_non_grad_params)

    @contextlib.contextmanager
    def _without_property_methods(self):
        """Temporarily remove the delegating property descriptors.

        The properties are defined on `FsdpOrthogonalizedOptimizer` and inherited by subclasses, so
        `delattr` must target the defining class (found via the MRO), not `type(self)` (which is the
        subclass and does not own the descriptors).
        """
        names = ["param_groups", "defaults", "state"]
        saved: dict[str, tuple[type, property]] = {}
        for name in names:
            for cls_ in type(self).__mro__:
                descriptor = cls_.__dict__.get(name)
                if descriptor is not None and isinstance(descriptor, property):
                    saved[name] = (cls_, descriptor)
                    try:
                        delattr(cls_, name)
                    except AttributeError:
                        pass
                    break
        try:
            yield
        finally:
            for name in names:
                self.__dict__.pop(name, None)
            for name, (cls_, descriptor) in saved.items():
                setattr(cls_, name, descriptor)

    # Mesh, group, and stream helpers
    # ===============================

    def _dp_group(self) -> dist.ProcessGroup:
        return self.dp_mesh.get_group()

    def _world_size(self) -> int:
        return self.dp_mesh.size()

    def _this_rank(self) -> int:
        return self.dp_mesh.get_local_rank()

    def _this_global_rank(self) -> int:
        return dist.get_global_rank(self._dp_group(), self._this_rank())

    def _init_collective_groups(self) -> dist.ProcessGroup:
        """Create (and cache) a duplicate owner-comm group for the DP group.

        A duplicate NCCL group with the same ranks lets owner P2P use an
        independent communicator/queue from FSDP's forward/backward collectives,
        so owner comm ordering is decoupled. The group is created once (a
        collective `new_group`) and initialized with a barrier so the first
        batched P2P may involve a subset of ranks.
        """
        ranks = tuple(dist.get_process_group_ranks(self._dp_group()))
        cached = self._shared_owner_group_cache.get(ranks)
        if cached is not None:
            return cached
        group = dist.new_group(ranks=list(ranks))
        type(self)._shared_owner_group_cache[ranks] = group
        # Initialize the communicator so the first batched P2P may involve a
        # subset of ranks; a barrier is a collective all ranks in the group run.
        if self._dp_group().size() > 1:
            if self.dp_mesh.device_type == "cuda":
                dist.barrier(group=group, device_ids=[torch.cuda.current_device()])
            else:
                dist.barrier(group=group)
        type(self)._shared_owner_group_initialized.add(ranks)
        return group

    def _owner_comm_stream(self, device: torch.device) -> torch.cuda.Stream | None:
        """Cached owner-comm stream (CUDA only; None on CPU or when disabled).

        Returns None (P2P on the default stream) when `use_owner_comm_stream`
        is False, e.g. for numerics tests on multi-rank-on-one-GPU dev boxes.
        """
        if device.type != "cuda" or not self.use_owner_comm_stream:
            return None
        cached = self._owner_comm_stream_cache.get(device)
        if cached is None:
            with torch.cuda.device(device):
                cached = torch.cuda.Stream()
            self._owner_comm_stream_cache[device] = cached
        return cached

    def _wait_for_dist_buffer(self, works: list[dist.Work]) -> None:
        """Wait for a batched P2P communication to complete.

        That means there is no (possibly asynchronous) communication, computation,
        or other memory access happening around it anymore.
        """
        for work in works:
            work.wait()

    # Shard planning and classification
    # =================================

    def _init_shard_plans(self, params: Sequence[torch.Tensor]) -> list[ShardPlan | None]:
        """Collect shard metadata for the model's parameters.

        The shard plans are built and cached, or retrieved from cache if
        available. Plans are derived from the owning `FsdpParameterGroup`'s
        `main_weight` DBuffer layout and are identical on every rank, so all
        ranks agree on owners.
        """
        plans: list[ShardPlan | None] = []
        for param in params:
            key = id(param)
            cached = self._shard_plans.get(key)
            if cached is not None:
                plans.append(cached)
                continue
            group = get_containing_parameter_group(param)
            if group is None:
                raise RuntimeError(
                    "FsdpOrthogonalizedOptimizer parameters must be FSDP-sharded; "
                    f"parameter {param!r} is not owned by an FsdpParameterGroup."
                )
            index = group.sharded_parameters.index(param)
            layout = group.main_weight.layout
            shape = layout.tensor_shapes[index]
            if len(shape) != 2:
                plans.append(None)
                continue
            tensor_flat_offset = layout.tensor_to_offset[index]
            rank_flat_shard_size = layout.size // self._world_size()
            plan = compute_shard_plan(
                shape, tensor_flat_offset, rank_flat_shard_size, self._world_size()
            )
            self._shard_plans[key] = plan
            plans.append(plan)
        return plans

    def _classify_params(self, plans: Sequence[ShardPlan | None]) -> dict[int, str]:
        """Classify parameters into fully local and sharded parameters that cross boundaries.

        This is useful to separate compute streams later. We have the local
        compute streams (fully-local Newton-Schulz) and owner-based compute
        streams (owner gather/scatter + owner Newton-Schulz).
        """
        classes: dict[int, str] = {}
        for index, plan in enumerate(plans):
            if plan is None:
                classes[index] = "non_matrix"
            elif plan.is_boundary():
                classes[index] = "boundary"
            else:
                classes[index] = "fully_local"
        return classes

    # Local orthogonalization-input computation
    # =========================================

    def _compute_orthogonalization_inputs(
        self, param: DTensor, grad: DTensor, group: dict[str, Any], lr: float
    ) -> torch.Tensor:
        """For the given parameter, apply weight decay and update momentum state, then produce and
        return the inputs for orthogonalization.
        """
        p_local = param.to_local()
        state = self.state[param]
        momentum = state["momentum_buffer"]
        mom_local = momentum.to_local()
        local_grad = grad.to_local()
        if local_grad.dtype != mom_local.dtype:
            local_grad = local_grad.to(dtype=mom_local.dtype)
        if local_grad.shape != mom_local.shape:
            local_grad = local_grad.reshape(mom_local.shape)

        self._inner._apply_weight_decay_inplace(p_local, local_grad, lr, group["weight_decay"])
        mom_local.lerp_(local_grad, 1 - group["momentum"])
        if self._inner.nesterov:
            pre_ns = local_grad.lerp(mom_local, group["momentum"])
        else:
            pre_ns = mom_local
        return pre_ns

    # Grouping and owner assignment
    # =============================

    def _group_updates(
        self, params: Sequence[DTensor], local_shards: Sequence[torch.Tensor]
    ) -> list[list[int]]:
        """Using the shard plans, group the updates into chunks.

        The updates are grouped into chunks by:
        - same collective group
        - same dtype and device of orthogonalization input shards
        - same dtype of parameter
        """
        chunks: dict[tuple, list[int]] = {}
        for index, (param, shard) in enumerate(zip(params, local_shards)):
            group = get_containing_parameter_group(param)
            collective_group = group.mesh.get_group() if group is not None else None
            key = (id(collective_group), shard.device, shard.dtype, param.dtype)
            chunks.setdefault(key, []).append(index)
        return list(chunks.values())

    def _assign_owner_work(self, plans: Sequence[ShardPlan]) -> dict[int, int]:
        """Assign the logical full, unsharded update tensor to one owner rank.

        Assignment has two rules:
        1. Only ranks that contain a non-empty part of the shard can be owners.
        2. Assignment is balanced around estimated full-tensor orthogonalization and update work.
           For an M x N matrix, with `num_steps` being the amount of iterations for the
           orthogonalization approximation:
           `M * N * (min(M, N) * num_steps + 1)`
        """
        return assign_owner_work(plans, self._num_ns_steps)

    # Owner-gather communication (P2P)
    # ================================

    def _pack_owner_work(
        self,
        plans: Sequence[ShardPlan],
        owners: dict[int, int],
        local_shards: Sequence[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> OwnerGatherPlan:
        """Pack all orthogonalization input shards for an owner into the owner's respective
        collective buffer.

        This sets up the buffers for communicating orthogonalization input shards to their owner.
        """
        return pack_owner_work(
            plans,
            owners,
            local_shards,
            self._world_size(),
            self._this_rank(),
            device=device,
            dtype=dtype,
        )

    def _send_to_owner(
        self, gather_plan: OwnerGatherPlan, device: torch.device, dtype: torch.dtype
    ) -> tuple[dict[int, torch.Tensor], list[dist.Work]]:
        """Send orthogonalization input shards to their respective owner.

        Uses peer-to-peer communication (`batch_isend_irecv`) to avoid memory
        allocations around setting up a large all-to-all buffer. Sends and recvs
        are issued on the owner-comm stream so they overlap local compute. The
        owner's own shard is not sent (kept locally for reconstruction).

        Args:
            gather_plan: This rank's owner-gather plan (send/recv buffers).
            device: Device for the send/recv buffers (the pre-NS device).
            dtype: Dtype for the send/recv buffers (the pre-NS dtype).
        """
        group = self._init_collective_groups()
        stream = self._owner_comm_stream(device)
        recv_buffers: dict[int, torch.Tensor] = {
            src: torch.empty(size, dtype=dtype, device=device)
            for src, size in gather_plan.recv_sizes.items()
            if size > 0
        }
        ops: list[dist.P2POp] = []
        for owner, buf in gather_plan.send_buffers.items():
            if buf.numel() == 0:
                continue
            # `owner` is a DP-group rank index (mesh local rank); pass it as
            # group_peer so P2POp resolves it within the owner-comm group.
            ops.append(dist.P2POp(dist.isend, buf, group_peer=owner, group=group))
        for src, buf in recv_buffers.items():
            ops.append(dist.P2POp(dist.irecv, buf, group_peer=src, group=group))

        default_stream = torch.cuda.current_stream() if stream is not None else None
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            if stream is not None:
                stream.wait_stream(default_stream)
            works = dist.batch_isend_irecv(ops) if ops else []
            for buf in list(gather_plan.send_buffers.values()) + list(recv_buffers.values()):
                if stream is not None:
                    buf.record_stream(stream)
        return recv_buffers, list(works or [])

    # Orthogonalization and update application
    # ========================================

    def _orthogonalize_with_precision(
        self, param: torch.Tensor, pre_ns: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """Run batched orthogonalization on the given orthogonalization inputs and return the
        result.

        Orthogonalization will use FP32 matrix multiplications in the given precision, by default
        `self._inner.fp32_matmul_prec`.
        """
        with eo_utils.fp32_matmul_precision(self._inner.fp32_matmul_prec):
            return self._inner.orthogonalize(param, pre_ns, **kwargs)

    def _apply_update(self, param: DTensor, update_shard: torch.Tensor, lr: float) -> None:
        """Update the given parameters in batched fashion with the result of orthogonalization."""
        p_local = param.to_local()
        if update_shard.dtype != p_local.dtype:
            update_shard = update_shard.to(dtype=p_local.dtype)
        self._inner.pre_weight_update_fn_inplace(p_local, update_shard)
        p_local.add_(update_shard, alpha=-lr)
        self._inner.post_weight_update_fn_inplace(p_local)

    def _orthogonalize_and_update(
        self, param: DTensor, pre_ns: torch.Tensor, lr: float, group: dict[str, Any]
    ) -> None:
        """Run orthogonalization on the given orthogonalization inputs and update the given
        parameter with the result of orthogonalization. (Fully-local path: no communication.)
        """
        group_kwargs = {k: v for k, v in group.items() if k != "params"}
        update = self._orthogonalize_with_precision(param, pre_ns, **group_kwargs)
        self._apply_update(param, update, lr)

    def _reconstruct_orthogonalization_input(
        self,
        param_index: int,
        plan: ShardPlan,
        gather_plan: OwnerGatherPlan,
        recv_buffers: dict[int, torch.Tensor],
        owner_rank: int,
    ) -> torch.Tensor:
        """Merge the gathered orthogonalization input shards back to the original, full, unsharded
        input tensor.
        """
        return reconstruct_orthogonalization_input(
            param_index, plan, gather_plan, recv_buffers, owner_rank
        )

    # Owner-scatter communication (P2P)
    # =================================

    def _pack_update_shards(
        self,
        full_updates: dict[int, torch.Tensor],
        plans: Sequence[ShardPlan],
        owners: dict[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> OwnerScatterPlan:
        """Set up the buffers for communication by packing update shards into their respective
        collective buffers.

        Pack all update shards for their destination into the destination's respective collective
        buffer. This sets up the buffers for communicating update shards to their destination.
        """
        return pack_update_shards(
            full_updates,
            plans,
            owners,
            self._world_size(),
            self._this_rank(),
            device=device,
            dtype=dtype,
        )

    def _send_to_destination(
        self, scatter_plan: OwnerScatterPlan, device: torch.device, dtype: torch.dtype
    ) -> tuple[dict[int, torch.Tensor], list[dist.Work]]:
        """Send update shards to their respective destination.

        Uses peer-to-peer communication (`batch_isend_irecv`) to avoid memory
        allocations around setting up a large all-to-all buffer. The owner's own
        update shard is not sent (applied directly).

        Args:
            scatter_plan: This rank's owner-scatter plan (send/recv buffers).
            device: Device for the send/recv buffers (the update device).
            dtype: Dtype for the send/recv buffers (the update dtype).
        """
        group = self._init_collective_groups()
        stream = self._owner_comm_stream(device)
        recv_buffers: dict[int, torch.Tensor] = {
            owner: torch.empty(size, dtype=dtype, device=device)
            for owner, size in scatter_plan.recv_sizes.items()
            if size > 0
        }
        ops: list[dist.P2POp] = []
        for dest, buf in scatter_plan.send_buffers.items():
            if buf.numel() == 0:
                continue
            ops.append(dist.P2POp(dist.isend, buf, group_peer=dest, group=group))
        for owner, buf in recv_buffers.items():
            ops.append(dist.P2POp(dist.irecv, buf, group_peer=owner, group=group))

        default_stream = torch.cuda.current_stream() if stream is not None else None
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            if stream is not None:
                stream.wait_stream(default_stream)
            works = dist.batch_isend_irecv(ops) if ops else []
            for buf in list(scatter_plan.send_buffers.values()) + list(recv_buffers.values()):
                if stream is not None:
                    buf.record_stream(stream)
        return recv_buffers, list(works or [])

    def _unpack_update_shards(
        self, scatter_plan: OwnerScatterPlan, recv_buffers: dict[int, torch.Tensor]
    ) -> dict[int, torch.Tensor]:
        """Unpack the packed update shards in the given buffer."""
        return unpack_update_shards(scatter_plan, recv_buffers)

    # Full step
    # =========

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single optimization step to update parameters.

        Separates collective (P2P) and local (NS) work into phases so that no
        rank is blocked waiting on another rank computing NS to reach P2P:

        1. Compute momentum updates locally for all params.
        2. P2P-send boundary pre-NS shards to owners - all P2P, no NS interleaved.
        3. Newton-Schulz + weight update locally (fully-local params overlap the
           owner gather; boundary params wait for their gather then NS+scatter).
        """
        loss = None if closure is None else closure()

        fsdp_parameter_groups: set[FsdpParameterGroup] = set()
        for group in self.param_groups:
            self._init_group(group)
            params = group["params"]
            plans = self._init_shard_plans(params)
            classes = self._classify_params(plans)
            lr = group["lr"]
            group_kwargs = {k: v for k, v in group.items() if k != "params"}

            # Non-2D parameters: plain momentum-SGD, no orthogonalization.
            for index, param in enumerate(params):
                if param.grad is None:
                    continue
                if classes[index] == "non_matrix":
                    self._step_non_matrix(param, param.grad, group, lr)
                    pg = get_containing_parameter_group(param)
                    if pg is not None:
                        fsdp_parameter_groups.add(pg)

            matrix_indices = [
                i for i, p in enumerate(params) if p.grad is not None and classes[i] != "non_matrix"
            ]
            if not matrix_indices:
                continue
            matrix_params = [params[i] for i in matrix_indices]
            matrix_plans = [plans[i] for i in matrix_indices if plans[i] is not None]
            # Assert no non-matrix plans were indexed.
            assert len(matrix_plans) == len(matrix_indices)
            matrix_plans = cast(list[ShardPlan], matrix_plans)
            owners = self._assign_owner_work(matrix_plans)
            self._owners.update({matrix_indices[k]: v for k, v in owners.items()})

            # Phase 1: local pre-NS shards for all matrix params.
            local_shards: list[torch.Tensor] = []
            for param in matrix_params:
                if param.grad is None:
                    local_shards.append(torch.empty(0, dtype=torch.float32, device=self._device()))
                    continue
                local_shards.append(
                    self._compute_orthogonalization_inputs(param, param.grad, group, lr)
                )
            # Separate fully-local and boundary params. Fully-local NS+update
            # overlaps the boundary owner gather.
            local_indices = [
                i for i in range(len(matrix_plans)) if not matrix_plans[i].is_boundary()
            ]
            boundary_indices_set = {
                i for i in range(len(matrix_plans)) if matrix_plans[i].is_boundary()
            }

            # Group boundary params by collective group, shard device/dtype,
            # and parameter dtype (see `_group_updates`) so each chunk uses
            # consistent P2P buffer metadata. A single optimizer param group
            # may span multiple FSDP groups and/or mixed dtypes (e.g. FP32 + BF16).
            for chunk_indices in self._group_updates(matrix_params, local_shards):
                chunk_boundary = [i for i in chunk_indices if i in boundary_indices_set]
                if not chunk_boundary:
                    continue
                shard_device = local_shards[chunk_boundary[0]].device
                shard_dtype = local_shards[chunk_boundary[0]].dtype
                self._step_boundary(
                    matrix_params,
                    matrix_plans,
                    owners,
                    chunk_boundary,
                    local_shards,
                    shard_device,
                    shard_dtype,
                    lr,
                    group_kwargs,
                )

            # Fully-local path: orthogonalize the full local matrix and update.
            # Runs on the default stream, overlapping the boundary P2P issued
            # above on the owner-comm stream.
            for i in local_indices:
                plan = matrix_plans[i]
                if plan.rank_row_count(self._this_rank()) == 0:
                    continue
                self._orthogonalize_and_update(matrix_params[i], local_shards[i], lr, group)

            for param in matrix_params:
                pg = get_containing_parameter_group(param)
                if pg is not None:
                    fsdp_parameter_groups.add(pg)

        for parameter_group in fsdp_parameter_groups:
            parameter_group.sync_model_weight_from_main_weight()
        return loss

    def _step_boundary(
        self,
        matrix_params: Sequence[DTensor],
        matrix_plans: Sequence[ShardPlan],
        owners: dict[int, int],
        boundary_indices: list[int],
        local_shards: Sequence[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        lr: float,
        group_kwargs: dict[str, Any],
    ) -> None:
        """Owner-compute P2P step for the boundary parameters of one chunk."""
        b_plans = [matrix_plans[i] for i in boundary_indices]
        b_params = [matrix_params[i] for i in boundary_indices]
        b_local = [local_shards[i] for i in boundary_indices]
        b_owners = {i: owners[boundary_indices[i]] for i in range(len(boundary_indices))}
        this_rank = self._this_rank()
        stream = self._owner_comm_stream(device)

        # Phase 2: pack + P2P-send pre-NS shards to owners (async on owner stream).
        gather_plan = self._pack_owner_work(b_plans, b_owners, b_local, device, dtype)
        recv_buffers, gather_works = self._send_to_owner(gather_plan, device)

        # Phase 3 (owner): wait gather, reconstruct, orthogonalize. Default
        # stream waits for the owner stream so the recv buffers are ready.
        if stream is not None:
            torch.cuda.current_stream(device).wait_stream(stream)
        self._wait_for_dist_buffer(gather_works)

        full_updates: dict[int, torch.Tensor] = {}
        for i in range(len(b_params)):
            if b_owners[i] != this_rank:
                continue
            plan = b_plans[i]
            if plan.rank_row_count(this_rank) == 0:
                continue
            full = self._reconstruct_orthogonalization_input(
                i, plan, gather_plan, recv_buffers, owner_rank=this_rank
            )
            full_updates[i] = self._orthogonalize_with_precision(full, **group_kwargs)

        # Phase 4: pack + P2P-send update shards from owners (async on owner stream).
        scatter_plan = self._pack_update_shards(full_updates, b_plans, b_owners, device, dtype)
        scatter_recv, scatter_works = self._send_to_destination(scatter_plan, device, dtype)
        if stream is not None:
            torch.cuda.current_stream(device).wait_stream(stream)
        self._wait_for_dist_buffer(scatter_works)
        received = self._unpack_update_shards(scatter_plan, scatter_recv)

        # Phase 5: apply local update shards.
        for i, param in enumerate(b_params):
            plan = b_plans[i]
            if b_owners[i] == this_rank:
                row_start, row_count = plan.rank_rows[this_rank]
                if row_count == 0:
                    continue
                update_shard = full_updates[i][row_start : row_start + row_count]
            else:
                update_shard = received.get(i)
                if update_shard is None:
                    continue
            self._apply_update(param, update_shard, lr)

    def _step_non_matrix(
        self, param: DTensor, grad: DTensor, group: dict[str, Any], lr: float
    ) -> None:
        """Plain momentum-SGD step for non-2D parameters (no orthogonalization)."""
        state = self.state[param]
        if len(state) == 0:
            state["momentum_buffer"] = torch.zeros_like(param.data)
        momentum = state["momentum_buffer"]
        p_local = param.to_local()
        mom_local = momentum.to_local()
        local_grad = grad.to_local()
        if local_grad.dtype != mom_local.dtype:
            local_grad = local_grad.to(dtype=mom_local.dtype)
        if local_grad.shape != mom_local.shape:
            local_grad = local_grad.reshape(mom_local.shape)
        self._inner._apply_weight_decay_inplace(p_local, local_grad, lr, group["weight_decay"])
        mom_local.lerp_(local_grad, 1 - group["momentum"])
        if self._inner.nesterov:
            update = local_grad.lerp(mom_local, group["momentum"])
        else:
            update = mom_local
        self._inner.pre_weight_update_fn_inplace(p_local, update)
        p_local.add_(update, alpha=-lr)
        self._inner.post_weight_update_fn_inplace(p_local)

    def _device(self) -> torch.device:
        return torch.device(self.dp_mesh.device_type)


class FsdpMuon(FsdpOrthogonalizedOptimizer):
    """Muon optimizer for all-`Flat` M-FSDPv2 parameters.

    Composes a `Muon` inner optimizer (an `OrthogonalizedOptimizer`) for the
    Newton-Schulz orthogonalization and update scaling. The inner `Muon`
    installs its own `scaled_orthogonalize_fn`, so the base
    `scaled_orthogonalize_fn` is ignored.
    """

    def __init__(
        self,
        params: ParamsT,
        inner_optimizer: Muon,
        dp_mesh: DeviceMesh,
        use_owner_comm_stream: bool = True,
    ) -> None:
        _require_emerging_optimizers()

        if hasattr(inner_optimizer, "num_ns_steps"):
            self._num_ns_steps = inner_optimizer.num_ns_steps
        else:
            # For older `emerging_optimizers` versions, we use introspection techniques to get a
            # sensible value for `num_ns_steps`.
            try:
                ortho_fn_vars = inspect.getclosurevars(inner_optimizer.scaled_orthogonalize_fn)
                self._num_ns_steps = ortho_fn_vars.nonlocals["num_ns_steps"]
            except KeyError:
                warnings.warn(
                    "Cannot access Muon closure non-locals; going with "
                    "`emerging_optimizers.orthogonalized_optimizer.Muon` default `num_ns_steps` "
                    "for compute cost estimation"
                )
                muon_sig = inspect.signature(inner_optimizer)
                self._num_ns_steps = muon_sig.parameters["num_ns_steps"].default
        super().__init__(
            params, inner_optimizer, dp_mesh=dp_mesh, use_owner_comm_stream=use_owner_comm_stream
        )
