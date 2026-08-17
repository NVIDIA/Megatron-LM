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

from .parameter_group import FsdpParameterGroup, get_containing_parameter_group
from .shard_plan import (
    OwnerGatherPlan,
    OwnerScatterPlan,
    ShardPlan,
    assign_owner_work,
    compute_shard_plan,
    pack_owner_work,
    pack_update_shards,
    reconstruct_full_tensor,
    unpack_update_shards,
)

logger = logging.getLogger(__name__)


def _require_emerging_optimizers() -> None:
    if not HAVE_EMERGING_OPTIMIZERS:
        raise ModuleNotFoundError(
            "Emerging-Optimizers is required for orthogonalized optimizer support. "
            "Please install the necessary dependencies with "
            "`pip install 'megatron_fsdp[emerging-optimizers]'`."
        )


@dataclasses.dataclass
class _BoundaryChunkState:
    """In-flight owner-gather state for one boundary chunk, between issue and finish.

    `FsdpOrthogonalizedOptimizer.step` issues every chunk's owner-gather first
    (so fully-local Newton-Schulz overlaps the gathers), then finishes each chunk
    (wait gather, orthogonalize, scatter, apply). This object carries everything
    `_finish_boundary_step` needs across that gap.
    """

    b_params: Sequence[DTensor]
    b_plans: Sequence[ShardPlan]
    b_owners: dict[int, int]
    gather_plan: OwnerGatherPlan
    recv_buffers: dict[int, torch.Tensor]
    gather_works: list[dist.Work]
    device: torch.device
    dtype: torch.dtype
    lr: float
    group_kwargs: dict[str, Any]
    # Only populated when `reconstruct_full_param` is enabled: a second owner-gather
    # round that collects each rank's local *weight* shard (instead of the pre-NS
    # shard) so the owner can reconstruct the full parameter for `orthogonalize`.
    weight_gather_plan: OwnerGatherPlan | None = None
    weight_recv_buffers: dict[int, torch.Tensor] | None = None
    weight_gather_works: list[dist.Work] | None = None


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
        reconstruct_full_param: Whether to also gather local weight shards and pass
            them to orthogonalization. When False, `None` is passed as the
            parameter to orthogonalize. Defaults to False, since it's not used in
            the standard case.
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
        reconstruct_full_param: bool = False,
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
        self.reconstruct_full_param: bool = reconstruct_full_param
        self._owner_comm_needed: bool | None = None
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
        self, param: torch.Tensor | None, pre_ns: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """Run batched orthogonalization on the given orthogonalization inputs and return the
        result.

        Orthogonalization will use FP32 matrix multiplications in the given precision, by default
        `self._inner.fp32_matmul_prec`.
        """
        with eo_utils.fp32_matmul_precision(self._inner.fp32_matmul_prec):
            # `param` (AKA `p`) is typed as `torch.Tensor` in
            # `emerging_optimizers.OrthogonalizedOptimizer`. However, we explicitly want to error if
            # the function tries to use it as a tensor, but we pass `None`. So we remove the `None`
            # type here to remove the type error for that parameter explicitly.
            param = cast(torch.Tensor, param)
            # The Newton-Schulz kernel is FP32-only, so cast accordingly.
            return self._inner.orthogonalize(param, pre_ns.to(torch.float32), **kwargs)

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

        For a fully-local parameter the owning rank's local shard *is* the full
        parameter, so no gather is needed: pass it directly when
        `reconstruct_full_param` is enabled, else `None` (matching the
        boundary path).
        """
        group_kwargs = {k: v for k, v in group.items() if k != "params"}
        param_arg = param.to_local() if self.reconstruct_full_param else None
        update = self._orthogonalize_with_precision(param_arg, pre_ns, **group_kwargs)
        self._apply_update(param, update, lr)

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

        1. Compute local pre-NS shards (weight decay + momentum + Nesterov) for
           all matrix params.
        2. P2P-send all boundary pre-NS shards to their owners (async, no NS).
        3. Newton-Schulz + weight update for fully-local params on the default
           stream, overlapping the boundary owner-gathers issued in phase 2.
        4. On each owner, wait gather, reconstruct, and orthogonalize to produce
           the full updates.
        5. P2P-send update shards from owners back to their destinations.
        6. Each rank applies its local update shard to its local weight shard.
        """
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None

        if self._world_size() > 1:
            if self._owner_comm_needed is None:
                has_boundary = False
                for group in self.param_groups:
                    plans = self._init_shard_plans(group["params"])
                    if any(p is not None and p.is_boundary() for p in plans):
                        has_boundary = True
                        break
                flag = torch.tensor(int(has_boundary), device=self._device(), dtype=torch.int)
                dist.all_reduce(flag, op=dist.ReduceOp.SUM, group=self._dp_group())
                self._owner_comm_needed = flag.item() > 0
            if self._owner_comm_needed:
                self._init_collective_groups()

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

            # Phase 2: issue all boundary owner-gathers (async on the owner-comm
            # stream) before any Newton-Schulz, so the fully-local NS in phase 3
            # overlaps the gathers. This matches the V1 pseudocode, which runs the
            # owner-gather before the local orthogonalization. Boundary params are
            # grouped by collective group, shard device/dtype, and parameter dtype
            # (see `_group_updates`) so each chunk uses consistent P2P metadata; a
            # single optimizer param group may span multiple FSDP groups and/or
            # mixed dtypes (e.g. FP32 + BF16).
            chunk_states: list[_BoundaryChunkState] = []
            for chunk_indices in self._group_updates(matrix_params, local_shards):
                chunk_boundary = [i for i in chunk_indices if i in boundary_indices_set]
                if not chunk_boundary:
                    continue
                shard_device = local_shards[chunk_boundary[0]].device
                shard_dtype = local_shards[chunk_boundary[0]].dtype
                chunk_states.append(
                    self._issue_owner_gather(
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
                )

            # Phase 3: fully-local Newton-Schulz + weight update on the default
            # stream, overlapping the boundary owner-gathers issued above.
            for i in local_indices:
                plan = matrix_plans[i]
                if plan.rank_row_count(self._this_rank()) == 0:
                    continue
                self._orthogonalize_and_update(matrix_params[i], local_shards[i], lr, group)

            # Phases 4-6: finish each boundary chunk - wait gather, orthogonalize
            # on owners, scatter updates, and apply local update shards.
            for state in chunk_states:
                self._finish_boundary_step(state)

            for param in matrix_params:
                pg = get_containing_parameter_group(param)
                if pg is not None:
                    fsdp_parameter_groups.add(pg)

        for parameter_group in fsdp_parameter_groups:
            parameter_group.sync_model_weight_from_main_weight()
        return loss

    def _issue_owner_gather(
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
    ) -> _BoundaryChunkState:
        """Pack and asynchronously P2P-send this chunk's pre-NS shards to owners.

        Returns the in-flight gather state; `step` finishes it later with
        `_finish_boundary_step` so fully-local Newton-Schulz overlaps the gathers.
        """
        b_plans = [matrix_plans[i] for i in boundary_indices]
        b_params = [matrix_params[i] for i in boundary_indices]
        b_local = [local_shards[i] for i in boundary_indices]
        b_owners = {i: owners[boundary_indices[i]] for i in range(len(boundary_indices))}
        gather_plan = self._pack_owner_work(b_plans, b_owners, b_local, device, dtype)
        recv_buffers, gather_works = self._send_to_owner(gather_plan, device, dtype)

        # Optionally also gather each rank's local *weight* shard so the owner can
        # reconstruct the full parameter and pass it to `orthogonalize` (some
        # subclasses read `p`). This is a second, parallel P2P round issued on the
        # same owner-comm stream so it overlaps the pre-NS gather; it reuses the
        # same shard plans/owners (only the shard content differs).
        weight_gather_plan = None
        weight_recv_buffers = None
        weight_gather_works = None
        if self.reconstruct_full_param:
            weight_local = [p.to_local() for p in b_params]
            weight_dtype = b_params[0].dtype
            weight_gather_plan = self._pack_owner_work(
                b_plans, b_owners, weight_local, device, weight_dtype
            )
            weight_recv_buffers, weight_gather_works = self._send_to_owner(
                weight_gather_plan, device, weight_dtype
            )
        return _BoundaryChunkState(
            b_params=b_params,
            b_plans=b_plans,
            b_owners=b_owners,
            gather_plan=gather_plan,
            recv_buffers=recv_buffers,
            gather_works=gather_works,
            device=device,
            dtype=dtype,
            lr=lr,
            group_kwargs=group_kwargs,
            weight_gather_plan=weight_gather_plan,
            weight_recv_buffers=weight_recv_buffers,
            weight_gather_works=weight_gather_works,
        )

    def _finish_boundary_step(self, state: _BoundaryChunkState) -> None:
        """Finish one boundary chunk: wait gather, orthogonalize, scatter, apply."""
        b_params = state.b_params
        b_plans = state.b_plans
        b_owners = state.b_owners
        gather_plan = state.gather_plan
        recv_buffers = state.recv_buffers
        device = state.device
        lr = state.lr
        group_kwargs = state.group_kwargs
        this_rank = self._this_rank()
        stream = self._owner_comm_stream(device)

        # Phase 4 (owner): wait gather, reconstruct, orthogonalize. Default
        # stream waits for the owner stream so the recv buffers are ready.
        if stream is not None:
            torch.cuda.current_stream(device).wait_stream(stream)
        self._wait_for_dist_buffer(state.gather_works)
        if state.weight_gather_works:
            self._wait_for_dist_buffer(state.weight_gather_works)

        full_updates: dict[int, torch.Tensor] = {}
        for i in range(len(b_params)):
            if b_owners[i] != this_rank:
                continue
            plan = b_plans[i]
            if plan.rank_row_count(this_rank) == 0:
                continue
            # Merge the gathered orthogonalization input shards back to the original, full,
            # unsharded input tensor.
            full = reconstruct_full_tensor(i, plan, gather_plan, recv_buffers, owner_rank=this_rank)
            # `full` is the reconstructed pre-NS matrix and is the
            # orthogonalization input (`pre_ns`). The `param` (`p`) argument is
            # unused by the default `OrthogonalizedOptimizer.orthogonalize`, so
            # by default (`reconstruct_full_param=False`) we pass `None`. When the
            # flag is enabled, reconstruct the full parameter from the gathered
            # weight shards (a second P2P round issued in `_issue_owner_gather`) and
            # pass it as `param` for subclasses that read `p`.
            if self.reconstruct_full_param and state.weight_gather_plan is not None:
                param_arg = reconstruct_full_tensor(
                    i,
                    plan,
                    state.weight_gather_plan,
                    state.weight_recv_buffers,
                    owner_rank=this_rank,
                )
            else:
                param_arg = None
            full_updates[i] = self._orthogonalize_with_precision(param_arg, full, **group_kwargs)

        # Phase 5: pack + P2P-send update shards from owners (async on owner stream).
        scatter_plan = self._pack_update_shards(
            full_updates, b_plans, b_owners, device, state.dtype
        )
        scatter_recv, scatter_works = self._send_to_destination(scatter_plan, device, state.dtype)
        if stream is not None:
            torch.cuda.current_stream(device).wait_stream(stream)
        self._wait_for_dist_buffer(scatter_works)
        received = self._unpack_update_shards(scatter_plan, scatter_recv)

        # Phase 6: apply local update shards.
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
        reconstruct_full_param: bool = False,
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
            params,
            inner_optimizer,
            dp_mesh=dp_mesh,
            use_owner_comm_stream=use_owner_comm_stream,
            reconstruct_full_param=reconstruct_full_param,
        )
