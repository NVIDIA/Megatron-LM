# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Owner-compute orthogonalized optimizer for Megatron-FSDP v2.

This module implements the Muon-style orthogonalizing optimizer step on top of M-FSDPv2's all-`Flat`
placements (`parameter=Flat, gradient=Flat, optimizer=Flat`). It reuses the `emerging_optimizers`
Newton-Schulz kernels through `OrthogonalizedOptimizer`.

Algorithm (per optimizer step, for 2D "matrix" parameters that cross FSDP rank boundaries), matching
an owner-compute + scatter design:

    1. Compute local orthogonalization-input ("pre-NS") shards: weight decay, momentum update, and
       (optional) Nesterov combination, all on each rank's local gradient shard.
    2. P2P-send the pre-NS shards to their owner ranks. Owners are balanced across ranks by an
       orthogonalization compute-cost heuristic so no single rank serializes all Newton-Schulz work.
       The owner's own shard is kept locally (no self-send).
    3. On each owner, reconstruct the full pre-NS matrix and run Newton-Schulz orthogonalization to
       produce the full update.
    4. P2P-send the update shards from owners back to their destination ranks.
    5. Each rank applies its local update shard to its local weight shard.

Fully local parameters (owned by a single rank) skip the communication and run Newton-Schulz
locally; their compute overlaps the boundary P2P. Non-2D parameters fall back to a plain
momentum-SGD step (no orthogonalization).

Communication is asynchronous and issued on a dedicated owner-comm stream using a dedicated
(duplicate) owner-comm process group, so owner P2P ordering is independent of FSDP's
forward/backward collectives. The synchronous waiting (`_wait_for_dist_buffer`) is deferred as late
as possible so local Newton-Schulz work overlaps owner gathers/scatters."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
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
    eo_utils = cast(Any, None)
    OrthogonalizedOptimizer = cast(Any, object)
    Muon = cast(Any, object)
    HAVE_EMERGING_OPTIMIZERS = False
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import ParamsT

from .parameter_group import FsdpParameterGroup, get_containing_parameter_group
from .placement import Flat
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


def _always_nvtx_decorator(message: str) -> Callable:
    """Wrap a function in one NVTX range without enabling global Megatron ranges."""

    def decorator(function: Callable) -> Callable:
        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            torch.cuda.nvtx.range_push(message)
            try:
                return function(*args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()

        return wrapper

    return decorator


def _require_emerging_optimizers() -> None:
    if not HAVE_EMERGING_OPTIMIZERS:
        raise ModuleNotFoundError(
            "Emerging-Optimizers is required for orthogonalized optimizer support. "
            "Please install the necessary dependencies with "
            "`pip install 'megatron_fsdp[emerging-optimizers]'`."
        )


@dataclasses.dataclass
class _BoundaryChunkState:
    """In-flight gather and scatter state for one boundary chunk.

    `FsdpOrthogonalizedOptimizer.step` issues every chunk's owner-gather first,
    then `_issue_boundary_update` fills the scatter fields for
    `_enqueue_boundary_update` to consume.
    """

    b_params: Sequence[DTensor]
    b_plans: Sequence[ShardPlan]
    b_owners: dict[int, int]
    gather_plan: OwnerGatherPlan
    recv_buffers: dict[int, torch.Tensor]
    gather_works: list[dist.Work]
    gather_event: torch.cuda.Event | None
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
    weight_gather_event: torch.cuda.Event | None = None

    # Populated by `_issue_boundary_update` while this chunk's scatter is in flight.
    full_updates: dict[int, torch.Tensor] = dataclasses.field(default_factory=dict)
    scatter_plan: OwnerScatterPlan | None = None
    scatter_recv: dict[int, torch.Tensor] = dataclasses.field(default_factory=dict)
    scatter_works: list[dist.Work] = dataclasses.field(default_factory=list)
    scatter_event: torch.cuda.Event | None = None


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
            their full value to orthogonalization. When False, the sharded parameter
            is passed for shape and tensor-parallel metadata only. Defaults to False,
            since the standard Muon path does not read parameter values.
        num_ns_steps: Newton-Schulz iteration count, also used by the owner
            load-balancing cost heuristic. If None, defaults to 1.
        max_params_per_owner_chunk: Maximum number of compatible parameters in
            one owner-communication chunk. None places all compatible parameters
            in one chunk.
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
        max_params_per_owner_chunk: int | None = 4,
    ) -> None:
        _require_emerging_optimizers()

        if num_ns_steps is None:
            num_ns_steps = 1
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")

        self.dp_mesh = dp_mesh
        self._num_ns_steps: int = num_ns_steps
        self._max_params_per_owner_chunk = max_params_per_owner_chunk
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

        # Assert all-`Flat` placements: this optimizer only supports the
        # all-Flat layout (parameter=Flat, gradient=Flat, optimizer=Flat).
        _seen_groups: set[FsdpParameterGroup] = set()
        for _param in self._all_params():
            _group = get_containing_parameter_group(_param)
            if _group is None or _group in _seen_groups:
                continue
            _seen_groups.add(_group)
            for _buf in (_group.main_weight, _group.model_weight, _group.main_grad):
                if _buf is None:
                    continue
                if not all(isinstance(_p, Flat) for _p in _buf.placements):
                    raise ValueError(
                        "FsdpOrthogonalizedOptimizer requires all-Flat placements "
                        "(parameter=Flat, gradient=Flat, optimizer=Flat), but "
                        f"{_group} has non-Flat placements "
                        f"{[type(_p).__name__ for _p in _buf.placements]}."
                    )

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
            index = next(i for i, fp in enumerate(group.fsdp_parameters) if fp.sharded is param)
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
        self,
        params: Sequence[torch.Tensor],
        local_shards: Sequence[torch.Tensor],
        plans: Sequence[ShardPlan],
    ) -> list[list[int]]:
        """Using the shard plans, group the updates into chunks.

        The updates are grouped into chunks by:
        - same communication requirement
        - same collective group
        - same dtype and device of orthogonalization input shards
        - same dtype of parameter
        - at most max_params_per_owner_chunk parameters per chunk when configured,
          so later owner gathers can overlap orthogonalization of earlier chunks
        """
        chunks: dict[tuple, list[list[int]]] = {}
        for index, (param, shard, plan) in enumerate(zip(params, local_shards, plans)):
            group = get_containing_parameter_group(param)
            collective_group = group.mesh.get_group() if group is not None else None
            key = (plan.is_boundary(), id(collective_group), shard.device, shard.dtype, param.dtype)
            compatible_chunks = chunks.setdefault(key, [])
            if (
                not compatible_chunks
                or self._max_params_per_owner_chunk is not None
                and len(compatible_chunks[-1]) >= self._max_params_per_owner_chunk
            ):
                compatible_chunks.append([])
            compatible_chunks[-1].append(index)
        return [chunk for compatible_chunks in chunks.values() for chunk in compatible_chunks]

    def _assign_owner_work(
        self, plans: Sequence[ShardPlan], chunks: Sequence[Sequence[int]]
    ) -> dict[int, int]:
        """Assign owners by descending NS cost with per-chunk load balancing."""
        return assign_owner_work(plans, self._num_ns_steps, chunks)

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
    ) -> tuple[dict[int, torch.Tensor], list[dist.Work], torch.cuda.Event | None]:
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
        completion_event = None
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            if stream is not None:
                assert default_stream is not None
                stream.wait_stream(default_stream)
            works = dist.batch_isend_irecv(ops) if ops else []
            for buf in list(gather_plan.send_buffers.values()) + list(recv_buffers.values()):
                if stream is not None:
                    buf.record_stream(stream)
            if stream is not None:
                completion_event = torch.cuda.Event()
                completion_event.record(stream)
        return recv_buffers, list(works or []), completion_event

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
            # `emerging_optimizers.OrthogonalizedOptimizer`. Some compatible inner optimizers may
            # accept `None`, so keep that internal flexibility while narrowing the call type here.
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
        parameter, so no gather is needed. Pass its local value when
        `reconstruct_full_param` is enabled; otherwise pass the sharded parameter
        so the inner optimizer can read shape and tensor-parallel metadata.
        """
        group_kwargs = {k: v for k, v in group.items() if k != "params"}
        param_arg = param.to_local() if self.reconstruct_full_param else param
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
    ) -> tuple[dict[int, torch.Tensor], list[dist.Work], torch.cuda.Event | None]:
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
        completion_event = None
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            if stream is not None:
                assert default_stream is not None
                stream.wait_stream(default_stream)
            works = dist.batch_isend_irecv(ops) if ops else []
            for buf in list(scatter_plan.send_buffers.values()) + list(recv_buffers.values()):
                if stream is not None:
                    buf.record_stream(stream)
            if stream is not None:
                completion_event = torch.cuda.Event()
                completion_event.record(stream)
        return recv_buffers, list(works or []), completion_event

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
    @_always_nvtx_decorator(message="mfsdp_muon_step")
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

            # Phase 1: local pre-NS shards for all matrix params.
            local_shards: list[torch.Tensor] = []
            for param in matrix_params:
                if param.grad is None:
                    local_shards.append(torch.empty(0, dtype=torch.float32, device=self._device()))
                    continue
                local_shards.append(
                    self._compute_orthogonalization_inputs(param, param.grad, group, lr)
                )

            chunks = self._group_updates(matrix_params, local_shards, matrix_plans)
            owners = self._assign_owner_work(matrix_plans, chunks)
            self._owners.update({matrix_indices[k]: v for k, v in owners.items()})

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
            for chunk_indices in chunks:
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

            # Submit all owner-scatter work before applying any received updates so
            # host-side Work.wait() does not stall later chunk launches.
            for state in chunk_states:
                self._issue_boundary_update(state)
            for state in chunk_states:
                self._enqueue_boundary_update(state)
            for state in chunk_states:
                self._wait_for_dist_buffer(state.scatter_works)

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
        `_issue_boundary_update` so fully-local Newton-Schulz overlaps the gathers.
        """
        b_plans = [matrix_plans[i] for i in boundary_indices]
        b_params = [matrix_params[i] for i in boundary_indices]
        b_local = [local_shards[i] for i in boundary_indices]
        b_owners = {i: owners[boundary_indices[i]] for i in range(len(boundary_indices))}
        gather_plan = self._pack_owner_work(b_plans, b_owners, b_local, device, dtype)
        recv_buffers, gather_works, gather_event = self._send_to_owner(
            gather_plan, device, dtype
        )

        # Optionally also gather each rank's local *weight* shard so the owner can
        # reconstruct the full parameter and pass it to `orthogonalize` (some
        # subclasses read `p`). This is a second, parallel P2P round issued on the
        # same owner-comm stream so it overlaps the pre-NS gather; it reuses the
        # same shard plans/owners (only the shard content differs).
        weight_gather_plan = None
        weight_recv_buffers = None
        weight_gather_works = None
        weight_gather_event = None
        if self.reconstruct_full_param:
            weight_local = [p.to_local() for p in b_params]
            weight_dtype = b_params[0].dtype
            weight_gather_plan = self._pack_owner_work(
                b_plans, b_owners, weight_local, device, weight_dtype
            )
            weight_recv_buffers, weight_gather_works, weight_gather_event = self._send_to_owner(
                weight_gather_plan, device, weight_dtype
            )
        return _BoundaryChunkState(
            b_params=b_params,
            b_plans=b_plans,
            b_owners=b_owners,
            gather_plan=gather_plan,
            recv_buffers=recv_buffers,
            gather_works=gather_works,
            gather_event=gather_event,
            device=device,
            dtype=dtype,
            lr=lr,
            group_kwargs=group_kwargs,
            weight_gather_plan=weight_gather_plan,
            weight_recv_buffers=weight_recv_buffers,
            weight_gather_works=weight_gather_works,
            weight_gather_event=weight_gather_event,
        )

    def _issue_boundary_update(self, state: _BoundaryChunkState) -> None:
        """Wait for one owner-gather, orthogonalize, and issue its owner-scatter."""
        b_params = state.b_params
        b_plans = state.b_plans
        b_owners = state.b_owners
        gather_plan = state.gather_plan
        recv_buffers = state.recv_buffers
        device = state.device
        group_kwargs = state.group_kwargs
        this_rank = self._this_rank()

        # Phase 4 (owner): wait only for this chunk's gather rather than the
        # tail of the shared owner-comm stream, then reconstruct and orthogonalize.
        if state.gather_event is not None:
            torch.cuda.current_stream(device).wait_event(state.gather_event)
        self._wait_for_dist_buffer(state.gather_works)
        if state.weight_gather_works:
            if state.weight_gather_event is not None:
                torch.cuda.current_stream(device).wait_event(state.weight_gather_event)
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
            # orthogonalization input (`pre_ns`). The default MCore Muon reads shape
            # and tensor-parallel attributes from `param` but does not read its values,
            # so pass the original sharded parameter unless the caller explicitly asks
            # us to reconstruct the full value.
            if self.reconstruct_full_param and state.weight_gather_plan is not None:
                assert state.weight_recv_buffers is not None
                param_arg = reconstruct_full_tensor(
                    i,
                    plan,
                    state.weight_gather_plan,
                    state.weight_recv_buffers,
                    owner_rank=this_rank,
                )
            else:
                param_arg = b_params[i]
            full_updates[i] = self._orthogonalize_with_precision(param_arg, full, **group_kwargs)

        # Phase 5: pack + P2P-send update shards from owners (async on owner stream).
        scatter_plan = self._pack_update_shards(
            full_updates, b_plans, b_owners, device, state.dtype
        )
        scatter_recv, scatter_works, scatter_event = self._send_to_destination(
            scatter_plan, device, state.dtype
        )
        state.full_updates = full_updates
        state.scatter_plan = scatter_plan
        state.scatter_recv = scatter_recv
        state.scatter_works = scatter_works
        state.scatter_event = scatter_event

    def _enqueue_boundary_update(self, state: _BoundaryChunkState) -> None:
        """Apply local update shards after the asynchronous owner-scatter."""
        # Depend only on this chunk's scatter, not later communication queued
        # on the shared owner-comm stream.
        if state.scatter_event is not None:
            torch.cuda.current_stream(state.device).wait_event(state.scatter_event)
        scatter_plan = cast(OwnerScatterPlan, state.scatter_plan)
        received = self._unpack_update_shards(scatter_plan, state.scatter_recv)

        # Phase 6: apply local update shards.
        this_rank = self._this_rank()
        for i, param in enumerate(state.b_params):
            plan = state.b_plans[i]
            if state.b_owners[i] == this_rank:
                row_start, row_count = plan.rank_rows[this_rank]
                if row_count == 0:
                    continue
                update_shard = state.full_updates[i][row_start : row_start + row_count]
            else:
                update_shard = received.get(i)
                if update_shard is None:
                    continue
            self._apply_update(param, update_shard, state.lr)

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
        max_params_per_owner_chunk: int | None = 4,
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
            num_ns_steps=self._num_ns_steps,
            max_params_per_owner_chunk=max_params_per_owner_chunk,
        )
