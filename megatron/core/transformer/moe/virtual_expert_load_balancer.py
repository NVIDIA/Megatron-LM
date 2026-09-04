# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Virtual-expert load balancing, replica planning, and runtime weight movement.

Every EP rank exposes ``2 * L`` runtime experts to HybridEP: its ``L`` native
experts followed by ``L`` replica slots. Each rank gathers the per-rank expert
histogram and independently derives the same placement. ``experts_to_copy[d, s]``
names the semantic expert whose weights destination ``d`` must hold in slot
``s`` (``-1`` if unused), and ``virtual_experts[token, k]`` rewrites each
semantic route to a rank-major runtime id: native ``d * 2L + local`` or replica
``d * 2L + L + slot``.

The weight bridge materializes a plan. Before expert compute every owner pushes
its selected weights into the peers' replica slots (MXFP8: rowwise storage in
forward, columnwise in backward); after expert backward the replica gradients
are reduced back into the owners' native wgrad staging, which autograd then
hands to the optimizer parameters. The runtime parameters TE executes against
alias the optimizer parameters (or their GTP gathers) for the natives and a
symmetric-memory arena for the slots, so no weight is ever copied locally.
"""

import functools
import gc
import math
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from megatron.core.fp8_utils import is_mxfp8tensor
from megatron.core.jit import jit_fuser

try:
    from megatron.core.transformer.moe.virtual_expert_triton import (
        MAX_REPLICA_WEIGHT_SMS,
        compile_replica_weight_kernels,
        launch_compact_routing_map,
        launch_replica_grad_reduce,
        launch_replica_placement,
        launch_replica_weight_prefetch,
    )

    _TRITON_AVAILABLE = True
except ImportError:
    MAX_REPLICA_WEIGHT_SMS = 32
    compile_replica_weight_kernels = None
    launch_compact_routing_map = None
    launch_replica_grad_reduce = None
    launch_replica_placement = None
    launch_replica_weight_prefetch = None
    _TRITON_AVAILABLE = False
from megatron.core.utils import nvtx_decorator

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig

# Push directions. MXFP8 forward GEMMs read the rowwise components, backward the columnwise.
FORWARD, BACKWARD = 0, 1
_MXFP8_COMPONENTS = (
    "_rowwise_data",
    "_rowwise_scale_inv",
    "_columnwise_data",
    "_columnwise_scale_inv",
)


# --------------------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class ReplicaPlan:
    """``virtual_experts``: int64 ``[num_tokens, router_topk]`` runtime ids;
    ``experts_to_copy``: int32 ``[ep_size, num_local_experts]`` semantic ids, ``-1`` if unused."""

    virtual_experts: torch.Tensor | None
    experts_to_copy: torch.Tensor


@dataclass(slots=True)
class ReplicaPlannerWorkspace:
    """Placement scratch for one ``(num_experts, ep_size)`` shape; every plan overwrites it."""

    num_experts: int
    ep_size: int
    num_local_experts: int
    gathered_counts: torch.Tensor  # [ep_size, num_experts] local routes per (source, expert)
    balance: torch.Tensor  # [ep_size] native load minus rank capacity
    allocation: torch.Tensor  # [num_experts, ep_size] routes of each expert per destination
    placement_grid_sync: torch.Tensor
    # Per-expert destination segment ends in this rank's local ordinal space, padded to a
    # power of two columns.
    destination_boundaries: torch.Tensor
    expert_replica_slots: torch.Tensor  # [num_experts, ep_size] slot holding an expert on a rank
    experts_to_copy: torch.Tensor  # [ep_size, num_local_experts]

    @classmethod
    def allocate(cls, *, num_experts, ep_size, device):
        """Allocate the scratch for one expert layout on ``device``."""
        if min(num_experts, ep_size) <= 0 or num_experts % ep_size:
            raise ValueError(
                "Replica planner needs a positive, even expert distribution, got "
                f"num_experts={num_experts}, ep_size={ep_size}."
            )
        int32 = dict(dtype=torch.int32, device=device)
        return cls(
            num_experts=num_experts,
            ep_size=ep_size,
            num_local_experts=num_experts // ep_size,
            gathered_counts=torch.empty((ep_size, num_experts), **int32),
            balance=torch.empty(ep_size, **int32),
            allocation=torch.empty((num_experts, ep_size), **int32),
            placement_grid_sync=torch.zeros(1, **int32),
            destination_boundaries=torch.empty(
                (num_experts, 1 << (ep_size - 1).bit_length()), **int32
            ),
            expert_replica_slots=torch.empty((num_experts, ep_size), **int32),
            experts_to_copy=torch.empty((ep_size, num_experts // ep_size), **int32),
        )


_planner_workspaces: dict = {}


def get_planner_workspace(*, num_experts: int, ep_size: int, device: torch.device):
    """Return the process-wide placement scratch for one expert layout; planning is
    stream-ordered, so every layer of a device can share it."""
    key = (num_experts, ep_size, device.index)
    workspace = _planner_workspaces.get(key)
    if workspace is None:
        workspace = _planner_workspaces[key] = ReplicaPlannerWorkspace.allocate(
            num_experts=num_experts, ep_size=ep_size, device=device
        )
    return workspace


@jit_fuser
def _map_routes(
    topk_indices: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    destination_boundaries: torch.Tensor,
    expert_replica_slots: torch.Tensor,
    num_local_experts: int,
    ep_size: int,
) -> torch.Tensor:
    """Sort the routes by expert (stably, so each expert's routes keep token order) and read
    every route's destination off one global array of segment ends."""
    flat = topk_indices.reshape(-1)
    sorted_experts, order = torch.sort(flat, stable=True)
    experts = sorted_experts.long()
    # Segment d of expert e ends, in sorted-position space, at the expert's bucket start plus
    # its destination boundary clipped to the routes this rank actually holds.
    bucket_start = torch.cumsum(tokens_per_expert, 0) - tokens_per_expert
    boundaries = (
        destination_boundaries[:, :ep_size].clamp(min=0).minimum(tokens_per_expert[:, None])
    )
    ends = (bucket_start[:, None] + boundaries).reshape(-1)
    positions = torch.arange(flat.numel(), device=flat.device, dtype=torch.int64)
    destination = torch.searchsorted(ends, positions, right=True) - experts * ep_size
    slot = expert_replica_slots.view(-1)[experts * ep_size + destination]
    runtime_local = torch.where(
        destination == experts // num_local_experts,
        experts % num_local_experts,
        num_local_experts + slot,
    )
    virtual = torch.empty_like(positions)
    virtual[order] = destination * (2 * num_local_experts) + runtime_local
    return virtual.view(topk_indices.shape)


def map_routes_to_runtime_experts(
    topk_indices: torch.Tensor, tokens_per_expert: torch.Tensor, workspace: ReplicaPlannerWorkspace
) -> torch.Tensor:
    """Turn this rank's semantic routes into rank-major runtime expert ids under the
    placement held by ``workspace``.

    A route's stable ordinal among this rank's routes to its expert, offset by the routes
    earlier ranks send that expert (already folded into ``destination_boundaries``), selects
    the destination segment; a remote destination runs it in the replica slot the placement
    assigned that expert there.
    """
    return _map_routes(
        topk_indices,
        tokens_per_expert,
        workspace.destination_boundaries,
        workspace.expert_replica_slots,
        workspace.num_local_experts,
        workspace.ep_size,
    )


def extract_semantic_routes(
    routing_map: torch.Tensor, probs: torch.Tensor, router_topk: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compact a dense ``[num_tokens, num_experts]`` routing map into top-k routes.

    Returns ``(token_probs, token_indices, tokens_per_expert)``: the first two are
    ``[num_tokens, router_topk]`` in ascending expert order (gradients flow back to the
    selected entries of ``probs``), the last is the int32 local histogram. The map, not
    the probabilities, is authoritative, so a selected zero-probability route survives.
    """
    num_tokens, num_experts = (int(size) for size in routing_map.shape)
    tokens_per_expert = torch.zeros(num_experts, dtype=torch.int32, device=routing_map.device)
    # Zero-filled so a token with fewer than router_topk selections leaves a route to
    # expert 0 rather than a stale id; the placement kernel's route-total check reports it.
    token_indices = torch.zeros(
        (num_tokens, router_topk), dtype=torch.int32, device=routing_map.device
    )
    launch_compact_routing_map(
        routing_map,
        token_indices,
        tokens_per_expert,
        num_tokens=num_tokens,
        router_topk=router_topk,
        num_experts=num_experts,
    )
    return torch.gather(probs, 1, token_indices.long()), token_indices, tokens_per_expert


def plan_replica_routes(
    topk_indices: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    ep_group: dist.ProcessGroup,
    workspace: ReplicaPlannerWorkspace,
    *,
    on_placement_ready: Callable[[ReplicaPlan], None] | None = None,
) -> ReplicaPlan:
    """Plan deterministic replica placement for one EP group.

    ``topk_indices`` (int32/int64 ``[num_tokens, router_topk]``) and ``tokens_per_expert``
    (int32 ``[num_experts]``) are this rank's semantic routes and histogram; every rank must
    route the same number of tokens. ``on_placement_ready`` runs as soon as
    ``experts_to_copy`` is final so the weight push can start ahead of the route mapping.
    """
    ep_size = dist.get_world_size(group=ep_group)
    if (
        (tokens_per_expert.numel(), ep_size) != (workspace.num_experts, workspace.ep_size)
        or topk_indices.dtype not in (torch.int32, torch.int64)
        or tokens_per_expert.dtype != torch.int32
        or not (topk_indices.is_contiguous() and tokens_per_expert.is_contiguous())
    ):
        raise ValueError("Replica planner inputs do not match the workspace shape or dtypes.")
    # The only cross-rank input; from here every rank computes the same placement.
    dist.all_gather_into_tensor(
        workspace.gathered_counts.view(-1), tokens_per_expert, group=ep_group
    )
    launch_replica_placement(
        workspace.gathered_counts,
        workspace.balance,
        workspace.allocation,
        workspace.destination_boundaries,
        workspace.experts_to_copy,
        workspace.expert_replica_slots,
        workspace.placement_grid_sync,
        rank_route_capacity=topk_indices.numel(),
        source_rank=dist.get_rank(group=ep_group),
        ep_size=ep_size,
        num_experts=workspace.num_experts,
        num_local_experts=workspace.num_local_experts,
    )
    # A plan owns its outputs: the backward push and reduction read experts_to_copy and
    # autograd saves virtual_experts, possibly past another forward of the same layer.
    plan = ReplicaPlan(None, workspace.experts_to_copy.clone())
    if on_placement_ready is not None:
        on_placement_ready(plan)
    plan.virtual_experts = map_routes_to_runtime_experts(topk_indices, tokens_per_expert, workspace)
    return plan


def map_replica_plan_to_hybridep(
    plan: ReplicaPlan, topk_probs: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter compact virtual routes into HybridEP's dense ``[num_tokens, num_experts]``
    boolean routing map and float32 probabilities (``num_experts`` = runtime experts)."""
    if plan.virtual_experts.shape != topk_probs.shape:
        raise ValueError(
            "Replica virtual experts and top-k probabilities must have the same shape, got "
            f"{tuple(plan.virtual_experts.shape)} and {tuple(topk_probs.shape)}."
        )
    dense_shape = (int(plan.virtual_experts.shape[0]), num_experts)
    routing_map = torch.zeros(dense_shape, dtype=torch.bool, device=plan.virtual_experts.device)
    dense_probs = torch.zeros(dense_shape, dtype=torch.float32, device=topk_probs.device)
    routing_map.scatter_(1, plan.virtual_experts, True)
    dense_probs.scatter_(1, plan.virtual_experts, topk_probs.to(torch.float32))
    return routing_map, dense_probs


# --------------------------------------------------------------------------------------
# Weight bridge
# --------------------------------------------------------------------------------------


def _wrap_mxfp8(template, shape, views, device) -> tuple[torch.Tensor, ...]:
    """Wrap raw ``(rowwise, rowwise_scale, columnwise, columnwise_scale)`` views as
    MXFP8 tensors carrying ``template``'s quantization metadata."""
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

    return tuple(
        MXFP8Tensor(
            shape=shape,
            dtype=template.dtype,
            rowwise_data=rowwise,
            rowwise_scale_inv=rowwise_scale,
            columnwise_data=columnwise,
            columnwise_scale_inv=columnwise_scale,
            fp8_dtype=template._fp8_dtype,
            quantizer=template._quantizer,
            with_gemm_swizzled_scales=template._with_gemm_swizzled_scales,
            requires_grad=False,
            device=device,
        )
        for rowwise, rowwise_scale, columnwise, columnwise_scale in views
    )


class _ReplicaWeightWorkspace:
    """Symmetric arenas and streams shared by every replica MoE layer of one EP group.

    The weight arena holds ``fc1 data, fc1 scales, fc2 data, fc2 scales`` with ``L``
    members per section (the scale sections are empty for BF16; MXFP8 keeps one because
    only one GEMM orientation is live at a time). The gradient arena holds ``fc1, fc2``.
    One layer's replicas are live at a time: the planner's histogram all-gather orders
    every push after the expert GEMMs that read the previous contents, and the
    reduction's exit rendezvous orders every slot rewrite after the owners' reads.
    """

    def __init__(self, group, device, config: tuple) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        self.config = config
        world_size, num_local_experts, member_shapes, mxfp8, grad_dtype, num_sms = config
        self.num_local_experts = num_local_experts
        self.member_shapes = member_shapes
        self.member_numels = tuple(math.prod(shape) for shape in member_shapes)
        self.mxfp8 = mxfp8
        self.grad_dtype = grad_dtype
        self.num_sms = num_sms
        # One E8M0 scale byte per 32 MXFP8 weight bytes, unpadded (the config requires
        # 128-aligned projections so TE's padded scale layout has this exact size).
        self.scale_numels = tuple(numel // 32 if mxfp8 else 0 for numel in self.member_numels)
        arena_numel = num_local_experts * sum(self.member_numels)
        try:
            # NCCL window registration needs the device communicator; create it here,
            # before training or graph capture. The backend choice is process-global.
            dist.barrier(group=group, device_ids=[device.index])
            if not group._get_backend(torch.device("cuda"))._comm_ptr():
                raise RuntimeError("ProcessGroupNCCL returned an invalid communicator pointer.")
            if symm_mem.get_backend(device) != "NCCL":
                symm_mem.set_backend("NCCL")
            self.weight_arena = symm_mem.empty(
                arena_numel + num_local_experts * sum(self.scale_numels),
                dtype=torch.uint8 if mxfp8 else torch.bfloat16,
                device=device,
            )
            self.weight_handle = symm_mem.rendezvous(self.weight_arena, group)
            self.grad_arena = symm_mem.empty(arena_numel, dtype=grad_dtype, device=device)
            self.grad_handle = symm_mem.rendezvous(self.grad_arena, group)
        except RuntimeError as exc:
            raise RuntimeError(
                "Replica weights could not allocate NCCL symmetric memory for the EP group; "
                "the EP group must lie within one NVLink domain."
            ) from exc
        self.weight_arena.zero_()
        self.grad_arena.zero_()
        self.weight_grid_barrier = torch.zeros(1, dtype=torch.int32, device=device)
        self.grad_grid_barrier = torch.zeros(1, dtype=torch.int32, device=device)
        # Two candidate weight streams: a CUDA-graph capture stream comes from the same
        # pool and may alias one of them.
        self.weight_streams = (torch.cuda.Stream(device=device), torch.cuda.Stream(device=device))
        self.grad_stream = torch.cuda.Stream(device=device)
        # Full native wgrad staging per projection: TE's GEMM overwrites it, the
        # reduction adds the replica partials, autograd hands it to the optimizer.
        self.native_grads = tuple(
            torch.empty((num_local_experts, *shape), dtype=grad_dtype, device=device)
            for shape in member_shapes
        )
        compile_replica_weight_kernels(
            world_size=world_size,
            num_local_experts=num_local_experts,
            member_numels=self.member_numels,
            num_sms=num_sms,
            device_index=device.index,
            grad_dtype=grad_dtype,
            mxfp8=mxfp8,
        )
        # No rank may enter the device-side rendezvous before every peer can launch.
        dist.barrier(group=group, device_ids=[device.index])

    def weight_stream(self, current_stream: torch.cuda.Stream) -> torch.cuda.Stream:
        """Return a weight stream distinct from ``current_stream``."""
        return next(s for s in self.weight_streams if s.cuda_stream != current_stream.cuda_stream)

    def slot_views(self, projection: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return the ``[L, *shape]`` weight slots of one projection and, for MXFP8, their
        ``[L, numel // 32]`` scale bytes."""
        count, shape = self.num_local_experts, self.member_shapes[projection]
        numel, scale_numel = self.member_numels[projection], self.scale_numels[projection]
        offset = count * sum(self.member_numels[:projection]) + count * sum(
            self.scale_numels[:projection]
        )
        data = self.weight_arena.narrow(0, offset, count * numel).view(count, *shape)
        if not self.mxfp8:
            return data, None
        scales = self.weight_arena.narrow(0, offset + count * numel, count * scale_numel)
        return data, scales.view(count, scale_numel)

    def grad_slots(self, projection: int) -> torch.Tensor:
        """Return the ``[L, *shape]`` replica gradient slots of one projection."""
        count, numel = self.num_local_experts, self.member_numels[projection]
        offset = count * sum(self.member_numels[:projection])
        return self.grad_arena.narrow(0, offset, count * numel).view(
            count, *self.member_shapes[projection]
        )

    def destroy(self) -> None:
        """Drop the NCCL window registrations while the process group is still alive."""
        if self.weight_arena is not None:
            torch.cuda.synchronize(self.weight_arena.device)
        self.weight_handle = self.grad_handle = self.weight_arena = self.grad_arena = None


_workspaces: dict = {}
_bridges = weakref.WeakSet()


def _get_workspace(group, device, config: tuple) -> _ReplicaWeightWorkspace:
    key = (id(group), device.index)
    workspace = _workspaces.get(key)
    if workspace is None:
        workspace = _workspaces[key] = _ReplicaWeightWorkspace(group, device, config)
    elif workspace.config != config:
        raise ValueError(
            "All replica MoE layers on an EP group must share one weight shape and launch "
            f"configuration; expected {workspace.config}, got {config}."
        )
    return workspace


def finalize_replica_weight_bridges() -> None:
    """Release every replica arena before its process group is destroyed."""
    for bridge in list(_bridges):
        bridge.destroy()
    for workspace in _workspaces.values():
        workspace.destroy()
    _workspaces.clear()
    # Symmetric-memory handles sit in reference cycles; deregister them now.
    gc.collect()


def _drop_grad(parameter: torch.nn.Parameter) -> None:
    """TE returns a dummy leaf grad once the fused wgrad is in ``main_grad``; drop it."""
    parameter.grad = None


class _ReplicaProjection:
    """One projection's optimizer parameters, runtime parameters and pointer tables.

    The ``2L`` runtime parameters are the natives followed by the replica slots. Their
    ``main_grad`` is the native staging or the slot's gradient arena member and carries
    ``overwrite_main_grad``, so TE's wgrad GEMM rewrites every member on each backward
    and the slots never need clearing (a planned slot always receives tokens).
    """

    def __init__(self, name, parameters, workspace: _ReplicaWeightWorkspace, index: int):
        self.name = name
        self.parameters = parameters
        self.index = index
        self.mxfp8 = workspace.mxfp8
        self.member_shape = workspace.member_shapes[index]
        self.device = parameters[0].device
        self.gtp_leader = (
            parameters[0] if getattr(parameters[0], "is_gtp_weight_remat", False) else None
        )
        self.virtual_grad = workspace.grad_slots(index)
        self.native_grad = workspace.native_grads[index]
        self.native_grad_bases = torch.tensor(
            [grad.data_ptr() for grad in self.native_grad], dtype=torch.int64, device=self.device
        )
        data, scales = workspace.slot_views(index)
        if self.mxfp8:
            template = parameters[0]
            if self.gtp_leader is not None:
                rowwise, columnwise = (
                    template._gtp_gather_quantizer.get_scale_shape(self.member_shape, columnwise=c)
                    for c in (False, True)
                )
            else:
                rowwise, columnwise = (
                    template._rowwise_scale_inv.shape,
                    template._columnwise_scale_inv.shape,
                )
            views = tuple(
                (data[i], scales[i].view(rowwise), data[i], scales[i].view(columnwise))
                for i in range(len(parameters))
            )
            self.virtual_weights = _wrap_mxfp8(template, self.member_shape, views, self.device)
            # GTP gathers hold one orientation; alias them into shells that always carry
            # both, so TE sees complete tensors while the live orientation stays exact.
            natives = _wrap_mxfp8(template, self.member_shape, views, self.device)
        else:
            self.virtual_weights = tuple(data)
            natives = (torch.empty(0, dtype=torch.bfloat16, device=self.device),) * len(data)
        sources = natives if self.gtp_leader is not None else parameters
        self.runtime_parameters = tuple(
            torch.nn.Parameter(weight) for weight in (*sources, *self.virtual_weights)
        )
        for parameter, grad in zip(
            self.runtime_parameters, (*self.native_grad, *self.virtual_grad)
        ):
            parameter.main_grad = grad
            parameter.grad_added_to_main_grad = True
            parameter.overwrite_main_grad = True
            parameter.register_post_accumulate_grad_hook(_drop_grad)
        # Per direction: the ``[components, L]`` device table the push reads (data, then
        # MXFP8 scales), its pinned mirror, the bound pointers and the copy's completion.
        rows = 2 if self.mxfp8 else 1
        self.tables = tuple(
            torch.empty((rows, len(parameters)), dtype=torch.int64, device=self.device)
            for _ in range(2)
        )
        self.host_tables = tuple(
            torch.empty((rows, len(parameters)), dtype=torch.int64, pin_memory=True)
            for _ in range(2)
        )
        self.bound = [None, None]
        self.copied = [torch.cuda.Event(), torch.cuda.Event()]

    def _components(self, direction: int) -> tuple[str, ...]:
        return _MXFP8_COMPONENTS[2 * direction : 2 * direction + 2] if self.mxfp8 else ("data",)

    def prepare(self, direction: int) -> None:
        """Materialize the source weights of ``direction`` and bind them for the push."""
        if self.gtp_leader is None:
            sources = self.parameters
        else:
            gathered = (
                self.gtp_leader.materialize_group_for_backward()
                if direction == BACKWARD
                else self.gtp_leader.materialize_group_for_forward()
            )
            sources = tuple(gathered) if isinstance(gathered, (list, tuple)) else (gathered,)
        components = self._components(direction)
        pointers = tuple(
            tuple(getattr(source, name).data_ptr() for name in components) for source in sources
        )
        if pointers == self.bound[direction]:
            return
        # A rebind is the exception (GTP gathers land in stable buffers); validate
        # the storage and refresh the table and the runtime parameters it describes.
        numel = math.prod(self.member_shape)
        expected = (
            ((numel, torch.uint8), (numel // 32, torch.uint8))
            if self.mxfp8
            else ((numel, torch.bfloat16),)
        )
        for source in sources:
            for name, (count, dtype) in zip(components, expected):
                storage = getattr(source, name)
                if (
                    not storage.is_contiguous()
                    or storage.numel() != count
                    or storage.dtype != dtype
                ):
                    raise ValueError(
                        f"{self.name} replica source {name} must be contiguous {dtype} with "
                        f"{count} elements, got {storage.dtype} {tuple(storage.shape)}."
                    )
        for parameter, source in zip(self.runtime_parameters, sources):
            if self.mxfp8:
                for name in components:
                    setattr(parameter, name, getattr(source, name))
            else:
                parameter.data = source
        # The pinned mirror may only be rewritten once its previous copy has landed.
        self.copied[direction].synchronize()
        self.host_tables[direction].copy_(torch.tensor(pointers, dtype=torch.int64).t())
        self.tables[direction].copy_(self.host_tables[direction], non_blocking=True)
        self.copied[direction].record(torch.cuda.current_stream(self.device))
        self.bound[direction] = pointers

    def destroy(self) -> None:
        for parameter in self.runtime_parameters:
            parameter.main_grad = None


class ReplicaWeightBridge:
    """Asynchronous replica weight push and gradient reduction for one MoE layer."""

    def __init__(
        self,
        *,
        experts: torch.nn.Module,
        group: dist.ProcessGroup,
        num_local_experts: int,
        grad_dtype: torch.dtype = torch.float32,
        num_sms: int | None = None,
    ) -> None:
        self.group = group
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self.num_local_experts = num_local_experts
        self.num_runtime_experts = 2 * num_local_experts
        self._experts_ref = weakref.ref(experts)
        self.last_plan = None
        self._prefetch_plan = None
        self._completed_plan = None
        self._backward_plan = None
        self._reduced: set[int] = set()
        self._destroyed = False

        linears = (experts.linear_fc1, experts.linear_fc2)
        parameters = tuple(
            tuple(linear.get_parameter(f"weight{i}") for i in range(num_local_experts))
            for linear in linears
        )
        member_shapes = tuple(
            (int(linear.out_features), int(linear.in_features)) for linear in linears
        )
        self.device = parameters[0][0].device
        config = (
            self.world_size,
            num_local_experts,
            member_shapes,
            is_mxfp8tensor(parameters[0][0]),
            grad_dtype,
            min(32 if num_sms is None else int(num_sms), MAX_REPLICA_WEIGHT_SMS),
        )
        self.workspace = _get_workspace(group, self.device, config)
        self.projections = [
            _ReplicaProjection(f"FC{i + 1}", parameters[i], self.workspace, i) for i in range(2)
        ]
        # CUDA events are created lazily on first record; materialize them before
        # training or graph capture.
        self.prefetch_done = torch.cuda.Event()
        self.grad_reduce_done = (torch.cuda.Event(), torch.cuda.Event())
        for event in (self.prefetch_done, *self.grad_reduce_done):
            event.record(torch.cuda.current_stream(self.device))
        _bridges.add(self)

    @property
    def runtime_fc1_weights(self) -> tuple[torch.nn.Parameter, ...]:
        """Native-then-replica FC1 runtime parameters."""
        return self.projections[0].runtime_parameters

    @property
    def runtime_fc2_weights(self) -> tuple[torch.nn.Parameter, ...]:
        """Native-then-replica FC2 runtime parameters."""
        return self.projections[1].runtime_parameters

    @property
    def source_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        """The optimizer-owned FC1 then FC2 parameters."""
        return tuple(parameter for p in self.projections for parameter in p.parameters)

    @torch.no_grad()
    @nvtx_decorator(message="replica_weight_push_start")
    def start_prefetch(self, plan: ReplicaPlan, direction: int = FORWARD) -> None:
        """Enqueue the owner push of the plan's FC1/FC2 weights on the weight stream."""
        if self._prefetch_plan is not None:
            raise RuntimeError("Replica weight prefetch is already outstanding.")
        if direction == FORWARD:
            # DDP/FSDP parameter hooks (all-gathers) must run before the push reads them.
            self._experts_ref().prepare_fused_impl_parameters()
        # Expert backward computes FC2 before FC1; keep GTP's linked gathers in that order.
        for projection in self.projections[:: -1 if direction == BACKWARD else 1]:
            projection.prepare(direction)
        workspace = self.workspace
        current_stream = torch.cuda.current_stream(self.device)
        weight_stream = workspace.weight_stream(current_stream)
        weight_stream.wait_stream(current_stream)
        tables = tuple(projection.tables[direction] for projection in self.projections)
        with torch.cuda.stream(weight_stream):
            launch_replica_weight_prefetch(
                sources=tuple(table[0] for table in tables),
                scale_sources=tuple(table[1] for table in tables) if workspace.mxfp8 else None,
                arena=workspace.weight_arena,
                peer_bases=workspace.weight_handle.buffer_ptrs_dev,
                signal_bases=workspace.weight_handle.signal_pad_ptrs_dev,
                experts_to_copy=plan.experts_to_copy,
                grid_barrier=workspace.weight_grid_barrier,
                rank=self.rank,
                world_size=self.world_size,
                num_local_experts=self.num_local_experts,
                member_numels=workspace.member_numels,
                num_sms=workspace.num_sms,
            )
            self.prefetch_done.record(weight_stream)
        self._prefetch_plan = plan

    @torch.no_grad()
    @nvtx_decorator(message="replica_weight_push_wait")
    def wait_prefetch(self, plan: ReplicaPlan) -> None:
        """Make the current stream wait for the push of ``plan``."""
        if self._prefetch_plan is None:
            # Waiting again for the resident plan is a no-op; anything else never started.
            if plan is None or plan is not self._completed_plan:
                raise RuntimeError("Replica weights require a started prefetch before use.")
        elif self._prefetch_plan is not plan:
            raise RuntimeError("Replica weight prefetch plan changed while outstanding.")
        torch.cuda.current_stream(self.device).wait_event(self.prefetch_done)
        self._completed_plan, self._prefetch_plan = plan, None

    def wait_prefetch_for_backward(self, plan: ReplicaPlan) -> None:
        """Wait for the backward push and remember the plan the expert backward reduces."""
        self.wait_prefetch(plan)
        self._backward_plan = plan

    @torch.no_grad()
    @nvtx_decorator(message="replica_grad_reduce_start")
    def start_grad_reduce(self, projection: int) -> None:
        """Enqueue the replica-gradient reduction of one projection (0 = FC1, 1 = FC2)."""
        if self._backward_plan is None:
            raise RuntimeError("Replica gradient reduction needs the backward plan.")
        if projection in self._reduced:
            raise RuntimeError(f"Replica gradient reduction of FC{projection + 1} started twice.")
        workspace = self.workspace
        workspace.grad_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(workspace.grad_stream):
            launch_replica_grad_reduce(
                arena=workspace.grad_arena,
                native_grads=tuple(p.native_grad_bases for p in self.projections),
                peer_bases=workspace.grad_handle.buffer_ptrs_dev,
                signal_bases=workspace.grad_handle.signal_pad_ptrs_dev,
                experts_to_copy=self._backward_plan.experts_to_copy,
                grid_barrier=workspace.grad_grid_barrier,
                rank=self.rank,
                world_size=self.world_size,
                num_local_experts=self.num_local_experts,
                member_numels=workspace.member_numels,
                num_sms=workspace.num_sms,
                projections=(projection,),
            )
            self.grad_reduce_done[projection].record(workspace.grad_stream)
        self._reduced.add(projection)

    def start_fc2_grad_reduce(self) -> None:
        """Start FC2's reduction from the expert backward, right behind FC2's wgrad GEMM."""
        self.start_grad_reduce(1)

    def start_pending_grad_reduces(self, plan: ReplicaPlan) -> None:
        """Start every reduction not yet started, FC2 first, once dispatch backward is done.

        FC2 normally starts from the FC2 op's wgrad store; FC1 starts here and hides behind
        the latent, shared-expert and router backward.
        """
        if plan is not self._backward_plan:
            raise RuntimeError("Replica gradient reduction is outstanding for another plan.")
        for projection in (1, 0):
            if projection not in self._reduced:
                self.start_grad_reduce(projection)

    @torch.no_grad()
    @nvtx_decorator(message="replica_grad_reduce_wait")
    def wait_grad_reduce(self, plan: ReplicaPlan) -> tuple[torch.Tensor | None, ...]:
        """Finish both reductions and return one full wgrad per source parameter.

        GTP parameters reduce-scatter the staging themselves (FC2 first, as their linked
        reduce-scatter chain expects) and return what their protocol returns.
        """
        if plan is not self._backward_plan or self._reduced != {0, 1}:
            raise RuntimeError("Replica gradient reduction of both projections must be started.")
        current_stream = torch.cuda.current_stream(self.device)
        for event in self.grad_reduce_done:
            current_stream.wait_event(event)
        self._backward_plan = None
        self._reduced.clear()
        grads = [tuple(projection.native_grad) for projection in self.projections]
        for index in (1, 0):
            leader = self.projections[index].gtp_leader
            if leader is not None:
                reduced = leader.wgrad_reduce_scatter(list(grads[index]))
                grads[index] = tuple(reduced) if isinstance(reduced, (list, tuple)) else (reduced,)
        return tuple(grad for projection in grads for grad in projection)

    def destroy(self) -> None:
        """Detach the layer's runtime parameters from the shared arenas."""
        if self._destroyed:
            return
        experts = self._experts_ref()
        if experts is not None:
            experts._fused_ops = None
            experts._replica_weight_bridge = None
        for projection in self.projections:
            projection.destroy()
        self.projections.clear()
        self.last_plan = self.workspace = None
        self._destroyed = True
        _bridges.discard(self)


class _ReplicaBackwardHook(torch.autograd.Function):
    """Run ``hook()`` when the gradient passes this point; the gradient itself is unchanged."""

    @staticmethod
    def forward(ctx, tensor, hook):
        ctx.hook = hook
        return tensor

    @staticmethod
    def backward(ctx, grad):
        ctx.hook()
        return grad, None


class _ReplicaWaitGradReduce(torch.autograd.Function):
    """Finish the replica reductions and hand the wgrads to the source parameters.

    Applied to the MoE layer input so its backward runs after every consumer of the
    input (router, shared experts, latent projection). ``context.plan`` is filled in
    by the dispatcher once routing has produced the plan.
    """

    @staticmethod
    def forward(ctx, hidden_states, *args):
        ctx.bridge, ctx.context = args[-2:]
        return hidden_states

    @staticmethod
    def backward(ctx, grad_hidden_states):
        from transformer_engine.pytorch.module.base import get_dummy_wgrad

        grads = []
        wgrads = ctx.bridge.wait_grad_reduce(ctx.context.plan)
        for parameter, wgrad in zip(ctx.bridge.source_parameters, wgrads):
            if wgrad is None or getattr(parameter, "is_gtp_weight_remat", False):
                grads.append(wgrad)  # GTP already reduce-scattered into the shard
            elif getattr(parameter, "main_grad", None) is None:
                grads.append(wgrad.clone())  # no fused accumulation: autograd owns a copy
            else:
                # Accumulate in main_grad's dtype and return a dummy so AccumulateGrad
                # still fires DDP's grad-ready hook without adding the dummy again.
                parameter.main_grad.add_(wgrad)
                parameter.grad_added_to_main_grad = True
                grads.append(
                    get_dummy_wgrad(
                        list(parameter.shape),
                        parameter.dtype,
                        zero=getattr(parameter, "zero_out_wgrad", False),
                    )
                )
        return (grad_hidden_states, *grads, None, None)


class VirtualExpertLoadBalancer:
    """Mixin that plans virtual experts and manages their runtime weights.

    Dispatch managers provide the transport-specific glue: they feed semantic routes into
    :meth:`setup_virtual_expert_metadata`, install the mapped routes returned by
    :meth:`prepare_virtual_expert_dispatch`, and bracket their combine operation with the
    corresponding pre/post helpers.
    """

    def initialize_virtual_expert_load_balancer(
        self,
        *,
        group: torch.distributed.ProcessGroup,
        num_local_experts: int,
        router_topk: int,
        num_experts: int,
        config: "TransformerConfig",
    ) -> None:
        """Initialize virtual-expert state without initializing the transport parent."""
        if not _TRITON_AVAILABLE:
            raise ImportError("--moe-virtual-expert-load-balance requires Triton.")
        world_size = torch.distributed.get_world_size(group=group)
        if num_experts != world_size * num_local_experts:
            raise ValueError(
                "Virtual-expert load balancing requires an even expert distribution: "
                f"num_experts={num_experts}, world_size={world_size}, "
                f"num_local_experts={num_local_experts}."
            )
        self.group = group
        self.config = config
        self.router_topk = router_topk
        self.semantic_num_experts = num_experts
        self.num_owned_experts = num_local_experts
        self.semantic_token_probs = None
        self.semantic_token_indices = None
        self.semantic_tokens_per_expert = None
        self._bridge = None
        self._plan = None
        self._context = None

    def bind_experts(self, experts: torch.nn.Module) -> None:
        """Bind the expert module and its optimizer-owned weights to runtime expert slots."""
        self._bridge = ReplicaWeightBridge(
            experts=experts,
            group=self.group,
            num_local_experts=self.num_owned_experts,
            grad_dtype=torch.bfloat16 if self.config.grad_reduce_in_bf16 else torch.float32,
            num_sms=self.config.moe_flex_dispatcher_num_sms,
        )
        experts.set_replica_weight_bridge(self._bridge)

    def wrap_layer_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Attach the replica-gradient completion hook to the whole MoE layer input."""
        if self._context is not None:
            raise RuntimeError("Virtual-expert layer input wrapped twice without a combine.")
        self._context = SimpleNamespace(plan=None)
        return _ReplicaWaitGradReduce.apply(
            hidden_states, *self._bridge.source_parameters, self._bridge, self._context
        )

    def setup_virtual_expert_metadata(self, routing_map: torch.Tensor, probs: torch.Tensor) -> None:
        """Extract compact semantic routes for the placement planner."""
        num_tokens = int(routing_map.shape[0])
        self.semantic_token_probs, self.semantic_token_indices, self.semantic_tokens_per_expert = (
            extract_semantic_routes(
                routing_map.reshape(num_tokens, self.semantic_num_experts),
                probs.reshape(num_tokens, self.semantic_num_experts),
                self.router_topk,
            )
        )
        self.num_local_tokens = num_tokens
        self.token_probs = self.semantic_token_probs

    def plan_dispatch(self, hidden_states: torch.Tensor) -> None:
        """Plan routes and begin the weight push before shared-expert compute."""
        if self._bridge is None or self._context is None or self._plan is not None:
            raise RuntimeError(
                "Virtual-expert planning needs bound experts, a wrapped layer input and a "
                "combined previous dispatch."
            )
        workspace = get_planner_workspace(
            num_experts=self.semantic_num_experts,
            ep_size=torch.distributed.get_world_size(group=self.group),
            device=hidden_states.device,
        )
        self._plan = self._context.plan = plan_replica_routes(
            self.semantic_token_indices,
            self.semantic_tokens_per_expert,
            self.group,
            workspace,
            on_placement_ready=self._start_prefetch,
        )

    def _start_prefetch(self, plan) -> None:
        self._bridge.last_plan = plan
        self._bridge.start_prefetch(plan)

    def prepare_virtual_expert_dispatch(
        self, hidden_states: torch.Tensor, *, num_runtime_experts: int, alignment: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Map planned routes to runtime experts and attach dispatch-backward work."""
        plan = self._plan
        if plan is None:
            raise RuntimeError("Virtual-expert dispatch requires plan_dispatch to run first.")
        routing_map, token_probs = map_replica_plan_to_hybridep(
            plan, self.semantic_token_probs, num_experts=self.semantic_num_experts * 2
        )
        num_permuted_tokens = self._get_rank_capacity(
            num_tokens=self.num_local_tokens,
            router_topk=self.router_topk,
            capacity_factor=self.config.moe_expert_rank_capacity_factor,
            num_runtime_experts=num_runtime_experts,
            alignment=alignment,
        )
        hidden_states = _ReplicaBackwardHook.apply(
            hidden_states, functools.partial(self._bridge.start_pending_grad_reduces, plan)
        )
        return hidden_states, routing_map, token_probs, num_permuted_tokens

    def prepare_virtual_expert_combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Attach the backward weight-push wait before transport combine."""
        if self._plan is None:
            raise RuntimeError("Virtual-expert combine requires a matching dispatch plan.")
        return _ReplicaBackwardHook.apply(
            hidden_states, functools.partial(self._bridge.wait_prefetch_for_backward, self._plan)
        )

    def finish_virtual_expert_combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Release transport routing metadata after combine."""
        self.token_probs = self.routing_map = None
        return hidden_states

    def finalize_output(self, output: torch.Tensor) -> torch.Tensor:
        """Start the backward weight push from the MoE layer output."""
        plan, self._plan, self._context = self._plan, None, None
        if plan is None:
            raise RuntimeError("Virtual-expert output finalization requires a combined plan.")
        return _ReplicaBackwardHook.apply(
            output, functools.partial(self._bridge.start_prefetch, plan, BACKWARD)
        )

    @staticmethod
    def _get_rank_capacity(
        *,
        num_tokens: int,
        router_topk: int,
        capacity_factor: float,
        num_runtime_experts: int,
        alignment: int,
    ) -> int:
        """Return a static, dropless route capacity for one transport rank."""
        num_routes = num_tokens * router_topk
        rank_capacity = int(num_routes * capacity_factor)
        if alignment > 1:
            rank_capacity = max(
                rank_capacity, num_routes + num_runtime_experts * (alignment - 1)
            )
            rank_capacity += -rank_capacity % alignment
        return rank_capacity
