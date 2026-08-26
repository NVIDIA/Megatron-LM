# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Deterministic replica planning for external expert-parallel transports.

The planner implements deterministic semantic placement without constructing
any transport-specific dispatch-buffer layout. Every rank gathers its fixed-size
expert histogram, independently computes the same placement, and emits
rank-major virtual expert ids plus the replica weights required by each rank.

The dimensions used throughout this file are ``num_tokens`` on one EP rank,
``router_topk`` routes per token, ``num_routes = num_tokens * router_topk``,
``num_experts`` semantic model experts, ``ep_size`` ranks, and
``num_experts_per_gpu = num_experts / ep_size`` native experts and replica
slots on each rank.

Planning has two related outputs. ``experts_to_copy[destination, slot]`` says
which semantic expert's weights must be copied into a destination rank's
replica slot. ``virtual_experts[token, k]`` rewrites every semantic route to a
rank-major runtime expert id understood by HybridEP or NCCL-EP. Each rank has
``2 * num_experts_per_gpu`` runtime experts: its native experts followed by an
equal number of replica slots.

At a high level every rank performs the same deterministic procedure:

1. Gather per-rank expert histograms and compute global expert totals.
2. Measure how far each native expert group is above or below rank capacity.
3. Pair overloaded groups with ranks that have room and assign migration
   quotas using deterministic tie-breaking rules.
4. Split experts across those quotas, then choose the replica weights needed
   by every destination rank.
5. Give each route a stable per-expert ordinal and use the allocation matrix
   to map that ordinal to a native expert or replica slot.

All scratch and output storage is supplied by ``ReplicaPlannerWorkspace``.
After its one-time distributed shape check, the hot path performs no tensor
allocation and can be captured in a CUDA graph.
"""

import gc
import math
import os
import weakref
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist

from megatron.core.fp8_utils import is_mxfp8tensor
from megatron.core.transformer.moe.replica_weight_cutedsl import (
    MAX_REPLICA_WEIGHT_SMS,
    compile_replica_weight_kernels,
    launch_replica_grad_reduce,
    launch_replica_weight_prefetch,
)
from megatron.core.utils import nvtx_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    from unittest.mock import MagicMock

    from megatron.core.utils import null_decorator

    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
    HAVE_TRITON = False


def _discard_runtime_parameter_grad(parameter: torch.nn.Parameter) -> None:
    """Drop TE's throwaway leaf grad after its fused wgrad reaches ``main_grad``."""
    parameter.grad = None


@dataclass(frozen=True, slots=True)
class ReplicaPlan:
    """Transport-facing result of replica placement planning.

    Attributes:
        virtual_experts: Rank-major runtime expert ids with shape
            ``[num_tokens, router_topk]``.
            A native id is ``destination * (2 * num_experts_per_gpu) +
            local_expert``; a replica id adds ``num_experts_per_gpu`` and the
            replica slot instead.
        experts_to_copy: Semantic expert ids assigned to each rank's replica
            slots, with shape ``[ep_size, num_experts_per_gpu]``. Unused slots
            contain ``-1``.
    """

    virtual_experts: torch.Tensor
    experts_to_copy: torch.Tensor


@dataclass(slots=True)
class ReplicaPlannerWorkspace:
    """Fixed-address scratch and output tensors for one planner shape.

    A workspace belongs to one fixed ``(num_tokens, router_topk, num_experts,
    ep_size)`` shape and CUDA device.
    Reusing it is what makes planner tensor addresses stable for CUDA graphs.
    ``ReplicaPlan`` returns views of the output fields below, so callers must
    consume a plan before invoking the planner again with the same workspace.
    """

    num_tokens: int
    router_topk: int
    num_experts: int
    ep_size: int
    num_local_experts: int
    # Global routing state. gathered_counts[source, expert] is the number of
    # local routes to expert on source; allocation[expert, destination] is the
    # final partition of that expert's global route stream across ranks.
    gathered_counts: torch.Tensor
    expert_totals: torch.Tensor
    balance: torch.Tensor
    receiver_quotas: torch.Tensor
    allocation: torch.Tensor
    # Inverse replica lookup produced by fused placement. This turns the
    # route mapper's replica-slot search into one indexed load.
    expert_replica_slots: torch.Tensor
    # Stable expert-bucket order of the original flattened route positions.
    sorted_route_indices: torch.Tensor
    sort_route_metadata: torch.Tensor
    sort_partition_counts: torch.Tensor
    sort_grid_sync: torch.Tensor
    sort_stream: torch.cuda.Stream
    local_count_cumsum: torch.Tensor
    # Planner outputs; these buffers are returned directly in ReplicaPlan.
    virtual_experts: torch.Tensor
    experts_to_copy: torch.Tensor
    # The object collective is a one-time validation and cannot be captured.
    # Once true, subsequent calls use only tensor collectives on the hot path.
    distributed_shape_validated: bool

    @classmethod
    def allocate(
        cls,
        *,
        num_tokens: int,
        router_topk: int,
        num_experts: int,
        ep_size: int,
        device: torch.device,
    ) -> "ReplicaPlannerWorkspace":
        """Allocate a reusable planner workspace for one fixed route shape.

        Args:
            num_tokens: Number of local tokens ``S`` on each EP rank.
            router_topk: Number of routes ``K`` selected for each token.
            num_experts: Total number of semantic experts in the model.
            ep_size: Number of ranks in the expert-parallel group.
            device: CUDA device on which all scratch and output tensors are
                allocated.

        Returns:
            A workspace containing fixed-address buffers sized for
            ``num_tokens * router_topk`` routes and
            ``num_experts / ep_size`` replica slots per rank.
        """
        if num_tokens <= 0 or router_topk <= 0 or num_experts <= 0 or ep_size <= 0:
            raise ValueError("Replica planner dimensions must all be positive.")
        if num_experts % ep_size != 0:
            raise ValueError(
                "Replica planner requires equal experts per rank, got "
                f"num_experts={num_experts}, ep_size={ep_size}."
            )
        num_local_experts = num_experts // ep_size
        num_routes = num_tokens * router_topk
        int32 = dict(dtype=torch.int32, device=device)
        return cls(
            num_tokens=num_tokens,
            router_topk=router_topk,
            num_experts=num_experts,
            ep_size=ep_size,
            num_local_experts=num_local_experts,
            gathered_counts=torch.empty((ep_size, num_experts), **int32),
            expert_totals=torch.empty(num_experts, **int32),
            balance=torch.empty(ep_size, **int32),
            receiver_quotas=torch.empty((ep_size, ep_size), **int32),
            allocation=torch.empty((num_experts, ep_size), **int32),
            expert_replica_slots=torch.empty((num_experts, ep_size), **int32),
            sorted_route_indices=torch.empty(num_routes, dtype=torch.int64, device=device),
            sort_route_metadata=torch.empty(num_routes, **int32),
            sort_partition_counts=torch.empty((256, num_experts), **int32),
            sort_grid_sync=torch.zeros(2, **int32),
            sort_stream=torch.cuda.Stream(device=device),
            local_count_cumsum=torch.empty(num_experts, **int32),
            virtual_experts=torch.empty(
                (num_tokens, router_topk), dtype=torch.int64, device=device
            ),
            experts_to_copy=torch.empty((ep_size, num_local_experts), **int32),
            distributed_shape_validated=False,
        )


@dataclass(frozen=True, slots=True)
class _ReplicaProjectionSpec:
    """Optimizer parameters and one runtime weight tensor per local expert."""

    parameters: tuple[torch.nn.Parameter, ...]
    source_tensors: tuple[torch.Tensor, ...]
    member_shape: tuple[int, int]
    weight_format: str
    gtp_leader: torch.nn.Parameter | None = None
    rowwise_scale_shape: tuple[int, ...] | None = None
    columnwise_scale_shape: tuple[int, ...] | None = None


def _parameter_storage(parameter: torch.nn.Parameter) -> torch.Tensor:
    """Return the BF16 storage used by a TE parameter, including tensor subclasses."""
    rowwise_data = getattr(parameter, "rowwise_data", None)
    return rowwise_data if rowwise_data is not None else parameter.data


def _validate_mxfp8_members(
    members: tuple[torch.Tensor, ...], *, member_shape: tuple[int, int], backend_name: str
) -> tuple[tuple[int, ...], tuple[int, ...], torch.device]:
    """Validate native MXFP8 member storage and return its two scale shapes."""
    rowwise_scale_shape = None
    columnwise_scale_shape = None
    device = None
    for index, member in enumerate(members):
        if not is_mxfp8tensor(member) or tuple(member.shape) != member_shape:
            raise ValueError(
                f"{backend_name} expected an MXFP8 expert with shape {member_shape}; "
                f"expert {index} has type={type(member).__name__}, shape={tuple(member.shape)}."
            )
        storage_tensors = (
            member._rowwise_data,
            member._rowwise_scale_inv,
            member._columnwise_data,
            member._columnwise_scale_inv,
        )
        if any(tensor is None for tensor in storage_tensors):
            raise ValueError(
                f"{backend_name} MXFP8 expert {index} requires rowwise and columnwise "
                "data plus scaling factors."
            )
        if any(
            tensor.dtype != torch.uint8 or not tensor.is_contiguous() for tensor in storage_tensors
        ):
            raise ValueError(
                f"{backend_name} MXFP8 expert {index} storage must be contiguous uint8."
            )
        if tuple(member._rowwise_data.shape) != member_shape:
            raise ValueError(
                f"{backend_name} MXFP8 rowwise data has shape "
                f"{tuple(member._rowwise_data.shape)}, expected {member_shape}."
            )
        if tuple(member._columnwise_data.shape) != member_shape:
            raise ValueError(
                f"{backend_name} MXFP8 columnwise data has shape "
                f"{tuple(member._columnwise_data.shape)}, expected {member_shape}."
            )
        current_rowwise_scale_shape = tuple(member._rowwise_scale_inv.shape)
        current_columnwise_scale_shape = tuple(member._columnwise_scale_inv.shape)
        if rowwise_scale_shape is None:
            rowwise_scale_shape = current_rowwise_scale_shape
            columnwise_scale_shape = current_columnwise_scale_shape
            device = member.device
        elif (
            current_rowwise_scale_shape != rowwise_scale_shape
            or current_columnwise_scale_shape != columnwise_scale_shape
            or member.device != device
        ):
            raise ValueError(
                f"{backend_name} MXFP8 experts must share scale shapes and one device."
            )
    if rowwise_scale_shape is None or columnwise_scale_shape is None or device is None:
        raise ValueError(f"{backend_name} did not find any MXFP8 expert weights.")
    return rowwise_scale_shape, columnwise_scale_shape, device


def _collect_replica_projection_specs(
    experts: torch.nn.Module, *, num_local_experts: int, backend_name: str
) -> tuple[list[_ReplicaProjectionSpec], torch.device]:
    """Collect independently allocated TE expert weights."""
    projection_specs = []
    device: torch.device | None = None
    for linear in (experts.linear_fc1, experts.linear_fc2):
        member_shape = (int(linear.out_features), int(linear.in_features))
        expected_numel = math.prod(member_shape)
        if getattr(linear, "single_grouped_weight", False):
            raise ValueError(
                f"{backend_name} requires discrete weight0..weightN expert parameters; "
                "moe_single_grouped_weight must be False."
            )
        parameters = tuple(
            linear.get_parameter(f"weight{index}") for index in range(num_local_experts)
        )
        if all(is_mxfp8tensor(parameter) for parameter in parameters):
            source_tensors = parameters
        else:
            source_tensors = tuple(_parameter_storage(parameter) for parameter in parameters)

        if len(source_tensors) != num_local_experts:
            raise ValueError(
                f"{backend_name} expected {num_local_experts} expert weights, "
                f"got {len(source_tensors)}."
            )
        gtp_members = tuple(
            bool(getattr(parameter, "is_gtp_weight_remat", False)) for parameter in parameters
        )
        if any(gtp_members) and not all(gtp_members):
            raise ValueError(
                f"{backend_name} requires every weight in a projection to use the same "
                "GTP layout."
            )
        gtp_leader = parameters[0] if all(gtp_members) else None
        if gtp_leader is not None:
            for index, parameter in enumerate(parameters):
                if tuple(parameter._unsharded_shape) != member_shape:
                    raise ValueError(
                        f"{backend_name} expected GTP expert {index} to materialize as "
                        f"{member_shape}, got {tuple(parameter._unsharded_shape)}."
                    )
        mxfp8_members = tuple(is_mxfp8tensor(source) for source in source_tensors)
        if any(mxfp8_members):
            if not all(mxfp8_members):
                raise ValueError(f"{backend_name} does not support mixed BF16 and MXFP8 experts.")
            if gtp_leader is None:
                rowwise_scale_shape, columnwise_scale_shape, projection_device = (
                    _validate_mxfp8_members(
                        source_tensors, member_shape=member_shape, backend_name=backend_name
                    )
                )
            else:
                quantizer = getattr(gtp_leader, "_gtp_gather_quantizer", None)
                if quantizer is None or not hasattr(quantizer, "get_scale_shape"):
                    raise ValueError(
                        f"{backend_name} GTP MXFP8 weights require a gather quantizer."
                    )
                rowwise_scale_shape = tuple(
                    quantizer.get_scale_shape(member_shape, columnwise=False)
                )
                columnwise_scale_shape = tuple(
                    quantizer.get_scale_shape(member_shape, columnwise=True)
                )
                projection_device = gtp_leader.device
            if device is None:
                device = projection_device
            elif projection_device != device:
                raise ValueError(f"{backend_name} FC1 and FC2 weights must share one device.")
            projection_specs.append(
                _ReplicaProjectionSpec(
                    parameters=parameters,
                    source_tensors=source_tensors,
                    member_shape=member_shape,
                    weight_format="mxfp8",
                    gtp_leader=gtp_leader,
                    rowwise_scale_shape=rowwise_scale_shape,
                    columnwise_scale_shape=columnwise_scale_shape,
                )
            )
            continue

        for index, source in enumerate(source_tensors):
            expected_source_numel = (
                math.prod(parameters[index]._sharded_padded_shape)
                if gtp_leader is not None
                else expected_numel
            )
            if source.dtype != torch.bfloat16 or source.numel() != expected_source_numel:
                raise ValueError(
                    f"{backend_name} weights require contiguous BF16 tensors with shape "
                    f"{member_shape}; expert {index} has dtype={source.dtype}, "
                    f"numel={source.numel()}."
                )
            if not source.is_contiguous():
                raise ValueError(
                    f"{backend_name} expert weight {index} must have contiguous storage."
                )
            if device is None:
                device = source.device
            elif source.device != device:
                raise ValueError(f"{backend_name} FC1 and FC2 weights must share one device.")
        projection_specs.append(
            _ReplicaProjectionSpec(
                parameters=parameters,
                source_tensors=source_tensors,
                member_shape=member_shape,
                weight_format="bf16",
                gtp_leader=gtp_leader,
            )
        )
    if device is None or device.type != "cuda":
        raise ValueError(f"{backend_name} expert weights must be CUDA tensors.")
    weight_formats = {spec.weight_format for spec in projection_specs}
    if len(weight_formats) != 1:
        raise ValueError(f"{backend_name} FC1 and FC2 weights must use one storage format.")
    return projection_specs, device


@dataclass(slots=True)
class _CuTeDSLReplicaProjection:
    """One registered projection and its virtual symmetric-memory views."""

    weight_format: str
    parameters: tuple[torch.nn.Parameter, ...]
    gtp_leader: torch.nn.Parameter | None
    source_tensors: tuple[torch.Tensor, ...]
    source_bases: torch.Tensor | None
    rowwise_data_bases: torch.Tensor | None
    rowwise_scale_bases: torch.Tensor | None
    columnwise_data_bases: torch.Tensor | None
    columnwise_scale_bases: torch.Tensor | None
    main_grad_bases: torch.Tensor
    member_shape: tuple[int, int]
    member_numel: int
    rowwise_scale_shape: tuple[int, ...] | None
    columnwise_scale_shape: tuple[int, ...] | None
    virtual_weight: tuple[torch.Tensor, ...]
    virtual_grad: torch.Tensor
    gtp_native_grad: torch.Tensor | None = None
    packed_runtime_weight: torch.Tensor | None = None
    packed_runtime_grad: torch.Tensor | None = None
    source_pointer_staging: tuple[torch.Tensor, ...] = ()
    gtp_rowwise_source_ptrs: tuple[tuple[int, int], ...] | None = None
    gtp_columnwise_source_ptrs: tuple[tuple[int, int], ...] | None = None
    runtime_parameters: tuple[torch.nn.Parameter, ...] | None = None
    source_storage_ptrs: tuple[tuple[int, ...], ...] | None = None
    source_main_grad_ptrs: tuple[int, ...] | None = None


class _ReplicaCuTeDSLWorkspace:
    """Fixed-shape symmetric arenas shared by every compatible MoE layer."""

    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        device: torch.device,
        world_size: int,
        num_local_experts: int,
        member_shapes: tuple[tuple[int, int], tuple[int, int]],
        weight_format: str,
        rowwise_scale_shapes: tuple[tuple[int, ...], tuple[int, ...]] | None,
        columnwise_scale_shapes: tuple[tuple[int, ...], tuple[int, ...]] | None,
        grad_dtype: torch.dtype,
        num_sms: int | None,
    ) -> None:
        try:
            import torch.distributed._symmetric_memory as symm_mem
        except ImportError as exc:
            raise ImportError(
                "Replica CuTeDSL weights require torch.distributed._symmetric_memory."
            ) from exc

        self.group = group
        self.device = device
        self.world_size = int(world_size)
        self.num_local_experts = int(num_local_experts)
        self.member_shapes = member_shapes
        self.member_numels = tuple(math.prod(shape) for shape in member_shapes)
        self.weight_format = weight_format
        self.rowwise_scale_shapes = rowwise_scale_shapes
        self.columnwise_scale_shapes = columnwise_scale_shapes
        if grad_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(
                "Replica CuTeDSL gradients must use torch.float32 or torch.bfloat16, "
                f"got {grad_dtype}."
            )
        self.grad_dtype = grad_dtype
        self.rowwise_scale_numels = (
            tuple(math.prod(shape) for shape in rowwise_scale_shapes)
            if rowwise_scale_shapes is not None
            else None
        )
        self.columnwise_scale_numels = (
            tuple(math.prod(shape) for shape in columnwise_scale_shapes)
            if columnwise_scale_shapes is not None
            else None
        )
        device_sms = torch.cuda.get_device_properties(device).multi_processor_count
        requested_sms = 32 if num_sms is None else int(num_sms)
        self.num_sms = min(requested_sms, MAX_REPLICA_WEIGHT_SMS, max(1, device_sms - 8))
        if self.num_sms <= 0:
            raise ValueError(f"Replica CuTeDSL num_sms must be positive, got {num_sms}.")

        arena_numel = self.num_local_experts * sum(self.member_numels)
        try:
            # Symmetric-memory backend selection is process-global and becomes
            # immutable after the first allocation.  NCCL-EP zero-copy payloads
            # require NCCLSymmetricMemory, whose window registration in turn
            # requires the device-specific process-group communicator to exist.
            # Materialize it once during bridge binding, before training or
            # graph capture; this is the same setup sequence used by TE's EP
            # bootstrap.
            dist.barrier(group=group, device_ids=[device.index])
            nccl_backend = group._get_backend(torch.device("cuda"))
            comm_ptr = nccl_backend._comm_ptr()
            if not isinstance(comm_ptr, int) or comm_ptr == 0:
                raise RuntimeError("ProcessGroupNCCL returned an invalid communicator pointer.")
            if symm_mem.get_backend(device) != "NCCL":
                symm_mem.set_backend("NCCL")
            if self.weight_format == "bf16":
                self.weight_arena = symm_mem.empty(arena_numel, dtype=torch.bfloat16, device=device)
                self.weight_handle = symm_mem.rendezvous(self.weight_arena, group)
                self.rowwise_arena = None
                self.rowwise_handle = None
                self.columnwise_arena = None
                self.columnwise_handle = None
            elif self.weight_format == "mxfp8":
                if self.rowwise_scale_numels is None or self.columnwise_scale_numels is None:
                    raise ValueError("MXFP8 replica weights require rowwise and columnwise scales.")
                # Forward consumes rowwise MXFP8 storage and backward consumes
                # columnwise storage only after an explicit orientation prefetch.
                # Retaining both would double the constant weight buffer despite
                # their disjoint lifetimes.
                shared_arena_numel = self.num_local_experts * sum(
                    member + max(rowwise_scale, columnwise_scale)
                    for member, rowwise_scale, columnwise_scale in zip(
                        self.member_numels, self.rowwise_scale_numels, self.columnwise_scale_numels
                    )
                )
                self.rowwise_arena = symm_mem.empty(
                    shared_arena_numel, dtype=torch.uint8, device=device
                )
                self.rowwise_handle = symm_mem.rendezvous(self.rowwise_arena, group)
                self.columnwise_arena = self.rowwise_arena
                self.columnwise_handle = self.rowwise_handle
                self.weight_arena = None
                self.weight_handle = None
            else:
                raise ValueError(f"Unsupported replica weight format {self.weight_format!r}.")
            self.grad_arena = symm_mem.empty(arena_numel, dtype=self.grad_dtype, device=device)
            self.grad_handle = symm_mem.rendezvous(self.grad_arena, group)
        except RuntimeError as exc:
            raise RuntimeError(
                "Replica CuTeDSL could not allocate PyTorch native symmetric memory for the "
                "EP group. The initial implementation requires a single NVLink domain."
            ) from exc

        if self.weight_arena is not None:
            self.weight_arena.zero_()
        if self.rowwise_arena is not None:
            self.rowwise_arena.zero_()
        if self.columnwise_arena is not None and self.columnwise_arena is not self.rowwise_arena:
            self.columnwise_arena.zero_()
        self.grad_arena.zero_()
        self.weight_grid_barrier = torch.zeros(1, dtype=torch.int32, device=device)
        self.columnwise_grid_barrier = torch.zeros(1, dtype=torch.int32, device=device)
        self.grad_grid_barrier = torch.zeros(1, dtype=torch.int32, device=device)
        self.weight_stream = torch.cuda.Stream(device=device, priority=0)
        # CUDA graph capture streams also come from PyTorch's stream pool and
        # may alias a stream allocated earlier. Keep a second candidate so the
        # weight branch never collapses onto the active planner stream.
        self.weight_stream_fallback = torch.cuda.Stream(device=device, priority=0)
        self.grad_stream = torch.cuda.Stream(device=device, priority=0)
        self.resident_bridge = None
        self.resident_plan = None
        self.resident_orientation = None
        self._gtp_native_projection_grad_storage = {}
        # TE's discrete BF16 grouped GEMM accepts at most 64 pointer-list
        # members. Large expert configurations use one packed runtime
        # GroupedTensor over this workspace-owned storage instead.
        self._packed_bf16_projection_storage = {}
        self._destroyed = False

        device_index = device.index
        if device_index is None:
            raise ValueError("Replica CuTeDSL workspace requires an indexed CUDA device.")
        compile_replica_weight_kernels(
            world_size=self.world_size,
            num_local_experts=self.num_local_experts,
            member_numels=self.member_numels,
            num_sms=self.num_sms,
            device_index=device_index,
            grad_dtype=self.grad_dtype,
            rowwise_scale_numels=self.rowwise_scale_numels,
            columnwise_scale_numels=self.columnwise_scale_numels,
        )
        # JIT time can vary substantially by rank on a cold cache. No rank may
        # enter the device-side cross-rank barrier until every peer has a
        # launchable kernel.
        dist.barrier(group=group, device_ids=[device.index])

    def select_weight_stream(self, current_stream: torch.cuda.Stream) -> torch.cuda.Stream:
        """Return a preallocated weight stream distinct from the active graph stream."""
        if self.weight_stream.cuda_stream != current_stream.cuda_stream:
            return self.weight_stream
        if self.weight_stream_fallback.cuda_stream != current_stream.cuda_stream:
            return self.weight_stream_fallback
        raise RuntimeError("Replica CuTeDSL weight streams alias the active CUDA stream.")

    def validate(
        self,
        *,
        world_size: int,
        num_local_experts: int,
        member_shapes: tuple[tuple[int, int], tuple[int, int]],
        weight_format: str,
        rowwise_scale_shapes: tuple[tuple[int, ...], tuple[int, ...]] | None,
        columnwise_scale_shapes: tuple[tuple[int, ...], tuple[int, ...]] | None,
        grad_dtype: torch.dtype,
        num_sms: int | None,
    ) -> None:
        """Reject heterogeneous layers instead of creating a shape-keyed memory pool."""
        requested_sms = 32 if num_sms is None else int(num_sms)
        device_sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        effective_sms = min(requested_sms, MAX_REPLICA_WEIGHT_SMS, max(1, device_sms - 8))
        actual = (
            int(world_size),
            int(num_local_experts),
            member_shapes,
            weight_format,
            rowwise_scale_shapes,
            columnwise_scale_shapes,
            grad_dtype,
            effective_sms,
        )
        expected = (
            self.world_size,
            self.num_local_experts,
            self.member_shapes,
            self.weight_format,
            self.rowwise_scale_shapes,
            self.columnwise_scale_shapes,
            self.grad_dtype,
            self.num_sms,
        )
        if actual != expected:
            raise ValueError(
                "All replica-planned MoE layers on an EP group must share one CuTeDSL "
                f"weight shape and launch configuration; expected {expected}, got {actual}."
            )

    def projection_views(self, projection_index: int) -> tuple[tuple, torch.Tensor]:
        """Return virtual runtime weights and gradients for one projection."""
        grad_offset = self.num_local_experts * sum(self.member_numels[:projection_index])
        member_numel = self.member_numels[projection_index]
        grad_numel = self.num_local_experts * member_numel
        member_shape = self.member_shapes[projection_index]
        virtual_grad = self.grad_arena.narrow(0, grad_offset, grad_numel).view(
            self.num_local_experts, *member_shape
        )
        if self.weight_format == "bf16":
            weight_offset = grad_offset
            weights = self.weight_arena.narrow(0, weight_offset, grad_numel).view(
                self.num_local_experts, *member_shape
            )
            return tuple(weights), virtual_grad

        rowwise_scale_numel = self.rowwise_scale_numels[projection_index]
        columnwise_scale_numel = self.columnwise_scale_numels[projection_index]
        projection_offset = self.num_local_experts * sum(
            member + max(rowwise_scale, columnwise_scale)
            for member, rowwise_scale, columnwise_scale in zip(
                self.member_numels[:projection_index],
                self.rowwise_scale_numels[:projection_index],
                self.columnwise_scale_numels[:projection_index],
            )
        )
        rowwise_data = self.rowwise_arena.narrow(
            0, projection_offset, self.num_local_experts * member_numel
        ).view(self.num_local_experts, *member_shape)
        rowwise_scale = self.rowwise_arena.narrow(
            0,
            projection_offset + self.num_local_experts * member_numel,
            self.num_local_experts * rowwise_scale_numel,
        ).view(self.num_local_experts, *self.rowwise_scale_shapes[projection_index])
        columnwise_data = self.columnwise_arena.narrow(
            0, projection_offset, self.num_local_experts * member_numel
        ).view(self.num_local_experts, *member_shape)
        columnwise_scale = self.columnwise_arena.narrow(
            0,
            projection_offset + self.num_local_experts * member_numel,
            self.num_local_experts * columnwise_scale_numel,
        ).view(self.num_local_experts, *self.columnwise_scale_shapes[projection_index])
        # The bridge wraps these raw views with source-matching TE metadata.
        return (
            tuple(
                (
                    rowwise_data[index],
                    rowwise_scale[index],
                    columnwise_data[index],
                    columnwise_scale[index],
                )
                for index in range(self.num_local_experts)
            ),
            virtual_grad,
        )

    def packed_bf16_projection_views(
        self, projection_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return shared packed native-plus-replica weight and wgrad storage."""
        cached = self._packed_bf16_projection_storage.get(projection_index)
        if cached is not None:
            return cached
        member_shape = self.member_shapes[projection_index]
        runtime_shape = (2 * self.num_local_experts, *member_shape)
        cached = (
            torch.empty(runtime_shape, dtype=torch.bfloat16, device=self.device),
            torch.empty(runtime_shape, dtype=self.grad_dtype, device=self.device),
        )
        self._packed_bf16_projection_storage[projection_index] = cached
        return cached

    def gtp_native_projection_grad_view(self, projection_index: int) -> torch.Tensor:
        """Return shared full-gradient staging for one GTP projection."""
        cached = self._gtp_native_projection_grad_storage.get(projection_index)
        if cached is None:
            cached = torch.empty(
                (self.num_local_experts, *self.member_shapes[projection_index]),
                dtype=self.grad_dtype,
                device=self.device,
            )
            self._gtp_native_projection_grad_storage[projection_index] = cached
        return cached

    def destroy(self) -> None:
        """Release symmetric registrations while their NCCL group is still alive."""
        if self._destroyed:
            return
        torch.cuda.synchronize(self.device)
        self.resident_bridge = None
        self.resident_plan = None
        self.resident_orientation = None
        self._gtp_native_projection_grad_storage.clear()
        self._packed_bf16_projection_storage.clear()
        # Handles own NCCL window registrations. Drop them before their backing
        # tensors and, critically, before model-parallel process-group teardown.
        self.weight_handle = None
        self.rowwise_handle = None
        self.columnwise_handle = None
        self.grad_handle = None
        self.weight_arena = None
        self.rowwise_arena = None
        self.columnwise_arena = None
        self.grad_arena = None
        self._destroyed = True


_replica_cutedsl_workspaces = weakref.WeakValueDictionary()
_replica_cutedsl_bridges = weakref.WeakSet()


def _get_replica_cutedsl_workspace(
    *,
    group: dist.ProcessGroup,
    device: torch.device,
    world_size: int,
    num_local_experts: int,
    member_shapes: tuple[tuple[int, int], tuple[int, int]],
    weight_format: str,
    rowwise_scale_shapes: tuple[tuple[int, ...], tuple[int, ...]] | None,
    columnwise_scale_shapes: tuple[tuple[int, ...], tuple[int, ...]] | None,
    grad_dtype: torch.dtype,
    num_sms: int | None,
) -> _ReplicaCuTeDSLWorkspace:
    """Return the one fixed-shape workspace owned by an EP group and device."""
    key = (id(group), device.index)
    workspace = _replica_cutedsl_workspaces.get(key)
    if workspace is None:
        workspace = _ReplicaCuTeDSLWorkspace(
            group=group,
            device=device,
            world_size=world_size,
            num_local_experts=num_local_experts,
            member_shapes=member_shapes,
            weight_format=weight_format,
            rowwise_scale_shapes=rowwise_scale_shapes,
            columnwise_scale_shapes=columnwise_scale_shapes,
            grad_dtype=grad_dtype,
            num_sms=num_sms,
        )
        _replica_cutedsl_workspaces[key] = workspace
    else:
        workspace.validate(
            world_size=world_size,
            num_local_experts=num_local_experts,
            member_shapes=member_shapes,
            weight_format=weight_format,
            rowwise_scale_shapes=rowwise_scale_shapes,
            columnwise_scale_shapes=columnwise_scale_shapes,
            grad_dtype=grad_dtype,
            num_sms=num_sms,
        )
    return workspace


class ReplicaCuTeDSLWeightBridge:
    """Dispatcher-independent asynchronous LSA weight and gradient bridge."""

    def __init__(
        self,
        *,
        experts: torch.nn.Module,
        group: dist.ProcessGroup,
        num_experts: int,
        num_local_experts: int,
        grad_dtype: torch.dtype = torch.float32,
        num_sms: int | None = None,
        hidden_dim: int | None = None,
        top_k: int | None = None,
        alignment: int | None = None,
        zero_copy: bool | None = None,
        num_blocks_permute: int | None = None,
        num_blocks_unpermute: int | None = None,
        num_sms_preprocessing: int | None = None,
    ) -> None:
        del (
            hidden_dim,
            top_k,
            alignment,
            zero_copy,
            num_blocks_permute,
            num_blocks_unpermute,
            num_sms_preprocessing,
        )
        self.group = group
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self.num_experts = int(num_experts)
        self.num_local_experts = int(num_local_experts)
        self.num_slots = self.num_local_experts
        self.num_runtime_experts = 2 * self.num_local_experts
        self._weight_backend_name = "Replica-CuTeDSL"
        self.last_plan = None
        self._prefetch_pending = False
        self._prefetch_plan = None
        self._grad_reduce_pending = False
        self._grad_reduce_plan = None
        self._experts_ref = weakref.ref(experts)
        self._destroyed = False

        if self.num_experts != self.world_size * self.num_local_experts:
            raise ValueError(
                "Replica CuTeDSL weights require an even expert distribution: "
                f"num_experts={self.num_experts}, world_size={self.world_size}, "
                f"num_local_experts={self.num_local_experts}."
            )
        projection_specs, self.device = _collect_replica_projection_specs(
            experts,
            num_local_experts=self.num_local_experts,
            backend_name=self._weight_backend_name,
        )
        # Replica wgrads are produced directly into ``main_grad`` and therefore
        # must not flow through PyTorch's leaf ``AccumulateGrad``. DDP sees this
        # marker during construction and retains a manually-triggered grad-ready
        # hook instead of registering the ordinary leaf hook.
        for spec in projection_specs:
            for parameter in spec.parameters:
                parameter._replica_managed_grad = True
        member_shapes = tuple(spec.member_shape for spec in projection_specs)
        self.weight_format = projection_specs[0].weight_format
        self.uses_packed_runtime_weights = self.weight_format == "bf16" and (
            self.num_runtime_experts > 64
            or any(spec.gtp_leader is not None for spec in projection_specs)
        )
        rowwise_scale_shapes = (
            tuple(spec.rowwise_scale_shape for spec in projection_specs)
            if self.weight_format == "mxfp8"
            else None
        )
        columnwise_scale_shapes = (
            tuple(spec.columnwise_scale_shape for spec in projection_specs)
            if self.weight_format == "mxfp8"
            else None
        )
        self.workspace = _get_replica_cutedsl_workspace(
            group=group,
            device=self.device,
            world_size=self.world_size,
            num_local_experts=self.num_local_experts,
            member_shapes=member_shapes,
            weight_format=self.weight_format,
            rowwise_scale_shapes=rowwise_scale_shapes,
            columnwise_scale_shapes=columnwise_scale_shapes,
            grad_dtype=grad_dtype,
            num_sms=num_sms,
        )
        self.prefetch_ready = torch.cuda.Event()
        self.prefetch_done = torch.cuda.Event()
        self.grad_reduce_ready = torch.cuda.Event()
        self.grad_reduce_done = torch.cuda.Event()
        # PyTorch creates CUDA event handles lazily on first record. Materialize
        # every reusable event during binding, before graph capture or training.
        initialization_stream = torch.cuda.current_stream(self.device)
        self.prefetch_ready.record(initialization_stream)
        self.prefetch_done.record(initialization_stream)
        self.grad_reduce_ready.record(initialization_stream)
        self.grad_reduce_done.record(initialization_stream)
        self.projections: list[_CuTeDSLReplicaProjection] = []
        for projection_index, spec in enumerate(projection_specs):
            virtual_storage, virtual_grad = self.workspace.projection_views(projection_index)
            packed_runtime_weight = None
            packed_runtime_grad = None
            gtp_native_grad = None
            if self.uses_packed_runtime_weights:
                packed_runtime_weight, packed_runtime_grad = (
                    self.workspace.packed_bf16_projection_views(projection_index)
                )
                native_storage = tuple(packed_runtime_weight[: self.num_local_experts])
            else:
                # GTP MXFP8 gather storage is stable and is bound directly before
                # runtime construction. Bootstrap distinct native wrappers over
                # the replica views instead of retaining an unused full weight copy.
                native_storage = virtual_storage if spec.gtp_leader is not None else None
                if spec.gtp_leader is not None:
                    gtp_native_grad = self.workspace.gtp_native_projection_grad_view(
                        projection_index
                    )

            def pointer_table() -> torch.Tensor:
                return torch.empty(self.num_local_experts, dtype=torch.int64, device=self.device)

            if spec.weight_format == "bf16":
                virtual_weight = virtual_storage
                native_weight = native_storage
                source_bases = pointer_table()
                rowwise_data_bases = None
                rowwise_scale_bases = None
                columnwise_data_bases = None
                columnwise_scale_bases = None
            else:
                from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

                def wrap_mxfp8_storage(storage_views):
                    weights = []
                    for source, storage in zip(spec.source_tensors, storage_views):
                        rowwise_data, rowwise_scale, columnwise_data, columnwise_scale = storage
                        weights.append(
                            MXFP8Tensor(
                                shape=spec.member_shape,
                                dtype=source.dtype,
                                rowwise_data=rowwise_data,
                                rowwise_scale_inv=rowwise_scale,
                                columnwise_data=columnwise_data,
                                columnwise_scale_inv=columnwise_scale,
                                fp8_dtype=source._fp8_dtype,
                                quantizer=source._quantizer,
                                with_gemm_swizzled_scales=source._with_gemm_swizzled_scales,
                                requires_grad=False,
                                device=self.device,
                            )
                        )
                    return tuple(weights)

                virtual_weight = wrap_mxfp8_storage(virtual_storage)
                native_weight = (
                    wrap_mxfp8_storage(native_storage) if native_storage is not None else None
                )
                source_bases = None
                rowwise_data_bases = pointer_table()
                rowwise_scale_bases = pointer_table()
                columnwise_data_bases = pointer_table()
                columnwise_scale_bases = pointer_table()
            main_grad_bases = pointer_table()
            source_pointer_staging = (
                tuple(
                    torch.empty(self.num_local_experts, dtype=torch.int64, pin_memory=True)
                    for _ in range(4)
                )
                if spec.gtp_leader is not None and spec.weight_format == "mxfp8"
                else ()
            )
            self.projections.append(
                _CuTeDSLReplicaProjection(
                    weight_format=spec.weight_format,
                    parameters=spec.parameters,
                    gtp_leader=spec.gtp_leader,
                    source_tensors=(
                        native_weight if native_weight is not None else spec.source_tensors
                    ),
                    source_bases=source_bases,
                    rowwise_data_bases=rowwise_data_bases,
                    rowwise_scale_bases=rowwise_scale_bases,
                    columnwise_data_bases=columnwise_data_bases,
                    columnwise_scale_bases=columnwise_scale_bases,
                    main_grad_bases=main_grad_bases,
                    member_shape=spec.member_shape,
                    member_numel=math.prod(spec.member_shape),
                    rowwise_scale_shape=spec.rowwise_scale_shape,
                    columnwise_scale_shape=spec.columnwise_scale_shape,
                    virtual_weight=virtual_weight,
                    virtual_grad=virtual_grad,
                    gtp_native_grad=gtp_native_grad,
                    packed_runtime_weight=packed_runtime_weight,
                    packed_runtime_grad=packed_runtime_grad,
                    source_pointer_staging=source_pointer_staging,
                )
            )
        _replica_cutedsl_bridges.add(self)

    @property
    def runtime_fc1_weights(self) -> tuple[torch.nn.Parameter, ...]:
        """Return stable native-then-virtual FC1 runtime parameters."""
        runtime_parameters = self.projections[0].runtime_parameters
        if runtime_parameters is None:
            raise RuntimeError("Replica CuTeDSL runtime weights were accessed before binding.")
        return runtime_parameters

    @property
    def runtime_fc2_weights(self) -> tuple[torch.nn.Parameter, ...]:
        """Return stable native-then-virtual FC2 runtime parameters."""
        runtime_parameters = self.projections[1].runtime_parameters
        if runtime_parameters is None:
            raise RuntimeError("Replica CuTeDSL runtime weights were accessed before binding.")
        return runtime_parameters

    @property
    def source_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        """Return the optimizer-owned FC1 and FC2 parameters."""
        return tuple(
            parameter for projection in self.projections for parameter in projection.parameters
        )

    def _bind_materialized_gtp_mxfp8_weights(
        self,
        projection: _CuTeDSLReplicaProjection,
        materialized_weights: tuple[torch.Tensor, ...],
        *,
        retain_for_grad: bool,
    ) -> None:
        """Bind stable GTP MXFP8 gather outputs directly as native replica sources."""
        data_field, scale_field = (
            ("_columnwise_data", "_columnwise_scale_inv")
            if retain_for_grad
            else ("_rowwise_data", "_rowwise_scale_inv")
        )
        expected_scale_shape = (
            projection.columnwise_scale_shape if retain_for_grad else projection.rowwise_scale_shape
        )
        source_ptrs = []
        for index, source in enumerate(materialized_weights):
            source_data = getattr(source, data_field, None)
            source_scale = getattr(source, scale_field, None)
            if (
                tuple(source.shape) != projection.member_shape
                or source.device != self.device
                or source_data is None
                or source_scale is None
                or source_data.dtype != torch.uint8
                or source_scale.dtype != torch.uint8
                or not source_data.is_contiguous()
                or not source_scale.is_contiguous()
                or tuple(source_data.shape) != projection.member_shape
                or tuple(source_scale.shape) != expected_scale_shape
            ):
                raise RuntimeError(
                    f"GTP MXFP8 {'backward' if retain_for_grad else 'forward'} gather "
                    f"returned invalid storage for replica expert {index}."
                )
            source_ptrs.append((source_data.data_ptr(), source_scale.data_ptr()))
        source_ptrs = tuple(source_ptrs)
        bound_ptrs_attr = (
            "gtp_columnwise_source_ptrs" if retain_for_grad else "gtp_rowwise_source_ptrs"
        )
        bound_ptrs = getattr(projection, bound_ptrs_attr)
        if bound_ptrs is not None:
            if source_ptrs != bound_ptrs:
                raise RuntimeError(
                    f"Replica CuTeDSL GTP {'backward' if retain_for_grad else 'forward'} "
                    "all-gather storage changed after direct binding; this would invalidate "
                    "CUDA-graph source pointers."
                )
            return

        if len(projection.source_pointer_staging) != 4:
            raise RuntimeError(
                "Replica CuTeDSL GTP MXFP8 direct binding lost its pinned pointer staging."
            )
        setattr(projection, bound_ptrs_attr, source_ptrs)
        component_offset = 2 if retain_for_grad else 0
        pointer_tables = (
            projection.rowwise_data_bases,
            projection.rowwise_scale_bases,
            projection.columnwise_data_bases,
            projection.columnwise_scale_bases,
        )
        for component_index in range(2):
            pointer_index = component_offset + component_index
            host_table = projection.source_pointer_staging[pointer_index]
            host_table.copy_(
                torch.tensor([ptrs[component_index] for ptrs in source_ptrs], dtype=torch.int64)
            )
            pointer_tables[pointer_index].copy_(host_table, non_blocking=True)

        runtime_parameters = projection.runtime_parameters
        for index, (destination, source) in enumerate(
            zip(projection.source_tensors, materialized_weights)
        ):
            source_data = getattr(source, data_field)
            source_scale = getattr(source, scale_field)
            setattr(destination, data_field, source_data)
            setattr(destination, scale_field, source_scale)
            if runtime_parameters is not None:
                setattr(runtime_parameters[index], data_field, source_data)
                setattr(runtime_parameters[index], scale_field, source_scale)

        if projection.source_storage_ptrs is not None:
            updated_storage_ptrs = [list(ptrs) for ptrs in projection.source_storage_ptrs]
            for expert_index, ptrs in enumerate(source_ptrs):
                updated_storage_ptrs[expert_index][component_offset] = ptrs[0]
                updated_storage_ptrs[expert_index][component_offset + 1] = ptrs[1]
            projection.source_storage_ptrs = tuple(tuple(ptrs) for ptrs in updated_storage_ptrs)

    def _publish_materialized_gtp_weights(
        self,
        projection: _CuTeDSLReplicaProjection,
        materialized_weights: tuple[torch.Tensor, ...],
        *,
        retain_for_grad: bool,
    ) -> None:
        """Publish one completed GTP gather to the replica runtime and owner-push."""
        if len(materialized_weights) != len(projection.source_tensors):
            raise RuntimeError(
                "GTP materialized an unexpected number of replica source weights: "
                f"expected {len(projection.source_tensors)}, got {len(materialized_weights)}."
            )
        if projection.weight_format == "bf16":
            torch._foreach_copy_(list(projection.source_tensors), list(materialized_weights))
            return
        self._bind_materialized_gtp_mxfp8_weights(
            projection, materialized_weights, retain_for_grad=retain_for_grad
        )

    def _materialize_gtp_source_weights(self, *, retain_for_grad: bool) -> None:
        """Complete GTP gathers before the replica owner-push reads native weights."""
        ordered_projections = (
            reversed(self.projections) if retain_for_grad else iter(self.projections)
        )
        for projection in ordered_projections:
            leader = projection.gtp_leader
            if leader is None:
                continue
            materialized = (
                leader.materialize_group_for_backward()
                if retain_for_grad
                else leader.materialize_group_for_forward()
            )
            materialized = (
                tuple(materialized) if isinstance(materialized, (list, tuple)) else (materialized,)
            )
            self._publish_materialized_gtp_weights(
                projection, materialized, retain_for_grad=retain_for_grad
            )

    def _stable_gtp_main_grads(
        self, projection: _CuTeDSLReplicaProjection
    ) -> tuple[torch.Tensor, ...]:
        """Return bridge-owned GTP gradient staging with stable device addresses."""
        if projection.gtp_leader is None:
            raise RuntimeError("Replica CuTeDSL requested GTP staging for a non-GTP projection.")
        if projection.gtp_native_grad is not None:
            return tuple(projection.gtp_native_grad)
        if projection.packed_runtime_grad is not None:
            return tuple(projection.packed_runtime_grad[: self.num_local_experts])
        raise RuntimeError("Replica CuTeDSL GTP weights lost their native gradient staging.")

    def prepare_runtime_parameters(self) -> None:
        """Late-bind optimizer storage and validate its subsequent stability.

        DistributedDataParallel constructs its persistent parameter and gradient
        buffers after the expert module (and therefore this bridge) is created.
        The first prepared forward binds those final addresses; only a later
        remapping would invalidate captured runtime pointers.
        """
        for projection in self.projections:
            if projection.gtp_leader is not None:
                source_tensors = projection.source_tensors
                main_grad_tensors = self._stable_gtp_main_grads(projection)
            else:
                source_tensors = (
                    tuple(_parameter_storage(parameter) for parameter in projection.parameters)
                    if projection.weight_format == "bf16"
                    else projection.source_tensors
                )
                main_grad_tensors = tuple(
                    getattr(parameter, "main_grad", None) for parameter in projection.parameters
                )
                if any(main_grad is None for main_grad in main_grad_tensors):
                    raise RuntimeError(
                        "Replica CuTeDSL weights require gradient-accumulation fusion and "
                        "initialized parameter.main_grad buffers."
                    )

            if projection.packed_runtime_weight is not None:
                source_tensors = tuple(projection.packed_runtime_weight[: self.num_local_experts])

            source_storage_ptrs = []
            for source in source_tensors:
                if projection.weight_format == "bf16":
                    if (
                        source.dtype != torch.bfloat16
                        or source.device != self.device
                        or source.numel() != projection.member_numel
                        or not source.is_contiguous()
                    ):
                        raise ValueError(
                            "Replica CuTeDSL requires contiguous BF16 expert weight storage on "
                            f"the weight device with {projection.member_numel} elements per expert."
                        )
                    source_storage_ptrs.append((source.data_ptr(),))
                else:
                    rowwise_data = source._rowwise_data
                    rowwise_scale = source._rowwise_scale_inv
                    columnwise_data = source._columnwise_data
                    columnwise_scale = source._columnwise_scale_inv
                    if (
                        tuple(source.shape) != projection.member_shape
                        or source.device != self.device
                        or any(
                            tensor is None
                            or tensor.dtype != torch.uint8
                            or not tensor.is_contiguous()
                            for tensor in (
                                rowwise_data,
                                rowwise_scale,
                                columnwise_data,
                                columnwise_scale,
                            )
                        )
                        or tuple(rowwise_scale.shape) != projection.rowwise_scale_shape
                        or tuple(columnwise_scale.shape) != projection.columnwise_scale_shape
                    ):
                        raise ValueError(
                            "Replica CuTeDSL requires stable contiguous MXFP8 data and scales "
                            "for every expert."
                        )
                    source_storage_ptrs.append(
                        (
                            rowwise_data.data_ptr(),
                            rowwise_scale.data_ptr(),
                            columnwise_data.data_ptr(),
                            columnwise_scale.data_ptr(),
                        )
                    )
            source_storage_ptrs = tuple(source_storage_ptrs)
            if projection.source_storage_ptrs is None:
                projection.source_storage_ptrs = source_storage_ptrs
                if projection.weight_format == "bf16":
                    projection.source_bases.copy_(
                        torch.tensor(
                            [ptrs[0] for ptrs in source_storage_ptrs],
                            dtype=torch.int64,
                            device=self.device,
                        )
                    )
                else:
                    pointer_tables = (
                        projection.rowwise_data_bases,
                        projection.rowwise_scale_bases,
                        projection.columnwise_data_bases,
                        projection.columnwise_scale_bases,
                    )
                    for pointer_index, pointer_table in enumerate(pointer_tables):
                        pointer_table.copy_(
                            torch.tensor(
                                [ptrs[pointer_index] for ptrs in source_storage_ptrs],
                                dtype=torch.int64,
                                device=self.device,
                            )
                        )
            elif source_storage_ptrs != projection.source_storage_ptrs:
                raise RuntimeError(
                    "Replica CuTeDSL parameter storage changed after runtime binding; this "
                    "would invalidate CUDA-graph source pointers."
                )

            main_grad_ptrs = []
            for main_grad in main_grad_tensors:
                if (
                    main_grad.dtype != self.workspace.grad_dtype
                    or main_grad.device != self.device
                    or main_grad.numel() != projection.member_numel
                    or not main_grad.is_contiguous()
                ):
                    raise ValueError(
                        "Replica CuTeDSL requires contiguous registered main-grad buffers "
                        f"with dtype {self.workspace.grad_dtype} on the weight device with one "
                        "expert's shape."
                    )
                main_grad_ptrs.append(main_grad.data_ptr())
            main_grad_ptrs = tuple(main_grad_ptrs)
            if projection.source_main_grad_ptrs is None:
                projection.source_main_grad_ptrs = main_grad_ptrs
                projection.main_grad_bases.copy_(
                    torch.tensor(main_grad_ptrs, dtype=torch.int64, device=self.device)
                )
            elif (
                main_grad_ptrs != projection.source_main_grad_ptrs and projection.gtp_leader is None
            ):
                raise RuntimeError(
                    "Replica CuTeDSL main-grad storage changed after runtime binding; this "
                    "would invalidate CUDA-graph destination pointers."
                )
            elif main_grad_ptrs != projection.source_main_grad_ptrs:
                raise RuntimeError(
                    "Replica CuTeDSL bridge-owned GTP gradient staging changed after runtime "
                    "binding; this would invalidate CUDA-graph destination pointers."
                )
            projection.source_tensors = source_tensors
            if projection.runtime_parameters is None:
                if projection.packed_runtime_weight is not None:
                    from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor

                    grouped_weight = GroupedTensor.make_grouped_tensor_from_rowwise_data(
                        num_tensors=self.num_runtime_experts,
                        tensor_shape=projection.member_shape,
                        rowwise_data=projection.packed_runtime_weight,
                        dtype=torch.bfloat16,
                    )
                    runtime_parameter = torch.nn.Parameter(grouped_weight, requires_grad=True)
                    runtime_parameter.main_grad = projection.packed_runtime_grad
                    runtime_parameter.grad_added_to_main_grad = True
                    runtime_parameter.overwrite_main_grad = True
                    runtime_parameter.register_post_accumulate_grad_hook(
                        _discard_runtime_parameter_grad
                    )
                    projection.runtime_parameters = (runtime_parameter,)
                else:
                    runtime_parameters = []
                    for weight, grad in zip(source_tensors, main_grad_tensors):
                        runtime_parameter = torch.nn.Parameter(weight, requires_grad=True)
                        runtime_parameter.main_grad = grad
                        runtime_parameter.grad_added_to_main_grad = True
                        runtime_parameter.overwrite_main_grad = projection.gtp_leader is not None
                        runtime_parameter.register_post_accumulate_grad_hook(
                            _discard_runtime_parameter_grad
                        )
                        runtime_parameters.append(runtime_parameter)
                    for weight, grad in zip(projection.virtual_weight, projection.virtual_grad):
                        runtime_parameter = torch.nn.Parameter(weight, requires_grad=True)
                        runtime_parameter.main_grad = grad
                        runtime_parameter.grad_added_to_main_grad = True
                        runtime_parameter.overwrite_main_grad = projection.gtp_leader is not None
                        runtime_parameter.register_post_accumulate_grad_hook(
                            _discard_runtime_parameter_grad
                        )
                        runtime_parameters.append(runtime_parameter)
                    projection.runtime_parameters = tuple(runtime_parameters)
            elif projection.packed_runtime_weight is not None:
                runtime_parameter = projection.runtime_parameters[0]
                if (
                    runtime_parameter.rowwise_data.data_ptr()
                    != projection.packed_runtime_weight.data_ptr()
                    or runtime_parameter.main_grad.data_ptr()
                    != projection.packed_runtime_grad.data_ptr()
                ):
                    raise RuntimeError(
                        "Replica CuTeDSL packed runtime storage changed after binding; "
                        "this would invalidate CUDA-graph pointers."
                    )
                runtime_parameter.grad_added_to_main_grad = True
                runtime_parameter.overwrite_main_grad = True
            else:
                expected_weights = source_tensors + tuple(projection.virtual_weight)
                expected_grads = main_grad_tensors + tuple(projection.virtual_grad)
                for runtime_parameter, expected_weight, expected_grad in zip(
                    projection.runtime_parameters, expected_weights, expected_grads
                ):
                    if projection.weight_format == "bf16":
                        weight_storage_matches = (
                            runtime_parameter.data_ptr() == expected_weight.data_ptr()
                        )
                    else:
                        weight_storage_matches = all(
                            getattr(runtime_parameter, field).data_ptr()
                            == getattr(expected_weight, field).data_ptr()
                            for field in (
                                "_rowwise_data",
                                "_rowwise_scale_inv",
                                "_columnwise_data",
                                "_columnwise_scale_inv",
                            )
                        )
                    if not weight_storage_matches:
                        raise RuntimeError(
                            "Replica CuTeDSL runtime weight storage changed after binding; "
                            "this would invalidate CUDA-graph weight pointers."
                        )
                    runtime_main_grad = getattr(runtime_parameter, "main_grad", None)
                    if (
                        runtime_main_grad is None
                        or runtime_main_grad.data_ptr() != expected_grad.data_ptr()
                    ):
                        if projection.gtp_leader is None:
                            raise RuntimeError(
                                "Replica CuTeDSL runtime main-grad storage changed after "
                                "binding; this would invalidate CUDA-graph gradient pointers."
                            )
                        runtime_parameter.main_grad = expected_grad
                    runtime_parameter.grad_added_to_main_grad = True
                    runtime_parameter.overwrite_main_grad = projection.gtp_leader is not None

    def _validate_plan(self, plan: ReplicaPlan) -> None:
        """Validate fixed device metadata without extracting any CUDA values."""
        experts_to_copy = plan.experts_to_copy
        expected_shape = (self.world_size, self.num_local_experts)
        if (
            experts_to_copy.dtype != torch.int32
            or experts_to_copy.device != self.device
            or tuple(experts_to_copy.shape) != expected_shape
            or not experts_to_copy.is_contiguous()
        ):
            raise ValueError(
                "Replica CuTeDSL experts_to_copy must be contiguous int32 with shape "
                f"{expected_shape} on {self.device}."
            )

    def prepare_source_weights(self, *, retain_for_grad: bool = False) -> None:
        """Materialize GTP weights and bind the replica runtime buffers."""
        experts = self._experts_ref()
        if experts is None:
            raise RuntimeError("Replica CuTeDSL experts were destroyed before prefetch.")
        experts.prepare_fused_impl_parameters()
        self._materialize_gtp_source_weights(retain_for_grad=retain_for_grad)
        for projection in self.projections:
            if projection.packed_runtime_weight is None or projection.gtp_leader is not None:
                continue
            sources = tuple(_parameter_storage(parameter) for parameter in projection.parameters)
            torch._foreach_copy_(list(projection.source_tensors), list(sources))
        self.prepare_runtime_parameters()

    def prepare_forward(self) -> None:
        """Validate that route planning started prefetch before expert entry."""
        if self.last_plan is None:
            raise RuntimeError("Replica CuTeDSL weights require a plan before expert compute.")

    @torch.no_grad()
    @nvtx_decorator(message="replica_cutedsl_weight_owner_push_start")
    def start_prefetch(self, plan: ReplicaPlan, *, retain_for_grad: bool = False) -> None:
        """Enqueue owner-push FC1/FC2 prefetch without blocking the caller."""
        if self._prefetch_pending:
            raise RuntimeError("Replica CuTeDSL prefetch is already outstanding.")
        self._validate_plan(plan)
        # A GTP parameter stores only its local shard. Materialization consumes
        # any one-weight-ahead gather (or performs the cold synchronous gather)
        # and stages the full native experts before CuTeDSL exchanges replicas.
        self.prepare_source_weights(retain_for_grad=retain_for_grad)
        # The CuTeDSL kernel has a device-side cross-rank rendezvous.  Keep an
        # opt-in host rendezvous available for debugging launch skew and stale
        # symmetric-memory signal state; it must remain opt-in because a host
        # collective cannot be captured into a CUDA graph.
        if os.environ.get("MCORE_REPLICA_PREFETCH_HOST_BARRIER", "0") == "1":
            dist.barrier(group=self.group, device_ids=[self.device.index])
        current_stream = torch.cuda.current_stream(self.device)
        weight_stream = self.workspace.select_weight_stream(current_stream)
        self.prefetch_ready.record(current_stream)
        weight_stream.wait_event(self.prefetch_ready)
        with torch.cuda.stream(weight_stream):
            orientation = (
                "columnwise"
                if self.weight_format == "mxfp8" and retain_for_grad
                else "rowwise" if self.weight_format == "mxfp8" else "full"
            )
            resident = (
                retain_for_grad
                and self.workspace.resident_bridge is self
                and self.workspace.resident_plan is plan
                and self.workspace.resident_orientation == orientation
            )
            if not resident:
                if self.weight_format == "bf16":
                    sources = tuple(projection.source_bases for projection in self.projections)
                    scale_sources = None
                    arena = self.workspace.weight_arena
                    handle = self.workspace.weight_handle
                    grid_barrier = self.workspace.weight_grid_barrier
                else:
                    rowwise = orientation == "rowwise"
                    sources = tuple(
                        (
                            projection.rowwise_data_bases
                            if rowwise
                            else projection.columnwise_data_bases
                        )
                        for projection in self.projections
                    )
                    scale_sources = tuple(
                        (
                            projection.rowwise_scale_bases
                            if rowwise
                            else projection.columnwise_scale_bases
                        )
                        for projection in self.projections
                    )
                    arena = (
                        self.workspace.rowwise_arena if rowwise else self.workspace.columnwise_arena
                    )
                    handle = (
                        self.workspace.rowwise_handle
                        if rowwise
                        else self.workspace.columnwise_handle
                    )
                    grid_barrier = (
                        self.workspace.weight_grid_barrier
                        if rowwise
                        else self.workspace.columnwise_grid_barrier
                    )
                launch_replica_weight_prefetch(
                    sources=sources,
                    scale_sources=scale_sources,
                    arena=arena,
                    peer_bases=handle.buffer_ptrs_dev,
                    signal_bases=handle.signal_pad_ptrs_dev,
                    experts_to_copy=plan.experts_to_copy,
                    grid_barrier=grid_barrier,
                    rank=self.rank,
                    world_size=self.world_size,
                    num_local_experts=self.num_local_experts,
                    member_numels=self.workspace.member_numels,
                    rowwise_scale_numels=(
                        self.workspace.rowwise_scale_numels if scale_sources is not None else None
                    ),
                    columnwise_scale_numels=(
                        self.workspace.columnwise_scale_numels
                        if scale_sources is not None
                        else None
                    ),
                    orientation=orientation if scale_sources is not None else None,
                    num_sms=self.workspace.num_sms,
                )
                self.workspace.resident_bridge = self
                self.workspace.resident_plan = plan
                self.workspace.resident_orientation = orientation
            self.prefetch_done.record(weight_stream)
        self._prefetch_pending = True
        self._prefetch_plan = plan

    @torch.no_grad()
    @nvtx_decorator(message="replica_cutedsl_weight_owner_push_wait")
    def wait_prefetch(self, plan: ReplicaPlan) -> None:
        """Insert the sole forward/restore consumer-stream dependency."""
        if not self._prefetch_pending:
            # The TE fused expert path can expose the same consumer boundary
            # through both its op-level and module-level hooks. Once this exact
            # plan is resident, a repeated wait is a no-op; starting a new
            # collective here would let rank-local hook timing desynchronize the
            # cross-rank transport. Every new forward is still refreshed by the
            # planner's explicit ``start_prefetch`` call.
            if not (
                self.workspace.resident_bridge is self and self.workspace.resident_plan is plan
            ):
                self.start_prefetch(plan)
        elif self._prefetch_plan is not plan:
            raise RuntimeError("Replica CuTeDSL prefetch plan changed while outstanding.")
        torch.cuda.current_stream(self.device).wait_event(self.prefetch_done)
        for projection in self.projections:
            if projection.packed_runtime_weight is not None:
                torch._foreach_copy_(
                    list(projection.packed_runtime_weight[self.num_local_experts :]),
                    list(projection.virtual_weight),
                )
        self._prefetch_pending = False
        self._prefetch_plan = None

    def prefetch(self, plan: ReplicaPlan) -> None:
        """Compatibility helper that preserves stream-asynchronous waiting."""
        self.start_prefetch(plan)
        self.wait_prefetch(plan)

    @torch.no_grad()
    @nvtx_decorator(message="replica_cutedsl_grad_reduce_start")
    def start_grad_reduce(self, plan: ReplicaPlan) -> None:
        """Enqueue direct peer reduction after expert wgrad production."""
        if self._grad_reduce_pending:
            raise RuntimeError("Replica CuTeDSL gradient reduction is already outstanding.")
        self._validate_plan(plan)
        self.prepare_runtime_parameters()
        for projection in self.projections:
            packed_grad = projection.packed_runtime_grad
            if packed_grad is None:
                continue
            replica_grads = packed_grad[self.num_local_experts :]
            if projection.gtp_leader is None:
                native_grads = packed_grad[: self.num_local_experts]
                main_grads = tuple(parameter.main_grad for parameter in projection.parameters)
                torch._foreach_add_(list(main_grads), list(native_grads))
            torch._foreach_copy_(list(projection.virtual_grad), list(replica_grads))
        current_stream = torch.cuda.current_stream(self.device)
        self.grad_reduce_ready.record(current_stream)
        self.workspace.grad_stream.wait_event(self.grad_reduce_ready)
        with torch.cuda.stream(self.workspace.grad_stream):
            launch_replica_grad_reduce(
                arena=self.workspace.grad_arena,
                main_grads=tuple(projection.main_grad_bases for projection in self.projections),
                peer_bases=self.workspace.grad_handle.buffer_ptrs_dev,
                signal_bases=self.workspace.grad_handle.signal_pad_ptrs_dev,
                experts_to_copy=plan.experts_to_copy,
                grid_barrier=self.workspace.grad_grid_barrier,
                rank=self.rank,
                world_size=self.world_size,
                num_local_experts=self.num_local_experts,
                member_numels=self.workspace.member_numels,
                num_sms=self.workspace.num_sms,
            )
            self.grad_reduce_done.record(self.workspace.grad_stream)
        for projection in self.projections:
            for parameter in projection.parameters:
                parameter.grad_added_to_main_grad = True
        self._grad_reduce_pending = True
        self._grad_reduce_plan = plan

    @torch.no_grad()
    @nvtx_decorator(message="replica_cutedsl_grad_reduce_wait")
    def wait_grad_reduce(self, plan: ReplicaPlan) -> None:
        """Finish replica reduction and notify DDP without materializing leaf grads."""
        if not self._grad_reduce_pending:
            self.start_grad_reduce(plan)
        elif self._grad_reduce_plan is not plan:
            raise RuntimeError("Replica CuTeDSL grad-reduction plan changed while outstanding.")
        torch.cuda.current_stream(self.device).wait_event(self.grad_reduce_done)
        self._grad_reduce_pending = False
        self._grad_reduce_plan = None

        # Expert backward computes FC2 before FC1. Preserve that reverse order
        # when handing full wgrads to GTP so its linked RS cascade remains valid.
        for projection in reversed(self.projections):
            if projection.gtp_leader is None:
                continue
            projection.gtp_leader.finalize_group_grads(
                list(self._stable_gtp_main_grads(projection))
            )

        # GTP finalization invokes its retained DDP hooks itself. Ordinary
        # replica parameters need the equivalent notification here because
        # their real gradients were already accumulated into ``main_grad`` by
        # the replica reduction. Returning full-sized zero tensors through
        # autograd would make AccumulateGrad issue one memcpy/add per expert.
        for projection in self.projections:
            if projection.gtp_leader is not None:
                continue
            for parameter in projection.parameters:
                parameter.grad = None
                parameter.grad_added_to_main_grad = True
                hook = getattr(parameter, "_replica_grad_accum_hook", None)
                if hook is not None:
                    hook()

    def reduce_grads(self, plan: ReplicaPlan) -> None:
        """Compatibility helper that preserves stream-asynchronous waiting."""
        self.start_grad_reduce(plan)
        self.wait_grad_reduce(plan)

    def destroy(self) -> None:
        """Detach layer-owned TE parameters from the shared symmetric arenas."""
        if self._destroyed:
            return
        experts = self._experts_ref()
        if experts is not None:
            experts._fused_ops = None
            experts._replica_weight_bridge = None
        for projection in self.projections:
            if projection.runtime_parameters is not None:
                for runtime_parameter in projection.runtime_parameters:
                    runtime_parameter.main_grad = None
                projection.runtime_parameters = None
        self.projections.clear()
        if self.workspace.resident_bridge is self:
            self.workspace.resident_bridge = None
            self.workspace.resident_plan = None
            self.workspace.resident_orientation = None
        self.last_plan = None
        self.workspace = None
        self._destroyed = True
        _replica_cutedsl_bridges.discard(self)


# Keep existing dispatcher-facing names as compatibility aliases. Weight
# movement is intentionally independent of the activation dispatcher.
NCCLEPReplicaWeightBridge = ReplicaCuTeDSLWeightBridge
HybridEPReplicaWeightBridge = ReplicaCuTeDSLWeightBridge


def finalize_replica_weight_bridges() -> None:
    """Release replica weight contexts before their process group is destroyed."""
    workspaces = list(_replica_cutedsl_workspaces.values())
    for bridge in list(_replica_cutedsl_bridges):
        bridge.destroy()
    for workspace in workspaces:
        workspace.destroy()
    _replica_cutedsl_workspaces.clear()
    # NCCLSymmetricMemory handles contain Python reference cycles. Collect them
    # now so their window deregistration runs before the process group is gone.
    gc.collect()


class _ReplicaStartWeightPrefetch(torch.autograd.Function):
    """Launch backward weight prefetch before activation-combine backward."""

    @staticmethod
    def forward(ctx, combined_hidden, bridge, plan):
        ctx.bridge = bridge
        ctx.plan = plan
        return combined_hidden

    @staticmethod
    def backward(ctx, grad_combined_hidden):
        ctx.bridge.start_prefetch(ctx.plan, retain_for_grad=True)
        return grad_combined_hidden, None, None


class _ReplicaWaitWeightPrefetch(torch.autograd.Function):
    """Wait for backward weight prefetch immediately before expert backward."""

    @staticmethod
    def forward(ctx, expert_output, bridge, plan):
        ctx.bridge = bridge
        ctx.plan = plan
        return expert_output

    @staticmethod
    def backward(ctx, grad_expert_output):
        ctx.bridge.wait_prefetch(ctx.plan)
        return grad_expert_output, None, None


class _ReplicaStartGradReduce(torch.autograd.Function):
    """Launch replica-gradient reduction immediately after expert backward."""

    @staticmethod
    def forward(ctx, dispatched_hidden, bridge, plan):
        ctx.bridge = bridge
        ctx.plan = plan
        return dispatched_hidden

    @staticmethod
    def backward(ctx, grad_dispatched_hidden):
        ctx.bridge.start_grad_reduce(ctx.plan)
        return grad_dispatched_hidden, None, None


class _ReplicaWaitGradReduce(torch.autograd.Function):
    """Finalize replica gradients after activation-dispatch backward."""

    @staticmethod
    def forward(ctx, hidden_states, *args):
        bridge, plan = args[-2:]
        ctx.bridge = bridge
        ctx.plan = plan
        ctx.num_source_parameters = len(args) - 2
        return hidden_states

    @staticmethod
    def backward(ctx, grad_hidden_states):
        ctx.bridge.wait_grad_reduce(ctx.plan)
        return (grad_hidden_states, *(None,) * ctx.num_source_parameters, None, None)


def start_replica_weight_prefetch_before_combine_backward(
    combined_hidden: torch.Tensor,
    bridge: NCCLEPReplicaWeightBridge | HybridEPReplicaWeightBridge,
    plan: ReplicaPlan,
) -> torch.Tensor:
    """Start weight communication before transport-combine backward."""
    return _ReplicaStartWeightPrefetch.apply(combined_hidden, bridge, plan)


def wait_replica_weight_prefetch_before_expert_backward(
    expert_output: torch.Tensor,
    bridge: NCCLEPReplicaWeightBridge | HybridEPReplicaWeightBridge,
    plan: ReplicaPlan,
) -> torch.Tensor:
    """Wait for weight communication immediately before expert backward."""
    return _ReplicaWaitWeightPrefetch.apply(expert_output, bridge, plan)


def start_replica_grad_reduce_after_expert_backward(
    dispatched_hidden: torch.Tensor,
    bridge: NCCLEPReplicaWeightBridge | HybridEPReplicaWeightBridge,
    plan: ReplicaPlan,
) -> torch.Tensor:
    """Start replica-gradient communication after expert backward."""
    return _ReplicaStartGradReduce.apply(dispatched_hidden, bridge, plan)


def wait_replica_grad_reduce_after_dispatch_backward(
    hidden_states: torch.Tensor,
    bridge: NCCLEPReplicaWeightBridge | HybridEPReplicaWeightBridge,
    plan: ReplicaPlan,
) -> torch.Tensor:
    """Wait for replica gradients before registered-parameter DDP hooks."""
    return _ReplicaWaitGradReduce.apply(hidden_states, *bridge.source_parameters, bridge, plan)


@triton.jit
def _initialize_allocation_kernel(
    gathered_tokens_per_expert,
    global_tokens_per_expert,
    expert_rank_allocations,
    EP_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_EXPERTS_PER_GPU: tl.constexpr,
    BLOCK_EP_SIZE: tl.constexpr,
):
    """Sum per-source counts and initialize every expert on its owner rank.

    ``expert_rank_allocations[expert, destination]`` records how many routes
    the destination will execute for that semantic expert. Initially the
    native owner executes all routes; later kernels migrate portions of
    overloaded experts elsewhere.

    Args:
        gathered_tokens_per_expert: Pointer to int32 ``[EP_SIZE, NUM_EXPERTS]``
            route counts, indexed by source rank and semantic expert.
        global_tokens_per_expert: Output pointer to int32 ``[NUM_EXPERTS]``
            global route counts.
        expert_rank_allocations: Output pointer to int32
            ``[NUM_EXPERTS, EP_SIZE]`` route allocations.
        EP_SIZE: Number of expert-parallel ranks.
        NUM_EXPERTS: Number of semantic experts.
        NUM_EXPERTS_PER_GPU: Native experts per rank, ``NUM_EXPERTS / EP_SIZE``.
        BLOCK_EP_SIZE: Power-of-two Triton block width covering ``EP_SIZE``.
    """
    expert = tl.program_id(0)
    ranks = tl.arange(0, BLOCK_EP_SIZE)
    counts = tl.load(
        gathered_tokens_per_expert + ranks * NUM_EXPERTS + expert, mask=ranks < EP_SIZE, other=0
    )
    total = tl.sum(counts, axis=0).to(tl.int32)
    tl.store(global_tokens_per_expert + expert, total)
    owner = expert // NUM_EXPERTS_PER_GPU
    values = tl.where(ranks == owner, total, 0)
    tl.store(expert_rank_allocations + expert * EP_SIZE + ranks, values, mask=ranks < EP_SIZE)


@triton.jit
def _compute_group_balance_kernel(
    global_tokens_per_expert,
    rank_load_balance,
    RANK_ROUTE_CAPACITY: tl.constexpr,
    NUM_EXPERTS_PER_GPU: tl.constexpr,
    BLOCK_NUM_EXPERTS_PER_GPU: tl.constexpr,
):
    """Compute native expert-group load relative to one rank's route capacity.

    A positive balance means an owner's native experts have more than the
    per-rank route capacity and must shed work. A negative balance is free
    capacity that can host replicas for an overloaded owner.

    Args:
        global_tokens_per_expert: Pointer to int32 ``[num_experts]`` global
            route counts.
        rank_load_balance: Output pointer to int32 ``[ep_size]`` signed
            load-minus-capacity values for each rank's native expert group.
        RANK_ROUTE_CAPACITY: Per-rank route capacity
            ``num_tokens * router_topk``.
        NUM_EXPERTS_PER_GPU: Number of native experts owned by each rank.
        BLOCK_NUM_EXPERTS_PER_GPU: Power-of-two Triton block width covering
            ``NUM_EXPERTS_PER_GPU``.
    """
    owner = tl.program_id(0)
    local_experts = tl.arange(0, BLOCK_NUM_EXPERTS_PER_GPU)
    counts = tl.load(
        global_tokens_per_expert + owner * NUM_EXPERTS_PER_GPU + local_experts,
        mask=local_experts < NUM_EXPERTS_PER_GPU,
        other=0,
    )
    tl.store(rank_load_balance + owner, tl.sum(counts, axis=0).to(tl.int32) - RANK_ROUTE_CAPACITY)


@triton.jit
def _choose_receiver_quotas_kernel(
    rank_load_balance, receiver_quotas, EP_SIZE: tl.constexpr, BLOCK_EP_SIZE: tl.constexpr
):
    """Greedily pair the lowest-index most-loaded and roomiest ranks.

    This is the rank-level balancing step. Equal balances choose the
    lowest rank, making the quota matrix identical on every EP rank.
    ``quotas[owner, receiver]`` records how many routes the receiver should
    take from that owner's experts; it does not choose the experts yet.

    Args:
        rank_load_balance: Pointer to int32 ``[EP_SIZE]`` signed group
            balances. Values are loaded into registers and not modified in
            global memory.
        receiver_quotas: Output pointer to int32 ``[EP_SIZE, EP_SIZE]``. The
            kernel clears it before writing owner-to-destination quotas.
        EP_SIZE: Number of EP ranks and maximum number of greedy pairings.
        BLOCK_EP_SIZE: Power-of-two Triton block width covering ``EP_SIZE``.
    """
    ranks = tl.arange(0, BLOCK_EP_SIZE)
    valid = ranks < EP_SIZE
    current_balance = tl.load(rank_load_balance + ranks, mask=valid, other=0)
    quota_owners = ranks[:, None]
    quota_destinations = ranks[None, :]
    tl.store(
        receiver_quotas + quota_owners * EP_SIZE + quota_destinations,
        0,
        mask=valid[:, None] & valid[None, :],
    )

    for _ in tl.static_range(0, EP_SIZE):
        maximum = tl.max(tl.where(valid, current_balance, -2147483648), axis=0)
        minimum = tl.min(tl.where(valid, current_balance, 2147483647), axis=0)
        overloaded = tl.min(
            tl.where(valid & (current_balance == maximum), ranks, BLOCK_EP_SIZE), axis=0
        )
        receiver = tl.min(
            tl.where(valid & (current_balance == minimum), ranks, BLOCK_EP_SIZE), axis=0
        )
        active = maximum > 0
        moved = tl.where(active, -minimum, 0).to(tl.int32)
        tl.store(receiver_quotas + overloaded * EP_SIZE + receiver, moved, mask=active)
        current_balance = tl.where(
            active & (ranks == overloaded), current_balance - moved, current_balance
        )
        current_balance = tl.where(active & (ranks == receiver), 0, current_balance)


@triton.jit
def _allocate_migrations_kernel(
    global_tokens_per_expert,
    receiver_quotas,
    expert_rank_allocations,
    EP_SIZE: tl.constexpr,
    NUM_EXPERTS_PER_GPU: tl.constexpr,
    BLOCK_EP_SIZE: tl.constexpr,
    BLOCK_NUM_EXPERTS_PER_GPU: tl.constexpr,
):
    """Assign each owner rank's receive quotas to its largest expert segments.

    For one owner, repeatedly pair its largest outstanding receiver quota with
    its largest remaining expert segment. Expert ties choose the lowest local
    expert id, matching the reference placement policy. A semantic expert may therefore be split
    between its owner and one or more replica destinations.

    Args:
        global_tokens_per_expert: Pointer to int32 ``[num_experts]`` global
            route counts, laid out as ``EP_SIZE`` contiguous native groups of
            ``NUM_EXPERTS_PER_GPU`` experts.
        receiver_quotas: Pointer to int32 ``[EP_SIZE, EP_SIZE]``
            owner-to-destination quotas from ``_choose_receiver_quotas_kernel``.
        expert_rank_allocations: In/out pointer to int32
            ``[num_experts, EP_SIZE]`` allocations. Each move subtracts routes
            from the owner and adds them to a remote destination.
        EP_SIZE: Number of expert-parallel ranks.
        NUM_EXPERTS_PER_GPU: Number of native experts owned by each rank.
        BLOCK_EP_SIZE: Power-of-two block width covering ``EP_SIZE``.
        BLOCK_NUM_EXPERTS_PER_GPU: Power-of-two block width covering the local
            experts owned by one rank.
    """
    owner = tl.program_id(0)
    ranks = tl.arange(0, BLOCK_EP_SIZE)
    local_experts = tl.arange(0, BLOCK_NUM_EXPERTS_PER_GPU)
    valid_ranks = ranks < EP_SIZE
    valid_local_experts = local_experts < NUM_EXPERTS_PER_GPU
    remaining = tl.load(
        global_tokens_per_expert + owner * NUM_EXPERTS_PER_GPU + local_experts,
        mask=valid_local_experts,
        other=0,
    ).to(tl.int32)
    quotas = tl.load(receiver_quotas + owner * EP_SIZE + ranks, mask=valid_ranks, other=0).to(
        tl.int32
    )

    for _ in tl.static_range(0, EP_SIZE + NUM_EXPERTS_PER_GPU):
        max_quota = tl.max(tl.where(valid_ranks, quotas, -1), axis=0)
        destination = tl.min(
            tl.where(valid_ranks & (quotas == max_quota), ranks, BLOCK_EP_SIZE), axis=0
        )
        max_remaining = tl.max(tl.where(valid_local_experts, remaining, -1), axis=0)
        local_expert = tl.min(
            tl.where(
                valid_local_experts & (remaining == max_remaining),
                local_experts,
                BLOCK_NUM_EXPERTS_PER_GPU,
            ),
            axis=0,
        )
        active = max_quota > 0
        moved = tl.where(active, tl.minimum(max_quota, max_remaining), 0).to(tl.int32)
        expert = owner * NUM_EXPERTS_PER_GPU + local_expert

        remote_ptr = expert_rank_allocations + expert * EP_SIZE + destination
        owner_ptr = expert_rank_allocations + expert * EP_SIZE + owner
        remote_value = tl.load(remote_ptr, mask=active, other=0)
        owner_value = tl.load(owner_ptr, mask=active, other=0)
        tl.store(remote_ptr, remote_value + moved, mask=active)
        tl.store(owner_ptr, owner_value - moved, mask=active)
        remaining = tl.where(active & (local_experts == local_expert), remaining - moved, remaining)
        quotas = tl.where(active & (ranks == destination), quotas - moved, quotas)


@triton.jit
def _select_replica_experts_kernel(
    expert_rank_allocations,
    experts_to_copy,
    expert_replica_slots,
    EP_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_EXPERTS_PER_GPU: tl.constexpr,
    BLOCK_NUM_EXPERTS: tl.constexpr,
    WRITE_REPLICA_LOOKUP: tl.constexpr,
):
    """Select destination replica slots by count descending, expert id descending.

    Only remote experts with a positive allocation need copied weights. The
    ordering is observable because it defines replica-slot ids, so the expert
    id tie-break deliberately runs in descending order for deterministic parity.
    Unused slots are filled with ``-1``.

    Args:
        expert_rank_allocations: Pointer to int32
            ``[NUM_EXPERTS, EP_SIZE]`` final route allocations.
        experts_to_copy: Output pointer to int32
            ``[EP_SIZE, NUM_EXPERTS_PER_GPU]`` semantic expert ids, indexed by
            destination rank and replica slot.
        expert_replica_slots: Optional output pointer to int32
            ``[NUM_EXPERTS, EP_SIZE]`` inverse slot assignments.
        EP_SIZE: Number of expert-parallel ranks.
        NUM_EXPERTS: Number of semantic experts.
        NUM_EXPERTS_PER_GPU: Replica slots per rank, equal to its native expert
            count.
        BLOCK_NUM_EXPERTS: Power-of-two block width covering ``NUM_EXPERTS``.
        WRITE_REPLICA_LOOKUP: Whether to write ``expert_replica_slots`` while
            selecting destination slots.
    """
    destination = tl.program_id(0)
    experts = tl.arange(0, BLOCK_NUM_EXPERTS)
    owner = experts // NUM_EXPERTS_PER_GPU
    valid = (experts < NUM_EXPERTS) & (owner != destination)
    counts = tl.load(
        expert_rank_allocations + experts * EP_SIZE + destination, mask=valid, other=-1
    )

    for slot in tl.static_range(0, NUM_EXPERTS_PER_GPU):
        maximum = tl.max(tl.where(valid, counts, -1), axis=0)
        expert = tl.max(tl.where(valid & (counts == maximum), experts, -1), axis=0)
        selected = tl.where(maximum > 0, expert, -1).to(tl.int32)
        tl.store(experts_to_copy + destination * NUM_EXPERTS_PER_GPU + slot, selected)
        if WRITE_REPLICA_LOOKUP:
            tl.store(
                expert_replica_slots + selected * EP_SIZE + destination, slot, mask=selected >= 0
            )
        counts = tl.where(experts == expert, -1, counts)


@triton.jit
def _cooperative_grid_barrier(grid_sync, num_programs):
    """Synchronize a cooperatively launched Triton grid."""
    generation = tl.atomic_add(grid_sync + 1, 0, sem="acquire", scope="gpu")
    arrival = tl.atomic_add(grid_sync, 1, sem="acq_rel", scope="gpu")
    if arrival == num_programs - 1:
        tl.atomic_xchg(grid_sync, 0, sem="release", scope="gpu")
        tl.atomic_add(grid_sync + 1, 1, sem="acq_rel", scope="gpu")
    else:
        observed_generation = generation
        while observed_generation == generation:
            observed_generation = tl.load(grid_sync + 1, cache_modifier=".cv", volatile=True)
        tl.atomic_add(grid_sync + 1, 0, sem="acquire", scope="gpu")


@triton.jit
def _stable_route_bucket_sort_kernel(
    flat_topk_indices,
    local_tokens_per_expert_cumsum,
    sorted_route_indices,
    route_metadata,
    partition_counts,
    grid_sync,
    NUM_ROUTES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_NUM_EXPERTS: tl.constexpr,
    BLOCK_NUM_ROUTES: tl.constexpr,
    BLOCK_WRITE_ROUTES: tl.constexpr,
    BLOCK_SCAN_PARTITIONS: tl.constexpr,
    NUM_PARTITION_CHUNKS: tl.constexpr,
    NUM_SCAN_EXPERTS: tl.constexpr,
):
    """Stably group route positions by expert without radix-sort scratch.

    A cooperative grid splits the original route stream into consecutive
    partitions. The first pass computes each route's partition-local ordinal
    and packs it with the expert id in one int32 scratch value. Programs then
    publish their histograms, synchronize, prefix-scan the partition counts,
    and scatter original positions to their final buckets. Consequently the
    result is identical to sorting unique ``expert * NUM_ROUTES + position``
    keys.
    """
    partition = tl.program_id(0)
    num_partitions = tl.num_programs(0)
    expert_offsets = tl.arange(0, BLOCK_NUM_EXPERTS)
    valid_experts = expert_offsets < NUM_EXPERTS
    routes_per_partition = tl.cdiv(NUM_ROUTES, num_partitions)
    partition_start = partition * routes_per_partition
    partition_end = tl.minimum(partition_start + routes_per_partition, NUM_ROUTES)
    partition_histogram = tl.zeros((BLOCK_NUM_EXPERTS,), dtype=tl.int32)
    tile_offsets = tl.arange(0, BLOCK_NUM_ROUTES)

    for route_start in tl.range(
        partition_start, partition_end, BLOCK_NUM_ROUTES, loop_unroll_factor=1
    ):
        route_positions = route_start + tile_offsets
        valid_routes = route_positions < partition_end
        route_experts = tl.load(
            flat_topk_indices + route_positions, mask=valid_routes, other=NUM_EXPERTS + tile_offsets
        ).to(tl.int32)
        ranks_in_tile = tl.inline_asm_elementwise(
            asm="""
            {
                .reg .b32 matching_lanes;
                .reg .b32 lower_lanes;
                match.sync.any.b32 matching_lanes, $1, 0xffffffff;
                mov.u32 lower_lanes, %lanemask_lt;
                and.b32 matching_lanes, matching_lanes, lower_lanes;
                popc.b32 $0, matching_lanes;
            }
            """,
            constraints="=r,r",
            args=[route_experts],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        safe_route_experts = tl.where(valid_routes, route_experts, 0)
        first_warp_counts = tl.histogram(
            route_experts, BLOCK_NUM_EXPERTS, mask=valid_routes & (tile_offsets < 32)
        )
        second_warp_counts = tl.histogram(
            route_experts,
            BLOCK_NUM_EXPERTS,
            mask=valid_routes & (tile_offsets >= 32) & (tile_offsets < 64),
        )
        preceding_warp_counts = tl.gather(first_warp_counts, safe_route_experts, axis=0)
        ranks_in_tile += tl.where(tile_offsets >= 32, preceding_warp_counts, 0)
        ordinals_before_tile = tl.gather(partition_histogram, safe_route_experts, axis=0)
        local_ordinals = ordinals_before_tile + ranks_in_tile
        tl.store(
            route_metadata + route_positions,
            local_ordinals * BLOCK_NUM_EXPERTS + route_experts,
            mask=valid_routes,
        )
        partition_histogram += first_warp_counts + second_warp_counts

    tl.store(
        partition_counts + partition * NUM_EXPERTS + expert_offsets,
        partition_histogram,
        mask=valid_experts,
    )

    _cooperative_grid_barrier(grid_sync, num_partitions)

    # Transpose the prefix work: programs scan expert columns across route
    # partitions, then publish all exclusive offsets through a second barrier.
    partition_offsets = tl.arange(0, BLOCK_SCAN_PARTITIONS)
    for scan_expert_offset in tl.static_range(0, NUM_SCAN_EXPERTS):
        scan_expert = partition + scan_expert_offset * num_partitions
        valid_scan_expert = scan_expert < NUM_EXPERTS
        preceding_chunks = tl.zeros((), dtype=tl.int32)
        for chunk in tl.static_range(0, NUM_PARTITION_CHUNKS):
            chunk_partitions = chunk * BLOCK_SCAN_PARTITIONS + partition_offsets
            valid_partitions = chunk_partitions < num_partitions
            counts_for_expert = tl.load(
                partition_counts + chunk_partitions * NUM_EXPERTS + scan_expert,
                mask=valid_scan_expert & valid_partitions,
                other=0,
            )
            inclusive_prefix = tl.cumsum(counts_for_expert, axis=0) + preceding_chunks
            tl.store(
                partition_counts + chunk_partitions * NUM_EXPERTS + scan_expert,
                inclusive_prefix - counts_for_expert,
                mask=valid_scan_expert & valid_partitions,
            )
            preceding_chunks += tl.sum(counts_for_expert, axis=0)

    _cooperative_grid_barrier(grid_sync, num_partitions)

    expert_starts = tl.load(
        local_tokens_per_expert_cumsum + expert_offsets - 1,
        mask=valid_experts & (expert_offsets > 0),
        other=0,
    ).to(tl.int32)
    running_counts = tl.load(
        partition_counts + partition * NUM_EXPERTS + expert_offsets, mask=valid_experts, other=0
    )

    write_tile_offsets = tl.arange(0, BLOCK_WRITE_ROUTES)

    for route_start in tl.range(
        partition_start, partition_end, BLOCK_WRITE_ROUTES, loop_unroll_factor=1
    ):
        route_positions = route_start + write_tile_offsets
        valid_routes = route_positions < partition_end
        packed_metadata = tl.load(route_metadata + route_positions, mask=valid_routes, other=0).to(
            tl.int32
        )
        route_experts = packed_metadata % BLOCK_NUM_EXPERTS
        local_ordinals = packed_metadata // BLOCK_NUM_EXPERTS
        safe_route_experts = tl.where(valid_routes, route_experts, 0)
        route_expert_starts = tl.gather(expert_starts, safe_route_experts, axis=0)
        route_running_counts = tl.gather(running_counts, safe_route_experts, axis=0)
        output_offsets = route_expert_starts + route_running_counts + local_ordinals
        tl.store(sorted_route_indices + output_offsets, route_positions, mask=valid_routes)


def _launch_stable_route_bucket_sort(
    workspace: ReplicaPlannerWorkspace,
    flat_topk_indices: torch.Tensor,
    *,
    num_experts: int,
    num_routes: int,
) -> None:
    """Launch the one-kernel stable expert-bucket route ordering."""

    # A cooperative launch must keep the complete grid resident.  The old
    # fixed 256-program launch can exceed the device's occupancy limit for
    # this register-heavy kernel, which then poisons the CUDA context and is
    # often reported by Triton at a later module-load call.  The 128-program
    # default is safe on the validated GB300 workload and retains substantially
    # more parallelism than the conservative 64-program debug setting.
    num_programs = min(128, num_routes)
    grid = (num_programs,)

    _stable_route_bucket_sort_kernel[grid](
        flat_topk_indices,
        workspace.local_count_cumsum,
        workspace.sorted_route_indices,
        workspace.sort_route_metadata,
        workspace.sort_partition_counts,
        workspace.sort_grid_sync,
        NUM_ROUTES=num_routes,
        NUM_EXPERTS=num_experts,
        BLOCK_NUM_EXPERTS=triton.next_power_of_2(num_experts),
        BLOCK_NUM_ROUTES=64,
        BLOCK_WRITE_ROUTES=128,
        BLOCK_SCAN_PARTITIONS=128,
        NUM_PARTITION_CHUNKS=2,
        NUM_SCAN_EXPERTS=triton.cdiv(num_experts, num_programs),
        launch_cooperative_grid=True,
        num_warps=2,
    )


@triton.jit
def _map_virtual_experts_kernel(
    flat_topk_indices,
    sorted_route_indices,
    local_tokens_per_expert_cumsum,
    gathered_tokens_per_expert,
    expert_rank_allocations,
    experts_to_copy,
    expert_replica_slots,
    virtual_experts,
    SOURCE_EP_RANK: tl.constexpr,
    EP_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_EXPERTS_PER_GPU: tl.constexpr,
    NUM_ROUTES: tl.constexpr,
    BLOCK_NUM_ROUTES: tl.constexpr,
    USE_REPLICA_LOOKUP: tl.constexpr,
):
    """Map globally ordered expert routes to rank-major native or replica ids.

    Routes for each expert form one deterministic global stream: source ranks
    are concatenated in rank order, and routes within a source use flattened
    token/top-k order. ``expert_rank_allocations[expert]`` partitions that
    stream into destination segments in rank order. This kernel computes each
    local route's global ordinal, finds its segment, and converts the
    destination to the transport's native-or-replica virtual id.

    Args:
        flat_topk_indices: Pointer to flattened int32/int64 ``[NUM_ROUTES]``
            semantic expert ids in original local route order.
        sorted_route_indices: Pointer to int64 ``[NUM_ROUTES]`` original route
            positions sorted by ``(semantic expert, original position)``.
        local_tokens_per_expert_cumsum: Pointer to int32 ``[NUM_EXPERTS]``
            inclusive prefix sums of this source rank's expert histogram.
        gathered_tokens_per_expert: Pointer to int32
            ``[EP_SIZE, NUM_EXPERTS]`` per-source expert histograms.
        expert_rank_allocations: Pointer to int32
            ``[NUM_EXPERTS, EP_SIZE]`` final route allocations; each expert row
            partitions its global ordinal stream by destination.
        experts_to_copy: Pointer to int32
            ``[EP_SIZE, NUM_EXPERTS_PER_GPU]`` replica-slot assignments.
        expert_replica_slots: Pointer to int32 ``[NUM_EXPERTS, EP_SIZE]``
            inverse replica-slot assignments emitted by fused placement.
        virtual_experts: Output pointer to int64 ``[NUM_ROUTES]`` rank-major
            runtime ids; the workspace exposes it as
            ``[num_tokens, router_topk]``.
        SOURCE_EP_RANK: Rank whose local routes this kernel is mapping.
        EP_SIZE: Number of expert-parallel ranks.
        NUM_EXPERTS: Number of semantic experts.
        NUM_EXPERTS_PER_GPU: Native experts and replica slots per rank.
        NUM_ROUTES: Number of local routes, ``num_tokens * router_topk``.
        BLOCK_NUM_ROUTES: Sorted routes processed by one Triton program.
        USE_REPLICA_LOOKUP: Use the fused inverse map instead of scanning every
            replica slot for each route.
    """
    positions = tl.program_id(0) * BLOCK_NUM_ROUTES + tl.arange(0, BLOCK_NUM_ROUTES)
    valid = positions < NUM_ROUTES

    # sorted in increasing expert number,
    routes = tl.load(sorted_route_indices + positions, mask=valid, other=0)
    # for each token in 'routes' which expert does it belong to?
    experts = tl.load(flat_topk_indices + routes, mask=valid, other=0).to(tl.int64)
    # for each token in 'routes', how many tokens before it for this gpu
    before_local = tl.load(
        local_tokens_per_expert_cumsum + experts - 1, mask=valid & (experts > 0), other=0
    )
    # encodes for each token, i am the nth token for my expert
    local_ordinal = positions - before_local

    # Counts from lower source ranks precede this rank's routes in the global
    # per-expert stream. SOURCE_EP_RANK is constexpr, so this loop is unrolled.
    before_rank = tl.zeros((BLOCK_NUM_ROUTES,), dtype=tl.int32)
    for source in tl.static_range(0, SOURCE_EP_RANK):
        before_rank += tl.load(
            gathered_tokens_per_expert + source * NUM_EXPERTS + experts, mask=valid, other=0
        )
    global_ordinal = local_ordinal + before_rank

    # expert_rank_allocations is a prefix partition of each expert's global
    # stream. The first boundary above the ordinal selects its destination.
    cumulative = tl.zeros((BLOCK_NUM_ROUTES,), dtype=tl.int32)
    destination = tl.zeros((BLOCK_NUM_ROUTES,), dtype=tl.int32)
    unresolved = valid
    for rank in tl.static_range(0, EP_SIZE):
        cumulative += tl.load(
            expert_rank_allocations + experts * EP_SIZE + rank, mask=valid, other=0
        )
        # for the nth token of an expert, if this rank is expected to carry more
        # than n tokens from this expert, mark this rank as needed
        choose = unresolved & (global_ordinal < cumulative)
        destination = tl.where(choose, rank, destination)
        unresolved = unresolved & ~choose

    # Native routes use the owner's local expert id. Remote routes look up the
    # slot holding that expert's copied weights on the selected destination.
    owner = experts // NUM_EXPERTS_PER_GPU
    owned_local = experts % NUM_EXPERTS_PER_GPU
    if USE_REPLICA_LOOKUP:
        replica_slot = tl.load(
            expert_replica_slots + experts * EP_SIZE + destination,
            mask=valid & (destination != owner),
            other=-1,
        ).to(tl.int64)
    else:
        replica_slot = tl.full((BLOCK_NUM_ROUTES,), -1, dtype=tl.int64)
        for slot in tl.static_range(0, NUM_EXPERTS_PER_GPU):
            copied = tl.load(
                experts_to_copy + destination * NUM_EXPERTS_PER_GPU + slot, mask=valid, other=-1
            ).to(tl.int64)
            replica_slot = tl.where((replica_slot < 0) & (copied == experts), slot, replica_slot)
    runtime_local = tl.where(destination == owner, owned_local, NUM_EXPERTS_PER_GPU + replica_slot)
    virtual = destination.to(tl.int64) * (2 * NUM_EXPERTS_PER_GPU) + runtime_local
    tl.store(virtual_experts + routes, virtual, mask=valid)


def _validate_inputs(
    topk_indices: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    ep_group: dist.ProcessGroup,
    workspace: ReplicaPlannerWorkspace,
) -> tuple[int, int, int, int]:
    """Validate the fixed-shape, even-expert planner contract.

    Args:
        topk_indices: Contiguous CUDA int32/int64 tensor
            ``[num_tokens, router_topk]`` containing the semantic expert
            selected for every local top-k route.
        tokens_per_expert: Contiguous CUDA int32 tensor ``[num_experts]``
            containing this rank's histogram of ``topk_indices``.
        ep_group: Initialized expert-parallel process group. Its world size is
            ``ep_size`` and determines ownership and collective participants.
        workspace: Reusable buffers whose fixed ``(num_tokens, router_topk,
            num_experts, ep_size)`` shape and CUDA device must match the inputs.

    Returns:
        ``(num_tokens, router_topk, num_experts, ep_size)`` as Python integers
        for kernel launch configuration.
    """
    if not HAVE_TRITON:
        raise ImportError("The replica planner requires Triton.")
    if not dist.is_initialized():
        raise RuntimeError("The replica planner requires initialized torch.distributed.")
    if not topk_indices.is_cuda or not tokens_per_expert.is_cuda:
        raise ValueError("Replica planner inputs must be CUDA tensors.")
    if topk_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"topk_indices must be int32 or int64, got {topk_indices.dtype}.")
    if tokens_per_expert.dtype != torch.int32:
        raise TypeError(f"tokens_per_expert must be int32, got {tokens_per_expert.dtype}.")
    if topk_indices.ndim != 2:
        raise ValueError(
            "topk_indices must have shape [num_tokens, router_topk], got "
            f"{tuple(topk_indices.shape)}."
        )
    if tokens_per_expert.ndim != 1:
        raise ValueError(
            "tokens_per_expert must have shape [num_experts], got "
            f"{tuple(tokens_per_expert.shape)}."
        )
    if not topk_indices.is_contiguous() or not tokens_per_expert.is_contiguous():
        raise ValueError("Replica planner inputs must be contiguous.")

    ep_size = dist.get_world_size(group=ep_group)
    num_tokens, router_topk = map(int, topk_indices.shape)
    num_experts = int(tokens_per_expert.numel())
    if num_experts % ep_size != 0:
        raise ValueError(
            "Replica planner requires equal experts per rank, got "
            f"num_experts={num_experts}, ep_size={ep_size}."
        )
    expected = (
        workspace.num_tokens,
        workspace.router_topk,
        workspace.num_experts,
        workspace.ep_size,
    )
    actual = (num_tokens, router_topk, num_experts, ep_size)
    if actual != expected:
        raise ValueError(
            f"Replica planner workspace shape mismatch: expected {expected}, got {actual}."
        )
    if workspace.num_local_experts != num_experts // ep_size:
        raise ValueError("Replica planner slot count must equal the number of local experts.")
    if workspace.gathered_counts.device != topk_indices.device:
        raise ValueError("Replica planner workspace and inputs must be on the same CUDA device.")
    return num_tokens, router_topk, num_experts, ep_size


def _launch_replica_placement(
    workspace: ReplicaPlannerWorkspace,
    local_tokens_per_expert: torch.Tensor,
    *,
    source_rank: int | None = None,
    rank_route_capacity: int,
    ep_size: int,
    num_experts: int,
    num_local_experts: int,
    use_fused_placement: bool = True,
    write_replica_lookup: bool = False,
    write_local_cumsum: bool = True,
) -> None:
    """Launch the uniform multi-program replica placement path.

    ``source_rank`` and ``use_fused_placement`` remain accepted for compatibility
    with the standalone benchmark; neither selects a placement kernel.
    """
    block_ep_size = triton.next_power_of_2(ep_size)
    block_num_experts_per_gpu = triton.next_power_of_2(num_local_experts)

    block_num_experts = triton.next_power_of_2(num_experts)
    _initialize_allocation_kernel[(num_experts,)](
        workspace.gathered_counts,
        workspace.expert_totals,
        workspace.allocation,
        EP_SIZE=ep_size,
        NUM_EXPERTS=num_experts,
        NUM_EXPERTS_PER_GPU=num_local_experts,
        BLOCK_EP_SIZE=block_ep_size,
    )
    _compute_group_balance_kernel[(ep_size,)](
        workspace.expert_totals,
        workspace.balance,
        RANK_ROUTE_CAPACITY=rank_route_capacity,
        NUM_EXPERTS_PER_GPU=num_local_experts,
        BLOCK_NUM_EXPERTS_PER_GPU=block_num_experts_per_gpu,
    )
    _choose_receiver_quotas_kernel[(1,)](
        workspace.balance, workspace.receiver_quotas, EP_SIZE=ep_size, BLOCK_EP_SIZE=block_ep_size
    )
    _allocate_migrations_kernel[(ep_size,)](
        workspace.expert_totals,
        workspace.receiver_quotas,
        workspace.allocation,
        EP_SIZE=ep_size,
        NUM_EXPERTS_PER_GPU=num_local_experts,
        BLOCK_EP_SIZE=block_ep_size,
        BLOCK_NUM_EXPERTS_PER_GPU=block_num_experts_per_gpu,
    )
    _select_replica_experts_kernel[(ep_size,)](
        workspace.allocation,
        workspace.experts_to_copy,
        workspace.expert_replica_slots,
        EP_SIZE=ep_size,
        NUM_EXPERTS=num_experts,
        NUM_EXPERTS_PER_GPU=num_local_experts,
        BLOCK_NUM_EXPERTS=block_num_experts,
        WRITE_REPLICA_LOOKUP=write_replica_lookup,
    )
    if write_local_cumsum:
        torch.cumsum(local_tokens_per_expert, dim=0, out=workspace.local_count_cumsum)


def _launch_replica_route_mapping(
    workspace: ReplicaPlannerWorkspace,
    flat_topk_indices: torch.Tensor,
    *,
    source_rank: int,
    ep_size: int,
    num_experts: int,
    num_local_experts: int,
    num_routes: int,
    use_fused_replica_lookup: bool,
    launch_sort: bool = True,
) -> None:
    """Launch stable route ordering and native-or-replica id mapping."""
    block_num_routes = 256
    if launch_sort:
        _launch_stable_route_bucket_sort(
            workspace, flat_topk_indices, num_experts=num_experts, num_routes=num_routes
        )
    _map_virtual_experts_kernel[(triton.cdiv(num_routes, block_num_routes),)](
        flat_topk_indices,
        workspace.sorted_route_indices,
        workspace.local_count_cumsum,
        workspace.gathered_counts,
        workspace.allocation,
        workspace.experts_to_copy,
        workspace.expert_replica_slots,
        workspace.virtual_experts,
        SOURCE_EP_RANK=source_rank,
        EP_SIZE=ep_size,
        NUM_EXPERTS=num_experts,
        NUM_EXPERTS_PER_GPU=num_local_experts,
        NUM_ROUTES=num_routes,
        BLOCK_NUM_ROUTES=block_num_routes,
        USE_REPLICA_LOOKUP=use_fused_replica_lookup,
    )


def plan_replica_routes(
    topk_indices: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    ep_group: dist.ProcessGroup,
    workspace: ReplicaPlannerWorkspace,
    *,
    use_fused_placement: bool = True,
    use_fused_replica_lookup: bool | None = None,
    overlap_route_sort: bool = True,
    on_placement_ready: Callable[[ReplicaPlan], None] | None = None,
) -> ReplicaPlan:
    """Plan deterministic replica placements for HybridEP or NCCL-EP.

    The route shape is fixed by ``workspace``. The returned tensors alias that
    workspace and remain valid until its next planner invocation. Call this
    once before CUDA-graph capture so the one-time object shape collective has
    completed; later calls use the graph-capturable tensor collective.

    Args:
        topk_indices: Contiguous CUDA int32/int64 tensor
            ``[num_tokens, router_topk]`` containing semantic expert ids in
            local token/top-k order.
        tokens_per_expert: Contiguous CUDA int32 tensor ``[num_experts]`` with
            this source rank's route count for every semantic expert.
        ep_group: Expert-parallel process group of size ``ep_size``. The
            function gathers histograms across this group and uses the group
            rank to place local routes in the global deterministic order.
        workspace: Fixed-address workspace allocated for the same
            ``(num_tokens, router_topk, num_experts, ep_size)`` shape. Its
            output buffers are overwritten.
        use_fused_placement: Compatibility switch for the optimized inverse
            replica lookup. Set this to ``False`` to run the preserved
            reference lookup path; placement always uses the scalable
            multi-program phase sequence.
        use_fused_replica_lookup: Have placement emit the inverse
            ``(expert, destination) -> replica slot`` map consumed by Phase 5.
            ``None`` follows ``use_fused_placement``.
        overlap_route_sort: Run stable local route ordering concurrently with
            placement on the workspace's fixed side stream. It is enabled only
            for EP sizes greater than four, where it overlaps the scalable
            placement phases without changing the EP=4 synchronization path.
        on_placement_ready: Optional callback invoked after
            ``experts_to_copy`` is ready. Phase 5 has already been enqueued as
            an independent sibling branch at this point; the replica runtime
            uses this boundary to start weight prefetch without making Phase 5
            wait for it.

    Returns:
        A ``ReplicaPlan`` whose ``virtual_experts`` tensor is int64
        ``[num_tokens, router_topk]`` and whose ``experts_to_copy`` tensor is
        int32 ``[ep_size, num_experts_per_gpu]``. Both tensors alias
        ``workspace`` and are valid only until its next invocation.
    """
    num_tokens, router_topk, num_experts, ep_size = _validate_inputs(
        topk_indices, tokens_per_expert, ep_group, workspace
    )
    num_local_experts = num_experts // ep_size
    num_routes = num_tokens * router_topk
    source_rank = dist.get_rank(group=ep_group)
    overlap_route_sort = overlap_route_sort and ep_size > 4
    if use_fused_replica_lookup is None:
        use_fused_replica_lookup = use_fused_placement
    if use_fused_replica_lookup and not use_fused_placement:
        raise ValueError("Fused replica lookup requires fused placement.")

    # A mismatch in tokens/top-k/experts would give ranks different kernel
    # shapes and can hang the following collective. Validate it once outside
    # the captured hot path.
    if not workspace.distributed_shape_validated:
        rank_shapes = [None] * ep_size
        local_shape = (num_tokens, router_topk, num_experts)
        dist.all_gather_object(rank_shapes, local_shape, group=ep_group)
        if any(shape != local_shape for shape in rank_shapes):
            raise ValueError(
                "Replica planner requires equal route capacity and expert count across ranks, "
                f"got {rank_shapes}."
            )
        workspace.distributed_shape_validated = True

    # Phase 1: collect the only cross-rank input. From here onward every rank
    # sees the same histograms and independently produces the same placement.
    dist.all_gather_into_tensor(
        workspace.gathered_counts.view(-1), tokens_per_expert, group=ep_group
    )

    # Route ordering depends only on local routes and their expert-count
    # prefix, while placement depends on the gathered histograms. Fork the
    # stable bucket sort onto its fixed workspace stream after producing that
    # small prefix, then join it before the route mapper consumes both results.
    if overlap_route_sort:
        torch.cumsum(tokens_per_expert, dim=0, out=workspace.local_count_cumsum)
        current_stream = torch.cuda.current_stream(topk_indices.device)
        workspace.sort_stream.wait_stream(current_stream)
        with torch.cuda.stream(workspace.sort_stream):
            _launch_stable_route_bucket_sort(
                workspace, topk_indices.reshape(-1), num_experts=num_experts, num_routes=num_routes
            )

    # Phases 2-4: construct allocation[expert, destination], then turn its
    # remote nonzero entries into a deterministic replica-weight list per rank.
    _launch_replica_placement(
        workspace,
        tokens_per_expert,
        rank_route_capacity=num_routes,
        ep_size=ep_size,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        write_replica_lookup=use_fused_replica_lookup,
        write_local_cumsum=not overlap_route_sort,
    )

    plan = ReplicaPlan(
        virtual_experts=workspace.virtual_experts, experts_to_copy=workspace.experts_to_copy
    )
    # Phase 5: allocations contain counts, not individual route identities.
    # Establish a stable per-expert route order, then locate every route in
    # the destination segments described by allocation. When placement and
    # the stable sort are independent branches, keep Phase 5 on the sort
    # branch and make it wait only for placement, then launch weight prefetch
    # as a sibling branch. Enqueuing Phase 5 after the prefetch callback makes
    # CUDA-graph capture incorrectly put weight completion on the mapping
    # critical path.
    if overlap_route_sort:
        workspace.sort_stream.wait_stream(current_stream)
        with torch.cuda.stream(workspace.sort_stream):
            _launch_replica_route_mapping(
                workspace,
                topk_indices.reshape(-1),
                source_rank=source_rank,
                ep_size=ep_size,
                num_experts=num_experts,
                num_local_experts=num_local_experts,
                num_routes=num_routes,
                use_fused_replica_lookup=use_fused_replica_lookup,
                launch_sort=False,
            )
        if on_placement_ready is not None:
            on_placement_ready(plan)
        current_stream.wait_stream(workspace.sort_stream)
    else:
        if on_placement_ready is not None:
            on_placement_ready(plan)
        _launch_replica_route_mapping(
            workspace,
            topk_indices.reshape(-1),
            source_rank=source_rank,
            ep_size=ep_size,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            num_routes=num_routes,
            use_fused_replica_lookup=use_fused_replica_lookup,
            launch_sort=True,
        )
    return plan


def map_replica_plan_to_hybridep(
    plan: ReplicaPlan, topk_probs: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a compact replica plan to HybridEP's dense routing inputs.

    The planner emits one virtual id and probability per top-k route. HybridEP
    consumes dense ``[num_tokens, ep_size * 2 * num_experts_per_gpu]`` tensors
    instead, so this is a scatter-only representation change; placement
    decisions are already final.

    Args:
        plan: Compact planner result. ``plan.virtual_experts`` is int64
            ``[num_tokens, router_topk]`` and supplies the dense column index
            for each route.
        topk_probs: CUDA tensor ``[num_tokens, router_topk]`` containing the
            router probability associated with each virtual route.
        num_experts: Number of runtime virtual experts,
            ``ep_size * 2 * num_experts_per_gpu`` (twice the semantic expert
            count), and therefore the dense output width.

    Returns:
        A boolean routing map ``[S, num_experts]`` and float32 dense
        probabilities ``[S, num_experts]`` for HybridEP dispatch.
    """
    if plan.virtual_experts.shape != topk_probs.shape:
        raise ValueError(
            "Replica virtual experts and top-k probabilities must have the same shape, got "
            f"{tuple(plan.virtual_experts.shape)} and {tuple(topk_probs.shape)}."
        )
    num_tokens = int(plan.virtual_experts.shape[0])
    routing_map = torch.zeros(
        (num_tokens, num_experts), dtype=torch.bool, device=plan.virtual_experts.device
    )
    dense_probs = torch.zeros(
        (num_tokens, num_experts), dtype=torch.float32, device=topk_probs.device
    )
    routing_map.scatter_(1, plan.virtual_experts, True)
    dense_probs.scatter_(1, plan.virtual_experts, topk_probs.to(torch.float32))
    return routing_map, dense_probs
