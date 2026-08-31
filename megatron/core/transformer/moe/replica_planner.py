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
from enum import Enum, auto

import torch
import torch.distributed as dist

from megatron.core.fp8_utils import is_mxfp8tensor
from megatron.core.transformer.moe.replica_weight_triton import (
    MAX_REPLICA_WEIGHT_SMS,
    compile_replica_weight_kernels,
    launch_replica_grad_reduce,
    launch_replica_weight_prefetch,
)
from megatron.core.utils import nvtx_decorator

try:
    from transformer_engine.pytorch.module.base import get_dummy_wgrad
except ImportError:
    get_dummy_wgrad = None

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


# Grid width for the planner's persistent and cooperative launches. It stays
# below the resident-block limit of every supported device while keeping enough
# programs to saturate one.
_MAX_PLANNER_PROGRAMS = 128


def _next_power_of_two(value: int) -> int:
    """Round a positive dimension up to the power of two Triton tiles use."""
    return 1 << (value - 1).bit_length()


def _route_partition_count(num_routes: int) -> int:
    """Return the route partitioning shared by route ranking and route mapping.

    Route ranking uses a cooperative launch, which must keep the complete grid
    resident. The default is safe on the validated GB300 workload and retains
    substantially more parallelism than a conservative 64-program setting.
    """
    return min(_MAX_PLANNER_PROGRAMS, num_routes)


def _discard_runtime_parameter_grad(parameter: torch.nn.Parameter) -> None:
    """Drop TE's throwaway leaf grad after its fused wgrad reaches runtime staging."""
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
    placement_grid_sync: torch.Tensor
    # Per-expert destination segment ends, rebased into this rank's local
    # ordinal space and padded to a power of two so the route mapper can
    # binary-search them.
    destination_boundaries: torch.Tensor
    # Inverse replica lookup produced by fused placement. This turns the
    # route mapper's replica-slot search into one indexed load.
    expert_replica_slots: torch.Tensor
    # Stable per-expert route ordinals, packed with the expert id and split
    # into the partition-local part and the per-partition prefix.
    sort_route_metadata: torch.Tensor
    sort_partition_counts: torch.Tensor
    sort_grid_sync: torch.Tensor
    sort_stream: torch.cuda.Stream
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
            placement_grid_sync=torch.zeros(2, **int32),
            destination_boundaries=torch.empty((num_experts, _next_power_of_two(ep_size)), **int32),
            expert_replica_slots=torch.empty((num_experts, ep_size), **int32),
            sort_route_metadata=torch.empty(num_routes, **int32),
            sort_partition_counts=torch.empty(
                (_route_partition_count(num_routes), num_experts), **int32
            ),
            sort_grid_sync=torch.zeros(2, **int32),
            sort_stream=torch.cuda.Stream(device=device),
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


def _bf16_storage_ptrs(
    sources: tuple[torch.Tensor, ...],
    *,
    member_numel: int,
    device: torch.device,
    label: str,
) -> tuple[tuple[int], ...]:
    """Validate BF16 runtime storage and return its pointer signature."""
    pointers = []
    for index, source in enumerate(sources):
        if (
            source.dtype != torch.bfloat16
            or source.device != device
            or source.numel() != member_numel
            or not source.is_contiguous()
        ):
            raise ValueError(
                f"{label} expert {index} requires contiguous BF16 storage with "
                f"{member_numel} elements on {device}; got dtype={source.dtype}, "
                f"shape={tuple(source.shape)}, device={source.device}."
            )
        pointers.append((source.data_ptr(),))
    return tuple(pointers)


def _mxfp8_storage_ptrs(
    sources: tuple[torch.Tensor, ...],
    *,
    member_shape: tuple[int, int],
    rowwise_scale_shape: tuple[int, ...],
    columnwise_scale_shape: tuple[int, ...],
    device: torch.device,
    label: str,
) -> tuple[tuple[int, int, int, int], ...]:
    """Validate complete MXFP8 runtime storage and return its pointer signature."""
    pointers = []
    for index, source in enumerate(sources):
        storage = (
            getattr(source, "_rowwise_data", None),
            getattr(source, "_rowwise_scale_inv", None),
            getattr(source, "_columnwise_data", None),
            getattr(source, "_columnwise_scale_inv", None),
        )
        shapes = (
            member_shape,
            rowwise_scale_shape,
            member_shape,
            columnwise_scale_shape,
        )
        if (
            tuple(source.shape) != member_shape
            or source.device != device
            or any(
                tensor is None
                or tensor.dtype != torch.uint8
                or not tensor.is_contiguous()
                or tuple(tensor.shape) != shape
                for tensor, shape in zip(storage, shapes)
            )
        ):
            actual = tuple(
                None if tensor is None else (tensor.dtype, tuple(tensor.shape))
                for tensor in storage
            )
            raise ValueError(
                f"{label} expert {index} requires contiguous MXFP8 data/scales with "
                f"shapes={shapes} on {device}; got shape={tuple(source.shape)}, "
                f"device={source.device}, storage={actual}."
            )
        pointers.append(tuple(tensor.data_ptr() for tensor in storage))
    return tuple(pointers)


def _validate_mxfp8_members(
    members: tuple[torch.Tensor, ...], *, member_shape: tuple[int, int], backend_name: str
) -> tuple[tuple[int, ...], tuple[int, ...], torch.device]:
    """Validate native MXFP8 member storage and return its two scale shapes."""
    if not members:
        raise ValueError(f"{backend_name} did not find any MXFP8 expert weights.")
    for index, member in enumerate(members):
        if not is_mxfp8tensor(member):
            raise ValueError(
                f"{backend_name} expected MXFP8 expert {index}, got {type(member).__name__}."
            )
    first = members[0]
    if first._rowwise_scale_inv is None or first._columnwise_scale_inv is None:
        raise ValueError(f"{backend_name} MXFP8 weights require rowwise and columnwise scales.")
    rowwise_scale_shape = tuple(first._rowwise_scale_inv.shape)
    columnwise_scale_shape = tuple(first._columnwise_scale_inv.shape)
    device = first.device
    _mxfp8_storage_ptrs(
        members,
        member_shape=member_shape,
        rowwise_scale_shape=rowwise_scale_shape,
        columnwise_scale_shape=columnwise_scale_shape,
        device=device,
        label=backend_name,
    )
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


class _WeightDirection(Enum):
    FORWARD = auto()
    BACKWARD = auto()


@dataclass(slots=True)
class _DirectionalBinding:
    data_bases: torch.Tensor
    scale_bases: torch.Tensor | None = None
    source_tensors: tuple[torch.Tensor, ...] | None = None
    source_ptrs: tuple[tuple[int, ...], ...] | None = None
    host_pointer_table: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class _PrefetchResources:
    sources: tuple[torch.Tensor, ...]
    scale_sources: tuple[torch.Tensor, ...] | None
    arena: torch.Tensor
    handle: object
    grid_barrier: torch.Tensor
    orientation: str


@dataclass(slots=True)
class _CuTeDSLReplicaProjection:
    """One projection and its stable native/virtual runtime storage.

    Expert backward writes native and replica wgrads into bridge-owned staging.
    The replica reduction accumulates the virtual contributions into the native
    staging, which is then handed to the optimizer parameters through autograd.
    """

    name: str
    device: torch.device
    weight_format: str
    parameters: tuple[torch.nn.Parameter, ...]
    gtp_leader: torch.nn.Parameter | None
    source_tensors: tuple[torch.Tensor, ...]
    forward: _DirectionalBinding
    backward: _DirectionalBinding
    native_grad_bases: torch.Tensor
    member_shape: tuple[int, int]
    member_numel: int
    rowwise_scale_shape: tuple[int, ...] | None
    columnwise_scale_shape: tuple[int, ...] | None
    virtual_weight: tuple[torch.Tensor, ...]
    virtual_grad: torch.Tensor
    native_grad: torch.Tensor | None = None
    runtime_parameters: tuple[torch.nn.Parameter, ...] | None = None
    source_storage_ptrs: tuple[tuple[int, ...], ...] | None = None
    native_grad_ptrs: tuple[int, ...] | None = None
    runtime_bound: bool = False

    def binding(self, direction: _WeightDirection) -> _DirectionalBinding:
        return self.backward if direction is _WeightDirection.BACKWARD else self.forward

    def bind_materialized_weights(
        self,
        materialized_weights: tuple[torch.Tensor, ...],
        direction: _WeightDirection,
    ) -> None:
        """Bind one stable directional GTP gather without copying its payload."""
        if len(materialized_weights) != len(self.parameters):
            raise RuntimeError(
                f"GTP materialized {len(materialized_weights)} {self.name} weights, "
                f"expected {len(self.parameters)}."
            )
        binding = self.binding(direction)
        direction_name = direction.name.lower()
        if self.weight_format == "bf16":
            for index, source in enumerate(materialized_weights):
                if (
                    tuple(source.shape) != self.member_shape
                    or source.dtype != torch.bfloat16
                    or source.device != self.device
                    or not source.is_contiguous()
                ):
                    raise RuntimeError(
                        f"GTP BF16 {direction_name} gather returned invalid {self.name} "
                        f"storage for replica expert {index}."
                    )
            source_ptrs = tuple((source.data_ptr(),) for source in materialized_weights)
        else:
            data_field, scale_field, scale_shape = (
                ("_columnwise_data", "_columnwise_scale_inv", self.columnwise_scale_shape)
                if direction is _WeightDirection.BACKWARD
                else ("_rowwise_data", "_rowwise_scale_inv", self.rowwise_scale_shape)
            )
            source_ptrs = []
            for index, source in enumerate(materialized_weights):
                data = getattr(source, data_field, None)
                scale = getattr(source, scale_field, None)
                if (
                    tuple(source.shape) != self.member_shape
                    or source.device != self.device
                    or data is None
                    or scale is None
                    or data.dtype != torch.uint8
                    or scale.dtype != torch.uint8
                    or not data.is_contiguous()
                    or not scale.is_contiguous()
                    or tuple(data.shape) != self.member_shape
                    or tuple(scale.shape) != scale_shape
                ):
                    raise RuntimeError(
                        f"GTP MXFP8 {direction_name} gather returned invalid {self.name} "
                        f"storage for replica expert {index}."
                    )
                source_ptrs.append((data.data_ptr(), scale.data_ptr()))
            source_ptrs = tuple(source_ptrs)

        if binding.source_ptrs is not None:
            if source_ptrs != binding.source_ptrs:
                raise RuntimeError(
                    f"Replica CuTeDSL GTP {direction_name} all-gather storage changed after "
                    "direct binding; this would invalidate CUDA-graph source pointers."
                )
        else:
            tables = (binding.data_bases, binding.scale_bases)
            tables = tuple(table for table in tables if table is not None)
            if binding.host_pointer_table is None or len(tables) != len(source_ptrs[0]):
                raise RuntimeError("Replica CuTeDSL GTP direct binding lost pointer storage.")
            binding.source_ptrs = source_ptrs
            for component, table in enumerate(tables):
                host_row = binding.host_pointer_table[component]
                host_row.copy_(
                    torch.tensor([ptrs[component] for ptrs in source_ptrs], dtype=torch.int64)
                )
                table.copy_(host_row, non_blocking=True)

        binding.source_tensors = materialized_weights
        if self.weight_format == "bf16":
            self.source_tensors = materialized_weights
            if self.runtime_parameters is not None:
                for parameter, source in zip(self.runtime_parameters, materialized_weights):
                    parameter.data = source
            return

        component_offset = 2 if direction is _WeightDirection.BACKWARD else 0
        for destination, source in zip(self.source_tensors, materialized_weights):
            setattr(destination, data_field, getattr(source, data_field))
            setattr(destination, scale_field, getattr(source, scale_field))
        if self.runtime_parameters is not None:
            for destination, source in zip(
                self.runtime_parameters, materialized_weights
            ):
                setattr(destination, data_field, getattr(source, data_field))
                setattr(destination, scale_field, getattr(source, scale_field))
        if self.source_storage_ptrs is not None:
            updated = [list(ptrs) for ptrs in self.source_storage_ptrs]
            for expert, ptrs in enumerate(source_ptrs):
                updated[expert][component_offset : component_offset + 2] = ptrs
            self.source_storage_ptrs = tuple(tuple(ptrs) for ptrs in updated)

    def _native_grads(self) -> tuple[torch.Tensor, ...]:
        if self.native_grad is None:
            raise RuntimeError(f"Replica CuTeDSL {self.name} lost native gradient staging.")
        return tuple(self.native_grad)

    def prepare_runtime_parameters(self, grad_dtype: torch.dtype) -> None:
        """Bind final DDP/GTP storage once, then validate pointer stability."""
        sources = (
            tuple(_parameter_storage(parameter) for parameter in self.parameters)
            if self.gtp_leader is None and self.weight_format == "bf16"
            else self.source_tensors
        )
        if len(sources) != len(self.parameters):
            raise RuntimeError(
                f"Replica CuTeDSL {self.name} expected {len(self.parameters)} native weights, "
                f"got {len(sources)}."
            )
        storage_ptrs = (
            _bf16_storage_ptrs(
                sources,
                member_numel=self.member_numel,
                device=self.device,
                label=f"Replica CuTeDSL {self.name}",
            )
            if self.weight_format == "bf16"
            else _mxfp8_storage_ptrs(
                sources,
                member_shape=self.member_shape,
                rowwise_scale_shape=self.rowwise_scale_shape,
                columnwise_scale_shape=self.columnwise_scale_shape,
                device=self.device,
                label=f"Replica CuTeDSL {self.name}",
            )
        )
        directional_gtp_bf16 = self.gtp_leader is not None and self.weight_format == "bf16"
        if not directional_gtp_bf16 and self.source_storage_ptrs is None:
            self.source_storage_ptrs = storage_ptrs
            if self.gtp_leader is None:
                tables = (
                    (self.forward.data_bases, 0),
                    (self.forward.scale_bases, 1),
                    (self.backward.data_bases, 2),
                    (self.backward.scale_bases, 3),
                )
                for table, component in tables:
                    if table is not None and component < len(storage_ptrs[0]):
                        table.copy_(
                            torch.tensor(
                                [ptrs[component] for ptrs in storage_ptrs], dtype=torch.int64,
                                device=self.device,
                            )
                        )
        elif not directional_gtp_bf16 and storage_ptrs != self.source_storage_ptrs:
            raise RuntimeError(
                f"Replica CuTeDSL {self.name} parameter storage changed after binding; "
                "this would invalidate CUDA-graph source pointers."
            )

        native_grads = self._native_grads()
        native_grad_ptrs = []
        for index, grad in enumerate(native_grads):
            if (
                grad.dtype != grad_dtype
                or grad.device != self.device
                or grad.numel() != self.member_numel
                or not grad.is_contiguous()
            ):
                raise ValueError(
                    f"Replica CuTeDSL {self.name} native grad {index} must be contiguous "
                    f"{grad_dtype} with {self.member_numel} elements on {self.device}; got "
                    f"dtype={grad.dtype}, shape={tuple(grad.shape)}, device={grad.device}."
                )
            native_grad_ptrs.append(grad.data_ptr())
        native_grad_ptrs = tuple(native_grad_ptrs)
        if self.native_grad_ptrs is None:
            self.native_grad_ptrs = native_grad_ptrs
            self.native_grad_bases.copy_(
                torch.tensor(native_grad_ptrs, dtype=torch.int64, device=self.device)
            )
        elif native_grad_ptrs != self.native_grad_ptrs:
            raise RuntimeError(
                f"Replica CuTeDSL {self.name} native-grad storage changed after binding; "
                "this would invalidate CUDA-graph destination pointers."
            )

        self.source_tensors = sources
        weights = sources + tuple(self.virtual_weight)
        grads = native_grads + tuple(self.virtual_grad)
        if not self.runtime_bound:
            self.bind_runtime_parameters(weights, grads)
        else:
            self.validate_runtime_parameters(weights, grads)

    def bind_runtime_parameters(self, weights, grads) -> None:
        """Create the stable native-then-virtual TE parameter sequence once."""
        runtime_parameters = []
        for weight, grad in zip(weights, grads):
            parameter = torch.nn.Parameter(weight, requires_grad=True)
            parameter.main_grad = grad
            parameter.grad_added_to_main_grad = True
            parameter.overwrite_main_grad = True
            parameter.register_post_accumulate_grad_hook(_discard_runtime_parameter_grad)
            runtime_parameters.append(parameter)
        self.runtime_parameters = tuple(runtime_parameters)
        self.runtime_bound = True

    def validate_runtime_parameters(self, weights, grads) -> None:
        """Validate stable storage while refreshing directional GTP gradients."""
        if self.runtime_parameters is None:
            raise RuntimeError(f"Replica CuTeDSL {self.name} lost its runtime parameters.")
        for parameter, weight, grad in zip(self.runtime_parameters, weights, grads):
            storage_matches = (
                parameter.data_ptr() == weight.data_ptr()
                if self.weight_format == "bf16"
                else all(
                    getattr(parameter, field).data_ptr()
                    == getattr(weight, field).data_ptr()
                    for field in (
                        "_rowwise_data",
                        "_rowwise_scale_inv",
                        "_columnwise_data",
                        "_columnwise_scale_inv",
                    )
                )
            )
            if not storage_matches:
                raise RuntimeError(
                    f"Replica CuTeDSL {self.name} runtime weight storage changed after binding."
                )
            runtime_grad = getattr(parameter, "main_grad", None)
            if runtime_grad is None or runtime_grad.data_ptr() != grad.data_ptr():
                raise RuntimeError(
                    f"Replica CuTeDSL {self.name} runtime main-grad storage changed after "
                    "binding."
                )
            parameter.grad_added_to_main_grad = True
            parameter.overwrite_main_grad = True

    def destroy(self) -> None:
        if self.runtime_parameters is not None:
            for parameter in self.runtime_parameters:
                parameter.main_grad = None
        self.runtime_parameters = None
        self.runtime_bound = False


@dataclass(frozen=True, slots=True)
class _ReplicaCuTeDSLWorkspaceConfig:
    world_size: int
    num_local_experts: int
    member_shapes: tuple[tuple[int, int], tuple[int, int]]
    weight_format: str
    rowwise_scale_shapes: tuple[tuple[int, ...], tuple[int, ...]] | None
    columnwise_scale_shapes: tuple[tuple[int, ...], tuple[int, ...]] | None
    grad_dtype: torch.dtype
    num_sms: int


class _ReplicaCuTeDSLWorkspace:
    """Fixed-shape symmetric arenas shared by every compatible MoE layer."""

    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        device: torch.device,
        config: _ReplicaCuTeDSLWorkspaceConfig,
    ) -> None:
        try:
            import torch.distributed._symmetric_memory as symm_mem
        except ImportError as exc:
            raise ImportError(
                "Replica CuTeDSL weights require torch.distributed._symmetric_memory."
            ) from exc

        self.group = group
        self.device = device
        self.config = config
        self.world_size = config.world_size
        self.num_local_experts = config.num_local_experts
        self.member_shapes = config.member_shapes
        self.member_numels = tuple(math.prod(shape) for shape in config.member_shapes)
        self.weight_format = config.weight_format
        self.rowwise_scale_shapes = config.rowwise_scale_shapes
        self.columnwise_scale_shapes = config.columnwise_scale_shapes
        self.grad_dtype = config.grad_dtype
        self.rowwise_scale_numels = (
            tuple(math.prod(shape) for shape in self.rowwise_scale_shapes)
            if self.rowwise_scale_shapes is not None
            else None
        )
        self.columnwise_scale_numels = (
            tuple(math.prod(shape) for shape in self.columnwise_scale_shapes)
            if self.columnwise_scale_shapes is not None
            else None
        )
        self.num_sms = config.num_sms

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
                weight_arena_numel = arena_numel
                weight_dtype = torch.bfloat16
            elif self.weight_format == "mxfp8":
                if self.rowwise_scale_numels is None or self.columnwise_scale_numels is None:
                    raise ValueError("MXFP8 replica weights require rowwise and columnwise scales.")
                # Forward consumes rowwise MXFP8 storage and backward consumes
                # columnwise storage only after an explicit orientation prefetch.
                # Retaining both would double the constant weight buffer despite
                # their disjoint lifetimes.
                weight_arena_numel = self.num_local_experts * sum(
                    member + max(rowwise_scale, columnwise_scale)
                    for member, rowwise_scale, columnwise_scale in zip(
                        self.member_numels, self.rowwise_scale_numels, self.columnwise_scale_numels
                    )
                )
                weight_dtype = torch.uint8
            else:
                raise ValueError(f"Unsupported replica weight format {self.weight_format!r}.")
            self.weight_arena = symm_mem.empty(
                weight_arena_numel, dtype=weight_dtype, device=device
            )
            self.weight_handle = symm_mem.rendezvous(self.weight_arena, group)
            self.grad_arena = symm_mem.empty(arena_numel, dtype=self.grad_dtype, device=device)
            self.grad_handle = symm_mem.rendezvous(self.grad_arena, group)
        except RuntimeError as exc:
            raise RuntimeError(
                "Replica CuTeDSL could not allocate PyTorch native symmetric memory for the "
                "EP group. The initial implementation requires a single NVLink domain."
            ) from exc

        self.weight_arena.zero_()
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
        self._native_projection_grad_storage = {}
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

    def validate(self, config: _ReplicaCuTeDSLWorkspaceConfig) -> None:
        """Reject heterogeneous layers instead of creating a shape-keyed memory pool."""
        if config != self.config:
            raise ValueError(
                "All replica-planned MoE layers on an EP group must share one CuTeDSL "
                f"weight shape and launch configuration; expected {self.config}, got {config}."
            )

    def projection_views(self, projection_index: int) -> tuple[tuple, torch.Tensor]:
        """Return virtual runtime weights and gradients for one projection."""
        grad_offset = self.num_local_experts * sum(self.member_numels[:projection_index])
        member_numel = self.member_numels[projection_index]
        grad_numel = self.num_local_experts * member_numel
        member_shape = self.member_shapes[projection_index]
        virtual_grad = self.grad_arena.narrow(0, grad_offset, grad_numel).view(
            self.num_local_experts, *member_shape)
        if self.weight_format == "bf16":
            weight_offset = grad_offset
            weights = self.weight_arena.narrow(0, weight_offset, grad_numel).view(
                self.num_local_experts, *member_shape)
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
        rowwise_data = self.weight_arena.narrow(
            0, projection_offset, self.num_local_experts * member_numel
        ).view(self.num_local_experts, *member_shape)
        rowwise_scale = self.weight_arena.narrow(
            0,
            projection_offset + self.num_local_experts * member_numel,
            self.num_local_experts * rowwise_scale_numel,
        ).view(self.num_local_experts, *self.rowwise_scale_shapes[projection_index])
        columnwise_data = self.weight_arena.narrow(
            0, projection_offset, self.num_local_experts * member_numel
        ).view(self.num_local_experts, *member_shape)
        columnwise_scale = self.weight_arena.narrow(
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

    def native_projection_grad_view(self, projection_index: int) -> torch.Tensor:
        """Return shared full-gradient staging for one projection."""
        cached = self._native_projection_grad_storage.get(projection_index)
        if cached is None:
            cached = torch.empty(
                (self.num_local_experts, *self.member_shapes[projection_index]),
                dtype=self.grad_dtype,
                device=self.device,
            )
            self._native_projection_grad_storage[projection_index] = cached
        return cached

    def destroy(self) -> None:
        """Release symmetric registrations while their NCCL group is still alive."""
        if self._destroyed:
            return
        torch.cuda.synchronize(self.device)
        self.resident_bridge = None
        self.resident_plan = None
        self.resident_orientation = None
        self._native_projection_grad_storage.clear()
        # Handles own NCCL window registrations. Drop them before their backing
        # tensors and, critically, before model-parallel process-group teardown.
        self.weight_handle = None
        self.grad_handle = None
        self.weight_arena = None
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
    if grad_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            "Replica CuTeDSL gradients must use torch.float32 or torch.bfloat16, "
            f"got {grad_dtype}."
        )
    requested_sms = 32 if num_sms is None else int(num_sms)
    device_sms = torch.cuda.get_device_properties(device).multi_processor_count
    effective_sms = min(requested_sms, MAX_REPLICA_WEIGHT_SMS, max(1, device_sms - 8))
    if effective_sms <= 0:
        raise ValueError(f"Replica CuTeDSL num_sms must be positive, got {num_sms}.")
    config = _ReplicaCuTeDSLWorkspaceConfig(
        int(world_size),
        int(num_local_experts),
        member_shapes,
        weight_format,
        rowwise_scale_shapes,
        columnwise_scale_shapes,
        grad_dtype,
        effective_sms,
    )
    key = (id(group), device.index)
    workspace = _replica_cutedsl_workspaces.get(key)
    if workspace is None:
        workspace = _ReplicaCuTeDSLWorkspace(group=group, device=device, config=config)
        _replica_cutedsl_workspaces[key] = workspace
    else:
        workspace.validate(config)
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
    ) -> None:
        self.group = group
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self.num_local_experts = int(num_local_experts)
        self.num_runtime_experts = 2 * self.num_local_experts
        self.last_plan = None
        self._prefetch_plan = None
        self._grad_reduce_plan = None
        self._experts_ref = weakref.ref(experts)
        self._destroyed = False

        if int(num_experts) != self.world_size * self.num_local_experts:
            raise ValueError(
                "Replica CuTeDSL weights require an even expert distribution: "
                f"num_experts={num_experts}, world_size={self.world_size}, "
                f"num_local_experts={self.num_local_experts}."
            )
        projection_specs, self.device = _collect_replica_projection_specs(
            experts,
            num_local_experts=self.num_local_experts,
            backend_name="Replica-CuTeDSL",
        )
        member_shapes = tuple(spec.member_shape for spec in projection_specs)
        self.weight_format = projection_specs[0].weight_format
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
            native_grad = self.workspace.native_projection_grad_view(projection_index)
            # GTP MXFP8 gather storage is stable and is bound directly before
            # runtime construction. Bootstrap distinct native wrappers over
            # the replica views instead of retaining an unused full weight copy.
            native_storage = (
                virtual_storage
                if spec.gtp_leader is not None and spec.weight_format == "mxfp8"
                else None
            )

            def pointer_table() -> torch.Tensor:
                return torch.empty(self.num_local_experts, dtype=torch.int64, device=self.device)

            def binding(data_bases, scale_bases=None, components=0):
                host_table = (
                    torch.empty(
                        (components, self.num_local_experts),
                        dtype=torch.int64,
                        pin_memory=True,
                    )
                    if components
                    else None
                )
                return _DirectionalBinding(data_bases, scale_bases, host_pointer_table=host_table)

            if spec.weight_format == "bf16":
                virtual_weight = virtual_storage
                native_weight = native_storage
                if spec.gtp_leader is None:
                    forward = backward = binding(pointer_table())
                else:
                    forward = binding(pointer_table(), components=1)
                    backward = binding(pointer_table(), components=1)
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
                components = 2 if spec.gtp_leader is not None else 0
                forward = binding(pointer_table(), pointer_table(), components)
                backward = binding(pointer_table(), pointer_table(), components)
            native_grad_bases = pointer_table()
            self.projections.append(
                _CuTeDSLReplicaProjection(
                    name=f"FC{projection_index + 1}",
                    device=self.device,
                    weight_format=spec.weight_format,
                    parameters=spec.parameters,
                    gtp_leader=spec.gtp_leader,
                    source_tensors=(
                        native_weight
                        if native_weight is not None
                        else (() if spec.gtp_leader is not None else spec.source_tensors)
                    ),
                    forward=forward,
                    backward=backward,
                    native_grad_bases=native_grad_bases,
                    member_shape=spec.member_shape,
                    member_numel=math.prod(spec.member_shape),
                    rowwise_scale_shape=spec.rowwise_scale_shape,
                    columnwise_scale_shape=spec.columnwise_scale_shape,
                    virtual_weight=virtual_weight,
                    virtual_grad=virtual_grad,
                    native_grad=native_grad,
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
        return tuple(parameter for p in self.projections for parameter in p.parameters)

    def _materialize_gtp_source_weights(self, direction: _WeightDirection) -> None:
        """Complete GTP gathers before the replica owner-push reads native weights."""
        ordered_projections = (
            reversed(self.projections)
            if direction is _WeightDirection.BACKWARD
            else iter(self.projections)
        )
        for projection in ordered_projections:
            leader = projection.gtp_leader
            if leader is None:
                continue
            materialized = (
                leader.materialize_group_for_backward()
                if direction is _WeightDirection.BACKWARD
                else leader.materialize_group_for_forward()
            )
            materialized = (
                tuple(materialized) if isinstance(materialized, (list, tuple)) else (materialized,)
            )
            projection.bind_materialized_weights(materialized, direction)

    def prepare_runtime_parameters(self) -> None:
        """Late-bind final DDP/GTP storage and validate subsequent stability."""
        for projection in self.projections:
            projection.prepare_runtime_parameters(self.workspace.grad_dtype)

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

    def prepare_source_weights(
        self, direction: _WeightDirection = _WeightDirection.FORWARD
    ) -> None:
        """Materialize GTP weights and bind the replica runtime buffers."""
        experts = self._experts_ref()
        if experts is None:
            raise RuntimeError("Replica CuTeDSL experts were destroyed before prefetch.")
        experts.prepare_fused_impl_parameters()
        self._materialize_gtp_source_weights(direction)
        self.prepare_runtime_parameters()

    def prepare_forward(self) -> None:
        """Validate that route planning started prefetch before expert entry."""
        if self.last_plan is None:
            raise RuntimeError("Replica CuTeDSL weights require a plan before expert compute.")

    def _prefetch_resources(self, direction: _WeightDirection) -> _PrefetchResources:
        backward = direction is _WeightDirection.BACKWARD
        orientation = (
            ("columnwise" if backward else "rowwise")
            if self.weight_format == "mxfp8"
            else (
                "full_backward"
                if backward
                and any(projection.gtp_leader is not None for projection in self.projections)
                else "full_forward"
            )
        )
        bindings = tuple(projection.binding(direction) for projection in self.projections)
        return _PrefetchResources(
            sources=tuple(binding.data_bases for binding in bindings),
            scale_sources=(
                tuple(binding.scale_bases for binding in bindings)
                if self.weight_format == "mxfp8"
                else None
            ),
            arena=self.workspace.weight_arena,
            handle=self.workspace.weight_handle,
            grid_barrier=(
                self.workspace.columnwise_grid_barrier
                if orientation == "columnwise"
                else self.workspace.weight_grid_barrier
            ),
            orientation=orientation,
        )

    @torch.no_grad()
    @nvtx_decorator(message="replica_cutedsl_weight_owner_push_start")
    def start_prefetch(
        self, plan: ReplicaPlan, direction: _WeightDirection = _WeightDirection.FORWARD
    ) -> None:
        """Enqueue owner-push FC1/FC2 prefetch without blocking the caller."""
        if self._prefetch_plan is not None:
            raise RuntimeError("Replica CuTeDSL prefetch is already outstanding.")
        self._validate_plan(plan)
        # A GTP parameter stores only its local shard. Materialization consumes
        # any one-weight-ahead gather (or performs the cold synchronous gather)
        # and stages the full native experts before CuTeDSL exchanges replicas.
        self.prepare_source_weights(direction)
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
            resources = self._prefetch_resources(direction)
            resident = (
                direction is _WeightDirection.BACKWARD
                and self.workspace.resident_bridge is self
                and self.workspace.resident_plan is plan
                and self.workspace.resident_orientation == resources.orientation
            )
            if not resident:
                has_scales = resources.scale_sources is not None
                launch_replica_weight_prefetch(
                    sources=resources.sources,
                    scale_sources=resources.scale_sources,
                    arena=resources.arena,
                    peer_bases=resources.handle.buffer_ptrs_dev,
                    signal_bases=resources.handle.signal_pad_ptrs_dev,
                    experts_to_copy=plan.experts_to_copy,
                    grid_barrier=resources.grid_barrier,
                    rank=self.rank,
                    world_size=self.world_size,
                    num_local_experts=self.num_local_experts,
                    member_numels=self.workspace.member_numels,
                    rowwise_scale_numels=(
                        self.workspace.rowwise_scale_numels if has_scales else None
                    ),
                    columnwise_scale_numels=(
                        self.workspace.columnwise_scale_numels if has_scales else None
                    ),
                    orientation=resources.orientation if has_scales else None,
                    num_sms=self.workspace.num_sms,
                )
                self.workspace.resident_bridge = self
                self.workspace.resident_plan = plan
                self.workspace.resident_orientation = resources.orientation
            self.prefetch_done.record(weight_stream)
        self._prefetch_plan = plan

    @torch.no_grad()
    @nvtx_decorator(message="replica_cutedsl_weight_owner_push_wait")
    def wait_prefetch(self, plan: ReplicaPlan) -> None:
        """Insert the sole forward/restore consumer-stream dependency."""
        if self._prefetch_plan is None:
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
        self._prefetch_plan = None

    @torch.no_grad()
    @nvtx_decorator(message="replica_cutedsl_grad_reduce_start")
    def start_grad_reduce(self, plan: ReplicaPlan) -> None:
        """Enqueue replica-gradient reduction into native wgrad staging."""
        if self._grad_reduce_plan is not None:
            raise RuntimeError("Replica CuTeDSL gradient reduction is already outstanding.")
        self._validate_plan(plan)
        self.prepare_runtime_parameters()
        current_stream = torch.cuda.current_stream(self.device)
        self.grad_reduce_ready.record(current_stream)
        self.workspace.grad_stream.wait_event(self.grad_reduce_ready)
        with torch.cuda.stream(self.workspace.grad_stream):
            launch_replica_grad_reduce(
                arena=self.workspace.grad_arena,
                native_grads=tuple(
                    projection.native_grad_bases for projection in self.projections
                ),
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
        self._grad_reduce_plan = plan

    @torch.no_grad()
    @nvtx_decorator(message="replica_cutedsl_grad_reduce_wait")
    def wait_grad_reduce(self, plan: ReplicaPlan) -> tuple[torch.Tensor | None, ...]:
        """Finish replica reduction and return source-parameter wgrads."""
        if self._grad_reduce_plan is None:
            self.start_grad_reduce(plan)
        elif self._grad_reduce_plan is not plan:
            raise RuntimeError("Replica CuTeDSL grad-reduction plan changed while outstanding.")
        torch.cuda.current_stream(self.device).wait_event(self.grad_reduce_done)
        self._grad_reduce_plan = None

        reduced_gtp_grads = [None] * len(self.projections)
        # Expert backward computes FC2 before FC1. Preserve that reverse order
        # when handing full wgrads to GTP so its linked RS cascade remains valid.
        for projection_index in reversed(range(len(self.projections))):
            projection = self.projections[projection_index]
            if projection.gtp_leader is None:
                continue
            reduced_gtp_grads[projection_index] = projection.gtp_leader.wgrad_reduce_scatter(
                list(projection._native_grads())
            )

        source_grads = []
        for projection_index, projection in enumerate(self.projections):
            if projection.gtp_leader is None:
                source_grads.extend(projection._native_grads())
                continue
            reduced_grads = reduced_gtp_grads[projection_index]
            if not isinstance(reduced_grads, (list, tuple)):
                reduced_grads = (reduced_grads,)
            if len(reduced_grads) != len(projection.parameters):
                raise RuntimeError(
                    "GTP returned a different number of reduced wgrads than source parameters."
                )
            source_grads.extend(reduced_grads)
        return tuple(source_grads)

    def destroy(self) -> None:
        """Detach layer-owned TE parameters from the shared symmetric arenas."""
        if self._destroyed:
            return
        experts = self._experts_ref()
        if experts is not None:
            experts._fused_ops = None
            experts._replica_weight_bridge = None
        for projection in self.projections:
            projection.destroy()
        self.projections.clear()
        if self.workspace.resident_bridge is self:
            self.workspace.resident_bridge = None
            self.workspace.resident_plan = None
            self.workspace.resident_orientation = None
        self.last_plan = None
        self.workspace = None
        self._destroyed = True
        _replica_cutedsl_bridges.discard(self)


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


class _ReplicaBackwardAction(Enum):
    START_WEIGHT_PREFETCH = auto()
    WAIT_WEIGHT_PREFETCH = auto()
    START_GRAD_REDUCE = auto()

    def run(self, bridge: ReplicaCuTeDSLWeightBridge, plan: ReplicaPlan) -> None:
        if self is _ReplicaBackwardAction.START_WEIGHT_PREFETCH:
            bridge.start_prefetch(plan, _WeightDirection.BACKWARD)
        elif self is _ReplicaBackwardAction.WAIT_WEIGHT_PREFETCH:
            bridge.wait_prefetch(plan)
        else:
            bridge.start_grad_reduce(plan)


class _ReplicaBackwardIdentity(torch.autograd.Function):
    """Run one communication boundary while passing its tensor gradient through."""

    @staticmethod
    def forward(ctx, tensor, bridge, plan, action):
        ctx.bridge = bridge
        ctx.plan = plan
        ctx.action = action
        return tensor

    @staticmethod
    def backward(ctx, grad):
        ctx.action.run(ctx.bridge, ctx.plan)
        return grad, None, None, None


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
        source_grads = ctx.bridge.wait_grad_reduce(ctx.plan)
        if len(source_grads) != ctx.num_source_parameters:
            raise RuntimeError(
                "Replica CuTeDSL returned a different number of wgrads than source parameters."
            )

        autograd_grads = []
        for parameter, source_grad in zip(ctx.bridge.source_parameters, source_grads):
            if source_grad is None or getattr(parameter, "is_gtp_weight_remat", False):
                autograd_grads.append(source_grad)
                continue

            main_grad = getattr(parameter, "main_grad", None)
            if main_grad is None or not hasattr(parameter, "grad_added_to_main_grad"):
                # AccumulateGrad may retain its input as parameter.grad. Give it
                # independent storage because the bridge reuses native staging.
                autograd_grads.append(source_grad.clone())
                continue

            if get_dummy_wgrad is None:
                raise RuntimeError(
                    "Replica CuTeDSL fused wgrad accumulation requires Transformer Engine."
                )
            # Accumulate the completed wgrad directly in main_grad's dtype. Return a
            # parameter-dtype dummy so AccumulateGrad still invokes DDP's grad-ready hook;
            # grad_added_to_main_grad prevents DDP from accumulating the dummy again.
            main_grad.add_(source_grad)
            parameter.grad_added_to_main_grad = True
            autograd_grads.append(
                get_dummy_wgrad(
                    list(parameter.shape),
                    parameter.dtype,
                    zero=getattr(parameter, "zero_out_wgrad", False),
                )
            )

        return grad_hidden_states, *autograd_grads, None, None


def start_replica_weight_prefetch_before_combine_backward(
    combined_hidden: torch.Tensor, bridge: ReplicaCuTeDSLWeightBridge, plan: ReplicaPlan
) -> torch.Tensor:
    """Start weight communication before transport-combine backward."""
    return _ReplicaBackwardIdentity.apply(
        combined_hidden, bridge, plan, _ReplicaBackwardAction.START_WEIGHT_PREFETCH
    )


def wait_replica_weight_prefetch_before_expert_backward(
    expert_output: torch.Tensor, bridge: ReplicaCuTeDSLWeightBridge, plan: ReplicaPlan
) -> torch.Tensor:
    """Wait for weight communication immediately before expert backward."""
    return _ReplicaBackwardIdentity.apply(
        expert_output, bridge, plan, _ReplicaBackwardAction.WAIT_WEIGHT_PREFETCH
    )


def start_replica_grad_reduce_after_expert_backward(
    dispatched_hidden: torch.Tensor,
    bridge: ReplicaCuTeDSLWeightBridge,
    plan: ReplicaPlan,
) -> torch.Tensor:
    """Start replica-gradient communication after expert backward."""
    return _ReplicaBackwardIdentity.apply(
        dispatched_hidden, bridge, plan, _ReplicaBackwardAction.START_GRAD_REDUCE
    )


def wait_replica_grad_reduce_after_dispatch_backward(
    hidden_states: torch.Tensor, bridge: ReplicaCuTeDSLWeightBridge, plan: ReplicaPlan
) -> torch.Tensor:
    """Wait for replica gradients before registered-parameter DDP hooks."""
    return _ReplicaWaitGradReduce.apply(hidden_states, *bridge.source_parameters, bridge, plan)


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
def _plan_replica_placement_kernel(
    gathered_tokens_per_expert,
    global_tokens_per_expert,
    rank_load_balance,
    receiver_quotas,
    expert_rank_allocations,
    destination_boundaries,
    experts_to_copy,
    expert_replica_slots,
    grid_sync,
    RANK_ROUTE_CAPACITY: tl.constexpr,
    SOURCE_EP_RANK: tl.constexpr,
    EP_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_EXPERTS_PER_GPU: tl.constexpr,
    BLOCK_EP_SIZE: tl.constexpr,
    BLOCK_NUM_EXPERTS_PER_GPU: tl.constexpr,
    BLOCK_NUM_EXPERTS: tl.constexpr,
):
    """Compute the whole deterministic placement in one cooperative launch.

    Program ``rank`` owns the semantic experts of one EP rank and the replica
    slots of that same rank. It runs the four placement steps in order, with a
    grid barrier wherever a step needs a value another program produced:

    1. Sum the gathered histograms for its native experts and publish the
       resulting group balance relative to one rank's route capacity.
    2. Replay the deterministic rank-pairing greedy over all balances. Every
       program derives the identical quota matrix, so only the row this program
       owns has to be kept and stored.
    3. Split its own quotas across its native experts. The allocation rows this
       step rewrites belong to this program alone, so the whole
       ``[expert, destination]`` tile stays in registers.
    4. After the allocation is globally visible, rank the remote experts routed
       to this program and record them in its replica slots.

    Step 3 also emits the route mapper's search table. Each expert's routes
    form one global stream ordered by source rank, so subtracting the routes
    contributed by lower source ranks rebases the destination segment ends
    into the local ordinal space the mapper works in.

    Keeping the steps in one kernel removes four launches and, more
    importantly, keeps the quota vector and the allocation tile in registers
    instead of round-tripping them through global memory once per greedy step.
    The greedy loops stay rolled: their bodies work on whole tiles, so
    unrolling them multiplies the instruction footprint without exposing any
    parallelism the sequential dependency would allow.
    """
    rank = tl.program_id(0)
    ranks = tl.arange(0, BLOCK_EP_SIZE)
    valid_ranks = ranks < EP_SIZE
    local_experts = tl.arange(0, BLOCK_NUM_EXPERTS_PER_GPU)
    valid_local_experts = local_experts < NUM_EXPERTS_PER_GPU
    native_experts = rank * NUM_EXPERTS_PER_GPU + local_experts

    # Step 1: global route totals for this rank's native experts, and the
    # amount by which their sum exceeds one rank's route capacity.
    source_counts = tl.load(
        gathered_tokens_per_expert + ranks[:, None] * NUM_EXPERTS + native_experts[None, :],
        mask=valid_ranks[:, None] & valid_local_experts[None, :],
        other=0,
    )
    native_totals = tl.sum(source_counts, axis=0).to(tl.int32)
    routes_before_source = tl.sum(
        tl.where(ranks[:, None] < SOURCE_EP_RANK, source_counts, 0), axis=0
    ).to(tl.int32)
    tl.store(global_tokens_per_expert + native_experts, native_totals, mask=valid_local_experts)
    tl.store(
        rank_load_balance + rank,
        tl.sum(native_totals, axis=0).to(tl.int32) - RANK_ROUTE_CAPACITY,
    )

    _cooperative_grid_barrier(grid_sync, EP_SIZE)

    # Step 2: greedily pair the lowest-index most-loaded and roomiest ranks.
    # Equal balances choose the lowest rank, making the quota matrix identical
    # on every EP rank. quotas[owner, receiver] records how many routes the
    # receiver should take from that owner's experts; it does not choose the
    # experts yet.
    balances = tl.load(rank_load_balance + ranks, mask=valid_ranks, other=0)
    quotas = tl.zeros((BLOCK_EP_SIZE,), dtype=tl.int32)
    for _ in tl.range(0, EP_SIZE, 1, loop_unroll_factor=1):
        maximum = tl.max(tl.where(valid_ranks, balances, -2147483648), axis=0)
        minimum = tl.min(tl.where(valid_ranks, balances, 2147483647), axis=0)
        overloaded = tl.min(
            tl.where(valid_ranks & (balances == maximum), ranks, BLOCK_EP_SIZE), axis=0
        )
        receiver = tl.min(
            tl.where(valid_ranks & (balances == minimum), ranks, BLOCK_EP_SIZE), axis=0
        )
        active = maximum > 0
        moved = tl.where(active, -minimum, 0).to(tl.int32)
        quotas = tl.where(active & (overloaded == rank) & (ranks == receiver), moved, quotas)
        balances = tl.where(active & (ranks == overloaded), balances - moved, balances)
        balances = tl.where(active & (ranks == receiver), 0, balances)
    tl.store(receiver_quotas + rank * EP_SIZE + ranks, quotas, mask=valid_ranks)

    # Step 3: assign this rank's receive quotas to its largest expert segments.
    # Repeatedly pair the largest outstanding receiver quota with the largest
    # remaining expert segment. Expert ties choose the lowest local expert id,
    # matching the reference placement policy. A semantic expert may therefore
    # be split between its owner and one or more replica destinations.
    # allocations[local_expert, destination] starts with the native owner
    # executing every route and is migrated away from there in place.
    remaining = native_totals
    allocations = tl.where(ranks[None, :] == rank, native_totals[:, None], 0)
    for _ in tl.range(0, EP_SIZE + NUM_EXPERTS_PER_GPU, 1, loop_unroll_factor=1):
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
        # One pass over the tile: the migrated routes leave the owner column
        # and arrive in the destination column of the same expert row.
        transfer = tl.where(
            ranks[None, :] == destination, moved, tl.where(ranks[None, :] == rank, -moved, 0)
        )
        allocations += tl.where((local_experts[:, None] == local_expert) & active, transfer, 0)
        remaining = tl.where(active & (local_experts == local_expert), remaining - moved, remaining)
        quotas = tl.where(active & (ranks == destination), quotas - moved, quotas)
    tl.store(
        expert_rank_allocations + native_experts[:, None] * EP_SIZE + ranks[None, :],
        allocations,
        mask=valid_local_experts[:, None] & valid_ranks[None, :],
    )
    tl.store(
        destination_boundaries + native_experts[:, None] * BLOCK_EP_SIZE + ranks[None, :],
        tl.cumsum(allocations, axis=1) - routes_before_source[:, None],
        mask=valid_local_experts[:, None],
    )

    _cooperative_grid_barrier(grid_sync, EP_SIZE)

    # Step 4: fill this rank's replica slots by count descending, expert id
    # descending. Only remote experts with a positive allocation need copied
    # weights. The ordering is observable because it defines replica-slot ids,
    # so the expert id tie-break deliberately runs in descending order for
    # deterministic parity. Unused slots are filled with -1.
    experts = tl.arange(0, BLOCK_NUM_EXPERTS)
    owner = experts // NUM_EXPERTS_PER_GPU
    valid_remote = (experts < NUM_EXPERTS) & (owner != rank)
    counts = tl.load(
        expert_rank_allocations + experts * EP_SIZE + rank, mask=valid_remote, other=-1
    )
    for slot in tl.range(0, NUM_EXPERTS_PER_GPU, 1, loop_unroll_factor=1):
        maximum = tl.max(tl.where(valid_remote, counts, -1), axis=0)
        expert = tl.max(tl.where(valid_remote & (counts == maximum), experts, -1), axis=0)
        selected = tl.where(maximum > 0, expert, -1).to(tl.int32)
        tl.store(experts_to_copy + rank * NUM_EXPERTS_PER_GPU + slot, selected)
        tl.store(expert_replica_slots + selected * EP_SIZE + rank, slot, mask=selected >= 0)
        counts = tl.where(experts == expert, -1, counts)


@triton.jit
def _rank_routes_within_experts_kernel(
    flat_topk_indices,
    route_metadata,
    partition_counts,
    grid_sync,
    NUM_ROUTES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_NUM_EXPERTS: tl.constexpr,
    BLOCK_NUM_ROUTES: tl.constexpr,
    BLOCK_SCAN_PARTITIONS: tl.constexpr,
    NUM_SCAN_EXPERTS: tl.constexpr,
):
    """Give every route its stable ordinal within its expert's local stream.

    A cooperative grid splits the original route stream into consecutive
    partitions. The first pass computes each route's partition-local ordinal
    and packs it with the expert id in one int32 scratch value. Programs then
    publish their histograms, synchronize, and prefix-scan the partition
    counts. A route's local ordinal is therefore
    ``partition_counts[partition, expert] + partition_local_ordinal``, which is
    exactly its position within the expert bucket a stable sort would produce.
    The route mapper reconstructs it from the same partitioning, so no sorted
    index array is ever materialized.
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
    # partitions and publish the exclusive offsets in place.
    partition_offsets = tl.arange(0, BLOCK_SCAN_PARTITIONS)
    valid_partitions = partition_offsets < num_partitions
    for scan_expert_offset in tl.static_range(0, NUM_SCAN_EXPERTS):
        scan_expert = partition + scan_expert_offset * num_partitions
        valid_scan = valid_partitions & (scan_expert < NUM_EXPERTS)
        counts_for_expert = tl.load(
            partition_counts + partition_offsets * NUM_EXPERTS + scan_expert,
            mask=valid_scan,
            other=0,
        )
        tl.store(
            partition_counts + partition_offsets * NUM_EXPERTS + scan_expert,
            tl.cumsum(counts_for_expert, axis=0) - counts_for_expert,
            mask=valid_scan,
        )


def _launch_route_ranking(
    workspace: ReplicaPlannerWorkspace,
    flat_topk_indices: torch.Tensor,
    *,
    num_experts: int,
    num_routes: int,
) -> None:
    """Launch the one-kernel stable per-expert route ranking.

    The intra-tile ranking reads ``match.sync`` lane masks and splits its
    histogram into a first and a second warp, so the tile must be exactly two
    warps wide: ``BLOCK_NUM_ROUTES`` and ``num_warps`` below are a correctness
    constraint, not a tuning choice.
    """
    num_programs = _route_partition_count(num_routes)
    _rank_routes_within_experts_kernel[(num_programs,)](
        flat_topk_indices,
        workspace.sort_route_metadata,
        workspace.sort_partition_counts,
        workspace.sort_grid_sync,
        NUM_ROUTES=num_routes,
        NUM_EXPERTS=num_experts,
        BLOCK_NUM_EXPERTS=triton.next_power_of_2(num_experts),
        BLOCK_NUM_ROUTES=64,
        BLOCK_SCAN_PARTITIONS=_next_power_of_two(num_programs),
        NUM_SCAN_EXPERTS=triton.cdiv(num_experts, num_programs),
        launch_cooperative_grid=True,
        num_warps=2,
    )


@triton.jit
def _map_virtual_experts_kernel(
    route_metadata,
    partition_counts,
    destination_boundaries,
    expert_replica_slots,
    virtual_experts,
    NUM_ROUTES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_EXPERTS_PER_GPU: tl.constexpr,
    EP_SIZE: tl.constexpr,
    BLOCK_NUM_EXPERTS: tl.constexpr,
    BLOCK_NUM_ROUTES: tl.constexpr,
    BLOCK_EP_SIZE: tl.constexpr,
    LOG2_BLOCK_EP_SIZE: tl.constexpr,
):
    """Map every local route to its rank-major native or replica expert id.

    Routes for each expert form one deterministic global stream: source ranks
    are concatenated in rank order, and routes within a source use flattened
    token/top-k order. ``destination_boundaries`` holds that stream's partition
    points already rebased into this rank's local ordinal space, so a route's
    destination is simply the number of boundaries it has passed. Because the
    boundaries are non-decreasing, the search is a branchless binary search
    instead of a scan over every rank.

    The grid matches the route-ranking partitioning so this kernel can finish
    the ranking with a single indexed load, read its input sequentially, and
    write ``virtual_experts`` in route order.
    """
    partition = tl.program_id(0)
    num_partitions = tl.num_programs(0)
    routes_per_partition = tl.cdiv(NUM_ROUTES, num_partitions)
    partition_start = partition * routes_per_partition
    partition_end = tl.minimum(partition_start + routes_per_partition, NUM_ROUTES)
    expert_offsets = tl.arange(0, BLOCK_NUM_EXPERTS)
    routes_before_partition = tl.load(
        partition_counts + partition * NUM_EXPERTS + expert_offsets,
        mask=expert_offsets < NUM_EXPERTS,
        other=0,
    )
    tile_offsets = tl.arange(0, BLOCK_NUM_ROUTES)

    for route_start in tl.range(
        partition_start, partition_end, BLOCK_NUM_ROUTES, loop_unroll_factor=1
    ):
        route_positions = route_start + tile_offsets
        valid_routes = route_positions < partition_end
        packed_metadata = tl.load(route_metadata + route_positions, mask=valid_routes, other=0).to(
            tl.int32
        )
        experts = packed_metadata % BLOCK_NUM_EXPERTS
        ordinals_in_partition = packed_metadata // BLOCK_NUM_EXPERTS
        safe_experts = tl.where(valid_routes, experts, 0)
        # This route is the nth of its expert among all routes on this rank.
        local_ordinal = (
            tl.gather(routes_before_partition, safe_experts, axis=0) + ordinals_in_partition
        )

        # Count the destination segments this ordinal has already passed. The
        # boundary row is padded up to BLOCK_EP_SIZE with the expert's local
        # route total, which no ordinal reaches, so the count never exceeds the
        # last real rank.
        boundary_base = destination_boundaries + safe_experts * BLOCK_EP_SIZE
        destination = tl.zeros((BLOCK_NUM_ROUTES,), dtype=tl.int32)
        for step in tl.static_range(0, LOG2_BLOCK_EP_SIZE):
            candidate = destination + (BLOCK_EP_SIZE >> (step + 1))
            boundary = tl.load(boundary_base + candidate - 1)
            destination = tl.where(local_ordinal >= boundary, candidate, destination)

        # Native routes use the owner's local expert id. Remote routes look up
        # the slot holding that expert's copied weights on the destination.
        owner = experts // NUM_EXPERTS_PER_GPU
        owned_local = experts % NUM_EXPERTS_PER_GPU
        replica_slot = tl.load(
            expert_replica_slots + safe_experts * EP_SIZE + destination,
            mask=valid_routes & (destination != owner),
            other=-1,
        )
        runtime_local = tl.where(
            destination == owner, owned_local, NUM_EXPERTS_PER_GPU + replica_slot
        )
        virtual = destination.to(tl.int64) * (2 * NUM_EXPERTS_PER_GPU) + runtime_local
        tl.store(virtual_experts + route_positions, virtual, mask=valid_routes)


@triton.jit
def _compact_routing_map_kernel(
    routing_map,
    token_indices,
    tokens_per_expert,
    num_tokens,
    ROUTER_TOPK: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_NUM_EXPERTS: tl.constexpr,
):
    """Compact a dense routing map into per-token semantic expert ids.

    The dense map already names every selected expert, so recovering the
    compact route list is a per-row prefix sum rather than a selection
    problem: the running count of set entries before an expert is the slot
    that expert belongs in. The same pass accumulates the local per-expert
    route histogram the planner gathers across the EP group.

    The grid is persistent because the histogram is published with one atomic
    update per program. Sizing the grid by tokens instead would make those
    updates, not the map itself, the dominant cost.
    """
    program = tl.program_id(0)
    experts = tl.arange(0, BLOCK_NUM_EXPERTS)
    valid_experts = experts < NUM_EXPERTS
    token_offsets = tl.arange(0, BLOCK_TOKENS)
    histogram = tl.zeros((BLOCK_NUM_EXPERTS,), dtype=tl.int32)
    tokens_per_program = tl.cdiv(num_tokens, tl.num_programs(0))
    program_start = program * tokens_per_program
    program_end = tl.minimum(program_start + tokens_per_program, num_tokens)

    for token_start in tl.range(program_start, program_end, BLOCK_TOKENS, loop_unroll_factor=1):
        tokens = token_start + token_offsets
        valid = (tokens[:, None] < program_end) & valid_experts[None, :]
        selected = tl.load(
            routing_map + tokens[:, None] * NUM_EXPERTS + experts[None, :], mask=valid, other=0
        ).to(tl.int32)
        slots = tl.cumsum(selected, axis=1) - selected
        tl.store(
            token_indices + tokens[:, None] * ROUTER_TOPK + slots,
            tl.broadcast_to(experts[None, :], (BLOCK_TOKENS, BLOCK_NUM_EXPERTS)),
            mask=(selected != 0) & (slots < ROUTER_TOPK),
        )
        histogram += tl.sum(selected, axis=0)

    tl.atomic_add(tokens_per_expert + experts, histogram, mask=valid_experts)


def extract_semantic_routes(
    routing_map: torch.Tensor, probs: torch.Tensor, router_topk: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover compact semantic routes from a dense flex-dispatcher routing map.

    The routing map is authoritative: reading the routes back out of the dense
    probabilities instead would silently change a selected zero-probability
    route whenever several unselected experts tie at zero.

    Args:
        routing_map: Bool tensor ``[num_tokens, num_experts]`` selecting exactly
            ``router_topk`` experts per token.
        probs: Router probabilities ``[num_tokens, num_experts]``. Gradients
            flow back to the selected entries.
        router_topk: Number of routes ``K`` selected for every token.

    Returns:
        ``(token_probs, token_indices, tokens_per_expert)``. The first two have
        shape ``[num_tokens, router_topk]`` and list each token's routes in
        ascending semantic expert order; the last is the int32 local route
        histogram over ``num_experts``.
    """
    num_tokens, num_experts = (int(size) for size in routing_map.shape)
    device = routing_map.device
    tokens_per_expert = torch.zeros(num_experts, dtype=torch.int32, device=device)
    if HAVE_TRITON and device.type == "cuda":
        # Zeroed rather than empty: a routing map that selects fewer than
        # router_topk experts for some token would otherwise leave stale slots,
        # and the planner indexes tables with these ids.
        token_indices = torch.zeros((num_tokens, router_topk), dtype=torch.int32, device=device)
        block_num_experts = _next_power_of_two(num_experts)
        block_tokens = min(32, max(1, 16384 // block_num_experts))
        num_programs = min(_MAX_PLANNER_PROGRAMS, triton.cdiv(num_tokens, block_tokens))
        _compact_routing_map_kernel[(num_programs,)](
            routing_map,
            token_indices,
            tokens_per_expert,
            num_tokens,
            ROUTER_TOPK=router_topk,
            NUM_EXPERTS=num_experts,
            BLOCK_TOKENS=block_tokens,
            BLOCK_NUM_EXPERTS=block_num_experts,
            num_warps=8,
        )
    else:
        # Reference formulation for CPU tests: the same prefix sum, expressed
        # as a scatter whose overflow column is discarded.
        slots = torch.cumsum(routing_map, dim=1, dtype=torch.int32) - 1
        slots = torch.where(routing_map & (slots < router_topk), slots, router_topk)
        expert_ids = torch.arange(num_experts, dtype=torch.int32, device=device)
        token_indices = torch.zeros(
            (num_tokens, router_topk + 1), dtype=torch.int32, device=device
        ).scatter_(1, slots.long(), expert_ids.expand(num_tokens, num_experts))[:, :router_topk]
        tokens_per_expert += routing_map.sum(dim=0, dtype=torch.int32)
    return torch.gather(probs, 1, token_indices.long()), token_indices, tokens_per_expert


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
    *,
    rank_route_capacity: int,
    source_rank: int,
    ep_size: int,
    num_experts: int,
    num_local_experts: int,
) -> None:
    """Launch deterministic single-kernel replica placement."""
    _plan_replica_placement_kernel[(ep_size,)](
        workspace.gathered_counts,
        workspace.expert_totals,
        workspace.balance,
        workspace.receiver_quotas,
        workspace.allocation,
        workspace.destination_boundaries,
        workspace.experts_to_copy,
        workspace.expert_replica_slots,
        workspace.placement_grid_sync,
        RANK_ROUTE_CAPACITY=rank_route_capacity,
        SOURCE_EP_RANK=source_rank,
        EP_SIZE=ep_size,
        NUM_EXPERTS=num_experts,
        NUM_EXPERTS_PER_GPU=num_local_experts,
        BLOCK_EP_SIZE=triton.next_power_of_2(ep_size),
        BLOCK_NUM_EXPERTS_PER_GPU=triton.next_power_of_2(num_local_experts),
        BLOCK_NUM_EXPERTS=triton.next_power_of_2(num_experts),
        launch_cooperative_grid=True,
        num_warps=1,
    )


def _launch_replica_route_mapping(
    workspace: ReplicaPlannerWorkspace,
    *,
    ep_size: int,
    num_experts: int,
    num_local_experts: int,
    num_routes: int,
) -> None:
    """Map the ranked routes to native-or-replica ids."""
    block_ep_size = triton.next_power_of_2(ep_size)
    _map_virtual_experts_kernel[(_route_partition_count(num_routes),)](
        workspace.sort_route_metadata,
        workspace.sort_partition_counts,
        workspace.destination_boundaries,
        workspace.expert_replica_slots,
        workspace.virtual_experts,
        NUM_ROUTES=num_routes,
        NUM_EXPERTS=num_experts,
        NUM_EXPERTS_PER_GPU=num_local_experts,
        EP_SIZE=ep_size,
        BLOCK_NUM_EXPERTS=triton.next_power_of_2(num_experts),
        BLOCK_NUM_ROUTES=256,
        BLOCK_EP_SIZE=block_ep_size,
        LOG2_BLOCK_EP_SIZE=block_ep_size.bit_length() - 1,
        num_warps=8,
    )


def plan_replica_routes(
    topk_indices: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    ep_group: dist.ProcessGroup,
    workspace: ReplicaPlannerWorkspace,
    *,
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

    # Route ranking depends only on local routes, while placement depends on
    # the gathered histograms. Fork the ranking onto its fixed workspace stream
    # before the gather so the collective's latency runs underneath it, then
    # join it before the route mapper consumes both results.
    current_stream = torch.cuda.current_stream(topk_indices.device)
    workspace.sort_stream.wait_stream(current_stream)
    with torch.cuda.stream(workspace.sort_stream):
        _launch_route_ranking(
            workspace,
            topk_indices.reshape(-1),
            num_experts=num_experts,
            num_routes=num_routes,
        )

    # Phase 1: collect the only cross-rank input. From here onward every rank
    # sees the same histograms and independently produces the same placement.
    dist.all_gather_into_tensor(
        workspace.gathered_counts.view(-1), tokens_per_expert, group=ep_group
    )

    # Phases 2-4: construct allocation[expert, destination], then turn its
    # remote nonzero entries into a deterministic replica-weight list per rank.
    _launch_replica_placement(
        workspace,
        rank_route_capacity=num_routes,
        source_rank=source_rank,
        ep_size=ep_size,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
    )

    plan = ReplicaPlan(
        virtual_experts=workspace.virtual_experts, experts_to_copy=workspace.experts_to_copy
    )
    # Phase 5: allocations contain counts, not individual route identities, so
    # finish the ranking started above and locate every route in the
    # destination segments the allocation describes. Keep Phase 5 on the
    # ranking branch and make it wait only for placement, then launch weight
    # prefetch as a sibling branch. Enqueuing Phase 5 after the prefetch
    # callback makes CUDA-graph capture incorrectly put weight completion on
    # the mapping critical path.
    workspace.sort_stream.wait_stream(current_stream)
    with torch.cuda.stream(workspace.sort_stream):
        _launch_replica_route_mapping(
            workspace,
            ep_size=ep_size,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            num_routes=num_routes,
        )
    if on_placement_ready is not None:
        on_placement_ready(plan)
    current_stream.wait_stream(workspace.sort_stream)
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
