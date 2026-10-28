# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Runtime expert state for MoEScheduler replica dispatch.

``ReplicaExpertRuntime`` binds native and replica weights to Transformer Engine,
coordinates GTP materialization, and connects replica gradients back to the
optimizer-owned parameters. Data movement is delegated to a pluggable
``ReplicaWeightTransport`` selected by the expert dispatcher.
"""

import math
import weakref
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import torch
import torch.distributed as dist

from megatron.core.fp8_utils import is_mxfp8tensor
from megatron.core.transformer.moe.replica_storage import (
    ReplicaWeightCache,
    get_replica_consumer_fence,
)
from megatron.core.transformer.moe.replica_weight_transport import (
    ReplicaGradDestination,
    ReplicaOwnership,
    ReplicaPlacement,
    ReplicaPreparedPlan,
    ReplicaTransportConfig,
    ReplicaWeightLayout,
    ReplicaWeightSource,
    ReplicaWeightTransport,
    finalize_replica_weight_transports,
)

_MXFP8_COMPONENTS = (
    "_rowwise_data",
    "_rowwise_scale_inv",
    "_columnwise_data",
    "_columnwise_scale_inv",
)


class _ReplicaFC2WgradStore:
    """Start FC2 replica reduction immediately after TE enqueues its wgrad GEMM."""

    context = None

    def __init__(self, runtime: "ReplicaExpertRuntime") -> None:
        self._runtime = runtime

    @staticmethod
    def delay_wgrad_compute() -> bool:
        """Request the TE callback that computes and then reduces FC2 wgrads."""
        return True

    def put(self, tensors: tuple, wgrad_gemm: Callable[..., None]) -> None:
        """Enqueue the wgrad GEMM before starting FC2 replica reduction."""
        wgrad_gemm(*tensors)
        self._runtime.start_fc2_grad_reduce()


@dataclass(frozen=True, eq=False)
class ReplicaPlan:
    """Transport-facing result of replica placement planning.

    Attributes:
        virtual_experts: Optional private MoonEP helper result. Runtime dispatch
            leaves this None and never interprets logical expert identities.
        experts_to_copy: Physical source-home slots assigned to each rank's replica
            slots, with shape ``[ep_size, num_replica_slots_per_gpu]``. Unused
            slots contain ``-1``.
        version: Dispatcher-local generation, incremented on every placement.

    Identity hashing and weak references let the runtime cache private plans
    without retaining finished microbatches. Tensors must remain immutable
    through the forward/backward lifetime, including CUDA graph replay.
    """

    virtual_experts: torch.Tensor | None
    experts_to_copy: torch.Tensor
    version: int = 0


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


def _bf16_ptrs(source: torch.Tensor, *, numel: int, device: torch.device, label: str) -> tuple[int]:
    """Validate one BF16 runtime member and return its pointer signature."""
    if (
        source.dtype != torch.bfloat16
        or source.device != device
        or source.numel() != numel
        or not source.is_contiguous()
    ):
        raise ValueError(
            f"{label} requires contiguous BF16 storage with {numel} elements on {device}; "
            f"got dtype={source.dtype}, shape={tuple(source.shape)}, device={source.device}."
        )
    return (source.data_ptr(),)


def _mxfp8_ptrs(
    source: torch.Tensor,
    *,
    member_shape: tuple[int, int],
    scale_shapes: tuple[tuple[int, ...] | None, tuple[int, ...] | None],
    device: torch.device,
    label: str,
    components: tuple[str, ...] = _MXFP8_COMPONENTS,
) -> tuple[int, ...]:
    """Validate the requested MXFP8 components of one member and return their pointers."""
    shapes = dict(
        zip(_MXFP8_COMPONENTS, (member_shape, scale_shapes[0], member_shape, scale_shapes[1]))
    )
    storage = tuple(getattr(source, name, None) for name in components)
    if (
        tuple(source.shape) != member_shape
        or source.device != device
        or any(
            tensor is None
            or tensor.dtype != torch.uint8
            or not tensor.is_contiguous()
            or tuple(tensor.shape) != shapes[name]
            for tensor, name in zip(storage, components)
        )
    ):
        actual = tuple(
            None if tensor is None else (tensor.dtype, tuple(tensor.shape)) for tensor in storage
        )
        raise ValueError(
            f"{label} requires contiguous MXFP8 {components} with shapes "
            f"{tuple(shapes[name] for name in components)} on {device}; got "
            f"shape={tuple(source.shape)}, device={source.device}, storage={actual}."
        )
    return tuple(tensor.data_ptr() for tensor in storage)


def _collect_replica_projection_specs(
    experts: torch.nn.Module, *, num_local_experts: int, backend_name: str
) -> tuple[list[_ReplicaProjectionSpec], torch.device]:
    """Collect independently allocated TE expert weights."""
    projection_specs = []
    device: torch.device | None = None
    for linear in (experts.linear_fc1, experts.linear_fc2):
        member_shape = (int(linear.out_features), int(linear.in_features))
        if getattr(linear, "single_grouped_weight", False):
            raise ValueError(
                f"{backend_name} requires discrete weight0..weightN expert parameters; "
                "moe_single_grouped_weight must be False."
            )
        parameters = tuple(
            linear.get_parameter(f"weight{index}") for index in range(num_local_experts)
        )
        gtp_members = tuple(
            bool(getattr(parameter, "is_gtp_weight_remat", False)) for parameter in parameters
        )
        if any(gtp_members) and not all(gtp_members):
            raise ValueError(
                f"{backend_name} requires every weight in a projection to use the same GTP layout."
            )
        gtp_leader = parameters[0] if all(gtp_members) else None
        if gtp_leader is not None:
            for index, parameter in enumerate(parameters):
                if tuple(parameter._unsharded_shape) != member_shape:
                    raise ValueError(
                        f"{backend_name} expected GTP expert {index} to materialize as "
                        f"{member_shape}, got {tuple(parameter._unsharded_shape)}."
                    )
        mxfp8_members = tuple(is_mxfp8tensor(parameter) for parameter in parameters)
        if any(mxfp8_members) and not all(mxfp8_members):
            raise ValueError(f"{backend_name} does not support mixed BF16 and MXFP8 experts.")
        projection_device = parameters[0].device
        if all(mxfp8_members):
            if gtp_leader is None:
                scales = (parameters[0]._rowwise_scale_inv, parameters[0]._columnwise_scale_inv)
                if any(scale is None for scale in scales):
                    raise ValueError(
                        f"{backend_name} MXFP8 weights require rowwise and columnwise scales."
                    )
                scale_shapes = tuple(tuple(scale.shape) for scale in scales)
                for index, parameter in enumerate(parameters):
                    _mxfp8_ptrs(
                        parameter,
                        member_shape=member_shape,
                        scale_shapes=scale_shapes,
                        device=projection_device,
                        label=f"{backend_name} expert {index}",
                    )
            else:
                quantizer = getattr(gtp_leader, "_gtp_gather_quantizer", None)
                if quantizer is None or not hasattr(quantizer, "get_scale_shape"):
                    raise ValueError(
                        f"{backend_name} GTP MXFP8 weights require a gather quantizer."
                    )
                scale_shapes = tuple(
                    tuple(quantizer.get_scale_shape(member_shape, columnwise=columnwise))
                    for columnwise in (False, True)
                )
            spec = _ReplicaProjectionSpec(
                parameters, parameters, member_shape, "mxfp8", gtp_leader, *scale_shapes
            )
        else:
            source_tensors = tuple(_parameter_storage(parameter) for parameter in parameters)
            for index, (parameter, source) in enumerate(zip(parameters, source_tensors)):
                _bf16_ptrs(
                    source,
                    numel=(
                        math.prod(parameter._sharded_padded_shape)
                        if gtp_leader is not None
                        else math.prod(member_shape)
                    ),
                    device=projection_device,
                    label=f"{backend_name} expert {index}",
                )
            spec = _ReplicaProjectionSpec(
                parameters, source_tensors, member_shape, "bf16", gtp_leader
            )
        if device is None:
            device = projection_device
        elif projection_device != device:
            raise ValueError(f"{backend_name} FC1 and FC2 weights must share one device.")
        projection_specs.append(spec)
    if device is None or device.type != "cuda":
        raise ValueError(f"{backend_name} expert weights must be CUDA tensors.")
    if len({spec.weight_format for spec in projection_specs}) != 1:
        raise ValueError(f"{backend_name} FC1 and FC2 weights must use one storage format.")
    return projection_specs, device


class _WeightDirection(Enum):
    FORWARD = auto()
    BACKWARD = auto()


@dataclass(slots=True)
class _DirectionalBinding:
    """Device pointer tables the push reads for one GEMM orientation."""

    data_bases: torch.Tensor
    scale_bases: torch.Tensor | None = None
    source_tensors: tuple[torch.Tensor, ...] | None = None
    source_ptrs: tuple[tuple[int, ...], ...] | None = None
    # Pinned staging for capture-safe GTP pointer updates; None without GTP.
    host_pointer_table: torch.Tensor | None = None


@dataclass(slots=True)
class _ReplicaProjection:
    """One projection and its stable native/virtual runtime storage.

    Regular DDP natives accumulate directly into persistent model main_grad.
    FSDP natives bind each current unsharded bucket. Replica partials are reduced
    into that owner-provided destination before autograd handoff.
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
    rowwise_scale_shape: tuple[int, ...] | None
    columnwise_scale_shape: tuple[int, ...] | None
    virtual_weight: tuple[torch.Tensor, ...]
    virtual_grad: torch.Tensor
    native_grad: torch.Tensor | tuple[torch.Tensor, ...]
    accumulates_main_grad: bool = False
    runtime_parameters: tuple[torch.nn.Parameter, ...] | None = None
    source_storage_ptrs: tuple[tuple[int, ...], ...] | None = None
    native_grad_ptrs: tuple[int, ...] | None = None

    @property
    def member_numel(self) -> int:
        """Return the element count of one expert projection."""
        return math.prod(self.member_shape)

    def binding(self, direction: _WeightDirection) -> _DirectionalBinding:
        """Select the native weight binding for the requested compute direction."""
        return self.backward if direction is _WeightDirection.BACKWARD else self.forward

    def _storage_ptrs(
        self, source: torch.Tensor, label: str, components: tuple[str, ...] = _MXFP8_COMPONENTS
    ) -> tuple[int, ...]:
        if self.weight_format == "bf16":
            return _bf16_ptrs(source, numel=self.member_numel, device=self.device, label=label)
        return _mxfp8_ptrs(
            source,
            member_shape=self.member_shape,
            scale_shapes=(self.rowwise_scale_shape, self.columnwise_scale_shape),
            device=self.device,
            label=label,
            components=components,
        )

    def bind_materialized_weights(
        self, materialized_weights: tuple[torch.Tensor, ...], direction: _WeightDirection
    ) -> None:
        """Bind one stable directional GTP gather without copying its payload."""
        if len(materialized_weights) != len(self.parameters):
            raise RuntimeError(
                f"GTP materialized {len(materialized_weights)} {self.name} weights, "
                f"expected {len(self.parameters)}."
            )
        binding = self.binding(direction)
        backward = direction is _WeightDirection.BACKWARD
        components = _MXFP8_COMPONENTS[2:] if backward else _MXFP8_COMPONENTS[:2]
        source_ptrs = tuple(
            self._storage_ptrs(
                source,
                f"GTP {direction.name.lower()} gather of {self.name} replica expert {index}",
                components,
            )
            for index, source in enumerate(materialized_weights)
        )
        if binding.source_ptrs is not None:
            if source_ptrs != binding.source_ptrs:
                raise RuntimeError(
                    f"Replica GTP {direction.name.lower()} all-gather storage of {self.name} "
                    "changed after direct binding; this would invalidate CUDA-graph source "
                    "pointers."
                )
        else:
            tables = tuple(
                table for table in (binding.data_bases, binding.scale_bases) if table is not None
            )
            if binding.host_pointer_table is None or len(tables) != len(source_ptrs[0]):
                raise RuntimeError(f"Replica GTP binding of {self.name} lost pointer storage.")
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
            for parameter, source in zip(self.runtime_parameters or (), materialized_weights):
                parameter.data = source
            return
        # MXFP8 gathers alias their orientation into the stable native wrappers.
        for destinations in (self.source_tensors, self.runtime_parameters or ()):
            for destination, source in zip(destinations, materialized_weights):
                for field in components:
                    setattr(destination, field, getattr(source, field))
        if self.source_storage_ptrs is not None:
            offset = 2 if backward else 0
            self.source_storage_ptrs = tuple(
                ptrs[:offset] + update + ptrs[offset + 2 :]
                for ptrs, update in zip(self.source_storage_ptrs, source_ptrs)
            )

    def prepare_runtime_parameters(self, grad_dtype: torch.dtype) -> None:
        """Bind final DDP/GTP storage once, then validate pointer stability."""
        fsdp_owned = all(getattr(p, "__fsdp_param__", False) for p in self.parameters)
        eager_fsdp = fsdp_owned and not torch.cuda.is_current_stream_capturing()
        directional = self.gtp_leader is not None and self.weight_format == "bf16"
        sources = (
            tuple(_parameter_storage(parameter) for parameter in self.parameters)
            if self.gtp_leader is None and self.weight_format == "bf16"
            else self.source_tensors
        )
        if len(sources) != len(self.parameters):
            raise RuntimeError(
                f"Replica expert runtime {self.name} expected {len(self.parameters)} native "
                f"weights, got {len(sources)}."
            )
        storage_ptrs = tuple(
            self._storage_ptrs(source, f"Replica expert runtime {self.name} expert {index}")
            for index, source in enumerate(sources)
        )
        # FSDP recreates MXFP8 transpose caches while unsharding for backward.
        # Refresh both transport pointer tables and the native TE wrappers.
        # Other owners retain the fixed-address contract, as does CUDA capture.
        storage_changed = (
            not directional
            and self.source_storage_ptrs is not None
            and storage_ptrs != self.source_storage_ptrs
        )
        if storage_changed:
            if not eager_fsdp:
                raise RuntimeError(
                    f"Replica expert runtime {self.name} parameter storage changed after binding; "
                    "this would invalidate CUDA-graph source pointers."
                )
        if eager_fsdp:
            # A freed bucket can be replaced at the same allocator address.
            # Refresh the Tensor/storage owners even when pointer tables match.
            for parameter, source in zip(self.runtime_parameters or (), sources):
                if self.weight_format == "bf16":
                    parameter.data = source
                else:
                    for field in _MXFP8_COMPONENTS:
                        setattr(parameter, field, getattr(source, field))
        # Directional GTP BF16 storage is tracked per binding instead.
        if not directional and (self.source_storage_ptrs is None or storage_changed):
            self.source_storage_ptrs = storage_ptrs
            if self.gtp_leader is None:
                bindings = (self.forward, self.backward)
                for direction, binding in enumerate(bindings):
                    host_table = binding.host_pointer_table
                    if host_table is None:
                        raise RuntimeError(
                            f"Replica pointer staging for {self.name} was not allocated."
                        )
                    component_offset = 2 * direction if self.weight_format == "mxfp8" else 0
                    for row, table in enumerate((binding.data_bases, binding.scale_bases)):
                        if table is None:
                            continue
                        component = component_offset + row
                        # Do not overwrite a pinned staging row that a previous
                        # asynchronous pointer-table upload may still be reading.
                        host_row = (
                            torch.empty_like(host_table[row], pin_memory=False)
                            if storage_changed
                            else host_table[row]
                        )
                        host_row.copy_(
                            torch.tensor(
                                [ptrs[component] for ptrs in storage_ptrs], dtype=torch.int64
                            )
                        )
                        table.copy_(host_row, non_blocking=not storage_changed)

        # Gradient ownership stays with DDP/FSDP (or a caller providing full
        # main_grad views). Never allocate a second native-gradient arena.
        native_grads = []
        for parameter in self.parameters:
            grad = getattr(parameter, "main_grad", None)
            if grad is None and callable(getattr(parameter, "get_main_grad", None)):
                grad = parameter.get_main_grad()
                parameter.main_grad = grad
            if grad is None or grad.numel() != self.member_numel:
                raise ValueError(
                    "Replica native GEMMs require owner-provided full-sized main_grad; "
                    "sharded GTP owners must expose their full wgrad destination."
                )
            native_grads.append(grad.view(self.member_shape))
        native_grads = tuple(native_grads)
        self.native_grad = native_grads
        policies = {bool(getattr(p, "overwrite_main_grad", False)) for p in self.parameters}
        if len(policies) != 1:
            raise ValueError("Native experts must agree on main_grad accumulation policy.")
        self.accumulates_main_grad = not policies.pop()
        for index, grad in enumerate(native_grads):
            if (
                grad.dtype != grad_dtype
                or grad.device != self.device
                or grad.numel() != self.member_numel
                or not grad.is_contiguous()
            ):
                raise ValueError(
                    f"Replica expert runtime {self.name} native grad {index} must be contiguous "
                    f"{grad_dtype} with {self.member_numel} elements on {self.device}; got "
                    f"dtype={grad.dtype}, shape={tuple(grad.shape)}, device={grad.device}."
                )
        native_grad_ptrs = tuple(grad.data_ptr() for grad in native_grads)
        if native_grad_ptrs != self.native_grad_ptrs:
            if self.native_grad_ptrs is not None and not eager_fsdp:
                raise RuntimeError(
                    f"Replica expert runtime {self.name} native-grad storage changed after binding."
                )
            self.native_grad_ptrs = native_grad_ptrs
            self.native_grad_bases.copy_(
                torch.tensor(native_grad_ptrs, dtype=torch.int64, device=self.device)
            )
        if eager_fsdp:
            for parameter, grad in zip(self.runtime_parameters or (), native_grads):
                parameter.main_grad = grad

        self.source_tensors = sources
        weights = sources + tuple(self.virtual_weight)
        grads = native_grads + tuple(self.virtual_grad)
        if self.runtime_parameters is None:
            self.bind_runtime_parameters(weights, grads)
        else:
            self.validate_runtime_parameters(weights, grads)

    def bind_runtime_parameters(self, weights, grads) -> None:
        """Create the stable native-then-virtual TE parameter sequence once."""
        runtime_parameters = []
        for weight, grad in zip(weights, grads):
            # These are runtime views, not optimizer-owned parameters. Replica
            # storage is restored to this microbatch's plan before backward,
            # potentially after other forwards reused it. Give BF16 wrappers
            # independent version counters, as native parameter.data views and
            # raw Peer-TMA writes already do. The dispatcher owns restoration
            # and stream ordering; TE still checks mutations of its own wrapper.
            runtime_weight = weight.data if self.weight_format == "bf16" else weight
            parameter = torch.nn.Parameter(runtime_weight, requires_grad=True)
            parameter.main_grad = grad
            # These runtime leaves have no DDP hooks. Without grad_added_to_main_grad,
            # TE returns None instead of allocating a dummy leaf gradient.
            parameter.overwrite_main_grad = not self.accumulates_main_grad
            runtime_parameters.append(parameter)
        self.runtime_parameters = tuple(runtime_parameters)

    def validate_runtime_parameters(self, weights, grads) -> None:
        """Validate that runtime parameters still alias the bound storage."""
        for parameter, weight, grad in zip(self.runtime_parameters, weights, grads):
            fields = ("data",) if self.weight_format == "bf16" else _MXFP8_COMPONENTS
            if any(
                getattr(parameter, field).data_ptr() != getattr(weight, field).data_ptr()
                for field in fields
            ):
                raise RuntimeError(
                    f"Replica expert runtime {self.name} runtime weight storage changed after "
                    "binding."
                )
            runtime_grad = getattr(parameter, "main_grad", None)
            if runtime_grad is None or runtime_grad.data_ptr() != grad.data_ptr():
                raise RuntimeError(
                    f"Replica expert runtime {self.name} runtime main-grad storage changed after "
                    "binding."
                )
            parameter.overwrite_main_grad = not self.accumulates_main_grad

    def destroy(self) -> None:
        """Release runtime gradient views without changing owner buffers."""
        for parameter in self.runtime_parameters or ():
            parameter.main_grad = None
        self.runtime_parameters = None


_replica_expert_runtimes = weakref.WeakSet()


class ReplicaExpertRuntime:
    """Bind replica expert parameters and coordinate a pluggable transport."""

    def __init__(
        self,
        *,
        experts: torch.nn.Module,
        group: dist.ProcessGroup,
        num_experts: int,
        num_local_home_experts: int,
        num_local_replica_slots: int,
        transport_factory: Callable[[ReplicaTransportConfig], ReplicaWeightTransport],
        grad_dtype: torch.dtype = torch.float32,
        num_sms: int | None = None,
        replica_storage: str = "shared",
        replica_mxfp8_storage: str = "reuse",
        replica_exchange_buffer_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self.group = group
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self.num_local_home_experts = int(num_local_home_experts)
        self.num_local_replica_slots = int(num_local_replica_slots)
        self.num_runtime_experts = self.num_local_home_experts + self.num_local_replica_slots
        self.last_plan = None
        self._prefetch_plan = None
        self._prefetch_handle = None
        self._completed_plan = None
        self._completed_handle = None
        self._prepared_plans: weakref.WeakKeyDictionary[ReplicaPlan, ReplicaPreparedPlan] = (
            weakref.WeakKeyDictionary()
        )
        self.ownership = ReplicaOwnership(home_experts=self.num_local_home_experts)
        self._backward_plan = None
        self._grad_reduce_plan = None
        self._grad_reduce_started: set[int] = set()
        self._grad_reduce_handles = {}
        self._experts_ref = weakref.ref(experts)
        self._destroyed = False

        if self.num_local_replica_slots <= 0:
            raise ValueError(
                "Replica weights require at least one replica slot per rank, "
                f"got {self.num_local_replica_slots}."
            )
        if int(num_experts) != self.world_size * self.num_local_home_experts:
            raise ValueError(
                "Replica weights require an even expert distribution: "
                f"num_experts={num_experts}, world_size={self.world_size}, "
                f"num_local_home_experts={self.num_local_home_experts}."
            )
        projection_specs, self.device = _collect_replica_projection_specs(
            experts, num_local_experts=self.num_local_home_experts, backend_name="Replica"
        )
        self.weight_format = projection_specs[0].weight_format
        mxfp8 = self.weight_format == "mxfp8"
        transport_config = ReplicaTransportConfig(
            group=group,
            device=self.device,
            world_size=self.world_size,
            num_local_home_experts=self.num_local_home_experts,
            num_local_replica_slots=self.num_local_replica_slots,
            member_shapes=tuple(spec.member_shape for spec in projection_specs),
            weight_format=self.weight_format,
            rowwise_scale_shapes=(
                tuple(spec.rowwise_scale_shape for spec in projection_specs) if mxfp8 else None
            ),
            columnwise_scale_shapes=(
                tuple(spec.columnwise_scale_shape for spec in projection_specs) if mxfp8 else None
            ),
            grad_dtype=grad_dtype,
            num_sms=num_sms,
            replica_storage=replica_storage,
            replica_mxfp8_storage=replica_mxfp8_storage,
            replica_exchange_buffer_bytes=replica_exchange_buffer_bytes,
        )
        self.transport: ReplicaWeightTransport = transport_factory(transport_config)
        self._consumer_fence = get_replica_consumer_fence(self.group, self.device)
        self._workspace_acquired = False
        self.weight_cache = (
            ReplicaWeightCache(self.group, self.device)
            if replica_storage == "layerwise" and (not mxfp8 or replica_mxfp8_storage == "separate")
            else None
        )

        def pointer_table() -> torch.Tensor:
            return torch.empty(self.num_local_home_experts, dtype=torch.int64, device=self.device)

        def binding(gtp: bool) -> _DirectionalBinding:
            del gtp
            # All pointer uploads stage through pinned host memory so the hot
            # path does not construct temporary CUDA tensors.
            components = 2 if mxfp8 else 1
            return _DirectionalBinding(
                pointer_table(),
                pointer_table() if mxfp8 else None,
                host_pointer_table=(
                    torch.empty(
                        (components, self.num_local_home_experts),
                        dtype=torch.int64,
                        pin_memory=True,
                    )
                    if components
                    else None
                ),
            )

        self.projections: list[_ReplicaProjection] = []
        for projection_index, spec in enumerate(projection_specs):
            virtual_storage, virtual_grad = self.transport.projection_views(projection_index)
            gtp = spec.gtp_leader is not None
            if mxfp8:
                virtual_weight = _wrap_mxfp8(spec, virtual_storage, self.device)
                # GTP gather storage is aliased into distinct native wrappers over
                # the replica views before use, instead of a second full copy.
                source_tensors = (
                    _wrap_mxfp8(
                        spec, (virtual_storage[0],) * len(spec.parameters), self.device, native=True
                    )
                    if gtp
                    else spec.source_tensors
                )
            else:
                virtual_weight = virtual_storage
                source_tensors = () if gtp else spec.source_tensors
            forward = binding(gtp)
            backward = binding(gtp) if gtp or mxfp8 else forward
            self.projections.append(
                _ReplicaProjection(
                    name=f"FC{projection_index + 1}",
                    device=self.device,
                    weight_format=spec.weight_format,
                    parameters=spec.parameters,
                    gtp_leader=spec.gtp_leader,
                    source_tensors=source_tensors,
                    forward=forward,
                    backward=backward,
                    native_grad_bases=pointer_table(),
                    member_shape=spec.member_shape,
                    rowwise_scale_shape=spec.rowwise_scale_shape,
                    columnwise_scale_shape=spec.columnwise_scale_shape,
                    virtual_weight=virtual_weight,
                    virtual_grad=virtual_grad,
                    native_grad=(),
                )
            )
        _replica_expert_runtimes.add(self)

    @property
    def runtime_fc1_weights(self) -> tuple[torch.nn.Parameter, ...]:
        """Return stable native-then-virtual FC1 runtime parameters."""
        return self._runtime_weights(0)

    @property
    def runtime_fc2_weights(self) -> tuple[torch.nn.Parameter, ...]:
        """Return stable native-then-virtual FC2 runtime parameters."""
        return self._runtime_weights(1)

    def make_fc2_wgrad_store(self) -> _ReplicaFC2WgradStore:
        """Create the TE callback that overlaps FC2 reduction with remaining backward work."""
        return _ReplicaFC2WgradStore(self)

    def _runtime_weights(self, projection_index: int) -> tuple[torch.nn.Parameter, ...]:
        runtime_parameters = self.projections[projection_index].runtime_parameters
        if runtime_parameters is None:
            raise RuntimeError("Replica runtime weights were accessed before binding.")
        return runtime_parameters

    @property
    def source_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        """Return the optimizer-owned FC1 and FC2 parameters."""
        return tuple(parameter for p in self.projections for parameter in p.parameters)

    def prepare_runtime_parameters(self) -> None:
        """Late-bind final DDP/GTP storage and validate subsequent stability."""
        for projection in self.projections:
            projection.prepare_runtime_parameters(self.transport.grad_dtype)

    def prepare_source_weights(self, direction: _WeightDirection) -> None:
        """Make plain weights ready and peek at GTP gathers for the transport push."""
        experts = self._experts_ref()
        if experts is None:
            raise RuntimeError("Replica experts were destroyed before prefetch.")
        backward = direction is _WeightDirection.BACKWARD
        experts.prepare_fused_impl_parameters(for_backward=backward)
        # Expert backward computes FC2 before FC1; keep GTP's linked gathers in
        # the order they will be consumed.
        for projection in reversed(self.projections) if backward else self.projections:
            leader = projection.gtp_leader
            if leader is None:
                continue
            peek = getattr(
                leader, "peek_group_for_backward" if backward else "peek_group_for_forward", None
            )
            if callable(peek):
                materialized = peek()
            else:
                # Compatibility with GTP revisions predating the non-consuming
                # peek protocol. Those revisions consume at push time.
                materialized = (
                    leader.materialize_group_for_backward()
                    if backward
                    else leader.materialize_group_for_forward()
                )
            if not isinstance(materialized, (list, tuple)):
                materialized = (materialized,)
            projection.bind_materialized_weights(tuple(materialized), direction)
        self.prepare_runtime_parameters()

    def consume_source_weights(self, direction: _WeightDirection) -> None:
        """Consume GTP gathers at the GEMMs after the transport-only peek."""
        backward = direction is _WeightDirection.BACKWARD
        projections = reversed(self.projections) if backward else self.projections
        for projection in projections:
            leader = projection.gtp_leader
            if leader is None:
                continue
            peek = getattr(
                leader, "peek_group_for_backward" if backward else "peek_group_for_forward", None
            )
            if not callable(peek):
                continue
            materialized = (
                leader.materialize_group_for_backward()
                if backward
                else leader.materialize_group_for_forward()
            )
            if not isinstance(materialized, (list, tuple)):
                materialized = (materialized,)
            # bind_materialized_weights verifies that the consuming gather uses
            # the same stable buffers read by the push.
            projection.bind_materialized_weights(tuple(materialized), direction)

    def consume_forward_source_weights(self) -> None:
        """Consume forward GTP gathers immediately before expert GEMMs."""
        self.consume_source_weights(_WeightDirection.FORWARD)

    def _validate_plan(self, plan: ReplicaPlan) -> None:
        """Validate fixed device metadata without extracting any CUDA values."""
        experts_to_copy = plan.experts_to_copy
        expected_shape = (self.world_size, self.num_local_replica_slots)
        if (
            experts_to_copy.dtype != torch.int32
            or experts_to_copy.device != self.device
            or tuple(experts_to_copy.shape) != expected_shape
            or not experts_to_copy.is_contiguous()
        ):
            raise ValueError(
                "Replica experts_to_copy must be contiguous int32 with shape "
                f"{expected_shape} on {self.device}."
            )

    def _transport_sources(self, direction: _WeightDirection) -> tuple[ReplicaWeightSource, ...]:
        """Expose backend-neutral source views plus peer-TMA pointer fast paths."""
        sources = []
        backward = direction is _WeightDirection.BACKWARD
        for projection in self.projections:
            binding = projection.binding(direction)
            source_tensors = binding.source_tensors or projection.source_tensors
            if projection.weight_format == "bf16":
                data = tuple(_parameter_storage(source) for source in source_tensors)
                scales = None
            else:
                data_field = "_columnwise_data" if backward else "_rowwise_data"
                scale_field = "_columnwise_scale_inv" if backward else "_rowwise_scale_inv"
                data = tuple(getattr(source, data_field) for source in source_tensors)
                scales = tuple(getattr(source, scale_field) for source in source_tensors)
            sources.append(
                ReplicaWeightSource(
                    data=data,
                    scales=scales,
                    layout=(
                        ReplicaWeightLayout.PLAIN
                        if projection.weight_format == "bf16"
                        else (
                            ReplicaWeightLayout.COLUMNWISE
                            if backward
                            else ReplicaWeightLayout.ROWWISE
                        )
                    ),
                    data_bases=binding.data_bases,
                    scale_bases=binding.scale_bases,
                )
            )
        return tuple(sources)

    def _prepare_transport_plan(self, plan: ReplicaPlan) -> ReplicaPreparedPlan:
        """Cache only scheduling metadata for this immutable microbatch plan."""
        prepared = self._prepared_plans.get(plan)
        if prepared is None:
            prepared = self.transport.prepare_plan(
                ReplicaPlacement(plan.experts_to_copy, self.ownership, plan.version)
            )
            self._prepared_plans[plan] = prepared
        return prepared

    def acquire_workspace(self) -> None:
        """Fence the last compute reader before any shared storage producer."""
        self._consumer_fence.acquire(self, self.device)
        self._workspace_acquired = True

    def release_workspace(self) -> None:
        """Called after forward consumption or native-gradient publication."""
        if self._workspace_acquired:
            self._consumer_fence.release(self, self.device)
            self._workspace_acquired = False

    @torch.no_grad()
    def start_prefetch(
        self, plan: ReplicaPlan, direction: _WeightDirection = _WeightDirection.FORWARD
    ) -> None:
        """Enqueue the owner push of FC1/FC2 weights without blocking the caller."""
        if self._prefetch_plan is not None:
            raise RuntimeError("Replica weight prefetch is already outstanding.")
        self._validate_plan(plan)
        prepared = self._prepare_transport_plan(plan)
        # A GTP parameter stores only its local shard. Materialization consumes
        # any one-weight-ahead gather (or performs the cold synchronous gather)
        # and stages the full native experts before the push reads them.
        self.prepare_source_weights(direction)
        sources = self._transport_sources(direction)
        layout = sources[0].layout if self.weight_cache is not None else None
        handle = (
            self.weight_cache.lookup(layout, plan.experts_to_copy, sources)
            if self.weight_cache is not None
            else None
        )
        if handle is None:
            handle = self.transport.start_weight_sync(sources=sources, plan=prepared)
            if self.weight_cache is not None:
                self.weight_cache.remember(layout, plan.experts_to_copy, sources, handle)
        self._prefetch_handle = handle
        self._prefetch_plan = plan
        self._completed_plan = None
        self._completed_handle = None

    @torch.no_grad()
    def wait_prefetch(self, plan: ReplicaPlan) -> None:
        """Make the current stream wait for the outstanding push of ``plan``."""
        if self._prefetch_plan is None:
            # A resident plan still needs a dependency on every consumer stream.
            if plan is None or plan is not self._completed_plan:
                raise RuntimeError("Replica weights require a started prefetch before use.")
        elif self._prefetch_plan is not plan:
            raise RuntimeError("Replica weight prefetch plan changed while outstanding.")
        handle = (
            self._prefetch_handle if self._prefetch_plan is not None else self._completed_handle
        )
        self.transport.wait_weight_sync(handle)
        self._completed_plan = plan
        self._completed_handle = handle
        self._prefetch_plan = None
        self._prefetch_handle = None

    def wait_prefetch_for_backward(self, plan: ReplicaPlan) -> None:
        """Wait for the backward push and bind its plan to expert backward."""
        self.wait_prefetch(plan)
        self.consume_source_weights(_WeightDirection.BACKWARD)
        # TE uses the native accumulation policy for the whole grouped GEMM.
        # Virtual slots are scratch for this backward, never microbatch accumulators.
        for projection in self.projections:
            if projection.accumulates_main_grad:
                projection.virtual_grad.zero_()
        self._backward_plan = plan

    def start_backward_prefetch(self, plan: ReplicaPlan) -> None:
        """Start the columnwise/GTP-backward replica weight transfer."""
        self.acquire_workspace()
        self.start_prefetch(plan, _WeightDirection.BACKWARD)

    @torch.no_grad()
    def start_grad_reduce(self, plan: ReplicaPlan, projection: int) -> None:
        """Enqueue one projection's replica-gradient reduction."""
        if self._grad_reduce_plan is not None and self._grad_reduce_plan is not plan:
            raise RuntimeError("Replica gradient reduction is outstanding for another plan.")
        if projection in self._grad_reduce_started:
            raise RuntimeError(f"Replica gradient reduction of FC{projection + 1} started twice.")
        self._validate_plan(plan)
        self._grad_reduce_handles[projection] = self.transport.start_grad_reduce(
            native_grads=tuple(
                ReplicaGradDestination(
                    tensors=tuple(projection.native_grad), bases=projection.native_grad_bases
                )
                for projection in self.projections
            ),
            plan=self._prepare_transport_plan(plan),
            projections=(projection,),
        )
        self._grad_reduce_plan = plan
        self._grad_reduce_started.add(projection)

    def start_fc2_grad_reduce(self) -> None:
        """Start FC2 reduction immediately behind its wgrad GEMM."""
        if self._backward_plan is None:
            raise RuntimeError("Replica FC2 gradient reduction needs the backward plan.")
        self.start_grad_reduce(self._backward_plan, 1)

    def start_pending_grad_reduces(self, plan: ReplicaPlan) -> None:
        """Start reductions not already issued by expert backward, FC2 first."""
        if self._grad_reduce_plan is not None and self._grad_reduce_plan is not plan:
            raise RuntimeError("Replica gradient reduction is outstanding for another plan.")
        for projection in (1, 0):
            if projection not in self._grad_reduce_started:
                self.start_grad_reduce(plan, projection)

    @torch.no_grad()
    def wait_grad_reduce(self, plan: ReplicaPlan) -> tuple[torch.Tensor | None, ...]:
        """Finish both projection reductions and return source-parameter wgrads."""
        if self._grad_reduce_plan is not plan or self._grad_reduce_started != {0, 1}:
            raise RuntimeError("Replica gradient reduction of both projections must be started.")
        for projection in (0, 1):
            self.transport.wait_grad_reduce(self._grad_reduce_handles[projection])
        self._grad_reduce_plan = None
        self._grad_reduce_started.clear()
        self._grad_reduce_handles.clear()
        self._backward_plan = None

        # Expert backward computes FC2 before FC1. Preserve that reverse order
        # when handing full wgrads to GTP so its linked RS cascade remains valid.
        source_grads = [tuple(projection.native_grad) for projection in self.projections]
        for index in reversed(range(len(self.projections))):
            leader = self.projections[index].gtp_leader
            if leader is None:
                continue
            reduced = leader.wgrad_reduce_scatter(list(source_grads[index]))
            reduced = tuple(reduced) if isinstance(reduced, (list, tuple)) else (reduced,)
            if len(reduced) != len(source_grads[index]):
                raise RuntimeError(
                    "GTP returned a different number of reduced wgrads than source parameters."
                )
            source_grads[index] = reduced
        return tuple(grad for grads in source_grads for grad in grads)

    def destroy(self) -> None:
        """Detach layer-owned TE parameters from transport-owned storage."""
        if self._destroyed:
            return
        experts = self._experts_ref()
        if experts is not None:
            experts.release_replica_expert_runtime(self)
        for projection in self.projections:
            projection.destroy()
        self.projections.clear()
        self.last_plan = None
        self.release_workspace()
        if self.weight_cache is not None:
            self.weight_cache.clear()
        self.transport.destroy()
        self._prepared_plans.clear()
        self._prefetch_handle = None
        self._completed_handle = None
        self._grad_reduce_handles.clear()
        self._prefetch_plan = self._completed_plan = self._backward_plan = None
        self._grad_reduce_plan = None
        self.transport = None
        self._destroyed = True
        _replica_expert_runtimes.discard(self)


def _wrap_mxfp8(
    spec: _ReplicaProjectionSpec,
    storage_views: tuple[tuple[torch.Tensor, ...], ...],
    device,
    *,
    native: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Create one wrapper per storage slot, independent of the number of homes.

    Replica slots share the projection's uniform TE format metadata. Native GTP
    placeholders preserve each source quantizer and are rebound before use.
    """
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

    return tuple(
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
            device=device,
        )
        for source, (rowwise_data, rowwise_scale, columnwise_data, columnwise_scale) in zip(
            spec.source_tensors if native else (spec.source_tensors[0],) * len(storage_views),
            storage_views,
        )
    )


def finalize_replica_expert_runtimes() -> None:
    """Release replica runtimes and transports before process-group teardown."""
    from megatron.core.transformer.moe.moonep_moe_scheduler import (
        finalize_moonep_planner_workspaces,
    )

    for runtime in list(_replica_expert_runtimes):
        runtime.destroy()
    finalize_replica_weight_transports()
    finalize_moonep_planner_workspaces()
