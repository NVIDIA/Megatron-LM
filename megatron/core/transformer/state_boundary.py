# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tensor contracts for extra state crossing existing execution boundaries.

These helpers describe data, not model construction or execution. Positions and
placement come from the caller's existing layer order. Feature codecs own the
meaning of their namespaced, versioned fields and restore fresh native state.
"""

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class TensorField:
    """One field in a batch-specific schema, including configured absent fields.

    ``differentiable`` is static gradient eligibility for this execution mode,
    independent of whether a producer tensor has a live edge (e.g. frozen weights).
    A no-grad evaluation uses a schema without backward eligibility.
    """

    key: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    layout: str
    differentiable: bool
    present: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.key, str)
            or not self.key
            or not isinstance(self.layout, str)
            or not self.layout
            or not isinstance(self.shape, tuple)
        ):
            raise ValueError("A state field requires a stable key, layout and tuple shape")
        if not isinstance(self.dtype, torch.dtype) or any(
            type(flag) is not bool for flag in (self.present, self.differentiable)
        ):
            raise ValueError("State fields require a torch dtype and boolean qualifications")
        if any(type(dim) is not int or dim < 0 for dim in self.shape):
            raise ValueError(f"Invalid shape for state field {self.key}")
        if self.differentiable and not (self.dtype.is_floating_point or self.dtype.is_complex):
            raise ValueError(f"Non-floating state field {self.key} cannot be differentiable")


@dataclass(frozen=True)
class TensorSchema:
    """Stable field order and separate present/gradient packing maps."""

    fields: tuple[TensorField, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.fields, tuple):
            raise TypeError("State schema fields must be an immutable tuple")
        keys = [field.key for field in self.fields]
        if len(set(keys)) != len(keys):
            raise ValueError("State schema contains duplicate field keys")

    @property
    def present_spec_indices(self) -> tuple[int, ...]:
        """Map packed tensor positions to complete schema positions."""
        return tuple(i for i, field in enumerate(self.fields) if field.present)

    @property
    def packed_fields(self) -> tuple[TensorField, ...]:
        """Return fields which have a tensor, in wire/argument order."""
        return tuple(self.fields[i] for i in self.present_spec_indices)

    @property
    def grad_tensor_indices(self) -> tuple[int, ...]:
        """Identify eligible gradients in the packed tensor tuple, not the schema."""
        return tuple(i for i, field in enumerate(self.packed_fields) if field.differentiable)

    @property
    def fingerprint(self) -> int:
        """Return a stable signed-int64-safe digest, independent of Python hash seeds."""
        values = tuple(
            (f.key, f.shape, str(f.dtype), f.layout, f.differentiable, f.present)
            for f in self.fields
        )
        return int.from_bytes(sha256(repr(values).encode("utf-8")).digest()[:8], "big") >> 1

    def validate(self, tensors: Sequence[Tensor]) -> None:
        """Validate concrete tensors without reading device values or autograd activity."""
        fields = self.packed_fields
        if len(fields) != len(tensors):
            raise ValueError("Packed state tensor count does not match its schema")
        for field, tensor in zip(fields, tensors):
            if (
                not isinstance(tensor, Tensor)
                or tuple(tensor.shape) != field.shape
                or tensor.dtype != field.dtype
                or tensor.layout != torch.strided
            ):
                raise ValueError(f"State field {field.key} has incompatible shape/dtype/layout")

    def detach_ineligible(self, tensors: Sequence[Tensor]) -> tuple[Tensor, ...]:
        """Apply input eligibility without changing eligible tensor identities."""
        self.validate(tensors)
        return tuple(
            tensor if field.differentiable or not tensor.requires_grad else tensor.detach()
            for field, tensor in zip(self.packed_fields, tensors)
        )

    def validate_peer(self, other: "TensorSchema") -> None:
        """Reject different identities, layouts, packing or gradient qualifications."""
        if self != other:
            raise ValueError("State boundary peer schemas do not match")


@dataclass(frozen=True)
class StateDependency:
    """A single source and its consumers in the host's existing execution order."""

    field: TensorField
    origin: Literal["prepared", "activation"]
    available_at: int
    source_pp_rank: int
    consumed_at: tuple[int, ...]
    delivery: Literal["local", "pipeline"]


@dataclass(frozen=True)
class BoundarySchema:
    """Extra inputs and outputs appended to the host's ordinary tensor arguments."""

    boundary_id: str
    inputs: tuple[TensorField, ...]
    outputs: tuple[TensorField, ...]

    def __post_init__(self) -> None:
        if not self.boundary_id:
            raise ValueError("A state boundary requires a stable identity")
        TensorSchema(self.inputs)
        TensorSchema(self.outputs)


@dataclass(frozen=True)
class StatePlacement:
    """Existing half-open position interval assigned to one physical/virtual chunk."""

    start: int
    end: int
    pp_rank: int
    chunk_id: int = 0


@dataclass(frozen=True)
class StateBoundary:
    """Existing region [start, end), or a pipeline cut with start == end."""

    boundary_id: str
    kind: Literal["pipeline", "checkpoint", "capture"]
    start: int
    end: int


class StateCodec(Protocol):
    """Translate explicit tensors and immutable metadata to fresh feature-native state."""

    def export(self, state: Any, fields: tuple[TensorField, ...]) -> tuple[Tensor, ...]:
        """Export present canonical tensors in the declared order without detaching."""
        ...

    def restore(
        self, fields: tuple[TensorField, ...], tensors: tuple[Tensor, ...], metadata: Any
    ) -> Any:
        """Restore independent state; metadata must not retain tensors or live payloads."""
        ...


def validate_shared_input(field: TensorField, previous: Tensor, tensor: Tensor) -> None:
    """Require one autograd edge, or equal immutable snapshots, for a shared input."""
    schema = TensorSchema((field,))
    schema.validate((previous,))
    schema.validate((tensor,))
    if previous is tensor:
        return
    if field.differentiable or previous.requires_grad or tensor.requires_grad:
        raise ValueError(f"Shared input {field.key} must refer to the same tensor in every codec")
    message = f"Shared input {field.key} has conflicting snapshots"
    if previous.device != tensor.device or tensor.is_meta:
        raise ValueError(message)
    if tensor.is_cuda:
        torch._assert_async((previous == tensor).all(), message)
    elif not torch.equal(previous, tensor):
        raise ValueError(message)


class CompositeStateCodec:
    """Dispatch declared fields to native codecs using one boundary implementation.

    The context factory, attribute names and fields come from the caller. Shared
    prepared inputs may be declared by multiple codecs and cross the boundary once. Neither
    field packing nor restoration needs to know which modules own those attributes.
    """

    def __init__(
        self,
        bindings: tuple[tuple[str, tuple[TensorField, ...], StateCodec], ...],
        factory: Callable,
        *,
        shared_inputs: tuple[str, ...] = (),
    ) -> None:
        self.bindings, self.factory = bindings, factory
        names = [name for name, _, _ in bindings]
        if len(set(names)) != len(names):
            raise ValueError("State codecs must declare unique context attributes")
        owners = {}
        for name, fields, _ in bindings:
            for field in fields:
                if field.key in owners and field.key not in shared_inputs:
                    raise ValueError(
                        f"Duplicate state field {field.key} owned by {owners[field.key]} and {name}"
                    )
                owners[field.key] = name

    def _partition(self, fields):
        selected = tuple(
            tuple(f for f in fields if f.key in {item.key for item in declared})
            for _, declared, _ in self.bindings
        )
        if {f.key for group in selected for f in group} != {f.key for f in fields}:
            raise ValueError("Boundary field has no registered state codec")
        return selected

    def export(self, state: Any, fields: tuple[TensorField, ...]) -> tuple[Tensor, ...]:
        """Export original edges in schema order, including interleaved namespaces."""
        values = {}
        for (name, _, codec), selected in zip(self.bindings, self._partition(fields)):
            if selected:
                tensors = codec.export(getattr(state, name), selected)
                TensorSchema(selected).validate(tensors)
                for field, tensor in zip((f for f in selected if f.present), tensors):
                    if field.key in values:
                        validate_shared_input(field, values[field.key], tensor)
                    values[field.key] = tensor
        return tuple(values[f.key] for f in fields if f.present)

    def restore(
        self, fields: tuple[TensorField, ...], tensors: tuple[Tensor, ...], metadata: tuple
    ) -> Any:
        """Create independent native state objects and attach them to a fresh context."""
        TensorSchema(fields).validate(tensors)
        validate_metadata(metadata)
        values, metadata = dict(zip((f.key for f in fields if f.present), tensors)), dict(metadata)
        context = self.factory()
        for (name, _, codec), selected in zip(self.bindings, self._partition(fields)):
            restored = codec.restore(
                selected,
                tuple(values[f.key] for f in selected if f.present),
                metadata.get(name, ()),
            )
            setattr(context, name, restored)
        return context


def compose_state_regions(
    boundary_id: str,
    regions: Sequence[tuple[str, "StateRegion"]],
    factory: Callable[[], Any],
    *,
    shared_inputs: tuple[str, ...] = (),
) -> "StateRegion":
    """Combine native declarations with explicitly shared prepared inputs.

    Each field has one owner unless the host declares it in shared_inputs.
    Shared fields must be inputs of every participating component; they may be
    retained or relayed through a pipeline boundary, but cannot introduce a
    second producer. Export verifies differentiable tensor identity so distinct
    autograd edges cannot silently replace each other. Non-differentiable
    metadata snapshots must have equal values.
    """

    def merge(groups):
        result = {}
        for group in groups:
            for field in group:
                previous = result.setdefault(field.key, field)
                if previous != field:
                    raise ValueError(
                        f"Shared state field has conflicting declarations: {field.key}"
                    )
        return tuple(result.values())

    regions = tuple(regions)
    for name, region in regions:
        inputs = {f.key for f in region.schema.inputs}
        for field in (*region.schema.outputs, *region.retained_fields):
            if field.key in shared_inputs and field.key not in inputs:
                raise ValueError(f"Shared input {field.key} cannot be produced by {name}")
    return StateRegion(
        BoundarySchema(
            boundary_id,
            merge(r.schema.inputs for _, r in regions),
            merge(r.schema.outputs for _, r in regions),
        ),
        CompositeStateCodec(
            tuple(
                (name, merge((r.schema.inputs, r.schema.outputs, r.retained_fields)), r.codec)
                for name, r in regions
            ),
            factory,
            shared_inputs=shared_inputs,
        ),
        tuple((name, r.input_metadata) for name, r in regions),
        tuple((name, r.output_metadata) for name, r in regions),
        merge(r.retained_fields for _, r in regions),
    )


def validate_metadata(metadata: Any) -> None:
    """Accept only immutable host scalars/tuples, never hidden activation references."""
    if metadata is None or type(metadata) in (str, bool, int, float):
        return
    if isinstance(metadata, tuple):
        for item in metadata:
            validate_metadata(item)
        return
    raise TypeError("State boundary metadata must contain only immutable scalars and tuples")


class TensorMappingCodec:
    """Codec for host-prepared arguments; integer inputs need no feature payload."""

    def export(
        self, state: Mapping[str, Tensor | None], fields: tuple[TensorField, ...]
    ) -> tuple[Tensor, ...]:
        """Extract present inputs while preserving gradients of trainable prepared values."""
        schema = TensorSchema(fields)
        for field in fields:
            if not field.present and state.get(field.key) is not None:
                raise ValueError(f"Configured absent field {field.key} has a value")
        tensors = tuple(state.get(field.key) for field in schema.packed_fields)
        schema.validate(tensors)
        return tensors

    def restore(
        self, fields: tuple[TensorField, ...], tensors: tuple[Tensor, ...], metadata: Any = ()
    ) -> dict[str, Tensor | None]:
        """Build a fresh mapping, restoring configured None slots without a tensor."""
        validate_metadata(metadata)
        TensorSchema(fields).validate(tensors)
        values = iter(tensors)
        return {field.key: next(values) if field.present else None for field in fields}


def prepare_boundaries(
    dependencies: Sequence[StateDependency],
    boundaries: Sequence[StateBoundary],
    placement: Sequence[StatePlacement],
) -> tuple[BoundarySchema, ...]:
    """Derive side schemas without changing placement, checkpoint groups or execution.

    Pipeline delivery follows adjacent logical chunks, including VPP. Checkpoint
    and capture regions exclude host preparation and carry only values they read
    or produce for an external reader. Unread relay values remain in parent state.
    """
    placement = tuple(placement)
    if not placement or any(p.start >= p.end or p.pp_rank < 0 for p in placement):
        raise ValueError("State placement requires nonempty ordered intervals")
    if any(a.end != b.start for a, b in zip(placement, placement[1:])):
        raise ValueError("State placement must be contiguous and ordered")
    TensorSchema(tuple(dep.field for dep in dependencies))
    if len({b.boundary_id for b in boundaries}) != len(boundaries):
        raise ValueError("State boundaries require unique identities")

    def owner(position: int) -> StatePlacement:
        for part in placement:
            if part.start <= position < part.end:
                return part
        raise ValueError(f"State position {position} is outside the existing placement")

    for dep in dependencies:
        if dep.origin not in ("prepared", "activation") or dep.delivery not in (
            "local",
            "pipeline",
        ):
            raise ValueError(f"Invalid origin/delivery for {dep.field.key}")
        source = owner(dep.available_at)
        if source.pp_rank != dep.source_pp_rank:
            raise ValueError(f"Source rank does not match placement for {dep.field.key}")
        if tuple(sorted(set(dep.consumed_at))) != dep.consumed_at:
            raise ValueError(f"Consumers must be unique and ordered for {dep.field.key}")
        for position in dep.consumed_at:
            if position < dep.available_at or (
                dep.origin == "activation" and position == dep.available_at
            ):
                raise ValueError(f"State source must precede its consumers: {dep.field.key}")
            if dep.delivery == "local" and owner(position) != source:
                raise ValueError(
                    f"Local state has a consumer in another rank or chunk: {dep.field.key}"
                )
            owner(position)

    schemas = []
    for boundary in boundaries:
        inputs, outputs = [], []
        if boundary.kind == "pipeline":
            if boundary.start != boundary.end or boundary.start not in {
                part.end for part in placement[:-1]
            }:
                raise ValueError("Pipeline boundary must be an existing placement cut")
            for dep in dependencies:
                if (
                    dep.delivery == "pipeline"
                    and dep.consumed_at
                    # A consumer at the cut belongs to the receiving chunk;
                    # a source at that cut is already local to the receiver.
                    and dep.available_at < boundary.start <= dep.consumed_at[-1]
                ):
                    inputs.append(dep.field)
            outputs = inputs.copy()
        elif boundary.kind in ("checkpoint", "capture"):
            if boundary.start >= boundary.end or owner(boundary.start) != owner(boundary.end - 1):
                raise ValueError("State region must stay within one rank and chunk")
            for dep in dependencies:
                reads = any(boundary.start <= p < boundary.end for p in dep.consumed_at)
                produced = boundary.start <= dep.available_at < boundary.end
                if reads and dep.origin == "prepared":
                    if dep.available_at > boundary.start:
                        raise ValueError(
                            f"Prepared field is not ready at region entry: {dep.field.key}"
                        )
                    inputs.append(dep.field)
                elif reads and not produced:
                    inputs.append(dep.field)
                if (
                    dep.origin == "activation"
                    and produced
                    and any(p >= boundary.end for p in dep.consumed_at)
                ):
                    outputs.append(dep.field)
        else:
            raise ValueError(f"Unknown state boundary kind: {boundary.kind}")
        schemas.append(BoundarySchema(boundary.boundary_id, tuple(inputs), tuple(outputs)))
    return tuple(schemas)


@dataclass(frozen=True)
class CheckpointBoundaryPolicy:
    """Output contract shared by checkpoint backends.

    Non-differentiable fields must not acquire autograd edges. An unused output
    keeps a None gradient; an explicitly supplied zero still executes backward.
    """

    outputs: TensorSchema
    strict: bool = False

    def apply(self, result: Tensor | tuple[Tensor, ...]) -> Tensor | tuple[Tensor, ...]:
        """Validate outputs and give each slot independent gradient eligibility.

        Eager execution can still create edges to closed-over parameters when
        every input is frozen. Repeated Tensor objects need distinct identities
        before a checkpoint backend assigns output numbers or marks a slot as
        non-differentiable. Zero-copy views separate those slots without changing
        values or duplicating the producer's computation.
        """
        single = isinstance(result, Tensor)
        tensors = (result,) if single else tuple(result)
        self.outputs.validate(tensors)
        seen, outputs = set(), []
        for field, tensor in zip(self.outputs.packed_fields, tensors):
            repeated = id(tensor) in seen
            seen.add(id(tensor))
            if repeated:
                tensor = tensor.view_as(tensor)
            outputs.append(tensor if field.differentiable else tensor.detach())
        return outputs[0] if single else tuple(outputs)


@dataclass(frozen=True)
class StateRegion:
    """Codec binding for one existing region; retained fields bypass its computation."""

    schema: BoundarySchema
    codec: StateCodec
    input_metadata: tuple = ()
    output_metadata: tuple = ()
    retained_fields: tuple[TensorField, ...] = ()

    def __post_init__(self) -> None:
        validate_metadata(self.input_metadata)
        validate_metadata(self.output_metadata)
        TensorSchema((*self.retained_fields, *self.schema.outputs))


def prepare_state_region(
    dependencies: Sequence[StateDependency],
    boundary: StateBoundary,
    placement: Sequence[StatePlacement],
    codec: StateCodec,
    *,
    input_metadata: tuple = (),
    output_metadata: tuple = (),
) -> StateRegion:
    """Bind a codec to planned inputs/outputs and fields that bypass a region.

    An incoming value stays in the parent state if a later region still consumes
    it, even when this region never reads it. Prepared values available at entry
    also survive until their last consumer. Pipeline schemas already contain all
    live relay fields, so they need no separate retained fields.
    """
    dependencies = tuple(dependencies)
    (schema,) = prepare_boundaries(dependencies, (boundary,), placement)
    retained = (
        tuple(
            dep.field
            for dep in dependencies
            if (
                dep.available_at < boundary.start
                or (dep.origin == "prepared" and dep.available_at == boundary.start)
            )
            and any(reader >= boundary.end for reader in dep.consumed_at)
        )
        if boundary.kind != "pipeline"
        else ()
    )
    return StateRegion(schema, codec, input_metadata, output_metadata, retained)


@dataclass(frozen=True)
class StaticTensorSchema:
    """An existing graph runner's concrete input contract, including tensor strides."""

    schema: TensorSchema
    strides: tuple[tuple[int, ...], ...]

    @classmethod
    def from_tensors(
        cls, fields: tuple[TensorField, ...], tensors: tuple[Tensor, ...]
    ) -> "StaticTensorSchema":
        """Snapshot host descriptors only; graph slots continue to own their buffers."""
        schema = TensorSchema(fields)
        schema.validate(tensors)
        return cls(schema, tuple(tensor.stride() for tensor in tensors))

    def validate(self, tensors: tuple[Tensor, ...]) -> None:
        """Reject replay inputs outside the captured profile before invoking the runner."""
        self.schema.validate(tensors)
        if tuple(tensor.stride() for tensor in tensors) != self.strides:
            raise ValueError("State graph inputs do not match captured strides")
