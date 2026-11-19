# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tensor contracts for extra state crossing existing execution boundaries.

The model declares the ordered fields each execution region reads and writes.
Activations belong to a single forward; the same contract is used for checkpoint,
graph and pipeline boundaries without encoding any attention architecture.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Literal, Mapping, Sequence, get_args

import torch
from torch import Tensor

# How a field relates to the sequence sharding of context parallelism:
# contiguous/zigzag: sequence on dimension zero, sharded like Megatron's CP layouts.
# replicated: identical on every CP rank. local: rank-specific, not a sequence shard.
# strided: no sequence semantics are declared (e.g. a pipeline's hidden activation).
StateLayout = Literal["contiguous", "zigzag", "replicated", "local", "strided"]


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
    layout: StateLayout
    differentiable: bool
    present: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key or not isinstance(self.shape, tuple):
            raise ValueError("A state field requires a stable key and tuple shape")
        if self.layout not in get_args(StateLayout):
            raise ValueError(f"Unsupported state layout: {self.layout}")
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
    """Validated form of an ordered field tuple, used inside the state boundaries.

    Public APIs take and return the tuple itself. Absent fields keep their slot in
    the mapping but not on the wire.
    """

    fields: tuple[TensorField, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.fields, tuple):
            raise TypeError("State schema fields must be an immutable tuple")
        keys = [field.key for field in self.fields]
        if len(set(keys)) != len(keys):
            raise ValueError("State schema contains duplicate field keys")

    @cached_property
    def packed_fields(self) -> tuple[TensorField, ...]:
        """Return fields which have a tensor, in wire/argument order."""
        return tuple(field for field in self.fields if field.present)

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

    def pack(self, state: Mapping[str, Tensor | None]) -> tuple[Tensor, ...]:
        """Select declared fields without copying or detaching eligible producer edges."""
        for field in self.fields:
            if not field.present and state.get(field.key) is not None:
                raise ValueError(f"Absent state field {field.key} has a tensor")
        return self.detach_ineligible(tuple(state[field.key] for field in self.packed_fields))

    def unpack(self, tensors: Sequence[Tensor]) -> dict[str, Tensor | None]:
        """Restore a fresh mapping, retaining explicit absent slots."""
        values = iter(self.detach_ineligible(tensors))
        return {field.key: next(values) if field.present else None for field in self.fields}


@dataclass(frozen=True)
class StateAccess:
    """The keys one layer reads before it runs and writes when it finishes.

    A key in both replaces the value the layer read; values never change in place.
    """

    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for keys in (self.reads, self.writes):
            if not isinstance(keys, tuple) or not all(isinstance(k, str) and k for k in keys):
                raise ValueError("A state access names its keys with a tuple of strings")
            if len(set(keys)) != len(keys):
                raise ValueError("A state access names a key twice")


class StateDataflow:
    """Live keys at each layer boundary, derived from every layer's reads and writes.

    Boundary ``b`` precedes layer ``b``; boundary ``len(layers)`` follows the last
    layer. ``inputs`` exist before layer 0 and ``outputs`` are consumed after the
    last layer. A value retires after its last reader, so a layer that only reads a
    key never republishes it, and a key crosses a boundary only while a later layer
    still reads it. Every value must be read, or be one of the ``outputs``.
    """

    def __init__(
        self,
        layers: Sequence[StateAccess],
        *,
        inputs: Sequence[str] = (),
        outputs: Sequence[str] = (),
    ) -> None:
        self.layers, self.inputs, self.outputs = tuple(layers), tuple(inputs), tuple(outputs)
        if len(set(self.inputs)) != len(self.inputs) or len(set(self.outputs)) != len(self.outputs):
            raise ValueError("State dataflow inputs and outputs must be unique")
        order = list(self.inputs)
        # Per key: the layer that wrote its current value (-1 for an input) and the
        # value's last reader so far. Each finished value is held at boundaries (start, end].
        current = {key: [-1, None] for key in self.inputs}
        spans = []

        def finish(key, start, end):
            if end is None:
                origin = "Input" if start < 0 else f"Layer {start}'s write of"
                raise ValueError(f"{origin} {key!r} is never read and is not an output")
            spans.append((key, start, end))

        for index, access in enumerate(self.layers):
            for key in access.reads:
                if key not in current:
                    raise ValueError(f"Layer {index} reads {key!r} before any layer writes it")
                current[key][1] = index
            for key in access.writes:
                if key in current:
                    finish(key, *current[key])
                else:
                    order.append(key)
                current[key] = [index, None]
        missing = [key for key in self.outputs if key not in current]
        if missing:
            raise ValueError(f"State dataflow outputs {missing} are never written")
        for key, (start, end) in current.items():
            finish(key, start, len(self.layers) if key in self.outputs else end)
        rank = {key: i for i, key in enumerate(order)}
        self._live = tuple(
            tuple(sorted((k for k, start, end in spans if start < b <= end), key=rank.get))
            for b in range(len(self.layers) + 1)
        )

    def live(self, boundary: int) -> tuple[str, ...]:
        """Keys held before layer ``boundary``, e.g. the state a pipeline boundary carries."""
        if not 0 <= boundary < len(self._live):
            raise ValueError(f"Boundary {boundary} is outside this dataflow")
        return self._live[boundary]

    def retired_after(self, layer: int) -> tuple[str, ...]:
        """Keys whose last reader is ``layer``."""
        following = set(self.live(layer + 1))
        return tuple(key for key in self.live(layer) if key not in following)

    def retire(self, layer: int, state: Mapping[str, Tensor | None]) -> dict[str, Tensor | None]:
        """Return a fresh mapping without the keys ``layer`` read last; others remain."""
        retired = set(self.retired_after(layer))
        return {key: value for key, value in state.items() if key not in retired}
