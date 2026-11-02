# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tensor contracts for extra state crossing existing execution boundaries.

The model declares ordered fields for each execution region. Activations belong
to a single forward; the same contract is used for checkpoint, graph and pipeline
boundaries without encoding any attention architecture.
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
    """Stable field order; absent fields keep their slot in the mapping but not on the wire."""

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
