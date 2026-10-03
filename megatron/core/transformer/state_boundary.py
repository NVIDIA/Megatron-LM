# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tensor contracts for extra state crossing existing execution boundaries.

These helpers describe data, not model construction or execution. Positions and
placement come from the caller's existing layer order.
"""

from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence

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
            raise ValueError(
                "A state field requires a stable key, layout and tuple shape"
            )
        if not isinstance(self.dtype, torch.dtype) or any(
            type(flag) is not bool for flag in (self.present, self.differentiable)
        ):
            raise ValueError(
                "State fields require a torch dtype and boolean qualifications"
            )
        if any(type(dim) is not int or dim < 0 for dim in self.shape):
            raise ValueError(f"Invalid shape for state field {self.key}")
        if self.differentiable and not (
            self.dtype.is_floating_point or self.dtype.is_complex
        ):
            raise ValueError(
                f"Non-floating state field {self.key} cannot be differentiable"
            )


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
        return tuple(
            i for i, field in enumerate(self.packed_fields) if field.differentiable
        )

    @property
    def fingerprint(self) -> int:
        """Return a stable signed-int64-safe digest, independent of Python hash seeds."""
        values = tuple(
            (f.key, f.shape, str(f.dtype), f.layout, f.differentiable, f.present)
            for f in self.fields
        )
        return (
            int.from_bytes(sha256(repr(values).encode("utf-8")).digest()[:8], "big")
            >> 1
        )

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
                raise ValueError(
                    f"State field {field.key} has incompatible shape/dtype/layout"
                )
