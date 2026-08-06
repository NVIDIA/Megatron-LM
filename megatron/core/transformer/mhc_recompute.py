# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fixed-address arena contracts for mHC selective recomputation.

The schedule owns one :class:`MHCRecomputeArena` per in-flight recompute
manager.  Partial CUDA Graph consumers bind their captured input tensors as
external slots.  mHC producers then write both the original forward value and
the backward recompute value directly into those slots.

The capture helper may pack several external slots into one backing allocation;
this module deliberately reasons in views so schedule liveness and physical
layout can evolve independently.
"""

import os
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Hashable, Tuple

import torch
from torch import Tensor

# Opt-in clobber detector for bring-up of new schedule shapes (e.g. VPP):
# checksum the slot after every producer write and verify it is unchanged when
class MHCRecomputePhase(IntEnum):
    """Backward consumer barriers in EP 1F1B execution order."""

    BEFORE_COMBINE_BWD = 0
    BEFORE_MLP_BWD = 1
    BEFORE_ATTN_BWD = 2


@dataclass(frozen=True)
class MHCRecomputeSlotMetadata:
    """Immutable tensor metadata recorded when an arena slot is bound."""

    shape: torch.Size
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout
    data_ptr: int
    storage_offset: int


class MHCRecomputeArenaSlot:
    """One fixed-address output view shared by producer and graph consumer."""

    def __init__(self, key: Hashable, tensor: Tensor, ordinal: int):
        if not isinstance(tensor, Tensor) or not tensor.is_cuda:
            raise TypeError("mHC recompute arena slots must be CUDA tensors")
        if not tensor.is_contiguous():
            raise ValueError("mHC recompute arena slots must be contiguous")

        self.key = key
        self.ordinal = ordinal
        self.consumer = tensor
        self.metadata = MHCRecomputeSlotMetadata(
            shape=tensor.shape,
            dtype=tensor.dtype,
            device=tensor.device,
            layout=tensor.layout,
            data_ptr=tensor.data_ptr(),
            storage_offset=tensor.storage_offset(),
        )

    @property
    def writer(self) -> Tensor:
        """Return a fresh detached TensorImpl over the captured bytes.

        A custom autograd Function may attach output AutogradMeta to the exact
        TensorImpl it returns. Reusing one persistent detached object would
        therefore turn the next recompute's ``out`` operand into a grad-requiring
        tensor. The physical contract is the storage address, so create a fresh
        detached view for every producer launch.
        """
        self.validate_address()
        return _fresh_view(self.consumer)

    def validate_address(self) -> None:
        """Fail before replay if the captured surface was rebound."""
        if self.consumer.data_ptr() != self.metadata.data_ptr:
            raise RuntimeError(
                f"mHC arena slot {self.key!r} changed address: expected "
                f"{self.metadata.data_ptr}, got {self.consumer.data_ptr()}"
            )

    def validate_output(self, output: Tensor) -> None:
        """Validate that a producer returned the arena bytes, not a copied result."""
        self.validate_address()
        if output.data_ptr() != self.metadata.data_ptr:
            raise RuntimeError(
                f"mHC arena producer for slot {self.key!r} did not direct-write: "
                f"expected pointer {self.metadata.data_ptr}, got {output.data_ptr()}"
            )
        if (
            output.shape != self.metadata.shape
            or output.dtype != self.metadata.dtype
            or output.device != self.metadata.device
            or output.layout != self.metadata.layout
        ):
            raise ValueError(f"mHC arena output metadata changed for slot {self.key!r}")


class MHCRecomputeArena:
    """Ordered fixed-address slots owned by one in-flight recompute manager."""

    def __init__(self):
        self._slots: Dict[Hashable, MHCRecomputeArenaSlot] = {}
        self._order: list[Hashable] = []

    def bind_external_slot(self, key: Hashable, tensor: Tensor) -> MHCRecomputeArenaSlot:
        """Bind a captured input view, preserving deterministic registration order."""
        existing = self._slots.get(key)
        if existing is not None:
            existing.validate_address()
            if (
                existing.consumer is not tensor
                and existing.consumer.data_ptr() != tensor.data_ptr()
            ):
                raise RuntimeError(f"mHC arena key {key!r} was rebound to a different tensor")
            return existing

        slot = MHCRecomputeArenaSlot(key, tensor, len(self._order))
        self._slots[key] = slot
        self._order.append(key)
        return slot

    @property
    def slots(self) -> Tuple[MHCRecomputeArenaSlot, ...]:
        """Return slots in producer registration order."""
        return tuple(self._slots[key] for key in self._order)

    def validate_addresses(self) -> None:
        """Validate every slot before a phase barrier reuses it."""
        for slot in self.slots:
            slot.validate_address()


def _fresh_view(backing: torch.Tensor) -> torch.Tensor:
    """New TensorImpl (own version counter) over ``backing``'s storage."""
    t = torch.empty(0, dtype=backing.dtype, device=backing.device)
    t.set_(backing.untyped_storage(), 0, backing.shape)
    return t


