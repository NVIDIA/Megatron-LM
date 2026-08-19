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

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Hashable, Tuple

import torch
from torch import Tensor


def uses_mhc_recompute_attn_cuda_graph_split(config) -> bool:
    """Whether attention-only TE graphs consume eager mHC recompute outputs.

    Single definition for a predicate that gates both the layer-side split
    (``TransformerLayer._uses_mhc_recompute_attn_cuda_graph_split``) and the
    capture-side arena validation (``TECudaGraphHelper._uses_mhc_direct_write_arena``).
    Two spellings drift, and the failure is asymmetric: one site would raise on a
    ``None`` field while the other silently returned False, disabling exactly the
    static-input validation that keeps a rebound buffer from being read by a
    captured backward.
    """
    from megatron.core.transformer.enums import CudaGraphModule

    return (
        config.mhc_recompute_attn_cuda_graph_split
        and config.cuda_graph_impl == "transformer_engine"
        and list(config.cuda_graph_modules or []) == [CudaGraphModule.attn]
        and config.recompute_granularity == "selective"
        and list(config.recompute_modules or []) == ["mhc"]
    )


class MHCRecomputePhase(IntEnum):
    """Backward consumer barriers in EP 1F1B execution order.

    Only ``BEFORE_COMBINE_BWD`` has a producer today -- ``add_checkpoint``
    rejects anything else -- so ``recompute_until``'s filter currently admits
    every checkpoint whatever phase it is asked for.

    The members name *barrier arguments* to ``recompute_until``, and both of these
    have callers: the schedule node passes ``BEFORE_COMBINE_BWD`` and
    ``recompute_now`` passes ``BEFORE_ATTN_BWD``. What is inert is the checkpoint
    side -- no checkpoint carries a phase other than ``BEFORE_COMBINE_BWD``, so
    the filter cannot yet discriminate between the two arguments.

    TODO: partition checkpoints across phases so ``recompute_until`` replays only
    what each barrier needs, and add the intermediate ``BEFORE_MLP_BWD`` barrier
    in the same change. It is not declared here because nothing would pass it:
    an unreachable member invites comparisons that can never be true.
    """

    BEFORE_COMBINE_BWD = 0
    BEFORE_ATTN_BWD = 1


@dataclass(frozen=True)
class MHCRecomputeSlotMetadata:
    """Immutable tensor metadata recorded when an arena slot is bound."""

    shape: torch.Size
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout
    data_ptr: int


class MHCRecomputeArenaSlot:
    """One fixed-address output view shared by producer and graph consumer."""

    def __init__(self, key: Hashable, tensor: Tensor):
        if not isinstance(tensor, Tensor) or not tensor.is_cuda:
            raise TypeError("mHC recompute arena slots must be CUDA tensors")
        if not tensor.is_contiguous():
            raise ValueError("mHC recompute arena slots must be contiguous")

        self.key = key
        self.consumer = tensor
        # storage_offset is deliberately not recorded: data_ptr already equals
        # storage.data_ptr() + storage_offset * itemsize, so pointer equality is
        # the offset check. Storing it too would read like a second, independent
        # invariant that nothing enforces.
        self.metadata = MHCRecomputeSlotMetadata(
            shape=tensor.shape,
            dtype=tensor.dtype,
            device=tensor.device,
            layout=tensor.layout,
            data_ptr=tensor.data_ptr(),
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

        slot = MHCRecomputeArenaSlot(key, tensor)
        self._slots[key] = slot
        return slot

    @property
    def slots(self) -> Tuple[MHCRecomputeArenaSlot, ...]:
        """Return slots in producer registration order.

        Dict insertion order is the registration order (guaranteed since 3.7), so
        no parallel list is kept: two structures that must stay in lockstep are a
        liability on an arena whose whole job is fixed addresses.
        """
        return tuple(self._slots.values())

    def validate_addresses(self) -> None:
        """Validate every slot before a phase barrier reuses it."""
        for slot in self.slots:
            slot.validate_address()


def _fresh_view(backing: torch.Tensor) -> torch.Tensor:
    """New TensorImpl (own version counter) over ``backing``'s bytes.

    The offset has to be carried through: a capture helper may pack several
    slots into one backing allocation, and a view built at offset 0 would hand
    the producer the base of the storage rather than this slot's bytes, so the
    direct write would land on a neighbour's region.
    """
    t = torch.empty(0, dtype=backing.dtype, device=backing.device)
    t.set_(backing.untyped_storage(), backing.storage_offset(), backing.shape)
    return t
