# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Lightweight topology for dynamic context-parallel groups."""

import weakref
from dataclasses import dataclass
from typing import Tuple

import torch

_LOGICAL_CP_TRANSPORT_GROUPS = weakref.WeakKeyDictionary()


@dataclass(frozen=True)
class LogicalCPGroup:
    """Describe a dynamic-CP subgroup without creating a ProcessGroup."""

    ranks: Tuple[int, ...]
    cp_size: int
    cp_rank: int

    def __post_init__(self) -> None:
        """Validate that the cached geometry matches the rank list."""
        if self.cp_size != len(self.ranks):
            raise ValueError(
                f"cp_size ({self.cp_size}) must match the number of ranks ({len(self.ranks)})."
            )
        if not 0 <= self.cp_rank < self.cp_size:
            raise ValueError(f"cp_rank ({self.cp_rank}) must be in [0, {self.cp_size}).")

    def size(self) -> int:
        """Match the ProcessGroup topology interface."""
        return self.cp_size

    def rank(self) -> int:
        """Match the ProcessGroup topology interface."""
        return self.cp_rank


def get_process_group_ranks(group) -> Tuple[int, ...]:
    """Return global ranks from either a logical group or a ProcessGroup."""
    if isinstance(group, LogicalCPGroup):
        return group.ranks
    return tuple(torch.distributed.get_process_group_ranks(group))


def set_logical_cp_transport_group(group: LogicalCPGroup, transport_group) -> None:
    """Associate a logical CP topology with an existing parent ProcessGroup."""
    _LOGICAL_CP_TRANSPORT_GROUPS[group] = weakref.ref(transport_group)


def get_logical_cp_transport_group(group):
    """Return the transport group for a logical CP group, or the group itself."""
    if not isinstance(group, LogicalCPGroup):
        return group
    transport_group_ref = _LOGICAL_CP_TRANSPORT_GROUPS.get(group)
    transport_group = transport_group_ref() if transport_group_ref is not None else None
    if transport_group is None:
        raise RuntimeError("Logical CP group requires a registered parent transport group.")
    return transport_group
