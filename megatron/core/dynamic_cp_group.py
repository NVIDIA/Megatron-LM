# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Lightweight groups for native dynamic context-parallel transport."""

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch


def build_bounded_peer_ring(
    parent_ranks: Sequence[int], group_start: int, cp_size: int
) -> Tuple[int, ...]:
    """Build a direct ring whose peers are at most two parent positions apart."""
    parent_ranks = tuple(parent_ranks)
    if len(set(parent_ranks)) != len(parent_ranks):
        raise ValueError("parent_ranks must be unique")
    if cp_size < 1 or group_start < 0 or group_start + cp_size > len(parent_ranks):
        raise ValueError(
            f"Invalid interval start={group_start}, size={cp_size}, "
            f"parent_size={len(parent_ranks)}"
        )

    offsets = [0, *range(1, cp_size, 2), *reversed(range(2, cp_size, 2))]
    if group_start % 2:
        offsets = [offsets[0], *reversed(offsets[1:])]
    return tuple(parent_ranks[group_start + offset] for offset in offsets)


@dataclass(frozen=True)
class LogicalCPGroup:
    """Describe a CP ring without creating a subgroup communicator."""

    ranks: Tuple[int, ...]
    cp_size: int
    cp_rank: int

    def __post_init__(self) -> None:
        if self.cp_size != len(self.ranks):
            raise ValueError("cp_size must match the rank count")
        if not 0 <= self.cp_rank < self.cp_size:
            raise ValueError("cp_rank is outside the logical group")

    @classmethod
    def from_parent_interval(
        cls, parent_ranks: Sequence[int], group_start: int, cp_size: int, rank: int
    ) -> "LogicalCPGroup":
        """Build the descriptor for one contiguous interval of the parent group."""
        ring = build_bounded_peer_ring(parent_ranks, group_start, cp_size)
        if rank not in ring:
            raise ValueError(
                f"Rank {rank} is outside interval [{group_start}, {group_start + cp_size})"
            )
        return cls(ranks=ring, cp_size=cp_size, cp_rank=ring.index(rank))

    def size(self) -> int:
        """Return the logical group size."""
        return self.cp_size

    def rank(self) -> int:
        """Return this process's rank in the logical ring."""
        return self.cp_rank


def get_process_group_ranks(group) -> Tuple[int, ...]:
    """Return ranks from a ProcessGroup or a logical CP descriptor."""
    if isinstance(group, LogicalCPGroup):
        return group.ranks
    return tuple(torch.distributed.get_process_group_ranks(group))
