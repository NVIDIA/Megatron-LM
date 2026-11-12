# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Evaluate dynamic-CP schedules with optional per-microbatch EP rendezvous.

Each rank executes one task in each microbatch. A CP group is treated as an
atomic task: its member ranks rendezvous when all of them have reached that
microbatch, then remain occupied for the predicted group wall time. Disjoint
groups, including groups from different microbatches, may overlap in time.
After one microbatch's attention groups finish, each fixed contiguous EP rank
group synchronizes independently; `ep_size=1` preserves the original fully
asynchronous no-DP-barrier behavior.
"""

from __future__ import annotations

class RankTimelineError(ValueError):
    """Raised when a schedule cannot map to the configured dynamic-CP groups."""

def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0

def uses_arbitrary_rank_layout(groups: Sequence[dict]) -> bool:
    """Use the runtime layout contract even in all-power-of-two rounds."""
    modes = {group["rank_layout"] for group in groups if "rank_layout" in group}
    if not modes <= {"nested", "arbitrary_contiguous"} or len(modes) > 1:
        raise RankTimelineError("invalid or conflicting rank_layout contracts")
    if modes:
        return "arbitrary_contiguous" in modes
    # Historical payloads predate an explicit runtime layout contract.
    return any(not _is_power_of_two(int(group["cp"])) for group in groups)

def rank_intervals(groups: Sequence[dict], world_size: int) -> tuple[tuple[int, ...], ...]:
    """Map groups in physical-rank order to contiguous rank blocks.

    Explicit ``rank_layout=arbitrary_contiguous`` enables arbitrary contiguous
    blocks independent of the CP sizes present in this round. Historical
    untagged schedules retain their original inferred layout semantics.
    """
    if world_size <= 0:
        raise RankTimelineError("world_size must be positive")
    arbitrary_layout = uses_arbitrary_rank_layout(groups)
    offset = 0
    result = []
    for slot, group in enumerate(groups, start=1):
        cp = int(group["cp"])
        if cp <= 0:
            raise RankTimelineError(f"slot {slot} has invalid CP{cp}")
        if not arbitrary_layout and world_size % cp:
            raise RankTimelineError(
                f"slot {slot} CP{cp} does not divide world size {world_size}"
            )
        if not arbitrary_layout and offset % cp:
            raise RankTimelineError(
                f"slot {slot} CP{cp} starts at unaligned rank {offset}"
            )
        end = offset + cp
        if end > world_size:
            raise RankTimelineError(
                f"slot {slot} ends at rank {end}, beyond world size {world_size}"
            )
        result.append(tuple(range(offset, end)))
        offset = end
    if offset != world_size:
        raise RankTimelineError(
            f"microbatch occupies {offset} ranks, expected exactly {world_size}"
        )
    return tuple(result)
