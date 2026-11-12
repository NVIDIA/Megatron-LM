# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

#!/usr/bin/env python3
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

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


class RankTimelineError(ValueError):
    """Raised when a schedule cannot map to the configured dynamic-CP groups."""


@dataclass(frozen=True)
class GroupTimeline:
    microbatch: int
    slot: int
    cp: int
    ranks: tuple[int, ...]
    cost_ms: float
    arrivals_ms: tuple[float, ...]
    start_ms: float
    finish_ms: float
    rendezvous_wait_gpu_ms: float


@dataclass(frozen=True)
class MicrobatchTimeline:
    microbatch: int
    earliest_start_ms: float
    latest_finish_ms: float
    ep_sync_wait_gpu_ms: float
    groups: tuple[GroupTimeline, ...]


@dataclass(frozen=True)
class ScheduleTimeline:
    world_size: int
    ep_size: int
    makespan_ms: float
    rank_work_ms: float
    balanced_wall_time_ms: float
    total_idle_gpu_ms: float
    rendezvous_wait_gpu_ms: float
    ep_sync_wait_gpu_ms: float
    final_sync_wait_gpu_ms: float
    final_rank_times_ms: tuple[float, ...]
    microbatches: tuple[MicrobatchTimeline, ...]


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


def evaluate_schedule(
    rounds: Sequence[Sequence[dict]],
    *,
    world_size: int,
    ep_size: int = 1,
    cost_field: str = "cost_ms",
) -> ScheduleTimeline:
    """Return the rank timeline with independent contiguous EP rendezvous."""
    if not rounds:
        raise RankTimelineError("schedule must contain at least one microbatch")
    if ep_size <= 0 or world_size % ep_size:
        raise RankTimelineError(
            f"ep_size must be a positive divisor of world_size, got "
            f"ep_size={ep_size}, world_size={world_size}"
        )

    available = [0.0] * world_size
    rank_work_ms = 0.0
    rendezvous_wait_gpu_ms = 0.0
    ep_sync_wait_gpu_ms = 0.0
    microbatches = []

    for microbatch, groups in enumerate(rounds, start=1):
        if not groups:
            raise RankTimelineError(f"microbatch {microbatch} is empty")
        intervals = rank_intervals(groups, world_size)
        group_rows = []
        for slot, (group, ranks) in enumerate(zip(groups, intervals), start=1):
            cp = int(group["cp"])
            cost_ms = float(group[cost_field])
            if not math.isfinite(cost_ms) or cost_ms <= 0.0:
                raise RankTimelineError(
                    f"microbatch {microbatch} slot {slot} has invalid cost {cost_ms}"
                )
            arrivals = tuple(available[rank] for rank in ranks)
            start_ms = max(arrivals)
            finish_ms = start_ms + cost_ms
            wait_gpu_ms = sum(start_ms - value for value in arrivals)
            for rank in ranks:
                available[rank] = finish_ms
            rank_work_ms += cp * cost_ms
            rendezvous_wait_gpu_ms += wait_gpu_ms
            group_rows.append(
                GroupTimeline(
                    microbatch=microbatch,
                    slot=slot,
                    cp=cp,
                    ranks=ranks,
                    cost_ms=cost_ms,
                    arrivals_ms=arrivals,
                    start_ms=start_ms,
                    finish_ms=finish_ms,
                    rendezvous_wait_gpu_ms=wait_gpu_ms,
                )
            )
        microbatch_ep_wait = 0.0
        for start in range(0, world_size, ep_size):
            end = start + ep_size
            synchronized = max(available[start:end])
            wait = sum(synchronized - value for value in available[start:end])
            available[start:end] = [synchronized] * ep_size
            microbatch_ep_wait += wait
        ep_sync_wait_gpu_ms += microbatch_ep_wait
        microbatches.append(
            MicrobatchTimeline(
                microbatch=microbatch,
                earliest_start_ms=min(row.start_ms for row in group_rows),
                latest_finish_ms=max(row.finish_ms for row in group_rows),
                ep_sync_wait_gpu_ms=microbatch_ep_wait,
                groups=tuple(group_rows),
            )
        )

    makespan_ms = max(available)
    final_sync_wait_gpu_ms = sum(makespan_ms - value for value in available)
    total_idle_gpu_ms = world_size * makespan_ms - rank_work_ms
    decomposed_idle = (
        rendezvous_wait_gpu_ms + ep_sync_wait_gpu_ms + final_sync_wait_gpu_ms
    )
    if not math.isclose(
        total_idle_gpu_ms,
        decomposed_idle,
        rel_tol=1.0e-10,
        abs_tol=1.0e-10,
    ):
        raise AssertionError(
            "rank-timeline idle decomposition does not close: "
            f"total={total_idle_gpu_ms}, components={decomposed_idle}"
        )

    return ScheduleTimeline(
        world_size=world_size,
        ep_size=ep_size,
        makespan_ms=makespan_ms,
        rank_work_ms=rank_work_ms,
        balanced_wall_time_ms=rank_work_ms / world_size,
        total_idle_gpu_ms=max(0.0, total_idle_gpu_ms),
        rendezvous_wait_gpu_ms=max(0.0, rendezvous_wait_gpu_ms),
        ep_sync_wait_gpu_ms=max(0.0, ep_sync_wait_gpu_ms),
        final_sync_wait_gpu_ms=max(0.0, final_sync_wait_gpu_ms),
        final_rank_times_ms=tuple(available),
        microbatches=tuple(microbatches),
    )


def schedule_makespan_ms(
    rounds: Sequence[Sequence[dict]],
    *,
    world_size: int,
    ep_size: int = 1,
    cost_field: str = "cost_ms",
) -> float:
    return evaluate_schedule(
        rounds,
        world_size=world_size,
        ep_size=ep_size,
        cost_field=cost_field,
    ).makespan_ms


def canonical_physical_order(groups: Iterable[dict]) -> list[dict]:
    """Return one valid dyadic rank ordering without changing group contents."""
    return sorted(
        groups,
        key=lambda group: (
            -int(group["cp"]),
            -float(group.get("cost_ms", 0.0)),
            tuple(int(value) for value in group.get("sample_ids", ())),
        ),
    )
