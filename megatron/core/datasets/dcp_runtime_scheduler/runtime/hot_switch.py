# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Portions derived from mazhuoru/DynamicUSP, BSD-3-Clause.
# See ../DYNAMIC_USP_NOTICE.txt for the upstream copyright and license.

"""Communication-agnostic DynamicUSP global-batch allocation and count search.

This ports the ``solve_gbs`` Stage 1 policy, not its attention-specific profiles
or optional fixed-topology Stage 2. Inputs are tokens, scheduling weights,
allowed group widths, a capacity predicate and a complete-layout objective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


class InfeasibleLayout(ValueError):
    """A candidate microbatch count did not admit a complete nonempty layout."""


@dataclass(frozen=True)
class HotSwitchOptions:
    """Generic allocation policy; none of these options describes communication."""

    candidate_count: int = 4
    outlier_threshold: float = 1.0
    pack_shorts_into_long_groups: bool = True
    column_offset: bool = True
    tier_affinity: bool = True
    heavy_middle: bool = True

    def __post_init__(self):
        if type(self.candidate_count) is not int or self.candidate_count < 1:
            raise ValueError("candidate_count must be a positive integer")
        if (
            isinstance(self.outlier_threshold, bool)
            or not math.isfinite(self.outlier_threshold)
            or self.outlier_threshold <= 0
        ):
            raise ValueError("outlier_threshold must be finite and positive")
        for name in (
            "pack_shorts_into_long_groups",
            "column_offset",
            "tier_affinity",
            "heavy_middle",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")


@dataclass
class Group:
    """One contiguous logical rank group in a candidate global-batch layout."""

    microbatch: int
    start: int
    size: int
    sample_ids: list[int]
    load: float = 0.0
    tokens: int = 0


class HotSwitchScheduler:
    """Allocate a global batch using the upstream deterministic hot-switch rules.

    ``fits(ids, size)`` owns physical padding and any additional capacity caps.
    ``objective(layout)`` owns time prediction; lower is better. The scheduler
    does not import a training framework or inspect either callback's internals.
    The original power-of-two, aligned-width policy is retained.
    """

    def __init__(
        self,
        world_size: int,
        group_sizes: Sequence[int],
        tokens_per_rank: int,
        options: HotSwitchOptions | None = None,
    ) -> None:
        sizes = tuple(sorted(set(group_sizes)))
        if world_size < 1 or tokens_per_rank < 1 or not sizes:
            raise ValueError("positive world/capacity and nonempty group_sizes required")
        if any(type(s) is not int or s < 1 or s & (s - 1) or world_size % s for s in sizes):
            raise ValueError("group_sizes must be power-of-two divisors of world_size")
        self.world_size = world_size
        self.sizes = sizes
        self.tokens_per_rank = tokens_per_rank
        self.options = options or HotSwitchOptions()
        self.last_candidates = []

    def _requirements(self, tokens, fits):
        required = {}
        for sample_id in tokens:
            size = next((s for s in self.sizes if fits([sample_id], s)), None)
            if size is None:
                raise InfeasibleLayout(f"sample {sample_id} exceeds every allowed group capacity")
            required[sample_id] = size
        return required

    def allocate(
        self,
        tokens: Mapping[int, int],
        weights: Mapping[int, float],
        count: int,
        fits: Callable[[list[int], int], bool],
    ) -> list[list[Group]]:
        """Allocate one candidate count, preserving upstream sample and rank ties."""
        if type(count) is not int or count < 1:
            raise ValueError("microbatch count must be a positive integer")
        if tokens.keys() != weights.keys() or not tokens:
            raise ValueError("tokens and weights must have the same nonempty sample IDs")
        if any(type(t) is not int or t <= 0 for t in tokens.values()) or any(
            isinstance(w, bool) or not math.isfinite(w) or w <= 0 for w in weights.values()
        ):
            raise ValueError("tokens and scheduling weights must be finite and positive")
        needed = self._requirements(tokens, fits)
        world, minimum = self.world_size, self.sizes[0]
        capacity = world * self.tokens_per_rank
        mean = sum(weights.values()) / (count * world)
        threshold = self.options.outlier_threshold * mean
        long_ids = sorted(
            (i for i in tokens if weights[i] > threshold or needed[i] > minimum),
            key=weights.__getitem__,
            reverse=True,
        )
        long_set = set(long_ids)
        short_ids = sorted(
            (i for i in tokens if i not in long_set), key=weights.__getitem__, reverse=True
        )
        free = [{s: (1 << (world // s)) - 1 for s in self.sizes} for _ in range(count)]
        column_load, column_tokens = [0.0] * count, [0] * count
        hosts = [set() for _ in range(count)]
        long_groups = []

        for sample_id in long_ids:
            ideal = max(minimum, math.ceil(weights[sample_id] / mean))
            rounded = 1 << (ideal - 1).bit_length()
            size = max(needed[sample_id], max(s for s in self.sizes if s <= rounded))
            if minimum > 1 and size <= minimum:
                short_ids.append(sample_id)
                continue
            eligible = [
                c
                for c in range(count)
                if free[c][size] and column_tokens[c] + tokens[sample_id] <= capacity
            ]
            if not eligible:
                short_ids.append(sample_id)
                continue
            column = min(
                eligible,
                key=lambda c: (
                    0 if self.options.tier_affinity and size in hosts[c] else 1,
                    column_load[c],
                    c,
                ),
            )
            start = (free[column][size].bit_length() - 1) * size
            for width in self.sizes:
                if width <= size:
                    free[column][width] &= ~(((1 << (size // width)) - 1) << (start // width))
                else:
                    free[column][width] &= ~(1 << (start // width))
            column_load[column] += weights[sample_id]
            column_tokens[column] += tokens[sample_id]
            hosts[column].add(size)
            long_groups.append(
                Group(
                    column, start, size, [sample_id], weights[sample_id] / size, tokens[sample_id]
                )
            )

        def can_take(group, sample_id):
            return column_tokens[group.microbatch] + tokens[sample_id] <= capacity and fits(
                group.sample_ids + [sample_id], group.size
            )

        def add(group, sample_id):
            group.sample_ids.append(sample_id)
            group.load += weights[sample_id] / group.size
            group.tokens += tokens[sample_id]
            column_tokens[group.microbatch] += tokens[sample_id]

        empty_count = sum(masks[minimum].bit_count() for masks in free)
        reserve = min(empty_count, len(short_ids))
        if self.options.pack_shorts_into_long_groups:
            for group in long_groups:
                budget = group.size * mean - weights[group.sample_ids[0]]
                remaining = []
                for sample_id in short_ids:
                    if (
                        len(short_ids) - (len(group.sample_ids) - 1) > reserve
                        and weights[sample_id] <= budget * 1.05
                        and can_take(group, sample_id)
                    ):
                        add(group, sample_id)
                        budget -= weights[sample_id]
                    else:
                        remaining.append(sample_id)
                short_ids = remaining
                group.load = sum(weights[i] for i in group.sample_ids) / group.size

        if minimum > 1:
            short_ids.sort(key=weights.__getitem__, reverse=True)
        empty_groups = [
            Group(c, start, minimum, [])
            for c in range(count)
            for start in range(0, world, minimum)
            if free[c][minimum] & (1 << (start // minimum))
        ]
        groups = long_groups + empty_groups
        creation = {id(g): index for index, g in enumerate(groups)}
        lpt_total = {id(g): 0.0 for g in empty_groups}
        for sample_id in short_ids:
            pool = empty_groups if self.options.pack_shorts_into_long_groups else groups
            eligible = [g for g in pool if can_take(g, sample_id)]
            if eligible:
                group = min(
                    eligible,
                    key=lambda g: (
                        lpt_total[id(g)] if self.options.pack_shorts_into_long_groups else g.load,
                        g.microbatch,
                        g.start,
                        creation[id(g)],
                    ),
                )
                add(group, sample_id)
                if self.options.pack_shorts_into_long_groups:
                    lpt_total[id(group)] += weights[sample_id]
            else:
                eligible = [g for g in long_groups if can_take(g, sample_id)]
                if not eligible:
                    raise InfeasibleLayout("group_capacity")
                group = min(eligible, key=lambda g: (g.load, creation[id(g)]))
                add(group, sample_id)

        for empty in sorted(
            (g for g in groups if not g.sample_ids), key=lambda g: (g.microbatch, g.start)
        ):
            donors = sorted(
                (g for g in groups if len(g.sample_ids) > 1),
                key=lambda g: (-g.load, g.microbatch, g.start, creation[id(g)]),
            )
            for donor in donors:
                sample_id = min(donor.sample_ids, key=weights.__getitem__)
                if (
                    not fits([sample_id], empty.size)
                    or donor.microbatch != empty.microbatch
                    and column_tokens[empty.microbatch] + tokens[sample_id] > capacity
                ):
                    continue
                donor.sample_ids.remove(sample_id)
                donor.load -= weights[sample_id] / donor.size
                donor.tokens -= tokens[sample_id]
                column_tokens[donor.microbatch] -= tokens[sample_id]
                add(empty, sample_id)
                break
            if not empty.sample_ids:
                raise InfeasibleLayout("coverage")

        rows = []
        for column in range(count):
            offset = (
                ((column * world // count) // self.sizes[-1]) * self.sizes[-1]
                if self.options.column_offset
                else 0
            )
            row = [g for g in groups if g.microbatch == column]
            for group in row:
                group.start = (group.start + offset) % world
            rows.append(sorted(row, key=lambda g: g.start))
        if self.options.heavy_middle and len(rows) > 1:
            smallest = min(g.size for row in rows for g in row)
            lightest = sorted(
                enumerate(rows),
                key=lambda entry: (
                    sum(g.size for g in entry[1] if g.size > smallest) / world,
                    max(g.load for g in entry[1]),
                    entry[0],
                ),
            )
            ordered = [None] * count
            for index, (_, row) in enumerate(lightest):
                ordered[index // 2 if index % 2 == 0 else count - 1 - index // 2] = row
            rows = ordered
        return rows

    def solve(
        self,
        tokens: Mapping[int, int],
        weights: Mapping[int, float],
        fits: Callable[[list[int], int], bool],
        objective: Callable[[list[list[Group]]], float],
    ) -> tuple[float, list[list[Group]]]:
        """Select the lowest-cost layout among the original bounded count candidates."""
        self.last_candidates = []
        required = self._requirements(tokens, fits)
        slots = sum(s for s in required.values() if s > self.sizes[0])
        lower = max(
            1,
            -(-sum(tokens.values()) // (self.world_size * self.tokens_per_rank)),
            -(-slots // self.world_size),
        )
        best = None
        for count in range(lower, min(lower + self.options.candidate_count, len(tokens) + 1)):
            try:
                layout = self.allocate(tokens, weights, count, fits)
            except InfeasibleLayout as error:
                self.last_candidates.append(dict(microbatches=count, rejection=str(error)))
                continue
            value = float(objective(layout))
            if not math.isfinite(value) or value < 0:
                raise ValueError("layout objective must be finite and nonnegative")
            self.last_candidates.append(dict(microbatches=count, predicted_ms=value))
            if best is None or value < best[0]:
                best = value, layout
        if best is None:
            raise InfeasibleLayout("no candidate count produced a complete feasible layout")
        return best
