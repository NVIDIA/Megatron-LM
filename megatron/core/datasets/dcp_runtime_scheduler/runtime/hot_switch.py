# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Portions adapted from DynamicUSP at commit
# 1ff97a4ae0beb5e088fadb9e112265bdd194b24d (src/hot_switch_solver: gbs/allocator.py,
# gbs/bounds.py, gbs/config.py, gbs/placement.py, gbs/search.py, models.py and validation.py),
# BSD-3-Clause license.
# See ../DYNAMIC_USP_NOTICE.txt for the upstream copyright and license text.

"""Communication-agnostic DynamicUSP global-batch allocation and count search.

This ports the ``solve_gbs`` Stage 1 policy: the allocation, which balances weight or,
with ``tier_time_factors``, predicted time, and the search over a bounded window of
microbatch counts, which are multiples of ``pp_multiple``. Its attention-specific profiles
are not ported; the optional fixed-topology Stage 2 is in ``stage2``. Inputs are tokens,
scheduling weights, allowed group widths, a capacity predicate and a complete-layout
objective.

Differences from upstream: the capacity predicate may limit the sequences of a group, the
count lower bound then also counts the sequences one microbatch can hold, the count search
goes on past its window, over a bounded number of counts, when every count of it is rejected
for capacity (see ``HotSwitchScheduler.solve``), and every count with a layout is scored with
the objective. ``../ORIGIN.json`` lists all differences.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


class InfeasibleLayout(ValueError):
    """A candidate microbatch count did not admit a complete nonempty layout."""


# The count search past a window of capacity rejections tries at most this many times
# ``candidate_count`` further counts to find one with a layout (``HotSwitchScheduler.solve``).
_WINDOWS_PAST_LIMIT = 4


def validate_tier_time_factors(factors) -> dict[int, float]:
    """Validated ``{group size: factor}`` copy of a time-balance table.

    Keys are positive integers or their decimal strings: a table read from a JSON file, or
    handed over as JSON, has string keys. A string key is at most nine ASCII digits without
    a leading zero, sign, space or underscore. Factors are finite numbers of at least one.
    """
    if not isinstance(factors, Mapping):
        raise ValueError("tier_time_factors must map group sizes to factors")
    table = {}
    for key, factor in factors.items():
        numeric = type(key) is int or type(key) is str and re.fullmatch("[1-9][0-9]{0,8}", key)
        size = int(key) if numeric else 0
        if size < 1 or size in table:
            raise ValueError("tier_time_factors keys must be distinct positive integer group sizes")
        try:
            valid = (
                not isinstance(factor, bool)
                and isinstance(factor, (int, float))
                and math.isfinite(factor)
                and factor >= 1
            )
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("tier_time_factors values must be finite numbers >= 1")
        table[size] = float(factor)
    return table


@dataclass(frozen=True)
class HotSwitchOptions:
    """Generic allocation policy; none of these options describes communication."""

    candidate_count: int = 4
    outlier_threshold: float = 1.0
    pack_shorts_into_long_groups: bool = True
    column_offset: bool = True
    tier_affinity: bool = True
    heavy_middle: bool = True
    # {group size: time per unit of per-rank weight, relative to the smallest size}. With it
    # Stage 1 balances predicted time instead of weight. A size without an entry counts as 1;
    # None, or a table of ones, keeps weight balance. An entry for a size the scheduler does
    # not form is not read.
    tier_time_factors: Mapping[int, float] | None = None
    # Also record the layout of every feasible candidate count in ``last_candidates``.
    keep_candidate_layouts: bool = False
    # Every candidate count is a multiple of this: the pipeline-parallel size, since the
    # interleaved pipeline schedule runs such counts only. One leaves the count free.
    pp_multiple: int = 1

    def __post_init__(self):
        if type(self.candidate_count) is not int or self.candidate_count < 1:
            raise ValueError("candidate_count must be a positive integer")
        if type(self.pp_multiple) is not int or self.pp_multiple < 1:
            raise ValueError("pp_multiple must be a positive integer")
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
            "keep_candidate_layouts",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.tier_time_factors is not None:
            table = validate_tier_time_factors(self.tier_time_factors)
            object.__setattr__(self, "tier_time_factors", table)


@dataclass
class Group:
    """One contiguous logical rank group in a candidate global-batch layout."""

    microbatch: int
    start: int
    size: int
    sample_ids: list[int]
    load: float = 0.0


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
        if not self.options.pack_shorts_into_long_groups and sizes[0] > 1:
            raise ValueError(
                "pack_shorts_into_long_groups=False is incompatible with a minimum group size "
                "greater than one"
            )
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
        needed = self._requirements(tokens, fits)
        table = self.options.tier_time_factors
        # Time balance is off without a table and for a table of ones on the sizes in use.
        timed = table is not None and any(table.get(size, 1.0) != 1.0 for size in self.sizes)

        def factor(size):
            return table.get(size, 1.0) if timed else 1.0

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
            if timed:
                # Smallest size whose predicted per-rank time fits the mean, else the widest.
                size = next(
                    (
                        s
                        for s in self.sizes
                        if s >= needed[sample_id] and weights[sample_id] * factor(s) / s <= mean
                    ),
                    self.sizes[-1],
                )
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
            column_load[column] += weights[sample_id] * factor(size)
            column_tokens[column] += tokens[sample_id]
            hosts[column].add(size)
            long_groups.append(Group(column, start, size, [sample_id], weights[sample_id] / size))

        def can_take(group, sample_id):
            return column_tokens[group.microbatch] + tokens[sample_id] <= capacity and fits(
                group.sample_ids + [sample_id], group.size
            )

        def add(group, sample_id):
            group.sample_ids.append(sample_id)
            group.load += weights[sample_id] / group.size
            column_tokens[group.microbatch] += tokens[sample_id]

        empty_count = sum(masks[minimum].bit_count() for masks in free)
        reserve = min(empty_count, len(short_ids))
        if self.options.pack_shorts_into_long_groups:
            for group in long_groups:
                budget = group.size * mean / factor(group.size) - weights[group.sample_ids[0]]
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
                group = min(eligible, key=lambda g: (g.load * factor(g.size), creation[id(g)]))
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
        *,
        sequences_per_microbatch: int | None = None,
    ) -> tuple[float, list[list[Group]]]:
        """Select the lowest-cost layout among the bounded count candidates.

        ``candidate_count`` counts are laid out from the lower bound on, every one a multiple
        of ``pp_multiple``, as upstream does. Upstream has no limit on the sequences of a
        group; two rules of this port keep the window on counts that can hold the samples
        (in full: ORIGIN.json, the adaptations of dynamic_usp):

        - ``sequences_per_microbatch``, the most sequences one microbatch can hold, adds
          ``ceil(samples / sequences_per_microbatch)`` to the lower bound: a necessary
          condition, so no feasible count is skipped.
        - When capacity rejects every count of the window, the search goes on past it, over
          at most ``4 * candidate_count`` counts, to the first count with a layout and lays
          out a window from that one. Upstream raises an error.
        """
        self.last_candidates = []
        required = self._requirements(tokens, fits)
        slots = sum(s for s in required.values() if s > self.sizes[0])
        step = self.options.pp_multiple

        def multiple(count):
            # ``count`` rounded up to a multiple of ``pp_multiple``.
            return -(-count // step) * step

        lower = multiple(
            max(
                1,
                -(-sum(tokens.values()) // (self.world_size * self.tokens_per_rank)),
                -(-slots // self.world_size),
            )
        )
        # A window spans ``candidate_count`` multiples of ``pp_multiple``.
        samples, window = len(tokens), self.options.candidate_count * step
        window_end = lower + window  # upstream's window is [lower, window_end)
        if sequences_per_microbatch is not None:
            if type(sequences_per_microbatch) is not int:
                raise ValueError("sequences_per_microbatch must be an integer")
            if sequences_per_microbatch < 1:
                raise InfeasibleLayout("no microbatch can hold a sequence")
            lower = max(lower, multiple(-(-samples // sequences_per_microbatch)))
            if lower >= window_end:
                window_end = lower + window
        best = None

        def counts(first, span):
            # The multiples from ``first``, itself one, below ``first + span``; none above
            # the number of samples.
            return range(first, min(first + span, samples + 1), step)

        def lay_out(count):
            """Lay out and score one count; True when it has a layout."""
            nonlocal best
            try:
                layout = self.allocate(tokens, weights, count, fits)
            except InfeasibleLayout as error:
                self.last_candidates.append(dict(microbatches=count, rejection=str(error)))
                return False
            value = float(objective(layout))
            record = dict(microbatches=count, predicted_ms=value)
            if self.options.keep_candidate_layouts:
                record["layout"] = [
                    [(g.start, g.size, list(g.sample_ids)) for g in row] for row in layout
                ]
            self.last_candidates.append(record)
            if best is None or value < best[0]:
                best = value, layout
            return True

        for count in counts(lower, window_end - lower):
            lay_out(count)
        if all(
            candidate.get("rejection") == "group_capacity" for candidate in self.last_candidates
        ):
            # A window cut short by the number of samples leaves no count to search.
            for count in counts(window_end, _WINDOWS_PAST_LIMIT * window):
                if lay_out(count):
                    for later in counts(count + step, window - step):
                        lay_out(later)
                    break
        if best is None:
            raise InfeasibleLayout("no candidate count produced a complete feasible layout")
        return best
