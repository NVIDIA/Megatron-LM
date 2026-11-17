# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounded exact packing of contiguous tensors into equal block-atomic segments."""

import math
from collections.abc import Iterator


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _earliest_start(end: int, length: int, alignment: int, shard: int) -> int | None:
    """Find the earliest start at or after ``end`` that permits only whole-block cuts."""
    boundary = (end // shard + 1) * shard
    if shard % alignment == 0:
        if end + length <= boundary:
            return end
        return _round_up(end, alignment)

    # Two interior boundaries would differ by shard, forcing alignment to divide shard.
    # Otherwise at most floor(shard / alignment) blocks fit on either side of one cut.
    blocks, per_shard = length // alignment, shard // alignment
    if blocks > 2 * per_shard:
        return None
    if end + length <= boundary:
        return end
    low = max(1, blocks - per_shard)
    high = min(blocks - 1, per_shard, (boundary - end) // alignment)
    if high >= low:
        return boundary - high * alignment
    if length <= shard:
        return boundary
    return boundary + shard - per_shard * alignment


class _SearchLimit(Exception):
    """The deterministic budget cannot accommodate another candidate search."""


class _PackingSearch:
    """Count interchangeable tensors instead of distinguishing their permutations."""

    def __init__(
        self,
        lengths: tuple[int, ...],
        alignments: tuple[int, ...],
        max_states: int | None,
        max_work: int | None,
    ) -> None:
        grouped: dict[tuple[int, int], list[int]] = {}
        for index, pair in enumerate(zip(lengths, alignments)):
            grouped.setdefault(pair, []).append(index)
        self.types = tuple(grouped)
        self.ids = tuple(tuple(indices) for indices in grouped.values())
        self.strides: list[int] = []
        self.states = 1
        for indices in self.ids:
            self.strides.append(self.states)
            self.states *= len(indices) + 1
        self.count = len(lengths)
        self.max_states = max_states
        self.remaining_work = max_work

    def pack(self, ranks: int, shard: int) -> tuple[int, ...] | None:
        """Decide feasibility, or stop before allocation when the budget is exhausted."""
        if self.max_states is not None and self.states > self.max_states:
            raise _SearchLimit
        # Reserve a conservative whole-candidate bound: two table initializations,
        # one state scan, and at most k type checks per state. Charging up front
        # bounds cumulative work without rank-dependent clocks or partial layouts.
        work = self.states * (len(self.types) + 3)
        if self.remaining_work is not None:
            if work > self.remaining_work:
                raise _SearchLimit
            self.remaining_work -= work

        capacity = ranks * shard
        ends = [capacity + 1] * self.states
        last = [-1] * self.states
        ends[0] = 0
        for state, end in enumerate(ends):
            if end > capacity:
                continue
            for kind, ((length, alignment), indices, stride) in enumerate(
                zip(self.types, self.ids, self.strides)
            ):
                used = state // stride % (len(indices) + 1)
                if used == len(indices):
                    continue
                start = _earliest_start(end, length, alignment, shard)
                if start is None:
                    continue
                next_end = start + length
                following = state + stride
                if next_end < ends[following]:
                    ends[following] = next_end
                    last[following] = kind
        if ends[-1] > capacity:
            return None
        offsets = [0] * self.count
        state = self.states - 1
        while state:
            kind = last[state]
            indices, stride = self.ids[kind], self.strides[kind]
            used = state // stride % (len(indices) + 1)
            offsets[indices[used - 1]] = ends[state] - self.types[kind][0]
            state -= stride
        return tuple(offsets)


def _greedy_bound(
    lengths: tuple[int, ...], alignments: tuple[int, ...], ranks: int
) -> tuple[tuple[int, ...], int]:
    """Retain the original padded builder's regular/conjugate/fragment packing."""
    chunk = math.lcm(*alignments)
    regular = [index for index, length in enumerate(lengths) if length >= chunk]
    fragments = sorted(
        (index for index, length in enumerate(lengths) if length < chunk),
        key=lambda index: -lengths[index],
    )
    offsets, end = [0] * len(lengths), 0
    while regular:
        index = regular.pop(0)
        length = lengths[index]
        offsets[index] = end
        remainder = length % chunk
        if not remainder:
            end += length
            continue
        gap = end + length
        end += _round_up(length, chunk)
        gap_end = end
        for other in regular:
            other_remainder = lengths[other] % chunk
            if other_remainder and remainder + other_remainder <= chunk:
                regular.remove(other)
                offsets[other] = end - other_remainder
                gap_end = offsets[other]
                end += lengths[other] // chunk * chunk
                break
        for fragment in fragments[:]:
            start = _round_up(gap, alignments[fragment])
            if start + lengths[fragment] <= gap_end:
                offsets[fragment] = start
                gap = start + lengths[fragment]
                fragments.remove(fragment)
    for fragment in fragments:
        offsets[fragment] = _round_up(end, alignments[fragment])
        end = offsets[fragment] + lengths[fragment]
    return tuple(offsets), _round_up(end, ranks * chunk) // ranks


def _candidate_shards(
    lengths: tuple[int, ...], alignments: tuple[int, ...], lower: int, upper: int
) -> Iterator[int]:
    """Skip sizes that cannot hold an individual tensor without splitting a block."""
    thresholds = tuple(
        alignment * ((length // alignment + 1) // 2)
        for length, alignment in zip(lengths, alignments)
    )
    breaks = sorted({threshold for threshold in thresholds if lower < threshold <= upper})
    start = lower
    for stop in [*breaks, upper + 1]:
        # Below this threshold a tensor must cross two boundaries, so its alignment
        # must divide the shard size. This does not impose the LCM of every tensor.
        multiple = math.lcm(
            1,
            *(
                alignment
                for alignment, threshold in zip(alignments, thresholds)
                if threshold > start
            ),
        )
        yield from range(_round_up(start, multiple), stop, multiple)
        start = stop


def _aligned_bound(
    lengths: tuple[int, ...], alignments: tuple[int, ...], ranks: int
) -> tuple[tuple[int, ...], int]:
    """Get a feasible upper bound from five deterministic orders, without subset search."""
    ids = tuple(range(len(lengths)))
    orders = (
        ids,
        tuple(sorted(ids, key=lambda index: -lengths[index])),
        tuple(sorted(ids, key=lambda index: lengths[index])),
        tuple(sorted(ids, key=lambda index: (-alignments[index], -lengths[index]))),
        tuple(sorted(ids, key=lambda index: (alignments[index], -lengths[index]))),
    )
    best_offsets: tuple[int, ...] = ()
    best_end: int | None = None
    for order in orders:
        offsets, end = [0] * len(lengths), 0
        for index in order:
            offsets[index] = _round_up(end, alignments[index])
            end = offsets[index] + lengths[index]
        if best_end is None or end < best_end:
            best_offsets, best_end = tuple(offsets), end
    assert best_end is not None
    quantum = ranks * math.lcm(*alignments)
    return best_offsets, _round_up(best_end, quantum) // ranks


def pack_block_atomic(
    lengths: tuple[int, ...],
    alignments: tuple[int, ...],
    ranks: int,
    *,
    max_states: int | None = 1 << 20,
    max_work: int | None = 10_000_000,
) -> tuple[tuple[int, ...], int]:
    """Improve a greedy equal-shard layout by exact search within fixed budgets.

    Inputs come from the layout builder: nonnegative lengths divisible by their
    positive alignments, and a positive rank count. Tensor IDs are preserved;
    tensors remain contiguous, but their physical order may change. Only cuts
    strictly inside tensors must align relative to those tensors' starts.

    Exactness follows from three reductions:

    * Let g be the GCD of nonempty tensors' alignments. Each rank contains a
      multiple of g useful elements. If its size S is not divisible by g, no
      tensor can span an entire rank and continue across it. Shrink each rank
      to g * floor(S / g), keeping incoming pieces first, outgoing pieces last,
      and whole tensors in between. Pieces still meet at rank boundaries. The
      same within-rank packing aligns all starts when S is divisible by g.
      Hence an optimum exists on the g grid.
    * For a fixed S, the earliest legal placement of a tensor is monotone in
      the previous end offset. Only the smallest end for each subset matters;
      every physical type order is considered by the recurrence. Tensors with
      identical (length, alignment) are interchangeable, so a mixed-radix state
      records how many of each type have been placed. Original IDs are restored
      in stable order when reconstructing the offsets.
    * Candidate sizes are visited in increasing order, skipping only those
      ruled out by an individual tensor. Feasibility is not monotone in S,
      so binary search would be invalid. The first feasible size is optimal.

    For k types with multiplicities c_i, P = product(c_i + 1) states cost
    O(M * k * P) time and O(P) space over M candidates, plus preprocessing and
    output. Cheap bounds can certify optimality without DP. Otherwise the best
    of the original greedy layout and five simple orders is retained. If either
    budget would be exceeded, return that valid bound with no padding regression
    against the original builder; optimality is then not guaranteed. ``max_work``
    reserves P * (k + 3) units per candidate across the entire search. Limits are
    deterministic, not wall-clock timeouts. None removes the corresponding limit.
    """
    for name, limit in (("max_states", max_states), ("max_work", max_work)):
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError(f"{name} must be a nonnegative integer or None, got {limit!r}.")
    nonempty = tuple(index for index, length in enumerate(lengths) if length)
    if not nonempty:
        return (0,) * len(lengths), 0
    if ranks == 1:
        linear_offsets, end = [], 0
        for length in lengths:
            linear_offsets.append(end)
            end += length
        # Put zero-sized tensors at the end to keep GlobalLayout's overlap check valid.
        return (
            tuple(offset if length else end for offset, length in zip(linear_offsets, lengths)),
            end,
        )

    unit = math.gcd(*(alignments[index] for index in nonempty))
    small_lengths = tuple(lengths[index] // unit for index in nonempty)
    small_alignments = tuple(alignments[index] // unit for index in nonempty)
    upper_offsets, upper = _aligned_bound(small_lengths, small_alignments, ranks)
    lower = max(max(small_alignments), (sum(small_lengths) + ranks - 1) // ranks)
    if next(_candidate_shards(small_lengths, small_alignments, lower, upper)) < upper:
        # Include empty tensors' original alignments in this baseline to exactly
        # preserve the old greedy algorithm. They do not constrain exact search.
        greedy_offsets, greedy_shard = _greedy_bound(lengths, alignments, ranks)
        if greedy_shard < upper * unit:
            upper = greedy_shard // unit
            upper_offsets = tuple(greedy_offsets[index] // unit for index in nonempty)
    search = _PackingSearch(small_lengths, small_alignments, max_states, max_work)
    for shard in _candidate_shards(small_lengths, small_alignments, lower, upper):
        try:
            offsets = upper_offsets if shard == upper else search.pack(ranks, shard)
        except _SearchLimit:
            offsets, shard = upper_offsets, upper
        if offsets is not None:
            result = [ranks * shard * unit] * len(lengths)
            for index, offset in zip(nonempty, offsets):
                result[index] = offset * unit
            return tuple(result), shard * unit
    raise AssertionError("Exact packing failed to recover its known feasible upper bound.")
