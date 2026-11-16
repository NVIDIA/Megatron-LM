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

"""Exact packing of contiguous tensors into equal block-atomic rank segments."""

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


def _pack_at_shard(
    lengths: tuple[int, ...],
    alignments: tuple[int, ...],
    ranks: int,
    shard: int,
    max_states: int | None,
) -> tuple[int, ...] | None:
    """Decide feasibility using the smallest attainable end offset for each subset."""
    count = len(lengths)
    states = 1 << count
    if max_states is not None and states > max_states:
        raise ValueError(
            f"Exact block-atomic packing needs {states:,} states for {count} tensors; "
            f"max_states={max_states:,}. Increase max_states explicitly if resources permit."
        )
    capacity = ranks * shard
    ends = [capacity + 1] * states
    last = [-1] * states
    ends[0] = 0
    for mask, end in enumerate(ends):
        if end > capacity:
            continue
        missing = (states - 1) ^ mask
        while missing:
            bit = missing & -missing
            missing ^= bit
            index = bit.bit_length() - 1
            start = _earliest_start(end, lengths[index], alignments[index], shard)
            if start is None:
                continue
            next_end = start + lengths[index]
            following = mask | bit
            if next_end < ends[following]:
                ends[following] = next_end
                last[following] = index
    if ends[-1] > capacity:
        return None
    offsets = [0] * count
    mask = states - 1
    while mask:
        index = last[mask]
        offsets[index] = ends[mask] - lengths[index]
        mask ^= 1 << index
    return tuple(offsets)


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
) -> tuple[tuple[int, ...], int]:
    """Return tensor offsets and the minimum feasible equal shard size.

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
      every physical tensor order is considered by the subset recurrence.
    * Candidate sizes are visited in increasing order, skipping only those
      ruled out by an individual tensor. Feasibility is not monotone in S,
      so binary search would be invalid. The first feasible size is optimal.

    A cheap feasible bound sometimes certifies optimality without subset search,
    even for large groups. Otherwise M tested sizes cost O(M * n * 2**n) time
    and O(2**n) space, in addition to polynomial preprocessing and output.
    M depends on normalized input magnitudes. ``max_states`` raises before a
    larger DP allocation; there is no heuristic fallback or runtime bound.
    """
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
    for shard in _candidate_shards(small_lengths, small_alignments, lower, upper):
        offsets = (
            upper_offsets
            if shard == upper
            else _pack_at_shard(small_lengths, small_alignments, ranks, shard, max_states)
        )
        if offsets is not None:
            result = [ranks * shard * unit] * len(lengths)
            for index, offset in zip(nonempty, offsets):
                result[index] = offset * unit
            return tuple(result), shard * unit
    raise AssertionError("Exact packing failed to recover its known feasible upper bound.")
