# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pure Triton histogram selection prototype; no inline assembly."""

import triton
import triton.language as tl


@triton.jit
def radix_topk(Scores, Lengths, Indices, WIDTH: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    """Select exact ordered float keys, resolving cutoff ties by position."""
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    length = tl.minimum(tl.load(Lengths + row), WIDTH)
    valid = offsets < length
    values = tl.load(Scores + row * WIDTH + offsets, mask=valid, other=float("-inf"))
    bits = values.to(tl.uint32, bitcast=True)
    ordered = bits ^ tl.where((bits & 0x80000000) != 0, 0xFFFFFFFF, 0x80000000).to(tl.uint32)
    active = valid
    remaining = tl.minimum(K, length)
    prefix = tl.full((), 0, tl.uint32)
    bins = tl.arange(0, 256)
    for shift in tl.static_range(24, -1, -8):
        digits = ((ordered >> shift) & 255).to(tl.int32)
        counts = tl.histogram(tl.where(active, digits, 256), 256)
        cumulative = tl.cumsum(counts, 0, reverse=True)
        crossing = tl.max(tl.where(cumulative >= remaining, bins, -1), 0)
        above = tl.sum(tl.where(bins > crossing, counts, 0), 0)
        remaining -= above
        active &= digits == crossing
        prefix |= crossing.to(tl.uint32) << shift
    greater = valid & (ordered > prefix)
    equal = valid & (ordered == prefix)
    count = tl.sum(greater.to(tl.int32), 0)
    equal_rank = tl.cumsum(equal.to(tl.int32), 0)
    selected = greater | (equal & (equal_rank <= tl.minimum(K, length) - count))
    rank = tl.cumsum(selected.to(tl.int32), 0) - 1
    slots = tl.arange(0, K)
    tl.store(Indices + row * K + slots, -1)
    tl.debug_barrier()
    tl.store(Indices + row * K + rank, offsets, mask=selected & (rank < K))


@triton.jit
def fallback_if_needed(
    Scores, Lengths, Indices, Flags, WIDTH: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr
):
    """Repair only rows rejected by the CuTe histogram fast path."""
    if tl.load(Flags + tl.program_id(0)):
        radix_topk(Scores, Lengths, Indices, WIDTH, K, BLOCK)
