# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Layout-aware validation of a PyTorch Distributed Checkpoint global save plan.

``torch.distributed.checkpoint.DefaultSavePlanner`` validates the merged global plan on the
coordinator rank with ``_validate_global_plan``. That function checks, for every tensor key, that
the chunks written by all ranks lie within the tensor, cover its volume exactly, and do not
overlap. The overlap check compares chunk pairs: older releases compare every pair, newer ones
sweep along the *largest* tensor dimension and compare each chunk with every chunk still active
on that dimension.

Megatron-Core lays MoE expert weights out with a leading expert axis of ``num_experts`` fragments
plus the (E)GTP / TP split of one feature axis. The largest dimension of such a tensor is a
feature dimension that is either not sharded at all (every chunk spans it, so nothing ever
leaves the sweep's active set) or split only (E)GTP ways. Either way the sweep keeps hundreds to
thousands of chunks active and the check degenerates to O(n^2) per key: with 512 experts and
EGTP 8 that is 4096 chunks, ~8.4M pair comparisons for a single key, and tens of minutes of
single-threaded Python for the whole model. Nothing waits on a collective with the coordinator
during this step, so the rest of the job idles until a fault-tolerance timeout kills it.

This module performs the same three checks and returns the same kind of verdict, but derives the
overlap algorithm from the chunk layout instead of the tensor shape:

1. Every dimension is *regular*, i.e. the distinct extents used along it are pairwise disjoint.
   Two chunks then overlap iff they have identical extents on every dimension, so overlap
   detection is a duplicate search: O(n).
2. Exactly one dimension is irregular (Megatron's flattened-range tensors: a regular grid of
   fragments, each fragment split into ranges along the last axis). Chunks in different grid
   cells cannot overlap; inside a cell a 1-D interval sweep finds every overlapping pair:
   O(n log n).
3. Anything else falls back to an exhaustive pairwise check for that key only. This keeps the
   verdict exact for arbitrary layouts; Megatron does not produce such layouts today.

Chunks with a zero-size dimension take the pairwise fallback as well, because torch treats a
degenerate box strictly inside another box as overlapping and the fast paths are written for
positive extents. Such chunks are removed before saving by ``filter_out_empty_flatten_tensor``.
"""

import math
from bisect import bisect_right, insort
from logging import getLogger
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
)
from torch.distributed.checkpoint.planner import SavePlan

logger = getLogger(__name__)

Extent = Tuple[int, int]  # (offset, size) along one dimension


def check_box_bounds(outer_box_size: Sequence[int], inner_box: ChunkStorageMetadata) -> bool:
    """True iff ``inner_box`` lies within a tensor of size ``outer_box_size``.

    Mirrors ``torch.distributed.checkpoint.default_planner._check_box_bounds``.
    """
    for i in range(len(outer_box_size)):
        if inner_box.offsets[i] < 0:
            return False
        if inner_box.sizes[i] < 0:
            return False
        if inner_box.offsets[i] + inner_box.sizes[i] > outer_box_size[i]:
            return False
    return True


def check_box_overlap(box0: ChunkStorageMetadata, box1: ChunkStorageMetadata) -> bool:
    """True iff the two boxes overlap.

    Mirrors ``torch.distributed.checkpoint.default_planner._check_box_overlap`` exactly,
    including its treatment of zero-size boxes, so the fallback path and torch agree.
    """
    for i in range(len(box0.offsets)):
        if box0.offsets[i] >= box1.offsets[i] + box1.sizes[i]:
            return False
        if box1.offsets[i] >= box0.offsets[i] + box0.sizes[i]:
            return False
    return True


def _pairwise_overlaps(
    chunks: Sequence[ChunkStorageMetadata],
) -> List[Tuple[ChunkStorageMetadata, ChunkStorageMetadata]]:
    """Exhaustive O(n^2) overlap search (torch's original algorithm)."""
    pairs = []
    for i, chunk0 in enumerate(chunks):
        for chunk1 in chunks[i + 1 :]:
            if check_box_overlap(chunk0, chunk1):
                pairs.append((chunk0, chunk1))
    return pairs


def _extents_disjoint(extents: Iterable[Extent]) -> bool:
    """True iff the given distinct positive-size extents are pairwise disjoint."""
    ordered = sorted(extents)
    for (off0, size0), (off1, _) in zip(ordered, ordered[1:]):
        if off1 < off0 + size0:
            return False
    return True


def find_overlapping_chunks(
    chunks: Sequence[ChunkStorageMetadata],
) -> List[Tuple[ChunkStorageMetadata, ChunkStorageMetadata]]:
    """Return every pair of overlapping chunks, choosing the algorithm from the layout.

    Args:
        chunks: chunks of one tensor key, as listed in ``TensorStorageMetadata.chunks``.

    Returns:
        List of overlapping ``(chunk_a, chunk_b)`` pairs; empty when the chunks are disjoint.
        The set of unordered pairs equals what torch's pairwise check reports.
    """
    if len(chunks) < 2:
        return []
    ndim = len(chunks[0].offsets)
    if ndim == 0:
        return []
    if any(size <= 0 for chunk in chunks for size in chunk.sizes):
        return _pairwise_overlaps(chunks)

    distinct_extents: List[Set[Extent]] = [set() for _ in range(ndim)]
    for chunk in chunks:
        for dim in range(ndim):
            distinct_extents[dim].add((chunk.offsets[dim], chunk.sizes[dim]))
    irregular_dims = [dim for dim in range(ndim) if not _extents_disjoint(distinct_extents[dim])]

    if not irregular_dims:
        # Regular grid on every dimension: overlap <=> identical extents on every dimension.
        seen: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], List[ChunkStorageMetadata]] = {}
        pairs: List[Tuple[ChunkStorageMetadata, ChunkStorageMetadata]] = []
        for chunk in chunks:
            cell = (tuple(chunk.offsets), tuple(chunk.sizes))
            previous = seen.setdefault(cell, [])
            pairs.extend((prev, chunk) for prev in previous)
            previous.append(chunk)
        return pairs

    if len(irregular_dims) == 1:
        # Regular grid on all other dimensions: only chunks in the same grid cell can overlap,
        # and inside a cell they overlap iff their extents on the irregular dimension do.
        irregular = irregular_dims[0]
        cells: Dict[Tuple[Extent, ...], List[ChunkStorageMetadata]] = {}
        for chunk in chunks:
            grid_cell = tuple(
                (chunk.offsets[dim], chunk.sizes[dim]) for dim in range(ndim) if dim != irregular
            )
            cells.setdefault(grid_cell, []).append(chunk)
        pairs = []
        for members in cells.values():
            if len(members) < 2:
                continue
            members.sort(key=lambda c: (c.offsets[irregular], c.sizes[irregular]))
            active: List[Tuple[int, int]] = []  # (end, index into members), sorted by end
            for idx, chunk in enumerate(members):
                start = chunk.offsets[irregular]
                end = start + chunk.sizes[irregular]
                cutoff = bisect_right(active, (start, len(members)))
                if cutoff:
                    del active[:cutoff]
                # Every remaining active chunk started at or before `start` and ends after it,
                # so it overlaps `chunk` on the irregular dimension and is identical elsewhere.
                pairs.extend((members[other], chunk) for _, other in active)
                insort(active, (end, idx))
        return pairs

    logger.debug(
        "global plan validation: %d chunks irregular on dims %s, using pairwise check",
        len(chunks),
        irregular_dims,
    )
    return _pairwise_overlaps(chunks)


def validate_global_plan(global_plan: List[SavePlan], metadata: Metadata) -> List[str]:
    """Validate the global save plan; return the list of problems found (empty if valid).

    Performs the three checks of torch's ``_validate_global_plan`` (chunk within bounds, no
    overlapping chunks, chunks cover the tensor volume exactly when more than one plan
    contributes) with layout-aware overlap detection. Messages follow torch's wording so
    downstream handling is unchanged.

    Args:
        global_plan: per-rank plans after deduplication.
        metadata: checkpoint metadata built from those plans.

    Returns:
        Error messages, one per violation. Empty when the plan is valid.
    """
    errors: List[str] = []
    for key, value in metadata.state_dict_metadata.items():
        if isinstance(value, BytesStorageMetadata):
            continue
        if len(value.size) == 0:
            continue
        chunks_volume = 0
        for chunk in value.chunks:
            if not check_box_bounds(value.size, chunk):
                msg = f"key:{key} has out of bounds chunk: tensor-size:{value.size} chunk: {chunk}"
                logger.warning(msg)
                errors.append(msg)
            chunks_volume += math.prod(chunk.sizes)

        for chunk0, chunk1 in find_overlapping_chunks(value.chunks):
            msg = f"key:{key} has overlapping chunks: {chunk0} {chunk1}"
            logger.warning(msg)
            errors.append(msg)

        tensor_volume = math.prod(value.size)
        if len(global_plan) > 1 and chunks_volume != tensor_volume:
            msg = (
                f"key:{key} invalid fill tensor-volume: "
                f"{tensor_volume} chunks-volume: {chunks_volume}"
            )
            logger.warning(msg)
            errors.append(msg)
    return errors
