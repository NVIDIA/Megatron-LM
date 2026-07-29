# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Transfer planning in logical weight coordinates."""

import math
from itertools import product
from typing import NamedTuple

from .utils import ParameterMetadata, _get_rank_in_group


class _Segment(NamedTuple):
    """A contiguous local interval and its position in the logical global weight."""

    local_start: int
    global_start: int
    length: int


def _storage_interval(metadata: ParameterMetadata, dim: int) -> tuple[int, int, int]:
    """Return local storage bounds in the unpadded TP-local weight."""
    local_size = metadata.shape[dim]
    if not metadata.is_gtp or dim != 0:
        return 0, local_size, local_size

    group = metadata.gtp_remat_group_ranks
    if not group:
        raise RuntimeError(f"{metadata.name}: missing GTP rematerialization group")

    tp_local_size = local_size * len(group) - metadata.gtp_pad_length
    if tp_local_size <= 0:
        raise RuntimeError(
            f"{metadata.name}: invalid GTP padding ({metadata.gtp_pad_length}) "
            f"for dim 0 size {local_size} and group size {len(group)}"
        )

    start = _get_rank_in_group(metadata.owner_rank, group) * local_size
    return start, min(start + local_size, tp_local_size), tp_local_size


def _tp_segments(metadata: ParameterMetadata, dim: int, tp_local_size: int) -> list[_Segment]:
    """Map a TP-local dimension to the logical global weight."""
    if not metadata.is_tp or metadata.partition_dim != dim:
        return [_Segment(0, 0, tp_local_size)]

    group = metadata.tensor_parallel_group_ranks
    if not group:
        raise RuntimeError(f"{metadata.name}: missing tensor-parallel group")

    tp_rank = _get_rank_in_group(metadata.owner_rank, group)
    tp_size = len(group)

    if metadata.partition_sizes is not None:
        if sum(metadata.partition_sizes) != tp_local_size:
            raise RuntimeError(
                f"{metadata.name}: partition_sizes sum to {sum(metadata.partition_sizes)}, "
                f"expected TP-local size {tp_local_size}"
            )

        segments = []
        local_offset = 0
        global_offset = 0
        for block_size in metadata.partition_sizes:
            segments.append(
                _Segment(local_offset, global_offset + tp_rank * block_size, block_size)
            )
            local_offset += block_size
            global_offset += block_size * tp_size
        return segments

    stride = max(1, metadata.partition_stride)
    if tp_local_size % stride:
        raise RuntimeError(
            f"{metadata.name}: TP-local size {tp_local_size} is not divisible by "
            f"partition_stride={stride}"
        )

    segment_size = tp_local_size // stride
    global_block_size = segment_size * tp_size
    return [
        _Segment(
            stride_idx * segment_size,
            stride_idx * global_block_size + tp_rank * segment_size,
            segment_size,
        )
        for stride_idx in range(stride)
    ]


def _local_segments(metadata: ParameterMetadata, dim: int) -> list[_Segment]:
    """Map local storage to the unpadded logical global weight.

    GTP always takes a contiguous dim-0 slice of the TP-local layout. Intersecting
    that slice with the TP segments naturally handles column, row, strided, and
    packed tensor-parallel layouts.
    """
    storage_start, storage_stop, tp_local_size = _storage_interval(metadata, dim)
    segments = []
    for tp_segment in _tp_segments(metadata, dim, tp_local_size):
        tp_stop = tp_segment.local_start + tp_segment.length
        start = max(storage_start, tp_segment.local_start)
        stop = min(storage_stop, tp_stop)
        if start < stop:
            segments.append(
                _Segment(
                    start - storage_start,
                    tp_segment.global_start + start - tp_segment.local_start,
                    stop - start,
                )
            )
    return segments


def _global_shape(metadata: ParameterMetadata) -> tuple[int, ...]:
    """Return the unpadded shape after materializing TP and GTP."""
    shape = []
    for dim in range(len(metadata.shape)):
        _, _, tp_local_size = _storage_interval(metadata, dim)
        if metadata.is_tp and metadata.partition_dim == dim:
            group = metadata.tensor_parallel_group_ranks
            if not group:
                raise RuntimeError(f"{metadata.name}: missing tensor-parallel group")
            tp_local_size *= len(group)
        shape.append(tp_local_size)
    return tuple(shape)


def _source_shards(
    all_src_metadata: list[ParameterMetadata], selected: ParameterMetadata
) -> list[ParameterMetadata]:
    """Find the TP x GTP shard grid containing the selected source replica."""
    by_rank = {metadata.owner_rank: metadata for metadata in all_src_metadata}
    pending = [selected.owner_rank]
    ranks = {selected.owner_rank}

    while pending:
        metadata = by_rank[pending.pop()]
        groups = []
        if metadata.is_tp:
            groups.append(metadata.tensor_parallel_group_ranks)
        if metadata.is_gtp:
            groups.append(metadata.gtp_remat_group_ranks)
        for group in groups:
            for rank in group or ():
                if rank in by_rank and rank not in ranks:
                    ranks.add(rank)
                    pending.append(rank)

    return [by_rank[rank] for rank in sorted(ranks)]


def _overlap(
    src_segments: list[_Segment], dst_segments: list[_Segment]
) -> list[tuple[slice, slice]]:
    """Intersect two segment lists in logical global coordinates."""
    overlaps = []
    for src in src_segments:
        for dst in dst_segments:
            start = max(src.global_start, dst.global_start)
            stop = min(src.global_start + src.length, dst.global_start + dst.length)
            if start < stop:
                overlaps.append(
                    (
                        slice(
                            src.local_start + start - src.global_start,
                            src.local_start + stop - src.global_start,
                        ),
                        slice(
                            dst.local_start + start - dst.global_start,
                            dst.local_start + stop - dst.global_start,
                        ),
                    )
                )
    return overlaps


def _rectangles_overlap(left: tuple[slice, ...], right: tuple[slice, ...]) -> bool:
    """Return whether two slice rectangles cover any common element."""
    return all(
        max(left_part.start, right_part.start) < min(left_part.stop, right_part.stop)
        for left_part, right_part in zip(left, right)
    )


def plan_sharded_transfer(
    param_name: str,
    all_src_metadata: list[ParameterMetadata],
    selected_src: ParameterMetadata,
    dst_metadata: ParameterMetadata,
) -> list[tuple[int, tuple[slice, ...], tuple[slice, ...]]]:
    """Plan a transfer by intersecting source and destination logical shards."""
    src_shards = _source_shards(all_src_metadata, selected_src)
    dst_shape = _global_shape(dst_metadata)
    dst_segments = [_local_segments(dst_metadata, dim) for dim in range(len(dst_metadata.shape))]

    expected = math.prod(sum(segment.length for segment in segments) for segments in dst_segments)

    transferred = 0
    ops = []
    for src in src_shards:
        src_shape = _global_shape(src)
        if src_shape != dst_shape:
            raise RuntimeError(
                f"{param_name}: logical shape mismatch: source rank {src.owner_rank} "
                f"has {src_shape}, destination rank {dst_metadata.owner_rank} has {dst_shape}"
            )

        overlaps_by_dim = [
            _overlap(_local_segments(src, dim), dst_segments[dim])
            for dim in range(len(dst_metadata.shape))
        ]
        for rectangle in product(*overlaps_by_dim):
            src_slice = tuple(slices[0] for slices in rectangle)
            dst_slice = tuple(slices[1] for slices in rectangle)
            if any(_rectangles_overlap(dst_slice, op[2]) for op in ops):
                raise RuntimeError(f"{param_name}: overlapping destination coverage")
            transferred += math.prod(part.stop - part.start for part in dst_slice)
            ops.append((src.owner_rank, src_slice, dst_slice))

    if transferred != expected:
        raise RuntimeError(
            f"{param_name}: covered {transferred} of {expected} destination elements "
            f"from source ranks {[metadata.owner_rank for metadata in src_shards]}"
        )

    return sorted(ops, key=lambda op: tuple(part.start for part in op[2]))
