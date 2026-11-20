# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Parameter layout dataclasses for optimizer-driven buffer layout.

These dataclasses describe how parameters are laid out in contiguous buffers.
Each distributed optimizer implementation (e.g., DistributedOptimizer) is
responsible for computing these layouts via a _compute_per_buffer_param_layout method,
applying its own padding, alignment, and bucket splitting rules. DDP and
buffers consume the resulting layouts without any optimizer-specific knowledge.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


def pad_to_divisor(value: int, divisor: int) -> int:
    """Round up ``value`` to the nearest multiple of ``divisor``."""
    return int(math.ceil(value / divisor) * divisor)


def pad_param_start(param_start_index: int) -> int:
    """Align parameter start index to a 64-element boundary."""
    return pad_to_divisor(param_start_index, 64)


def bucket_end_divisor(data_parallel_world_size: int, pad_for_high_nccl_busbw: bool) -> int:
    """Divisor used to pad bucket ends for DP-divisibility (and optional NCCL busbw)."""
    if pad_for_high_nccl_busbw:
        return math.lcm(data_parallel_world_size, 128, 2**16)
    return math.lcm(data_parallel_world_size, 128)


def pad_bucket_end(
    bucket_end_index: int, data_parallel_world_size: int, pad_for_high_nccl_busbw: bool
) -> int:
    """Pad bucket end for DP-divisibility (and optionally high NCCL bus bandwidth)."""
    return pad_to_divisor(
        bucket_end_index, bucket_end_divisor(data_parallel_world_size, pad_for_high_nccl_busbw)
    )


@dataclass(frozen=True)
class BufferKey:
    """Identifies a distinct parameter buffer.

    Each unique combination of these fields corresponds to a separate contiguous
    buffer in DDP. Parameters are grouped into buffers by these dimensions.

    Attributes:
        param_dtype: Storage dtype (torch.uint8 for FP8/NVFP4 parameters, else param.dtype).
        grad_dtype: Gradient reduction dtype.
        is_expert_parallel: Whether the buffer holds expert-parallel parameters,
            which use a separate data-parallel group.
        is_managed_by_layer_wise_optimizer: Whether parameters in this buffer are
            managed by :class:`LayerWiseDistributedOptimizer` (shard-aligned layout
            so each whole param lives in one shard). Non-LayerWise params get
            :class:`DistributedOptimizer`'s byte-level layout in a separate buffer.
    """

    param_dtype: torch.dtype
    grad_dtype: torch.dtype
    is_expert_parallel: bool
    is_managed_by_layer_wise_optimizer: bool = False


@dataclass
class PerBufferParamLayout:
    """Layout for parameters within a single contiguous buffer.

    Describes how parameters should be laid out in the contiguous buffer.

    Attributes:
        param_index_map: Mapping from parameter to (start_index, end_index, bucket_id) in buffer.
        bucket_indices: List of (start_index, end_index) for each bucket.
        per_bucket_numel_unpadded: Number of unpadded elements per bucket.
        param_indices: The index of each param among same-dtype params (using the "fake"
            high-precision dtype for FP8/NVFP4 params). Needed for loading non-native-fp8
            checkpoints in native-fp8 mode. Order matches param_index_map iteration order.
        num_optimizer_shards: Number of optimizer shards. Set by the distributed optimizer
            that computes the layout so that shard assignment at runtime uses the same
            value. ``None`` for non-distributed-optimizer layouts.
        param_to_owner_shard: Element-granular LayerWise layouts only: the shard (rank of the
            buffer's data-parallel group) that owns each param's optimizer state. Params may
            then cross shard boundaries, and after each bucket's reduce-scatter the pieces are
            routed to their owner (:func:`compute_owner_exchange_plan`). ``None`` for every
            other layout, where a LayerWise param's owner is the shard its range lies in.
    """

    param_index_map: Dict[torch.nn.Parameter, Tuple[int, int, int]] = field(default_factory=dict)
    bucket_indices: List[Tuple[int, int]] = field(default_factory=list)
    per_bucket_numel_unpadded: List[int] = field(default_factory=list)
    param_indices: List[int] = field(default_factory=list)
    num_optimizer_shards: Optional[int] = None
    param_to_owner_shard: Optional[Dict[torch.nn.Parameter, int]] = None


@dataclass(frozen=True)
class OwnerExchangePlan:
    """One rank's side of a bucket's exchange between DP slice holders and param owners.

    Entries are ``(start, end, peer)``: bucket-local element offsets and a rank of the bucket's
    data-parallel group. ``to_owners`` are ranges of this rank's slice that belong to params
    owned by ``peer``: stage A sends them (reduced grads), stage A^-1 receives them (updated
    params). ``from_slices`` are ranges of this rank's owned params that lie in ``peer``'s
    slice: stage A receives them, stage A^-1 sends them. Both are ordered by offset, the order
    in which the peer posts the matching side, as NCCL pairs sends and receives in order.
    """

    to_owners: Tuple[Tuple[int, int, int], ...] = ()
    from_slices: Tuple[Tuple[int, int, int], ...] = ()


def owner_exchange_runs(
    param_ranges: List[Tuple[int, int, int]], bucket_start: int, bucket_end: int, dp_size: int
) -> List[List[Tuple[int, int, int]]]:
    """Cut a bucket's params at its DP slice boundaries.

    Args:
        param_ranges: ``(start, end, owner)`` of every param in the bucket, in buffer offsets.
        bucket_start: Buffer offset of the bucket.
        bucket_end: Buffer end of the bucket (a multiple of ``dp_size`` elements past the start).
        dp_size: Size of the bucket's data-parallel group.

    Returns:
        Per slice, the runs ``(start, end, owner)`` (bucket-local offsets, ordered) of maximal
        sequences of consecutive pieces with the same owner. A run spans the alignment gaps
        between its pieces, which are buffer padding, so one send/recv covers it.
    """
    numel = bucket_end - bucket_start
    shard = numel // dp_size
    assert shard * dp_size == numel, f"bucket of {numel} elements is not divisible by {dp_size}"
    runs: List[List[Tuple[int, int, int]]] = [[] for _ in range(dp_size)]
    for start, end, owner in sorted(param_ranges):
        assert bucket_start <= start <= end <= bucket_end
        assert 0 <= owner < dp_size
        if end == start:
            continue
        first = (start - bucket_start) // shard
        last = (end - 1 - bucket_start) // shard
        for slice_rank in range(first, last + 1):
            lo = max(start, bucket_start + slice_rank * shard) - bucket_start
            hi = min(end, bucket_start + (slice_rank + 1) * shard) - bucket_start
            slice_runs = runs[slice_rank]
            if slice_runs and slice_runs[-1][2] == owner:
                assert slice_runs[-1][1] <= lo
                slice_runs[-1] = (slice_runs[-1][0], hi, owner)
            else:
                slice_runs.append((lo, hi, owner))
    return runs


def compute_owner_exchange_plan(
    param_ranges: List[Tuple[int, int, int]],
    bucket_start: int,
    bucket_end: int,
    dp_size: int,
    rank: int,
) -> OwnerExchangePlan:
    """This rank's :class:`OwnerExchangePlan` for one bucket (see :func:`owner_exchange_runs`).

    Every rank derives its plan from the same layout, so the sends of one rank and the receives
    of its peers match one to one and in the same order.
    """
    to_owners, from_slices = [], []
    for slice_rank, slice_runs in enumerate(
        owner_exchange_runs(param_ranges, bucket_start, bucket_end, dp_size)
    ):
        for lo, hi, owner in slice_runs:
            if owner == slice_rank:
                continue
            if slice_rank == rank:
                to_owners.append((lo, hi, owner))
            if owner == rank:
                from_slices.append((lo, hi, slice_rank))
    return OwnerExchangePlan(to_owners=tuple(to_owners), from_slices=tuple(from_slices))


@dataclass(frozen=True)
class OwnerExchangeSplits:
    """One rank's split sizes for a bucket's slice <-> owner exchange as one all-to-all.

    The bucket is owner-major, so it is a sequence of one contiguous segment per owner (see
    :func:`owner_exchange_segments`), and a rank's slice meets each segment in one contiguous
    range. Sizes are in elements and indexed by rank of the bucket's data-parallel group.

    Attributes:
        segment: Bucket-local ``(start, end)`` of this rank's owner segment.
        slice_to_owner_sizes: Length of ``slice(this rank) & segment(owner)`` per owner; they
            sum to the slice size and follow the slice's element order.
        owner_from_slice_sizes: Length of ``slice(rank) & segment(this rank)`` per slice rank;
            they sum to the segment size and follow the segment's element order.
        has_remote_pieces: Whether any rank of the group holds a piece owned by another rank.
            It is a property of the bucket, so every rank agrees on whether the collective is
            issued.
    """

    segment: Tuple[int, int] = (0, 0)
    slice_to_owner_sizes: Tuple[int, ...] = ()
    owner_from_slice_sizes: Tuple[int, ...] = ()
    has_remote_pieces: bool = False


def owner_exchange_segments(
    param_ranges: List[Tuple[int, int, int]], bucket_start: int, bucket_end: int, dp_size: int
) -> List[Tuple[int, int]]:
    """Cut an owner-major bucket into one contiguous segment per owner.

    Args:
        param_ranges: ``(start, end, owner)`` of every param in the bucket, in buffer offsets.
            Owners must not decrease along the buffer.
        bucket_start: Buffer offset of the bucket.
        bucket_end: Buffer end of the bucket.
        dp_size: Size of the bucket's data-parallel group.

    Returns:
        Per owner, the bucket-local ``(start, end)`` of its segment. The segments tile the
        bucket: a segment runs from the owner's first param to the next owner's first param,
        so it also holds the alignment gaps behind the owner's params, and the last owner's
        segment holds the padding at the bucket end. An owner without a param in the bucket
        has an empty segment.
    """
    numel = bucket_end - bucket_start
    # boundaries[o] = bucket-local offset of the first param owned by a rank >= o.
    boundaries = [numel] * (dp_size + 1)
    previous_owner, previous_end = 0, bucket_start
    for start, end, owner in sorted(param_ranges):
        assert bucket_start <= start <= end <= bucket_end
        assert 0 <= owner < dp_size
        assert previous_end <= start, "params of a bucket must not overlap"
        assert previous_owner <= owner, "an element-granular bucket must be owner-major"
        for boundary_owner in range(previous_owner + 1, owner + 1):
            boundaries[boundary_owner] = start - bucket_start
        previous_owner, previous_end = owner, end
    # The segments tile the bucket from its first element.
    boundaries[0] = 0
    return [(boundaries[owner], boundaries[owner + 1]) for owner in range(dp_size)]


def compute_owner_exchange_splits(
    param_ranges: List[Tuple[int, int, int]],
    bucket_start: int,
    bucket_end: int,
    dp_size: int,
    rank: int,
) -> OwnerExchangeSplits:
    """This rank's :class:`OwnerExchangeSplits` for one bucket.

    Every rank derives the same ``dp_size x dp_size`` matrix of piece lengths from the same
    layout, so the send sizes of one rank are the receive sizes of its peers.
    """
    numel = bucket_end - bucket_start
    shard = numel // dp_size
    assert shard * dp_size == numel, f"bucket of {numel} elements is not divisible by {dp_size}"
    segments = owner_exchange_segments(param_ranges, bucket_start, bucket_end, dp_size)

    def piece(slice_rank: int, owner: int) -> int:
        segment_start, segment_end = segments[owner]
        lo = max(segment_start, slice_rank * shard)
        hi = min(segment_end, (slice_rank + 1) * shard)
        return max(0, hi - lo)

    has_remote_pieces = any(
        piece(slice_rank, owner) > 0
        for slice_rank in range(dp_size)
        for owner in range(dp_size)
        if owner != slice_rank
    )
    return OwnerExchangeSplits(
        segment=segments[rank],
        slice_to_owner_sizes=tuple(piece(rank, owner) for owner in range(dp_size)),
        owner_from_slice_sizes=tuple(piece(slice_rank, rank) for slice_rank in range(dp_size)),
        has_remote_pieces=has_remote_pieces,
    )


@dataclass
class FullParamLayout:
    """Layout for all parameters across all buffer groups in a model chunk.

    Maps BufferKey to per-buffer PerBufferParamLayout objects. Each PerBufferParamLayout has its
    own independent index space since different buffer groups are physically
    separate buffers.

    Attributes:
        layouts: Mapping from BufferKey to PerBufferParamLayout.
    """

    layouts: Dict[BufferKey, PerBufferParamLayout] = field(default_factory=dict)
