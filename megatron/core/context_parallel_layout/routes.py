# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""THD context-parallel route helpers."""

import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from megatron.core.context_parallel_layout.types import CpPartitionMode, ThdCpRoute
from megatron.core.context_parallel_layout.utils import (
    get_packed_seq_params_cp_partition_cu_seqlens,
)
from megatron.core.utils import nvtx_range

if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams

# (global_start, length, local_start): a run of packed tokens that one rank holds locally.
_ThdLayoutSegment = Tuple[int, int, int]
# (source_local_start, target_local_start, length, global_start)
_ThdLayoutIntersection = Tuple[int, int, int, int]


def _materialize_thd_cu_seqlens_to_list(cu_seqlens: torch.Tensor) -> List[int]:
    if cu_seqlens.dim() != 1:
        raise ValueError(f"cu_seqlens must be 1-D, got shape {tuple(cu_seqlens.shape)}.")
    return cu_seqlens.detach().to(device="cpu", dtype=torch.long).tolist()


def _compact_thd_cu_seqlens_list(cu: List[int], source: torch.Tensor) -> List[int]:
    if not cu or cu[0] != 0:
        raise ValueError(f"cu_seqlens must start at 0, got {source}.")

    compact_cu: List[int] = [cu[0]]
    prev = cu[0]
    for value in cu[1:]:
        if value < prev:
            raise ValueError(f"cu_seqlens must be nondecreasing, got {source}.")
        if value != prev:
            compact_cu.append(value)
        prev = value
    return compact_cu


def _compact_thd_cu_seqlens_to_list(cu_seqlens: torch.Tensor) -> List[int]:
    return _compact_thd_cu_seqlens_list(_materialize_thd_cu_seqlens_to_list(cu_seqlens), cu_seqlens)


def _materialize_compact_thd_qkv_cu_seqlens(
    cu_q: torch.Tensor, cu_kv: Optional[torch.Tensor]
) -> Tuple[List[int], List[int]]:
    """Materialize compact Q/KV boundaries with one host transfer."""
    if cu_kv is None or cu_kv is cu_q:
        host_q = _compact_thd_cu_seqlens_to_list(cu_q)
        return host_q, host_q

    if cu_q.device != cu_kv.device:
        # Preserve the pre-existing mixed-device behavior. Joint materialization is
        # only possible for co-located tensors; otherwise copy each side separately.
        return _compact_thd_cu_seqlens_to_list(cu_q), _compact_thd_cu_seqlens_to_list(cu_kv)
    if cu_q.dim() != 1:
        raise ValueError(f"cu_seqlens must be 1-D, got shape {tuple(cu_q.shape)}.")
    if cu_kv.dim() != 1:
        raise ValueError(f"cu_seqlens must be 1-D, got shape {tuple(cu_kv.shape)}.")

    q_numel = cu_q.numel()
    joint_cu = torch.cat((cu_q.detach(), cu_kv.detach()))
    joint_host = _materialize_thd_cu_seqlens_to_list(joint_cu)
    host_q = _compact_thd_cu_seqlens_list(joint_host[:q_numel], cu_q)
    host_kv = _compact_thd_cu_seqlens_list(joint_host[q_numel:], cu_kv)
    return host_q, host_kv


def _validate_thd_route_partitioning(cu: List[int], cp_size: int) -> None:
    total_tokens = cu[-1]
    if total_tokens % cp_size != 0:
        raise ValueError(
            f"Contiguous CP partitioning requires total_tokens={total_tokens} "
            f"to be divisible by cp_size={cp_size}."
        )

    chunk_divisor = 2 * cp_size
    bad_seq_lens = [
        seq_end - seq_start
        for seq_start, seq_end in zip(cu[:-1], cu[1:])
        if (seq_end - seq_start) % chunk_divisor != 0
    ]
    if bad_seq_lens:
        raise ValueError(
            "All packed sequence lengths must be divisible by "
            f"2 * cp_size ({chunk_divisor}) for zigzag CP layout conversion, "
            f"got {bad_seq_lens}."
        )


def _build_thd_layout_segments(
    cu: List[int], cp_size: int, cp_rank: int, cp_partition_mode: CpPartitionMode
) -> Tuple[List[_ThdLayoutSegment], int]:
    total_tokens = cu[-1]
    if cp_partition_mode == "contiguous":
        part_len = total_tokens // cp_size
        if part_len == 0:
            return [], 0
        return [(cp_rank * part_len, part_len, 0)], part_len

    if cp_partition_mode != "zigzag":
        raise ValueError(
            f"Unsupported context-parallel partition mode {cp_partition_mode!r} "
            f"for THD layout segments with cp_size={cp_size}, rank={cp_rank}."
        )

    segments: List[_ThdLayoutSegment] = []
    local_start = 0
    for seq_start, seq_end in zip(cu[:-1], cu[1:]):
        seq_len = seq_end - seq_start
        chunk_len = seq_len // (2 * cp_size)
        first_chunk = cp_rank
        second_chunk = 2 * cp_size - cp_rank - 1
        segments.append((seq_start + first_chunk * chunk_len, chunk_len, local_start))
        segments.append((seq_start + second_chunk * chunk_len, chunk_len, local_start + chunk_len))
        local_start += 2 * chunk_len

    return segments, local_start


def _restrict_thd_layout_segments(
    segments: List[_ThdLayoutSegment], window_start: int, window_length: int
) -> List[_ThdLayoutSegment]:
    """Keep the part of a rank's layout that falls into one window of its local rows.

    The window is a contiguous range of the CP rank's local token order, i.e. one
    sequence-parallel shard. Local starts of the returned segments are relative to
    the window, so the result describes the rows the shard owner actually holds.
    """
    window_end = window_start + window_length
    restricted: List[_ThdLayoutSegment] = []
    for global_start, length, local_start in segments:
        local_end = local_start + length
        overlap_start = max(local_start, window_start)
        overlap_end = min(local_end, window_end)
        if overlap_start < overlap_end:
            restricted.append(
                (
                    global_start + overlap_start - local_start,
                    overlap_end - overlap_start,
                    overlap_start - window_start,
                )
            )
    return restricted


def _build_thd_tp_cp_layout_segments(
    cu: List[int], cp_size: int, tp_size: int, cp_partition_mode: CpPartitionMode
) -> Tuple[List[List[_ThdLayoutSegment]], int]:
    """Return the local segments of every (cp_rank, tp_rank) sequence-parallel shard.

    The result is indexed by the logical rank ``cp_rank * tp_size + tp_rank`` and also
    returns the shard length shared by all ranks. Sequence parallelism splits each CP
    rank's local packed sequence into ``tp_size`` equal contiguous windows.
    """
    segments_by_logical_rank: List[List[_ThdLayoutSegment]] = []
    shard_length: Optional[int] = None
    for cp_rank in range(cp_size):
        segments, local_length = _build_thd_layout_segments(cu, cp_size, cp_rank, cp_partition_mode)
        if local_length % tp_size != 0:
            raise ValueError(
                "Sequence-parallel THD CP layout conversion requires the CP rank-local token "
                f"count ({local_length}) to be divisible by tp_size={tp_size} "
                f"(cp_size={cp_size}, layout={cp_partition_mode!r})."
            )
        rank_shard_length = local_length // tp_size
        if shard_length is None:
            shard_length = rank_shard_length
        elif shard_length != rank_shard_length:
            raise ValueError(
                "THD CP layouts must give every CP rank the same local token count, got "
                f"{shard_length * tp_size} and {local_length} for layout {cp_partition_mode!r}."
            )
        for tp_rank in range(tp_size):
            segments_by_logical_rank.append(
                _restrict_thd_layout_segments(
                    segments, tp_rank * rank_shard_length, rank_shard_length
                )
            )
    return segments_by_logical_rank, shard_length or 0


def _intersect_thd_layout_segments(
    source_segments: List[_ThdLayoutSegment], target_segments: List[_ThdLayoutSegment]
) -> List[_ThdLayoutIntersection]:
    intersections: List[_ThdLayoutIntersection] = []
    source_index = 0
    target_index = 0
    while source_index < len(source_segments) and target_index < len(target_segments):
        source_global_start, source_len, source_local_start = source_segments[source_index]
        target_global_start, target_len, target_local_start = target_segments[target_index]
        source_global_end = source_global_start + source_len
        target_global_end = target_global_start + target_len

        overlap_start = max(source_global_start, target_global_start)
        overlap_end = min(source_global_end, target_global_end)
        if overlap_start < overlap_end:
            intersections.append(
                (
                    source_local_start + overlap_start - source_global_start,
                    target_local_start + overlap_start - target_global_start,
                    overlap_end - overlap_start,
                    overlap_start,
                )
            )

        if source_global_end <= target_global_end:
            source_index += 1
        else:
            target_index += 1

    return intersections


def _build_thd_layout_side_route(
    local_segments: List[_ThdLayoutSegment],
    target_segments_by_rank: List[List[_ThdLayoutSegment]],
    *,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], List[int]]:
    """Order one rank's local rows by communication peer.

    ``target_segments_by_rank`` must be ordered by the communication group's rank order.
    Within one peer, rows are ordered by global token position; both sides of an exchange
    derive their order from the same intersections, so the k-th row sent to a peer is the
    k-th row that peer expects from us. For the zigzag and contiguous layouts every rank's
    local order is already monotonic in global position, so this equals the previous
    target-local ordering and the CP-only routes are unchanged.
    """
    row_order: List[int] = []
    split_sizes: List[int] = []
    for peer_segments in target_segments_by_rank:
        intersections = _intersect_thd_layout_segments(local_segments, peer_segments)
        intersections.sort(key=lambda item: item[3])
        split_size = 0
        for source_row, _, length, _ in intersections:
            row_order.extend(range(source_row, source_row + length))
            split_size += length
        split_sizes.append(split_size)

    if all(row == index for index, row in enumerate(row_order)):
        return None, split_sizes
    return torch.tensor(row_order, device=device, dtype=torch.long), split_sizes


def _build_thd_cp_partition_route_from_host(
    cu: List[int], cp_size: int, cp_rank: int, *, device: torch.device
) -> ThdCpRoute:
    """Build a THD CP route from already compact host boundaries."""
    _validate_thd_route_partitioning(cu, cp_size)

    zigzag_segments_by_rank: List[List[_ThdLayoutSegment]] = []
    zigzag_lengths: List[int] = []
    contiguous_segments_by_rank: List[List[_ThdLayoutSegment]] = []
    for rank in range(cp_size):
        zigzag_segments, zigzag_length = _build_thd_layout_segments(cu, cp_size, rank, "zigzag")
        contiguous_segments, contiguous_length = _build_thd_layout_segments(
            cu, cp_size, rank, "contiguous"
        )
        if zigzag_length != contiguous_length:
            raise ValueError(
                "THD CP layout conversion must preserve local token count, "
                f"got zigzag={zigzag_length}, contiguous={contiguous_length} "
                f"for cp_size={cp_size}, rank={rank}."
            )
        zigzag_segments_by_rank.append(zigzag_segments)
        zigzag_lengths.append(zigzag_length)
        contiguous_segments_by_rank.append(contiguous_segments)

    zigzag_index, zigzag_split_sizes = _build_thd_layout_side_route(
        zigzag_segments_by_rank[cp_rank], contiguous_segments_by_rank, device=device
    )
    contiguous_index, contiguous_split_sizes = _build_thd_layout_side_route(
        contiguous_segments_by_rank[cp_rank], zigzag_segments_by_rank, device=device
    )

    local_length = zigzag_lengths[cp_rank]
    if sum(zigzag_split_sizes) != local_length:
        raise ValueError(
            "Zigzag THD CP route split sizes do not match the local token count: "
            f"splits={zigzag_split_sizes}, local_length={local_length}."
        )
    if sum(contiguous_split_sizes) != local_length:
        raise ValueError(
            "Contiguous THD CP route split sizes do not match the local token count: "
            f"splits={contiguous_split_sizes}, local_length={local_length}."
        )

    return ThdCpRoute(
        zigzag_index=zigzag_index,
        zigzag_split_sizes=zigzag_split_sizes,
        contiguous_index=contiguous_index,
        contiguous_split_sizes=contiguous_split_sizes,
    )


def build_thd_cp_partition_route(
    cu_seqlens: torch.Tensor, cp_size: int, cp_rank: int, *, device: Optional[torch.device] = None
) -> ThdCpRoute:
    """Precompute the rank-local THD CP layout route for a microbatch.

    The route stores both zigzag and contiguous layout views and can be reused
    for either conversion direction over tensors with the same THD sequence
    axis in the same microbatch.
    """
    if cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {cp_size}.")
    if not 0 <= cp_rank < cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}.")
    if device is None:
        device = cu_seqlens.device

    with nvtx_range("cp_layout/thd/route"):
        cu = _compact_thd_cu_seqlens_to_list(cu_seqlens)
        return _build_thd_cp_partition_route_from_host(cu, cp_size, cp_rank, device=device)


def _validate_group_rank_by_logical_rank(
    group_rank_by_logical_rank: Tuple[int, ...], cp_size: int, tp_size: int
) -> None:
    group_size = cp_size * tp_size
    if sorted(group_rank_by_logical_rank) != list(range(group_size)):
        raise ValueError(
            "group_rank_by_logical_rank must be a permutation of the TP x CP group ranks "
            f"(group_size={group_size}), got {group_rank_by_logical_rank}."
        )


def _build_thd_tp_cp_partition_route_from_host(
    cu: List[int],
    cp_size: int,
    cp_rank: int,
    tp_size: int,
    tp_rank: int,
    group_rank_by_logical_rank: Tuple[int, ...],
    *,
    device: torch.device,
) -> ThdCpRoute:
    """Build the fused TP x CP THD route of one sequence-parallel shard from host boundaries.

    The shard held by ``(tp_rank, cp_rank)`` is exchanged directly with every other
    ``(tp_rank', cp_rank')`` shard of the TP x CP group, replacing the TP gather ->
    CP all-to-all -> TP scatter composition with a single all-to-all-v. The result is a
    plain :class:`ThdCpRoute` whose split sizes follow the TP x CP group's rank order.
    """
    _validate_thd_route_partitioning(cu, cp_size)
    _validate_group_rank_by_logical_rank(group_rank_by_logical_rank, cp_size, tp_size)

    zigzag_by_logical_rank, zigzag_shard_length = _build_thd_tp_cp_layout_segments(
        cu, cp_size, tp_size, "zigzag"
    )
    contiguous_by_logical_rank, contiguous_shard_length = _build_thd_tp_cp_layout_segments(
        cu, cp_size, tp_size, "contiguous"
    )
    if zigzag_shard_length != contiguous_shard_length:
        raise ValueError(
            "THD CP layout conversion must preserve the sequence-parallel shard length, got "
            f"zigzag={zigzag_shard_length}, contiguous={contiguous_shard_length} for "
            f"cp_size={cp_size}, tp_size={tp_size}."
        )

    group_size = cp_size * tp_size
    zigzag_by_group_rank: List[List[_ThdLayoutSegment]] = [[] for _ in range(group_size)]
    contiguous_by_group_rank: List[List[_ThdLayoutSegment]] = [[] for _ in range(group_size)]
    for logical_rank, group_rank in enumerate(group_rank_by_logical_rank):
        zigzag_by_group_rank[group_rank] = zigzag_by_logical_rank[logical_rank]
        contiguous_by_group_rank[group_rank] = contiguous_by_logical_rank[logical_rank]

    local_logical_rank = cp_rank * tp_size + tp_rank
    zigzag_index, zigzag_split_sizes = _build_thd_layout_side_route(
        zigzag_by_logical_rank[local_logical_rank], contiguous_by_group_rank, device=device
    )
    contiguous_index, contiguous_split_sizes = _build_thd_layout_side_route(
        contiguous_by_logical_rank[local_logical_rank], zigzag_by_group_rank, device=device
    )

    for name, split_sizes in (
        ("Zigzag", zigzag_split_sizes),
        ("Contiguous", contiguous_split_sizes),
    ):
        if sum(split_sizes) != zigzag_shard_length:
            raise ValueError(
                f"{name} THD TP x CP route split sizes do not match the sequence-parallel shard "
                f"length: splits={split_sizes}, shard_length={zigzag_shard_length}."
            )

    return ThdCpRoute(
        zigzag_index=zigzag_index,
        zigzag_split_sizes=zigzag_split_sizes,
        contiguous_index=contiguous_index,
        contiguous_split_sizes=contiguous_split_sizes,
    )


def build_thd_tp_cp_partition_route(
    cu_seqlens: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    tp_size: int,
    tp_rank: int,
    group_rank_by_logical_rank: Tuple[int, ...],
    *,
    device: Optional[torch.device] = None,
) -> ThdCpRoute:
    """Precompute the fused TP x CP THD route of this rank's sequence-parallel shard.

    ``group_rank_by_logical_rank[cp_rank * tp_size + tp_rank]`` is the rank of that
    coordinate inside the TP x CP communication group (see
    :func:`get_tp_cp_group_rank_by_logical_rank`).
    """
    if cp_size < 1 or tp_size < 1:
        raise ValueError(f"cp_size and tp_size must be >= 1, got {cp_size} and {tp_size}.")
    if not 0 <= cp_rank < cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}.")
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"tp_rank must be in [0, {tp_size}), got {tp_rank}.")
    if device is None:
        device = cu_seqlens.device

    with nvtx_range("cp_layout/thd/tp_cp_route"):
        cu = _compact_thd_cu_seqlens_to_list(cu_seqlens)
        return _build_thd_tp_cp_partition_route_from_host(
            cu, cp_size, cp_rank, tp_size, tp_rank, tuple(group_rank_by_logical_rank), device=device
        )


@lru_cache(maxsize=None)
def build_tp_cp_group_rank_by_logical_rank(
    cp_global_ranks: Tuple[int, ...],
    tp_global_ranks: Tuple[int, ...],
    tp_cp_global_ranks: Tuple[int, ...],
    current_global_rank: int,
) -> Tuple[int, ...]:
    """Map logical ``cp_rank * tp_size + tp_rank`` coordinates to TP x CP group ranks.

    The TP and CP groups of the calling rank span a Cartesian product inside the
    TP x CP group: the global rank at logical coordinate ``(cp_rank, tp_rank)`` is
    ``cp_global_ranks[cp_rank] + tp_global_ranks[tp_rank] - current_global_rank``.
    Raises ``RuntimeError`` when the supplied groups do not form that product
    (for example a dynamic CP sub-group paired with the static TP x CP group).
    """
    group_rank_by_global_rank = {
        global_rank: group_rank for group_rank, global_rank in enumerate(tp_cp_global_ranks)
    }
    if len(group_rank_by_global_rank) != len(cp_global_ranks) * len(tp_global_ranks):
        raise RuntimeError("TP and CP process groups do not form the expected Cartesian product")
    group_rank_by_logical_rank = []
    for cp_global_rank in cp_global_ranks:
        for tp_global_rank in tp_global_ranks:
            target_global_rank = cp_global_rank + tp_global_rank - current_global_rank
            if target_global_rank not in group_rank_by_global_rank:
                raise RuntimeError(
                    "TP and CP process groups do not form the expected Cartesian product"
                )
            group_rank_by_logical_rank.append(group_rank_by_global_rank[target_global_rank])
    if len(set(group_rank_by_logical_rank)) != len(group_rank_by_logical_rank):
        raise RuntimeError("TP and CP process groups do not form the expected Cartesian product")
    return tuple(group_rank_by_logical_rank)


def get_tp_cp_group_rank_by_logical_rank(
    cp_group: torch.distributed.ProcessGroup,
    tp_group: torch.distributed.ProcessGroup,
    tp_cp_group: torch.distributed.ProcessGroup,
) -> Tuple[int, ...]:
    """Return the logical-to-group rank mapping of the calling rank's TP x CP group."""
    return build_tp_cp_group_rank_by_logical_rank(
        cp_global_ranks=tuple(torch.distributed.get_process_group_ranks(cp_group)),
        tp_global_ranks=tuple(torch.distributed.get_process_group_ranks(tp_group)),
        tp_cp_global_ranks=tuple(torch.distributed.get_process_group_ranks(tp_cp_group)),
        current_global_rank=torch.distributed.get_rank(),
    )


def resolve_tp_cp_group_rank_by_logical_rank(
    cp_group: Optional[torch.distributed.ProcessGroup],
    tp_group: Optional[torch.distributed.ProcessGroup],
    tp_cp_group: Optional[torch.distributed.ProcessGroup],
) -> Optional[Tuple[int, ...]]:
    """Return the TP x CP rank mapping, or None when the groups cannot be fused.

    None means the caller has to fall back to composing TP and CP collectives: no
    TP x CP group was supplied, or the CP group is not the one the TP x CP group was
    built from (dynamic context parallelism swaps in a sub-group per microbatch).
    """
    if cp_group is None or tp_group is None or tp_cp_group is None:
        return None
    if tp_cp_group.size() != cp_group.size() * tp_group.size():
        return None
    try:
        return get_tp_cp_group_rank_by_logical_rank(cp_group, tp_group, tp_cp_group)
    except RuntimeError:
        return None


def _thd_route_host_cu_seqlens(packed_seq_params: "PackedSeqParams") -> Optional[List[int]]:
    """Return the compact host boundaries stored by a previous route prebuild, if any."""
    host_cu = getattr(packed_seq_params, "thd_cp_host_cu_seqlens_q", None)
    if host_cu is None:
        return None
    return list(host_cu)


def get_thd_cp_partition_route(
    packed_seq_params: Optional["PackedSeqParams"],
    source_partition_mode: CpPartitionMode,
    target_partition_mode: CpPartitionMode,
) -> Optional[ThdCpRoute]:
    """Return the precomputed THD CP partition route for one direction.

    The fallback below is intentionally only a compatibility path: it performs
    a blocking device-to-host copy while compacting ``cu_seqlens`` and mutates
    ``packed_seq_params`` by storing the resulting route. Production callers
    should prebuild routes when constructing the batch.
    """
    if source_partition_mode == target_partition_mode:
        return None
    if source_partition_mode not in ("zigzag", "contiguous") or target_partition_mode not in (
        "zigzag",
        "contiguous",
    ):
        raise ValueError(
            f"Unsupported CP partition mode conversion "
            f"{source_partition_mode!r} -> {target_partition_mode!r} for THD route."
        )
    if packed_seq_params is None or getattr(packed_seq_params, "qkv_format", None) != "thd":
        return None

    route = getattr(packed_seq_params, "cp_partition_route", None)
    if route is not None:
        return route

    warnings.warn(
        "THD PackedSeqParams is missing precomputed context-parallel layout routes. "
        "This lookup will attempt to build them from packed_seq_params.cp_group as "
        "a compatibility fallback. The fallback synchronizes cu_seqlens to CPU "
        "and mutates packed_seq_params.cp_partition_route, so it should not be "
        "used on the steady-state forward path. Callers should prebuild THD CP "
        "routes when constructing the batch; a future release will require the "
        "routes to be present before layout conversion.",
        FutureWarning,
        stacklevel=2,
    )
    prebuild_thd_cp_partition_routes(packed_seq_params)
    return getattr(packed_seq_params, "cp_partition_route", None)


def get_thd_tp_cp_partition_route(
    packed_seq_params: Optional["PackedSeqParams"],
    source_partition_mode: CpPartitionMode,
    target_partition_mode: CpPartitionMode,
    *,
    cp_group: Optional[torch.distributed.ProcessGroup],
    tp_group: Optional[torch.distributed.ProcessGroup],
    tp_cp_group: Optional[torch.distributed.ProcessGroup],
) -> Optional[ThdCpRoute]:
    """Return the fused TP x CP route of this rank's sequence-parallel THD shard.

    Like :func:`get_thd_cp_partition_route`, this returns the route stored on
    ``packed_seq_params`` (``tp_cp_partition_route``) when present and otherwise
    builds and caches one. Building reuses the compact host boundaries left by the
    CP route prebuild when present, so it does not add a device-to-host copy in that
    case. Returns None when no conversion is needed, the input is not THD, or the
    groups cannot be fused (see :func:`resolve_tp_cp_group_rank_by_logical_rank`),
    in which case the caller falls back to the composed CP-only conversion.
    """
    if source_partition_mode == target_partition_mode:
        return None
    if packed_seq_params is None or getattr(packed_seq_params, "qkv_format", None) != "thd":
        return None
    if tp_group is None or tp_group.size() <= 1 or cp_group is None or cp_group.size() <= 1:
        return None
    group_rank_by_logical_rank = resolve_tp_cp_group_rank_by_logical_rank(
        cp_group, tp_group, tp_cp_group
    )
    if group_rank_by_logical_rank is None:
        return None

    route = getattr(packed_seq_params, "tp_cp_partition_route", None)
    if route is not None:
        return route

    cu_q = get_packed_seq_params_cp_partition_cu_seqlens(packed_seq_params)
    if cu_q is None:
        return None
    host_cu = _thd_route_host_cu_seqlens(packed_seq_params)
    if host_cu is None:
        warnings.warn(
            "THD PackedSeqParams is missing the precomputed TP x CP layout route and the "
            "host cu_seqlens of a CP route prebuild. Building the route here synchronizes "
            "cu_seqlens to CPU; prebuild routes when constructing the batch "
            "(prebuild_thd_cp_partition_routes with tp_group and tp_cp_group).",
            FutureWarning,
            stacklevel=2,
        )
        host_cu = _compact_thd_cu_seqlens_to_list(cu_q)
    with nvtx_range("cp_layout/thd/tp_cp_route"):
        route = _build_thd_tp_cp_partition_route_from_host(
            host_cu,
            cp_group.size(),
            cp_group.rank(),
            tp_group.size(),
            tp_group.rank(),
            group_rank_by_logical_rank,
            device=cu_q.device,
        )
    packed_seq_params.tp_cp_partition_route = route
    return route


def prebuild_thd_cp_partition_routes(
    packed_seq_params: Optional["PackedSeqParams"],
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    *,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    tp_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    device: Optional[torch.device] = None,
) -> None:
    """Prebuild the THD CP layout route for a packed microbatch.

    When ``tp_group`` (with more than one rank) and ``tp_cp_group`` are supplied and
    form a Cartesian product with ``cp_group``, the fused TP x CP route used by
    sequence-parallel layout conversion is prebuilt as well, from the same host copy
    of ``cu_seqlens``.
    """
    if packed_seq_params is None or getattr(packed_seq_params, "qkv_format", None) != "thd":
        return
    if cp_group is None:
        cp_group = getattr(packed_seq_params, "cp_group", None)
    if cp_group is None or cp_group.size() <= 1:
        return
    cp_size = cp_group.size()
    cp_rank = cp_group.rank()
    if not 0 <= cp_rank < cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}.")
    cu_q = get_packed_seq_params_cp_partition_cu_seqlens(packed_seq_params)
    if cu_q is None:
        return
    if device is None:
        device = cu_q.device

    # Also expose the compacted cu_seqlens as host integer lists. Consumers that
    # derive per-token layout metadata from them (e.g. the DSA packed-CP position
    # builders) can then work entirely on the host instead of re-deriving the same
    # spans on the device, where the data-dependent shapes force a
    # device-to-host readback behind the whole queued iteration. Doing the copy
    # here is cheap for the same reason the route build is: at batch-construction
    # time the CUDA queue is still shallow.
    # getattr: the pre-existing contract of this function (see the unit tests) is
    # any object carrying the q-side fields, so the kv-side reads must not widen it.
    cu_kv_padded = getattr(packed_seq_params, "cu_seqlens_kv_padded", None)
    cu_kv = (
        cu_kv_padded
        if cu_kv_padded is not None
        else getattr(packed_seq_params, "cu_seqlens_kv", None)
    )
    with nvtx_range("cp_layout/thd/route"):
        host_q, host_kv = _materialize_compact_thd_qkv_cu_seqlens(cu_q, cu_kv)
        route = _build_thd_cp_partition_route_from_host(host_q, cp_size, cp_rank, device=device)

    packed_seq_params.cp_partition_route = route
    packed_seq_params.thd_cp_host_cu_seqlens_q = host_q
    packed_seq_params.thd_cp_host_cu_seqlens_kv = host_kv

    if tp_group is None or tp_group.size() <= 1:
        return
    group_rank_by_logical_rank = resolve_tp_cp_group_rank_by_logical_rank(
        cp_group, tp_group, tp_cp_group
    )
    if group_rank_by_logical_rank is None:
        return
    with nvtx_range("cp_layout/thd/tp_cp_route"):
        packed_seq_params.tp_cp_partition_route = _build_thd_tp_cp_partition_route_from_host(
            host_q,
            cp_size,
            cp_rank,
            tp_group.size(),
            tp_group.rank(),
            group_rank_by_logical_rank,
            device=device,
        )
