# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""THD context-parallel route helpers."""

import warnings
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from megatron.core.context_parallel_layout.types import CpPartitionMode, ThdCpRoute
from megatron.core.context_parallel_layout.utils import (
    get_packed_seq_params_cp_partition_cu_seqlens,
)
from megatron.core.utils import nvtx_range

if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams

_ThdLayoutSegment = Tuple[int, int, int]


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


def _intersect_thd_layout_segments(
    source_segments: List[_ThdLayoutSegment], target_segments: List[_ThdLayoutSegment]
) -> List[Tuple[int, int, int]]:
    intersections: List[Tuple[int, int, int]] = []
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
    row_order: List[int] = []
    split_sizes: List[int] = []
    for peer_rank in range(len(target_segments_by_rank)):
        intersections = _intersect_thd_layout_segments(
            local_segments, target_segments_by_rank[peer_rank]
        )
        intersections.sort(key=lambda item: item[1])
        split_size = 0
        for source_row, _, length in intersections:
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


def prebuild_thd_cp_partition_routes(
    packed_seq_params: Optional["PackedSeqParams"],
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    *,
    device: Optional[torch.device] = None,
) -> None:
    """Prebuild the THD CP layout route for a packed microbatch."""
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
