# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""THD context-parallel partition indices and route tensor helpers."""

import warnings
from contextlib import contextmanager
from typing import Any, List, Optional, Tuple

import torch

from megatron.core.context_parallel_layout.metadata import (
    get_packed_seq_params_cp_partition_cu_seqlens,
)
from megatron.core.context_parallel_layout import CpPartitionMode


_THD_CP_ROUTE_HEADER_SIZE = 6
_THD_CP_ROUTE_ATTRS = {
    ("zigzag", "contiguous"): "cp_partition_route_zigzag_to_contiguous",
    ("contiguous", "zigzag"): "cp_partition_route_contiguous_to_zigzag",
}


@contextmanager
def _cp_layout_nvtx_range(message: str):
    active = torch.cuda.is_available()
    if active:
        torch.cuda.nvtx.range_push(message)
    try:
        yield
    finally:
        if active:
            torch.cuda.nvtx.range_pop()


def get_thd_context_parallel_rank_indices(
    cu_seqlens: torch.Tensor, cp_size: int, cp_rank: int, cp_partition_mode: str
) -> torch.Tensor:
    """Return global THD token indices owned by one CP rank in a layout.

    Args:
        cu_seqlens: Global packed-sequence cumulative lengths before CP partitioning.
        cp_size: Context-parallel group size.
        cp_rank: Context-parallel rank.
        cp_partition_mode: Either ``"zigzag"`` or ``"contiguous"``.

    The returned indices are ordered exactly as the rank-local THD tensor is stored.
    ``"zigzag"`` follows Megatron's per-sequence load-balanced chunk order; ``"contiguous"``
    partitions the flattened packed THD buffer into rank-contiguous spans.
    """
    if cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {cp_size}.")
    if not 0 <= cp_rank < cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}.")
    if cu_seqlens.dim() != 1:
        raise ValueError(f"cu_seqlens must be 1-D, got shape {tuple(cu_seqlens.shape)}.")

    cu = cu_seqlens.to(dtype=torch.long)
    if cu.numel() == 0 or cu[0].item() != 0:
        raise ValueError(f"cu_seqlens must start at 0, got {cu_seqlens}.")

    if torch.any(torch.diff(cu) < 0):
        raise ValueError(f"cu_seqlens must be nondecreasing, got {cu_seqlens}.")

    nonduplicate_boundaries = torch.ones(cu.numel(), device=cu.device, dtype=torch.bool)
    nonduplicate_boundaries[1:] = cu[1:] != cu[:-1]
    cu = cu[nonduplicate_boundaries]

    total_tokens = int(cu[-1].item())
    if cp_partition_mode == "contiguous":
        if total_tokens % cp_size != 0:
            raise ValueError(
                f"Contiguous CP partitioning requires total_tokens={total_tokens} "
                f"to be divisible by cp_size={cp_size}."
            )
        part_len = total_tokens // cp_size
        rank_start = cp_rank * part_len
        return torch.arange(rank_start, rank_start + part_len, device=cu.device, dtype=torch.long)
    if cp_partition_mode != "zigzag":
        raise ValueError(
            f"Unsupported context-parallel partition mode {cp_partition_mode!r} "
            f"for THD rank indices with cp_size={cp_size}, cp_rank={cp_rank}, "
            f"cu_seqlens_shape={tuple(cu_seqlens.shape)}."
        )

    positions = torch.arange(total_tokens, device=cu.device, dtype=torch.long)
    if total_tokens == 0:
        return positions

    seq_lens = torch.diff(cu)

    chunk_divisor = 2 * cp_size
    if torch.any(seq_lens % chunk_divisor != 0):
        raise ValueError(
            "All packed sequence lengths must be divisible by "
            f"2 * cp_size ({chunk_divisor}) for zigzag CP layout conversion, "
            f"got {seq_lens}."
        )

    seq_idx = torch.bucketize(positions, cu[1:], right=True)
    global_starts = cu[:-1]
    pos_in_seq = positions - global_starts[seq_idx]
    chunk_lens = (seq_lens // chunk_divisor)[seq_idx]
    chunk = pos_in_seq // chunk_lens
    offset = pos_in_seq - chunk * chunk_lens

    owner = torch.where(chunk < cp_size, chunk, 2 * cp_size - chunk - 1)
    local_slot = torch.where(chunk < cp_size, torch.zeros_like(chunk), torch.ones_like(chunk))

    local_starts = (global_starts // cp_size)[seq_idx]
    local_pos = local_starts + local_slot * chunk_lens + offset

    rank_mask = owner == cp_rank
    rank_positions = positions[rank_mask]
    rank_local_pos = local_pos[rank_mask]
    return rank_positions[torch.argsort(rank_local_pos)]


_ThdLayoutSegment = Tuple[int, int, int]


def _compact_thd_cu_seqlens_to_list(cu_seqlens: torch.Tensor) -> List[int]:
    if cu_seqlens.dim() != 1:
        raise ValueError(f"cu_seqlens must be 1-D, got shape {tuple(cu_seqlens.shape)}.")

    cu = cu_seqlens.detach().to(device="cpu", dtype=torch.long).tolist()
    if not cu or cu[0] != 0:
        raise ValueError(f"cu_seqlens must start at 0, got {cu_seqlens}.")

    compact_cu: List[int] = [cu[0]]
    prev = cu[0]
    for value in cu[1:]:
        if value < prev:
            raise ValueError(f"cu_seqlens must be nondecreasing, got {cu_seqlens}.")
        if value != prev:
            compact_cu.append(value)
        prev = value
    return compact_cu


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


def _append_range(rows: List[int], start: int, length: int) -> None:
    rows.extend(range(start, start + length))


def _row_list_is_identity(rows: List[int]) -> bool:
    return all(row == index for index, row in enumerate(rows))


def _thd_cp_partition_route_attr_name(
    source_partition_mode: CpPartitionMode, target_partition_mode: CpPartitionMode
) -> str:
    try:
        return _THD_CP_ROUTE_ATTRS[(source_partition_mode, target_partition_mode)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported CP partition mode conversion "
            f"{source_partition_mode!r} -> {target_partition_mode!r} for THD route."
        ) from exc


def _encode_thd_cp_partition_route(
    *,
    cp_size: int,
    cp_rank: int,
    local_source_length: int,
    local_target_length: int,
    send_rows_list: List[int],
    recv_rows_list: List[int],
    input_split_sizes: List[int],
    output_split_sizes: List[int],
    device: torch.device,
) -> torch.Tensor:
    send_rows_payload = [] if _row_list_is_identity(send_rows_list) else send_rows_list
    recv_rows_payload = [] if _row_list_is_identity(recv_rows_list) else recv_rows_list
    payload = (
        [
            cp_size,
            cp_rank,
            local_source_length,
            local_target_length,
            len(send_rows_payload),
            len(recv_rows_payload),
        ]
        + input_split_sizes
        + output_split_sizes
        + send_rows_payload
        + recv_rows_payload
    )
    return torch.tensor(payload, device=device, dtype=torch.long)


def _split_sizes_from_route_tensor(
    route_tensor: torch.Tensor, start: int, end: int
) -> List[int]:
    return route_tensor[start:end].detach().to(device="cpu", dtype=torch.long).tolist()


def decode_thd_cp_partition_route(
    route_tensor: torch.Tensor, cp_size: int, cp_rank: int
) -> Tuple[int, int, Optional[torch.Tensor], Optional[torch.Tensor], List[int], List[int]]:
    """Decode a THD CP route tensor into local conversion metadata.

    The tensor layout is:
    ``[cp_size, cp_rank, local_source_len, local_target_len, send_rows_len,
    recv_rows_len, input_splits..., output_splits..., send_rows..., recv_rows...]``.
    Empty send/recv row payloads denote identity row order.
    """
    if route_tensor is None:
        raise ValueError("THD CP partition route tensor must not be None.")
    if route_tensor.dim() != 1:
        raise ValueError(
            f"THD CP partition route tensor must be 1-D, got shape {tuple(route_tensor.shape)}."
        )
    if route_tensor.numel() < _THD_CP_ROUTE_HEADER_SIZE:
        raise ValueError(
            f"THD CP partition route tensor is too short: {route_tensor.numel()} values."
        )

    (
        route_cp_size,
        route_cp_rank,
        local_source_length,
        local_target_length,
        send_rows_len,
        recv_rows_len,
    ) = route_tensor[:_THD_CP_ROUTE_HEADER_SIZE].detach().cpu().tolist()
    route_cp_size = int(route_cp_size)
    route_cp_rank = int(route_cp_rank)
    local_source_length = int(local_source_length)
    local_target_length = int(local_target_length)
    send_rows_len = int(send_rows_len)
    recv_rows_len = int(recv_rows_len)
    if route_cp_size != cp_size or route_cp_rank != cp_rank:
        raise ValueError(
            "THD CP partition route tensor does not match the requested CP rank/size: "
            f"route cp_size={route_cp_size}, cp_rank={route_cp_rank}; "
            f"requested cp_size={cp_size}, cp_rank={cp_rank}."
        )
    split_start = _THD_CP_ROUTE_HEADER_SIZE
    input_split_start = split_start
    output_split_start = input_split_start + cp_size
    send_rows_start = output_split_start + cp_size
    recv_rows_start = send_rows_start + send_rows_len
    expected_numel = recv_rows_start + recv_rows_len
    if route_tensor.numel() != expected_numel:
        raise ValueError(
            "THD CP partition route tensor has inconsistent length: "
            f"got {route_tensor.numel()}, expected {expected_numel}."
        )

    input_split_sizes = _split_sizes_from_route_tensor(
        route_tensor, input_split_start, output_split_start
    )
    output_split_sizes = _split_sizes_from_route_tensor(
        route_tensor, output_split_start, send_rows_start
    )
    send_rows = None if send_rows_len == 0 else route_tensor[send_rows_start:recv_rows_start]
    recv_rows = None if recv_rows_len == 0 else route_tensor[recv_rows_start:expected_numel]
    return (
        local_source_length,
        local_target_length,
        send_rows,
        recv_rows,
        input_split_sizes,
        output_split_sizes,
    )


def build_thd_cp_partition_route(
    cu_seqlens: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    source_partition_mode: CpPartitionMode,
    target_partition_mode: CpPartitionMode,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Precompute one THD CP layout conversion route as a tensor.

    The route depends only on packed sequence metadata, CP rank/size, and the
    source/target partition modes.  It can be reused for every tensor that has
    the same THD sequence axis in the same microbatch.
    """
    if source_partition_mode not in ("zigzag", "contiguous") or target_partition_mode not in (
        "zigzag",
        "contiguous",
    ):
        raise ValueError(
            f"Unsupported CP partition mode conversion "
            f"{source_partition_mode!r} -> {target_partition_mode!r} for THD route: "
            f"cp_size={cp_size}, cp_rank={cp_rank}, cu_seqlens_shape={tuple(cu_seqlens.shape)}."
        )
    if source_partition_mode == target_partition_mode:
        raise ValueError("A THD CP partition route is only needed when partition modes differ.")
    _thd_cp_partition_route_attr_name(source_partition_mode, target_partition_mode)
    if device is None:
        device = cu_seqlens.device

    with _cp_layout_nvtx_range(
        f"cp_layout/thd/route/{source_partition_mode}_to_{target_partition_mode}"
    ):
        cu = _compact_thd_cu_seqlens_to_list(cu_seqlens)
        _validate_thd_route_partitioning(cu, cp_size)

        source_segments_by_rank: List[List[_ThdLayoutSegment]] = []
        source_lengths: List[int] = []
        target_segments_by_rank: List[List[_ThdLayoutSegment]] = []
        target_lengths: List[int] = []
        for rank in range(cp_size):
            source_segments, source_length = _build_thd_layout_segments(
                cu, cp_size, rank, source_partition_mode
            )
            target_segments, target_length = _build_thd_layout_segments(
                cu, cp_size, rank, target_partition_mode
            )
            source_segments_by_rank.append(source_segments)
            source_lengths.append(source_length)
            target_segments_by_rank.append(target_segments)
            target_lengths.append(target_length)

        local_source_segments = source_segments_by_rank[cp_rank]
        local_target_segments = target_segments_by_rank[cp_rank]

        send_rows_list: List[int] = []
        input_split_sizes: List[int] = []
        for dst_rank in range(cp_size):
            intersections = _intersect_thd_layout_segments(
                local_source_segments, target_segments_by_rank[dst_rank]
            )
            intersections.sort(key=lambda item: item[1])
            input_split_size = 0
            for source_row, _, length in intersections:
                _append_range(send_rows_list, source_row, length)
                input_split_size += length
            input_split_sizes.append(input_split_size)

        recv_rows_list: List[int] = []
        output_split_sizes: List[int] = []
        for src_rank in range(cp_size):
            intersections = _intersect_thd_layout_segments(
                source_segments_by_rank[src_rank], local_target_segments
            )
            intersections.sort(key=lambda item: item[1])
            output_split_size = 0
            for _, target_row, length in intersections:
                _append_range(recv_rows_list, target_row, length)
                output_split_size += length
            output_split_sizes.append(output_split_size)

        assert len(send_rows_list) == source_lengths[cp_rank]
        assert len(recv_rows_list) == target_lengths[cp_rank]
        return _encode_thd_cp_partition_route(
            cp_size=cp_size,
            cp_rank=cp_rank,
            local_source_length=source_lengths[cp_rank],
            local_target_length=target_lengths[cp_rank],
            send_rows_list=send_rows_list,
            recv_rows_list=recv_rows_list,
            input_split_sizes=input_split_sizes,
            output_split_sizes=output_split_sizes,
            device=device,
        )


def get_thd_cp_partition_route(
    packed_seq_params: Optional[Any],
    source_partition_mode: CpPartitionMode,
    target_partition_mode: CpPartitionMode,
) -> Optional[torch.Tensor]:
    """Return the precomputed THD CP partition route tensor for one direction."""
    if source_partition_mode == target_partition_mode:
        return None
    if packed_seq_params is None or getattr(packed_seq_params, "qkv_format", None) != "thd":
        return None

    attr_name = _thd_cp_partition_route_attr_name(source_partition_mode, target_partition_mode)
    route = getattr(packed_seq_params, attr_name, None)
    if route is not None:
        return route

    warnings.warn(
        "THD PackedSeqParams is missing precomputed context-parallel layout routes. "
        "This lookup will attempt to build them from packed_seq_params.cp_group as "
        "a compatibility fallback. Callers should prebuild THD CP routes when "
        "constructing the batch; a future release will require the routes to be "
        "present before layout conversion.",
        FutureWarning,
        stacklevel=2,
    )
    prebuild_thd_cp_partition_routes(packed_seq_params)
    return getattr(packed_seq_params, attr_name, None)


def prebuild_thd_cp_partition_routes(
    packed_seq_params: Optional[Any],
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    *,
    device: Optional[torch.device] = None,
) -> None:
    """Best-effort prebuild of THD CP layout route tensors for a packed microbatch."""
    if packed_seq_params is None or getattr(packed_seq_params, "qkv_format", None) != "thd":
        return
    if cp_group is None:
        cp_group = getattr(packed_seq_params, "cp_group", None)
    if cp_group is None or cp_group.size() <= 1:
        return
    cp_size = cp_group.size()
    cp_rank = cp_group.rank()
    cu_seqlens = get_packed_seq_params_cp_partition_cu_seqlens(packed_seq_params)
    if cu_seqlens is None:
        return
    if device is None:
        device = cu_seqlens.device

    for source_partition_mode, target_partition_mode in _THD_CP_ROUTE_ATTRS:
        attr_name = _thd_cp_partition_route_attr_name(source_partition_mode, target_partition_mode)
        try:
            route = build_thd_cp_partition_route(
                cu_seqlens,
                cp_size,
                cp_rank,
                source_partition_mode,
                target_partition_mode,
                device=device,
            )
        except ValueError:
            # Some batches/layouts may never need the opposite route.  Preserve
            # lazy block-time validation for the path that actually uses it.
            setattr(packed_seq_params, attr_name, None)
            continue
        setattr(packed_seq_params, attr_name, route)
