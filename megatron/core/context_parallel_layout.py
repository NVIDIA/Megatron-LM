# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Context parallel sequence partition-mode helpers."""

from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import torch


CpPartitionMode = Literal["zigzag", "contiguous"]


@dataclass(frozen=True)
class ThdCPPartitionRoute:
    """Precomputed local routing for one THD CP partition-mode conversion."""

    source_partition_mode: CpPartitionMode
    target_partition_mode: CpPartitionMode
    cp_size: int
    cp_rank: int
    local_source_length: int
    local_target_length: int
    send_rows: torch.Tensor
    recv_rows: torch.Tensor
    input_split_sizes: List[int]
    output_split_sizes: List[int]
    send_rows_are_identity: bool
    recv_rows_are_identity: bool


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


def get_context_parallel_layout_chunk_indices(
    cp_size: int, cp_rank: int, cp_partition_mode: str
) -> torch.Tensor:
    """Return the two global chunk indices owned by this CP rank in ``cp_partition_mode``."""
    if cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {cp_size}.")
    if not 0 <= cp_rank < cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}.")

    if cp_partition_mode == "zigzag":
        return torch.tensor([cp_rank, 2 * cp_size - cp_rank - 1], dtype=torch.long)
    if cp_partition_mode == "contiguous":
        return torch.tensor([2 * cp_rank, 2 * cp_rank + 1], dtype=torch.long)
    raise ValueError(f"Unsupported context-parallel partition mode {cp_partition_mode!r}.")


################################################################################
# Layer-to-CP-partition-mode mapping
################################################################################
#
# ``None`` is a meaningful result here: it means the module is token-layout
# agnostic and preserves whichever CP partition mode it receives.  It must not
# be used as the fallback for an unrecognized module type; unknown types should
# fail loudly so new layer implementations add an explicit partition-mode policy.


def get_required_cp_partition_mode_for_layer(
    layer: Any, config: Any, *, cp_comm_type: Optional[str] = None
) -> Optional[CpPartitionMode]:
    """Return the CP partition mode required by a layer or attention-like module.

    The helper intentionally uses light duck-typing instead of importing concrete
    modules, because several of those modules already import this file.
    """
    if cp_comm_type is None:
        cp_comm_type = getattr(config, "cp_comm_type", None)

    if layer is None:
        raise ValueError("Cannot determine CP partition mode for None.")

    module_name = layer.__class__.__name__
    module_type_names = {cls.__name__ for cls in type(layer).__mro__}
    if hasattr(layer, "inner_layer"):
        return get_required_cp_partition_mode_for_layer(
            layer.inner_layer, getattr(layer, "config", config), cp_comm_type=cp_comm_type
        )
    if hasattr(layer, "self_attention"):
        return get_required_cp_partition_mode_for_layer(
            layer.self_attention, getattr(layer, "config", config), cp_comm_type=cp_comm_type
        )
    if module_type_names & {"IdentityOp", "IdentityFuncOp"}:
        return None
    if "MambaLayer" in module_type_names:
        # MambaContextParallel currently undoes/redoes Megatron's attention
        # load-balancing layout internally, so it expects zigzag inputs.
        return "zigzag"
    if module_type_names & {"MLPLayer", "MoETransformerLayer"}:
        return None
    if "GatedDeltaNet" in module_type_names:
        mode = getattr(config, "linear_cp_mode", "chunkwise")
        if mode in {"chunkwise", "headwise"}:
            return "contiguous"
        raise ValueError(f"Unsupported GatedDeltaNet linear_cp_mode: {mode!r}.")
    if module_type_names & {"DSv4HybridAttention", "DSv4HybridSelfAttention"}:
        return "contiguous"

    # Preserve current standard-attention behavior.  Ring/P2P needs zigzag for
    # causal load balancing, and TE A2A currently still expects zigzag input.
    # ``cp_comm_type`` is deliberately part of this policy surface so TE A2A can
    # switch to contiguous here once the backend stops requiring zigzag.
    del cp_comm_type
    if module_type_names & {
        "SelfAttention",
        "CrossAttention",
        "MultiLatentAttention",
        "MLASelfAttention",
        "FusedMLASelfAttention",
        "AbsorbedMLASelfAttention",
    }:
        return "zigzag"
    raise ValueError(
        f"Cannot determine CP partition mode for layer/module type {module_name!r}."
    )


def build_cp_partition_mode_plan(
    layers: Any,
    config: Any,
    cp_stage_entry_partition_mode: Optional[CpPartitionMode],
    *,
    owner_name: str,
) -> Tuple[Optional[CpPartitionMode], List[Optional[CpPartitionMode]], Optional[CpPartitionMode]]:
    """Build a local immutable CP partition-mode plan for a block-like module.

    The stage entry partition mode is an external pipeline boundary property.
    It must not be inferred from local layers.
    """
    if (
        getattr(config, "context_parallel_size", 1) == 1
        and not getattr(config, "dynamic_context_parallel", False)
    ):
        return None, [None] * len(layers), None

    if cp_stage_entry_partition_mode is None:
        raise ValueError(
            f"cp_stage_entry_partition_mode must be provided for {owner_name}. "
            "A block cannot infer its input tensor partition mode from its local layers."
        )
    if cp_stage_entry_partition_mode not in ("zigzag", "contiguous"):
        raise ValueError(
            f"Unsupported cp_stage_entry_partition_mode {cp_stage_entry_partition_mode!r}."
        )

    current_partition_mode = cp_stage_entry_partition_mode
    cp_partition_mode_plan: List[Optional[CpPartitionMode]] = []
    for layer in layers:
        layer_config = getattr(layer, "config", config)
        required_partition_mode = get_required_cp_partition_mode_for_layer(layer, layer_config)
        cp_partition_mode_plan.append(required_partition_mode)
        if required_partition_mode is not None:
            current_partition_mode = required_partition_mode

    return cp_stage_entry_partition_mode, cp_partition_mode_plan, current_partition_mode


def get_cp_partition_mode_before_local_index(
    cp_stage_entry_partition_mode: Optional[CpPartitionMode],
    cp_partition_mode_plan: List[Optional[CpPartitionMode]],
    local_index: int,
) -> Optional[CpPartitionMode]:
    """Return the CP partition mode immediately before a local layer index."""
    current_partition_mode = cp_stage_entry_partition_mode
    for required_partition_mode in cp_partition_mode_plan[:local_index]:
        if required_partition_mode is not None:
            current_partition_mode = required_partition_mode
    return current_partition_mode


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
        raise ValueError(f"Unsupported context-parallel partition mode {cp_partition_mode!r}.")

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
    cu: List[int],
    cp_size: int,
    cp_rank: int,
    cp_partition_mode: CpPartitionMode,
) -> Tuple[List[_ThdLayoutSegment], int]:
    total_tokens = cu[-1]
    if cp_partition_mode == "contiguous":
        part_len = total_tokens // cp_size
        if part_len == 0:
            return [], 0
        return [(cp_rank * part_len, part_len, 0)], part_len

    if cp_partition_mode != "zigzag":
        raise ValueError(f"Unsupported context-parallel partition mode {cp_partition_mode!r}.")

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
    source_segments: List[_ThdLayoutSegment],
    target_segments: List[_ThdLayoutSegment],
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


def _row_list_to_tensor(rows: List[int], device: torch.device) -> torch.Tensor:
    if not rows:
        return torch.empty(0, device=device, dtype=torch.long)
    return torch.tensor(rows, device=device, dtype=torch.long)


def build_thd_cp_partition_route(
    cu_seqlens: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    source_partition_mode: CpPartitionMode,
    target_partition_mode: CpPartitionMode,
    *,
    device: Optional[torch.device] = None,
) -> ThdCPPartitionRoute:
    """Precompute local send/receive indices for one THD CP layout conversion.

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
            f"{source_partition_mode!r} -> {target_partition_mode!r}."
        )
    if source_partition_mode == target_partition_mode:
        raise ValueError("A THD CP partition route is only needed when partition modes differ.")
    if device is None:
        device = cu_seqlens.device

    with _cp_layout_nvtx_range(
        f"cp_layout/build_thd_route/{source_partition_mode}_to_{target_partition_mode}"
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
        send_rows = _row_list_to_tensor(send_rows_list, device)
        recv_rows = _row_list_to_tensor(recv_rows_list, device)
        return ThdCPPartitionRoute(
            source_partition_mode=source_partition_mode,
            target_partition_mode=target_partition_mode,
            cp_size=cp_size,
            cp_rank=cp_rank,
            local_source_length=source_lengths[cp_rank],
            local_target_length=target_lengths[cp_rank],
            send_rows=send_rows,
            recv_rows=recv_rows,
            input_split_sizes=input_split_sizes,
            output_split_sizes=output_split_sizes,
            send_rows_are_identity=_row_list_is_identity(send_rows_list),
            recv_rows_are_identity=_row_list_is_identity(recv_rows_list),
        )


def _thd_route_cache_key(
    cu_seqlens: torch.Tensor,
    device: torch.device,
    cp_size: int,
    cp_rank: int,
    source_partition_mode: CpPartitionMode,
    target_partition_mode: CpPartitionMode,
) -> Tuple[Any, ...]:
    device_index = device.index if device.index is not None else -1
    return (
        source_partition_mode,
        target_partition_mode,
        cp_size,
        cp_rank,
        device.type,
        device_index,
        cu_seqlens.data_ptr(),
        cu_seqlens.storage_offset(),
        tuple(cu_seqlens.shape),
        getattr(cu_seqlens, "_version", None),
    )


def get_or_build_thd_cp_partition_route(
    packed_seq_params: Optional[Any],
    cp_group: Optional[torch.distributed.ProcessGroup],
    source_partition_mode: CpPartitionMode,
    target_partition_mode: CpPartitionMode,
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Optional[ThdCPPartitionRoute]:
    """Return a cached THD CP partition route for one packed microbatch."""
    if source_partition_mode == target_partition_mode:
        return None
    cp_size = cp_group.size() if cp_group is not None else 1
    if cp_size == 1:
        return None
    cp_rank = cp_group.rank()
    if cu_seqlens is None:
        cu_seqlens = get_packed_seq_params_cp_partition_cu_seqlens(packed_seq_params)
    if cu_seqlens is None:
        return None
    if device is None:
        device = cu_seqlens.device

    if packed_seq_params is None:
        return build_thd_cp_partition_route(
            cu_seqlens,
            cp_size,
            cp_rank,
            source_partition_mode,
            target_partition_mode,
            device=device,
        )

    cache = getattr(packed_seq_params, "cp_partition_route_cache", None)
    if cache is None:
        cache = {}
        packed_seq_params.cp_partition_route_cache = cache
    key = _thd_route_cache_key(
        cu_seqlens, device, cp_size, cp_rank, source_partition_mode, target_partition_mode
    )
    route = cache.get(key)
    if route is None:
        route = build_thd_cp_partition_route(
            cu_seqlens,
            cp_size,
            cp_rank,
            source_partition_mode,
            target_partition_mode,
            device=device,
        )
        cache[key] = route
    return route


def prebuild_thd_cp_partition_route_cache(
    packed_seq_params: Optional[Any],
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    *,
    device: Optional[torch.device] = None,
) -> None:
    """Best-effort prebuild of THD CP layout routes for a packed microbatch.

    ``get_or_build_thd_cp_partition_route`` remains the authoritative lazy
    lookup used by model blocks.  This helper lets model forward paths move the
    first CPU route construction before the first layout conversion.
    """
    if packed_seq_params is None or getattr(packed_seq_params, "qkv_format", None) != "thd":
        return
    if cp_group is None:
        cp_group = getattr(packed_seq_params, "cp_group", None)
    if cp_group is None or cp_group.size() <= 1:
        return

    for source_partition_mode, target_partition_mode in (
        ("zigzag", "contiguous"),
        ("contiguous", "zigzag"),
    ):
        try:
            get_or_build_thd_cp_partition_route(
                packed_seq_params,
                cp_group,
                source_partition_mode,
                target_partition_mode,
                device=device,
            )
        except ValueError:
            # Some batches/layouts may never need the opposite route.  Preserve
            # lazy block-time validation for the path that actually uses it.
            continue


def zigzag_to_contiguous_chunks(
    x: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
    seq_dim: int = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    thd_cp_partition_route: Optional[ThdCPPartitionRoute] = None,
) -> torch.Tensor:
    """Permute CP chunks from Megatron zigzag layout to contiguous-time layout.

    SBHD tensors have two equal chunks per rank along ``seq_dim`` and use a
    chunk-level all-to-all. THD tensors pass global ``cu_seqlens`` and use one
    packed-token all-to-all over the whole local THD tensor.
    """
    if cu_seqlens is not None:
        return _zigzag_contiguous_thd_swap(
            x,
            cp_group,
            seq_dim,
            cu_seqlens,
            source_partition_mode="zigzag",
            target_partition_mode="contiguous",
            thd_cp_partition_route=thd_cp_partition_route,
        )
    return _zigzag_contiguous_chunk_swap(x, cp_group, seq_dim, to_contiguous=True)


def contiguous_to_zigzag_chunks(
    x: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
    seq_dim: int = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    thd_cp_partition_route: Optional[ThdCPPartitionRoute] = None,
) -> torch.Tensor:
    """Inverse of :func:`zigzag_to_contiguous_chunks`."""
    if cu_seqlens is not None:
        return _zigzag_contiguous_thd_swap(
            x,
            cp_group,
            seq_dim,
            cu_seqlens,
            source_partition_mode="contiguous",
            target_partition_mode="zigzag",
            thd_cp_partition_route=thd_cp_partition_route,
        )
    return _zigzag_contiguous_chunk_swap(x, cp_group, seq_dim, to_contiguous=False)


def convert_cp_partition_mode(
    x: torch.Tensor,
    cp_group: Optional[torch.distributed.ProcessGroup],
    *,
    source_partition_mode: Optional[str],
    target_partition_mode: Optional[str],
    seq_dim: int = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    sequence_parallel: bool = False,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    tp_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    thd_cp_partition_route: Optional[ThdCPPartitionRoute] = None,
) -> torch.Tensor:
    """Convert a sequence tensor between CP zigzag and contiguous layouts.

    With sequence parallel enabled, the baseline path gathers the full CP-local
    sequence on each TP rank, performs the CP layout conversion, then scatters
    back to the original SP sharding.  ``tp_cp_group`` is accepted for the
    future direct TPxCP all-to-all implementation.
    """
    del tp_cp_group

    if source_partition_mode == target_partition_mode:
        return x

    cp_size = cp_group.size() if cp_group is not None else 1
    if cp_size == 1:
        return x

    if source_partition_mode not in ("zigzag", "contiguous") or target_partition_mode not in (
        "zigzag",
        "contiguous",
    ):
        raise ValueError(
            f"Unsupported CP partition mode conversion "
            f"{source_partition_mode!r} -> {target_partition_mode!r}."
        )

    if sequence_parallel and tp_group is not None and tp_group.size() > 1:
        from megatron.core.tensor_parallel.mappings import (
            gather_from_sequence_parallel_region,
            scatter_to_sequence_parallel_region,
        )

        moved = x.movedim(seq_dim, 0) if seq_dim != 0 else x
        gathered = gather_from_sequence_parallel_region(moved, group=tp_group)
        converted = _convert_cp_partition_mode_full_sequence(
            gathered,
            cp_group,
            source_partition_mode=source_partition_mode,
            target_partition_mode=target_partition_mode,
            seq_dim=0,
            cu_seqlens=cu_seqlens,
            thd_cp_partition_route=thd_cp_partition_route,
        )
        scattered = scatter_to_sequence_parallel_region(converted, group=tp_group)
        return scattered.movedim(0, seq_dim).contiguous() if seq_dim != 0 else scattered

    return _convert_cp_partition_mode_full_sequence(
        x,
        cp_group,
        source_partition_mode=source_partition_mode,
        target_partition_mode=target_partition_mode,
        seq_dim=seq_dim,
        cu_seqlens=cu_seqlens,
        thd_cp_partition_route=thd_cp_partition_route,
    )


def get_packed_seq_params_cp_partition_cu_seqlens(
    packed_seq_params: Optional[Any],
) -> Optional[torch.Tensor]:
    """Return THD cumulative sequence lengths used for CP layout conversion."""
    if packed_seq_params is None or getattr(packed_seq_params, "qkv_format", None) != "thd":
        return None
    return (
        packed_seq_params.cu_seqlens_q_padded
        if packed_seq_params.cu_seqlens_q_padded is not None
        else packed_seq_params.cu_seqlens_q
    )


def replace_packed_seq_params_cp_partition_mode(
    packed_seq_params: Optional[Any],
    cp_partition_mode: Optional[CpPartitionMode],
) -> Optional[Any]:
    """Return packed-sequence metadata annotated with the current CP partition mode."""
    if packed_seq_params is None or getattr(packed_seq_params, "qkv_format", None) != "thd":
        return packed_seq_params
    if getattr(packed_seq_params, "cp_partition_mode", None) == cp_partition_mode:
        return packed_seq_params
    return replace(packed_seq_params, cp_partition_mode=cp_partition_mode)


def convert_cp_partition_mode_nested(
    value: Any,
    cp_group: Optional[torch.distributed.ProcessGroup],
    *,
    source_partition_mode: Optional[str],
    target_partition_mode: Optional[str],
    seq_dim: Union[int, Callable[[torch.Tensor], int]] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    sequence_parallel: bool = False,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    tp_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    thd_cp_partition_route: Optional[ThdCPPartitionRoute] = None,
) -> Any:
    """Recursively convert tensors inside a nested value between CP layouts.

    ``None`` and non-tensor leaves are returned unchanged.  Lists and tuples are
    traversed recursively while preserving their container type.
    """
    if source_partition_mode == target_partition_mode:
        return value
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(
            convert_cp_partition_mode_nested(
                part,
                cp_group,
                source_partition_mode=source_partition_mode,
                target_partition_mode=target_partition_mode,
                seq_dim=seq_dim,
                cu_seqlens=cu_seqlens,
                sequence_parallel=sequence_parallel,
                tp_group=tp_group,
                tp_cp_group=tp_cp_group,
                thd_cp_partition_route=thd_cp_partition_route,
            )
            for part in value
        )
    if isinstance(value, list):
        return [
            convert_cp_partition_mode_nested(
                part,
                cp_group,
                source_partition_mode=source_partition_mode,
                target_partition_mode=target_partition_mode,
                seq_dim=seq_dim,
                cu_seqlens=cu_seqlens,
                sequence_parallel=sequence_parallel,
                tp_group=tp_group,
                tp_cp_group=tp_cp_group,
                thd_cp_partition_route=thd_cp_partition_route,
            )
            for part in value
        ]
    if not torch.is_tensor(value):
        return value

    resolved_seq_dim = seq_dim(value) if callable(seq_dim) else seq_dim
    return convert_cp_partition_mode(
        value,
        cp_group,
        source_partition_mode=source_partition_mode,
        target_partition_mode=target_partition_mode,
        seq_dim=resolved_seq_dim,
        cu_seqlens=cu_seqlens,
        sequence_parallel=sequence_parallel,
        tp_group=tp_group,
        tp_cp_group=tp_cp_group,
        thd_cp_partition_route=thd_cp_partition_route,
    )


def _convert_cp_partition_mode_full_sequence(
    x: torch.Tensor,
    cp_group: Optional[torch.distributed.ProcessGroup],
    *,
    source_partition_mode: CpPartitionMode,
    target_partition_mode: CpPartitionMode,
    seq_dim: int,
    cu_seqlens: Optional[torch.Tensor],
    thd_cp_partition_route: Optional[ThdCPPartitionRoute] = None,
) -> torch.Tensor:
    """Convert a tensor whose sequence dim contains the full CP-local sequence."""
    if source_partition_mode == "zigzag" and target_partition_mode == "contiguous":
        return zigzag_to_contiguous_chunks(
            x,
            cp_group,
            seq_dim=seq_dim,
            cu_seqlens=cu_seqlens,
            thd_cp_partition_route=thd_cp_partition_route,
        )
    if source_partition_mode == "contiguous" and target_partition_mode == "zigzag":
        return contiguous_to_zigzag_chunks(
            x,
            cp_group,
            seq_dim=seq_dim,
            cu_seqlens=cu_seqlens,
            thd_cp_partition_route=thd_cp_partition_route,
        )
    raise ValueError(
        f"Unsupported CP partition mode conversion "
        f"{source_partition_mode!r} -> {target_partition_mode!r}."
    )


def _pack_thd_cp_route_send_buffer(
    x: torch.Tensor,
    route: ThdCPPartitionRoute,
) -> torch.Tensor:
    if route.local_source_length == 0:
        return x.narrow(0, 0, 0)
    if route.send_rows_are_identity:
        return x
    return x.index_select(0, route.send_rows)


def _scatter_thd_cp_route_recv_buffer(
    recv_buf: torch.Tensor,
    route: ThdCPPartitionRoute,
    out_shape: Tuple[int, ...],
) -> torch.Tensor:
    if route.recv_rows_are_identity:
        return recv_buf
    out = recv_buf.new_empty(out_shape)
    if route.recv_rows.numel() > 0:
        out.index_copy_(0, route.recv_rows, recv_buf)
    return out


def _zigzag_contiguous_thd_swap(
    x: torch.Tensor,
    cp_group: Optional[torch.distributed.ProcessGroup],
    seq_dim: int,
    cu_seqlens: torch.Tensor,
    source_partition_mode: str,
    target_partition_mode: str,
    thd_cp_partition_route: Optional[ThdCPPartitionRoute] = None,
) -> torch.Tensor:
    """Single-all-to-all THD permutation between zigzag and contiguous layouts.

    The packed THD tensor stays packed: we first group local tokens by their
    target CP rank, exchange those groups once, then scatter received tokens
    back into the target rank-local order.
    """
    cp_size = cp_group.size() if cp_group is not None else 1
    if cp_size == 1:
        return x
    cp_rank = cp_group.rank()
    from megatron.core.tensor_parallel.mappings import all_to_all

    with _cp_layout_nvtx_range(
        f"cp_layout/thd_swap/{source_partition_mode}_to_{target_partition_mode}"
    ):
        if seq_dim != 0:
            x = x.movedim(seq_dim, 0)
        x = x.contiguous()

        route = thd_cp_partition_route
        if route is None:
            route = build_thd_cp_partition_route(
                cu_seqlens,
                cp_size,
                cp_rank,
                source_partition_mode,
                target_partition_mode,
                device=x.device,
            )
        if (
            route.source_partition_mode != source_partition_mode
            or route.target_partition_mode != target_partition_mode
            or route.cp_size != cp_size
            or route.cp_rank != cp_rank
        ):
            raise ValueError(
                "THD CP partition route does not match the requested conversion: "
                f"route={route.source_partition_mode}->{route.target_partition_mode}, "
                f"cp_size={route.cp_size}, cp_rank={route.cp_rank}; "
                f"requested={source_partition_mode}->{target_partition_mode}, "
                f"cp_size={cp_size}, cp_rank={cp_rank}."
            )
        if route.send_rows.device != x.device or route.recv_rows.device != x.device:
            route = build_thd_cp_partition_route(
                cu_seqlens,
                cp_size,
                cp_rank,
                source_partition_mode,
                target_partition_mode,
                device=x.device,
            )

        if x.size(0) != route.local_source_length:
            raise ValueError(
                f"Local THD tensor length ({x.size(0)}) does not match {source_partition_mode} "
                f"rank-{cp_rank} partition length ({route.local_source_length})."
            )

        with _cp_layout_nvtx_range("cp_layout/thd_swap/pack"):
            send_buf = _pack_thd_cp_route_send_buffer(x, route)
            if not send_buf.is_contiguous():
                send_buf = send_buf.contiguous()

        with _cp_layout_nvtx_range("cp_layout/thd_swap/all_to_all"):
            recv_buf = all_to_all(
                cp_group, send_buf, route.output_split_sizes, route.input_split_sizes
            )

        with _cp_layout_nvtx_range("cp_layout/thd_swap/scatter"):
            out_shape = (route.local_target_length,) + tuple(x.shape[1:])
            out = _scatter_thd_cp_route_recv_buffer(recv_buf, route, out_shape)

        if seq_dim != 0:
            out = out.movedim(0, seq_dim)
        return out.contiguous()


def _zigzag_contiguous_chunk_swap(
    x: torch.Tensor,
    cp_group: Optional[torch.distributed.ProcessGroup],
    seq_dim: int,
    to_contiguous: bool,
) -> torch.Tensor:
    """Single-all-to-all chunk permutation between zigzag and contiguous layouts.

    Each rank holds exactly two chunks along ``seq_dim``. The mapping from
    local (rank, slot) to (rank, slot) in the target layout is deterministic
    and depends only on ``cp_size`` and ``cp_rank``, so we pack send data in
    destination-rank order and use one ``all_to_all_single`` with unequal
    splits to route each chunk to its target rank.
    """
    cp_size = cp_group.size() if cp_group is not None else 1
    if cp_size == 1:
        return x
    cp_rank = cp_group.rank()
    from megatron.core.tensor_parallel.mappings import all_to_all

    # Work with seq_dim at position 0.
    if seq_dim != 0:
        x = x.movedim(seq_dim, 0)
    x = x.contiguous()

    seq_len_local = x.size(0)
    assert seq_len_local % 2 == 0, (
        f"zigzag/contiguous chunk swap requires an even local sequence length, "
        f"got {seq_len_local}."
    )
    chunk_len = seq_len_local // 2

    def _rank_to_chunks(rank: int, in_zigzag: bool) -> Tuple[int, int]:
        """Global chunk indices at (slot 0, slot 1) for this rank."""
        if in_zigzag:
            return (rank, 2 * cp_size - rank - 1)
        return (2 * rank, 2 * rank + 1)

    def _chunk_to_dest(chunk_idx: int, target_zigzag: bool) -> Tuple[int, int]:
        """Destination (rank, slot) for a given global chunk index in the target layout."""
        if target_zigzag:
            if chunk_idx < cp_size:
                return chunk_idx, 0
            return 2 * cp_size - chunk_idx - 1, 1
        return chunk_idx // 2, chunk_idx % 2

    source_in_zigzag = to_contiguous
    target_in_zigzag = not to_contiguous

    local_chunk_indices = _rank_to_chunks(cp_rank, source_in_zigzag)
    local_dests = [_chunk_to_dest(c, target_in_zigzag) for c in local_chunk_indices]

    # Pack the send buffer so chunks are ordered by (dst_rank, dst_slot).
    local_slot_order = sorted(range(2), key=lambda s: local_dests[s])
    local_chunks = [x[:chunk_len], x[chunk_len:]]
    send_buf = torch.cat([local_chunks[s] for s in local_slot_order], dim=0).contiguous()

    input_split_chunks = [0] * cp_size
    for dst_rank, _ in local_dests:
        input_split_chunks[dst_rank] += 1

    # Mirror every source rank's packing logic so we know which received chunk
    # belongs in which local target slot.
    output_split_chunks = [0] * cp_size
    recv_dst_slots_per_source: List[List[int]] = [[] for _ in range(cp_size)]
    for src in range(cp_size):
        src_chunks = _rank_to_chunks(src, source_in_zigzag)
        src_dests = [_chunk_to_dest(c, target_in_zigzag) for c in src_chunks]
        src_slot_order = sorted(range(2), key=lambda s: src_dests[s])
        for s in src_slot_order:
            dst_rank, dst_slot = src_dests[s]
            if dst_rank == cp_rank:
                output_split_chunks[src] += 1
                recv_dst_slots_per_source[src].append(dst_slot)

    input_split_sizes = [n * chunk_len for n in input_split_chunks]
    output_split_sizes = [n * chunk_len for n in output_split_chunks]

    recv_buf = all_to_all(cp_group, send_buf, output_split_sizes, input_split_sizes)

    # Reassemble local chunks in target-layout slot order.
    target_slots: List[Optional[torch.Tensor]] = [None, None]
    offset = 0
    for src in range(cp_size):
        for dst_slot in recv_dst_slots_per_source[src]:
            target_slots[dst_slot] = recv_buf[offset : offset + chunk_len]
            offset += chunk_len
    assert all(t is not None for t in target_slots), "Incomplete chunk reassembly in CP swap"

    out = torch.cat(target_slots, dim=0)
    if seq_dim != 0:
        out = out.movedim(0, seq_dim)
    return out.contiguous()
