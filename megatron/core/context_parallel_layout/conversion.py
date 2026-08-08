# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tensor operations for converting between CP partition modes."""

import copy
from typing import Any, Callable, List, Optional, Tuple, Union

import torch

from megatron.core.context_parallel_layout.metadata import (
    get_packed_seq_params_cp_partition_cu_seqlens,
)
from megatron.core.context_parallel_layout.routes import (
    _cp_layout_nvtx_range,
    build_thd_cp_partition_route,
    decode_thd_cp_partition_route,
    get_thd_cp_partition_route,
)
from megatron.core.context_parallel_layout import CpPartitionMode


class CpPartitionModeConverter:
    """Convert tensors across one CP layout edge."""

    def __init__(
        self,
        *,
        cp_group: Optional[torch.distributed.ProcessGroup],
        packed_seq_params: Optional[Any],
        source_partition_mode: Optional[CpPartitionMode],
        target_partition_mode: Optional[CpPartitionMode],
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        self.cp_group = cp_group
        self.packed_seq_params = packed_seq_params
        self.source_partition_mode = source_partition_mode
        self.target_partition_mode = target_partition_mode
        self.tp_group = tp_group

    @property
    def conversion_needed(self) -> bool:
        """Return whether this edge needs a real layout conversion."""
        return (
            self.source_partition_mode != self.target_partition_mode
            and self.cp_group is not None
            and self.cp_group.size() > 1
        )

    def assert_no_dense_attention_inputs(
        self,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> None:
        """Reject dense attention tensors when this edge would reorder tokens."""
        if not self.conversion_needed:
            return
        if attention_mask is not None:
            self._raise_unsupported_dense_attention(
                "an explicit attention_mask", hidden_states=hidden_states
            )
        if attention_bias is not None:
            self._raise_unsupported_dense_attention(
                "attention_bias", hidden_states=hidden_states
            )

    def convert(
        self,
        value: Any,
        *,
        seq_dim: Union[int, Callable[[torch.Tensor], int]] = 0,
        sequence_parallel: bool = False,
    ) -> Any:
        """Convert a tensor or nested tensor container across this layout edge."""
        if not self.conversion_needed or value is None:
            return value
        # Nested values may contain optional tensors; traverse containers while
        # preserving their original shape.
        if isinstance(value, tuple):
            return tuple(
                self.convert(part, seq_dim=seq_dim, sequence_parallel=sequence_parallel)
                for part in value
            )
        if isinstance(value, list):
            return [
                self.convert(part, seq_dim=seq_dim, sequence_parallel=sequence_parallel)
                for part in value
            ]
        if not torch.is_tensor(value):
            return value

        resolved_seq_dim = seq_dim(value) if callable(seq_dim) else seq_dim
        return convert_cp_partition_mode(
            value,
            self.cp_group,
            source_partition_mode=self.source_partition_mode,
            target_partition_mode=self.target_partition_mode,
            seq_dim=resolved_seq_dim,
            cu_seqlens=get_packed_seq_params_cp_partition_cu_seqlens(self.packed_seq_params),
            sequence_parallel=sequence_parallel,
            tp_group=self.tp_group,
            thd_cp_partition_route=get_thd_cp_partition_route(
                self.packed_seq_params,
                self.source_partition_mode,
                self.target_partition_mode,
            ),
        )

    def _raise_unsupported_dense_attention(
        self,
        tensor_name: str,
        *,
        hidden_states: Optional[torch.Tensor],
    ) -> None:
        hidden_shape = tuple(hidden_states.shape) if hidden_states is not None else None
        raise NotImplementedError(
            "Changing CP partition mode with "
            f"{tensor_name} is not supported yet: "
            f"source={self.source_partition_mode!r}, "
            f"target={self.target_partition_mode!r}, "
            f"qkv_format={getattr(self.packed_seq_params, 'qkv_format', None)!r}, "
            f"hidden_shape={hidden_shape}."
        )


def convert_module_input_tensors_cp_partition_mode(
    *,
    hidden_states: torch.Tensor,
    packed_seq_params: Optional[Any],
    cp_group: Optional[torch.distributed.ProcessGroup],
    tp_group: Optional[torch.distributed.ProcessGroup],
    target_partition_mode: CpPartitionMode,
    sequence_parallel: bool,
    attention_mask: Optional[torch.Tensor] = None,
    attention_bias: Optional[torch.Tensor] = None,
    key_value_states: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor,
    Optional[Any],
    Optional[CpPartitionModeConverter],
]:
    """Convert a module's rank-local sequence tensors to a target CP layout.

    This helper performs the common "entry conversion" pattern used by modules
    that need to consume a different CP layout than their caller supplied.  It
    returns a converter for the opposite edge so the module output can be
    converted back to the original input layout.
    """
    if cp_group is None or cp_group.size() <= 1:
        return hidden_states, packed_seq_params, None

    source_partition_mode = getattr(packed_seq_params, "cp_partition_mode", None)
    if source_partition_mode is None:
        raise ValueError(
            "PackedSeqParams.cp_partition_mode is required before module input CP layout "
            "conversion when context parallelism is active."
        )
    if source_partition_mode == target_partition_mode:
        return hidden_states, packed_seq_params, None

    input_to_target_converter = CpPartitionModeConverter(
        cp_group=cp_group,
        packed_seq_params=packed_seq_params,
        source_partition_mode=source_partition_mode,
        target_partition_mode=target_partition_mode,
        tp_group=tp_group,
    )
    input_to_target_converter.assert_no_dense_attention_inputs(
        attention_mask=attention_mask,
        attention_bias=attention_bias,
        hidden_states=hidden_states,
    )
    if key_value_states is not None:
        raise NotImplementedError(
            "Changing CP partition mode with cross-attention key/value states is not supported "
            f"yet: source={source_partition_mode!r}, target={target_partition_mode!r}."
        )
    hidden_states = input_to_target_converter.convert(
        hidden_states,
        seq_dim=0,
        sequence_parallel=sequence_parallel,
    )

    local_packed_seq_params = packed_seq_params
    if packed_seq_params is not None:
        local_packed_seq_params = copy.copy(packed_seq_params)
        local_packed_seq_params.cp_partition_mode = target_partition_mode
    target_to_input_converter = CpPartitionModeConverter(
        cp_group=cp_group,
        packed_seq_params=local_packed_seq_params,
        source_partition_mode=target_partition_mode,
        target_partition_mode=source_partition_mode,
        tp_group=tp_group,
    )
    return (
        hidden_states,
        local_packed_seq_params,
        target_to_input_converter,
    )


def zigzag_to_contiguous_chunks(
    x: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
    seq_dim: int = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    thd_cp_partition_route: Optional[torch.Tensor] = None,
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
    thd_cp_partition_route: Optional[torch.Tensor] = None,
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
    thd_cp_partition_route: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convert a sequence tensor between CP zigzag and contiguous layouts.

    With sequence parallel enabled, the baseline path gathers the full CP-local
    sequence on each TP rank, performs the CP layout conversion, then scatters
    back to the original SP sharding.
    """
    # TODO(yuzhongw): implement a direct TPxCP layout conversion path if the
    # gather-convert-scatter fallback becomes a bottleneck.

    if source_partition_mode == target_partition_mode:
        return x

    cp_size = cp_group.size() if cp_group is not None else 1
    if cp_size == 1:
        return x

    if source_partition_mode not in ("zigzag", "contiguous") or target_partition_mode not in (
        "zigzag",
        "contiguous",
    ):
        cp_rank = cp_group.rank() if cp_group is not None else 0
        raise ValueError(
            f"Unsupported CP partition mode conversion "
            f"{source_partition_mode!r} -> {target_partition_mode!r}; "
            f"shape={tuple(x.shape)}, seq_dim={seq_dim}, cp_size={cp_size}, cp_rank={cp_rank}."
        )

    if sequence_parallel and tp_group is not None and tp_group.size() > 1:
        from megatron.core.tensor_parallel.mappings import (
            gather_from_sequence_parallel_region,
            scatter_to_sequence_parallel_region,
        )

        moved = x.movedim(seq_dim, 0) if seq_dim != 0 else x
        # This gather is only used to run a duplicated CP layout permutation before
        # scattering back to SP shards. Its backward must split, not reduce-scatter;
        # otherwise every TP rank contributes the same full-sequence gradient.
        gathered = gather_from_sequence_parallel_region(
            moved,
            tensor_parallel_output_grad=False,
            group=tp_group,
        )
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


def _convert_cp_partition_mode_full_sequence(
    x: torch.Tensor,
    cp_group: Optional[torch.distributed.ProcessGroup],
    *,
    source_partition_mode: CpPartitionMode,
    target_partition_mode: CpPartitionMode,
    seq_dim: int,
    cu_seqlens: Optional[torch.Tensor],
    thd_cp_partition_route: Optional[torch.Tensor] = None,
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
        f"{source_partition_mode!r} -> {target_partition_mode!r}; "
        f"shape={tuple(x.shape)}, seq_dim={seq_dim}."
    )


def _pack_thd_cp_route_send_buffer(
    x: torch.Tensor, local_source_length: int, send_rows: Optional[torch.Tensor]
) -> torch.Tensor:
    if local_source_length == 0:
        return x.narrow(0, 0, 0)
    if send_rows is None:
        return x
    return x.index_select(0, send_rows)


def _scatter_thd_cp_route_recv_buffer(
    recv_buf: torch.Tensor, recv_rows: Optional[torch.Tensor], out_shape: Tuple[int, ...]
) -> torch.Tensor:
    if recv_rows is None:
        return recv_buf
    out = recv_buf.new_empty(out_shape)
    if recv_rows.numel() > 0:
        out.index_copy_(0, recv_rows, recv_buf)
    return out


def _zigzag_contiguous_thd_swap(
    x: torch.Tensor,
    cp_group: Optional[torch.distributed.ProcessGroup],
    seq_dim: int,
    cu_seqlens: torch.Tensor,
    source_partition_mode: str,
    target_partition_mode: str,
    thd_cp_partition_route: Optional[torch.Tensor] = None,
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

    conversion_name = f"{source_partition_mode}_to_{target_partition_mode}"
    with _cp_layout_nvtx_range(f"cp_layout/thd/swap/{conversion_name}"):
        if seq_dim != 0:
            x = x.movedim(seq_dim, 0)
        x = x.contiguous()

        route = thd_cp_partition_route
        if route is None or route.device != x.device:
            route = build_thd_cp_partition_route(
                cu_seqlens,
                cp_size,
                cp_rank,
                source_partition_mode,
                target_partition_mode,
                device=x.device,
            )
        (
            local_source_length,
            local_target_length,
            send_rows,
            recv_rows,
            input_split_sizes,
            output_split_sizes,
        ) = decode_thd_cp_partition_route(route, cp_size, cp_rank)

        if x.size(0) != local_source_length:
            raise ValueError(
                f"Local THD tensor length ({x.size(0)}) does not match {source_partition_mode} "
                f"rank-{cp_rank} partition length ({local_source_length})."
            )

        with _cp_layout_nvtx_range(f"cp_layout/thd/pack/{conversion_name}"):
            send_buf = _pack_thd_cp_route_send_buffer(x, local_source_length, send_rows)
            if not send_buf.is_contiguous():
                send_buf = send_buf.contiguous()

        with _cp_layout_nvtx_range(f"cp_layout/thd/all_to_all/{conversion_name}"):
            recv_buf = all_to_all(cp_group, send_buf, output_split_sizes, input_split_sizes)

        with _cp_layout_nvtx_range(f"cp_layout/thd/scatter/{conversion_name}"):
            out_shape = (local_target_length,) + tuple(x.shape[1:])
            out = _scatter_thd_cp_route_recv_buffer(recv_buf, recv_rows, out_shape)

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

    # TODO(yuzhongw): cache this small SBHD permutation plan by
    # (cp_size, cp_rank, source layout, target layout, device) instead of
    # rebuilding Python lists on every conversion.
    source_in_zigzag = to_contiguous
    target_in_zigzag = not to_contiguous
    source_partition_mode = "zigzag" if source_in_zigzag else "contiguous"
    target_partition_mode = "zigzag" if target_in_zigzag else "contiguous"
    conversion_name = f"{source_partition_mode}_to_{target_partition_mode}"

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

    with _cp_layout_nvtx_range(f"cp_layout/sbhd/all_to_all/{conversion_name}"):
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
