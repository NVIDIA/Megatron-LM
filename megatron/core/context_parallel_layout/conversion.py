# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tensor operations for converting between CP partition modes."""

import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, Union, cast

import torch

from megatron.core.context_parallel_layout.routes import (
    build_thd_cp_partition_route,
    get_thd_cp_partition_route,
)
from megatron.core.context_parallel_layout.types import CpPartitionMode, ThdCpRoute
from megatron.core.context_parallel_layout.utils import (
    get_packed_seq_params_cp_partition_cu_seqlens,
)
from megatron.core.tensor_parallel.mappings import all_to_all
from megatron.core.utils import nvtx_range

if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams


class CpPartitionModeConverter:
    """Convert tensors across one CP layout edge."""

    def __init__(
        self,
        *,
        packed_seq_params: Optional["PackedSeqParams"],
        source_partition_mode: CpPartitionMode,
        target_partition_mode: CpPartitionMode,
        config: Any,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        tp_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        self.cp_group = cp_group
        self.packed_seq_params = packed_seq_params
        self.source_partition_mode = source_partition_mode
        self.target_partition_mode = target_partition_mode
        self.config = config
        self.tp_group = tp_group
        self.tp_cp_group = tp_cp_group
        if (
            self.conversion_needed
            and getattr(self.packed_seq_params, "qkv_format", None) == "thd"
            and self.config.cuda_graph_impl == "full_iteration"
        ):
            raise ValueError(
                "Full-iteration CUDA graph is not supported for THD CP layout conversion: "
                f"source={self.source_partition_mode!r}, target={self.target_partition_mode!r}."
            )

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
            self._raise_unsupported_dense_attention("attention_bias", hidden_states=hidden_states)

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
        converted = convert_cp_partition_mode(
            x=value,
            source_partition_mode=self.source_partition_mode,
            target_partition_mode=self.target_partition_mode,
            seq_dim=resolved_seq_dim,
            cu_seqlens=get_packed_seq_params_cp_partition_cu_seqlens(self.packed_seq_params),
            sequence_parallel=sequence_parallel,
            cp_group=self.cp_group,
            tp_group=self.tp_group,
            tp_cp_group=self.tp_cp_group,
            thd_cp_partition_route=get_thd_cp_partition_route(
                self.packed_seq_params, self.source_partition_mode, self.target_partition_mode
            ),
        )
        if self.packed_seq_params is not None:
            self.packed_seq_params.cp_partition_mode = self.target_partition_mode
        return converted

    def _raise_unsupported_dense_attention(
        self, tensor_name: str, *, hidden_states: Optional[torch.Tensor]
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
    packed_seq_params: Optional["PackedSeqParams"],
    target_partition_mode: CpPartitionMode,
    sequence_parallel: bool,
    config: Any,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    tp_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    attention_mask: Optional[torch.Tensor] = None,
    attention_bias: Optional[torch.Tensor] = None,
    key_value_states: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[CpPartitionModeConverter]]:
    """Convert a module's rank-local sequence tensors to a target CP layout.

    This helper performs the common "entry conversion" pattern used by modules
    that need to consume a different CP layout than their caller supplied.  It
    returns a converter for the opposite edge so the module output can be
    converted back to the original input layout.
    """
    if cp_group is None or cp_group.size() <= 1:
        return hidden_states, None

    source_partition_mode = getattr(config, "cp_partition_mode", None)
    if source_partition_mode is None:
        raise ValueError(
            "config.cp_partition_mode is required before module input CP layout conversion when "
            "context parallelism is active."
        )
    if source_partition_mode == target_partition_mode:
        return hidden_states, None

    input_to_target_converter = CpPartitionModeConverter(
        cp_group=cp_group,
        packed_seq_params=packed_seq_params,
        source_partition_mode=source_partition_mode,
        target_partition_mode=target_partition_mode,
        config=config,
        tp_group=tp_group,
        tp_cp_group=tp_cp_group,
    )
    input_to_target_converter.assert_no_dense_attention_inputs(
        attention_mask=attention_mask, attention_bias=attention_bias, hidden_states=hidden_states
    )
    if key_value_states is not None:
        raise NotImplementedError(
            "Changing CP partition mode with cross-attention key/value states is not supported "
            f"yet: source={source_partition_mode!r}, target={target_partition_mode!r}."
        )
    hidden_states = input_to_target_converter.convert(
        value=hidden_states, seq_dim=0, sequence_parallel=sequence_parallel
    )

    target_to_input_converter = CpPartitionModeConverter(
        cp_group=cp_group,
        packed_seq_params=packed_seq_params,
        source_partition_mode=target_partition_mode,
        target_partition_mode=source_partition_mode,
        config=config,
        tp_group=tp_group,
        tp_cp_group=tp_cp_group,
    )
    return (hidden_states, target_to_input_converter)


def convert_cp_partition_mode(
    x: torch.Tensor,
    *,
    source_partition_mode: Optional[str],
    target_partition_mode: Optional[str],
    seq_dim: int = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    sequence_parallel: bool = False,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    tp_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    thd_cp_partition_route: Optional[ThdCpRoute] = None,
) -> torch.Tensor:
    """Convert a sequence tensor between CP zigzag and contiguous layouts.

    SBHD tensors use one unified all-to-all-v redistribution path over CP or
    TPxCP. THD tensors use their packed-token CP route and, when sequence
    parallelism shards the packed sequence, retain the naive TP gather/scatter
    fallback.
    """

    if source_partition_mode == target_partition_mode:
        return x

    cp_size = cp_group.size() if cp_group is not None else 1
    if cp_size == 1:
        return x
    assert cp_group is not None

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
    source_layout = cast(CpPartitionMode, source_partition_mode)
    target_layout = cast(CpPartitionMode, target_partition_mode)

    if cu_seqlens is None:
        moved = x.movedim(seq_dim, 0) if seq_dim != 0 else x
        converted = _redistribute_sbhd_layout(
            input_=moved,
            cp_group=cp_group,
            source_layout=source_layout,
            target_layout=target_layout,
            sequence_parallel=sequence_parallel,
            tp_group=tp_group,
            tp_cp_group=tp_cp_group,
        )
        return converted.movedim(0, seq_dim).contiguous() if seq_dim != 0 else converted

    if sequence_parallel and tp_group is not None and tp_group.size() > 1:
        from megatron.core.tensor_parallel.mappings import (
            gather_from_sequence_parallel_region,
            scatter_to_sequence_parallel_region,
        )

        # TODO(yuzhongw): replace the naive THD TP gather -> CP all-to-all -> TP scatter
        # fallback with a direct packed THD TPxCP redistribution path.
        warnings.warn(
            "THD CP layout conversion with sequence parallelism uses the naive "
            "TP gather -> CP all-to-all -> TP scatter fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
        moved = x.movedim(seq_dim, 0) if seq_dim != 0 else x
        # This gather is only used to run a duplicated CP layout permutation before
        # scattering back to SP shards. Its backward must split, not reduce-scatter;
        # otherwise every TP rank contributes the same full-sequence gradient.
        gathered = gather_from_sequence_parallel_region(
            input_=moved, tensor_parallel_output_grad=False, group=tp_group
        )
        converted = _redistribute_thd_layout(
            x=gathered,
            cp_group=cp_group,
            seq_dim=0,
            cu_seqlens=cu_seqlens,
            source_partition_mode=source_layout,
            target_partition_mode=target_layout,
            thd_cp_partition_route=thd_cp_partition_route,
        )
        scattered = scatter_to_sequence_parallel_region(input_=converted, group=tp_group)
        return scattered.movedim(0, seq_dim).contiguous() if seq_dim != 0 else scattered

    return _redistribute_thd_layout(
        x=x,
        cp_group=cp_group,
        seq_dim=seq_dim,
        cu_seqlens=cu_seqlens,
        source_partition_mode=source_layout,
        target_partition_mode=target_layout,
        thd_cp_partition_route=thd_cp_partition_route,
    )


def _pack_thd_cp_route_send_buffer(
    x: torch.Tensor, send_index: Optional[torch.Tensor]
) -> torch.Tensor:
    if send_index is None:
        return x
    return x.index_select(0, send_index)


def _scatter_thd_cp_route_recv_buffer(
    recv_buf: torch.Tensor, recv_index: Optional[torch.Tensor], out_shape: Tuple[int, ...]
) -> torch.Tensor:
    if recv_index is None:
        return recv_buf
    out = recv_buf.new_empty(out_shape)
    if recv_index.numel() > 0:
        out.index_copy_(0, recv_index, recv_buf)
    return out


def _redistribute_thd_layout(
    x: torch.Tensor,
    cp_group: Optional[torch.distributed.ProcessGroup],
    seq_dim: int,
    cu_seqlens: torch.Tensor,
    source_partition_mode: str,
    target_partition_mode: str,
    thd_cp_partition_route: Optional[ThdCpRoute] = None,
) -> torch.Tensor:
    """Single-all-to-all THD permutation between zigzag and contiguous layouts.

    The packed THD tensor stays packed: we first group local tokens by their
    target CP rank, exchange those groups once, then scatter received tokens
    back into the target rank-local order.
    """
    cp_size = cp_group.size() if cp_group is not None else 1
    if cp_size == 1:
        return x
    assert cp_group is not None
    cp_rank = cp_group.rank()
    conversion_name = f"{source_partition_mode}_to_{target_partition_mode}"
    with nvtx_range(f"cp_layout/thd/swap/{conversion_name}"):
        if seq_dim != 0:
            x = x.movedim(seq_dim, 0)
        x = x.contiguous()

        route = thd_cp_partition_route
        if route is None:
            route = build_thd_cp_partition_route(
                cu_seqlens=cu_seqlens, cp_size=cp_size, cp_rank=cp_rank, device=x.device
            )

        if source_partition_mode == "zigzag" and target_partition_mode == "contiguous":
            send_index = route.zigzag_index
            recv_index = route.contiguous_index
            input_split_sizes = route.zigzag_split_sizes
            output_split_sizes = route.contiguous_split_sizes
        elif source_partition_mode == "contiguous" and target_partition_mode == "zigzag":
            send_index = route.contiguous_index
            recv_index = route.zigzag_index
            input_split_sizes = route.contiguous_split_sizes
            output_split_sizes = route.zigzag_split_sizes
        else:
            raise ValueError(
                f"Unsupported CP partition mode conversion "
                f"{source_partition_mode!r} -> {target_partition_mode!r} for THD route."
            )

        local_source_length = sum(input_split_sizes)
        local_target_length = sum(output_split_sizes)

        if x.size(0) != local_source_length:
            raise ValueError(
                f"Local THD tensor length ({x.size(0)}) does not match {source_partition_mode} "
                f"rank-{cp_rank} partition length ({local_source_length})."
            )
        if local_target_length != x.size(0):
            raise ValueError(
                "THD CP layout conversion must preserve the local token count, "
                f"got source={local_source_length}, target={local_target_length}, "
                f"cp_size={cp_size}, cp_rank={cp_rank}, "
                f"source_layout={source_partition_mode!r}, "
                f"target_layout={target_partition_mode!r}."
            )

        with nvtx_range(f"cp_layout/thd/pack/{conversion_name}"):
            send_buf = _pack_thd_cp_route_send_buffer(x=x, send_index=send_index)
            if not send_buf.is_contiguous():
                send_buf = send_buf.contiguous()

        with nvtx_range(f"cp_layout/thd/all_to_all/{conversion_name}"):
            recv_buf = all_to_all(
                group=cp_group,
                input_=send_buf,
                output_split_sizes_=output_split_sizes,
                input_split_sizes=input_split_sizes,
            )

        with nvtx_range(f"cp_layout/thd/scatter/{conversion_name}"):
            out_shape = (local_target_length,) + tuple(x.shape[1:])
            out = _scatter_thd_cp_route_recv_buffer(
                recv_buf=recv_buf, recv_index=recv_index, out_shape=out_shape
            )

        if seq_dim != 0:
            out = out.movedim(0, seq_dim)
        return out.contiguous()


@dataclass(frozen=True)
class _SbhdLayoutRedistributionPlan:
    """Rank-local SBHD all-to-all plan expressed in sequence-segment counts."""

    send_slots: tuple[int, ...]
    input_segment_counts: tuple[int, ...]
    output_segment_counts: tuple[int, ...]
    receive_permutation: tuple[int, ...]


def _sbhd_segments_per_rank(tp_size: int) -> int:
    """Return two SBHD segments for CP-only conversion and one for even-TP SP conversion."""
    if tp_size == 1:
        return 2
    if tp_size % 2 != 0:
        raise ValueError(
            "Sequence-parallel SBHD CP layout conversion requires an even tensor-parallel size, "
            f"got {tp_size}"
        )
    return 1


def _local_sbhd_segment_ids(
    layout: CpPartitionMode, cp_size: int, cp_rank: int, tp_size: int = 1, tp_rank: int = 0
) -> tuple[int, ...]:
    """Return the atomic SBHD sequence segments owned by one TP×CP rank."""
    segments_per_rank = _sbhd_segments_per_rank(tp_size=tp_size)
    if layout == "contiguous":
        first_segment = segments_per_rank * (cp_rank * tp_size + tp_rank)
        return tuple(range(first_segment, first_segment + segments_per_rank))
    if layout == "zigzag":
        segments_per_cp_half = tp_size * segments_per_rank // 2
        front_start = cp_rank * segments_per_cp_half
        back_start = (2 * cp_size - cp_rank - 1) * segments_per_cp_half
        cp_segments = tuple(range(front_start, front_start + segments_per_cp_half)) + tuple(
            range(back_start, back_start + segments_per_cp_half)
        )
        sp_start = segments_per_rank * tp_rank
        return cp_segments[sp_start : sp_start + segments_per_rank]
    raise ValueError(f"Unsupported CP layout: {layout}")


@lru_cache(maxsize=None)
def _sbhd_segment_owner(
    segment_id: int, layout: CpPartitionMode, cp_size: int, tp_size: int
) -> tuple[int, int]:
    for cp_rank in range(cp_size):
        for tp_rank in range(tp_size):
            if segment_id in _local_sbhd_segment_ids(
                layout=layout, cp_size=cp_size, cp_rank=cp_rank, tp_size=tp_size, tp_rank=tp_rank
            ):
                return cp_rank, tp_rank
    raise ValueError(
        f"SBHD segment {segment_id} is not present in the {layout} layout for "
        f"{cp_size=} and {tp_size=}"
    )


@lru_cache(maxsize=None)
def _build_sbhd_group_rank_by_logical_rank(
    cp_global_ranks: tuple[int, ...],
    tp_global_ranks: tuple[int, ...],
    tp_cp_global_ranks: tuple[int, ...],
    current_global_rank: int,
) -> tuple[int, ...]:
    """Map logical ``cp_rank * tp_size + tp_rank`` coordinates to group ranks for SBHD."""
    group_rank_by_global_rank = {
        global_rank: group_rank for group_rank, global_rank in enumerate(tp_cp_global_ranks)
    }
    group_rank_by_logical_rank = []
    for cp_global_rank in cp_global_ranks:
        for tp_global_rank in tp_global_ranks:
            target_global_rank = cp_global_rank + tp_global_rank - current_global_rank
            if target_global_rank not in group_rank_by_global_rank:
                raise RuntimeError(
                    "TP and CP process groups do not form the expected Cartesian product"
                )
            group_rank_by_logical_rank.append(group_rank_by_global_rank[target_global_rank])
    return tuple(group_rank_by_logical_rank)


def _get_sbhd_group_rank_by_logical_rank(
    cp_group: torch.distributed.ProcessGroup,
    tp_group: torch.distributed.ProcessGroup,
    tp_cp_group: torch.distributed.ProcessGroup,
) -> tuple[int, ...]:
    return _build_sbhd_group_rank_by_logical_rank(
        cp_global_ranks=tuple(torch.distributed.get_process_group_ranks(cp_group)),
        tp_global_ranks=tuple(torch.distributed.get_process_group_ranks(tp_group)),
        tp_cp_global_ranks=tuple(torch.distributed.get_process_group_ranks(tp_cp_group)),
        current_global_rank=torch.distributed.get_rank(),
    )


@lru_cache(maxsize=None)
def _build_sbhd_layout_redistribution_plan(
    source_layout: CpPartitionMode,
    target_layout: CpPartitionMode,
    cp_size: int,
    cp_rank: int,
    tp_size: int = 1,
    tp_rank: int = 0,
    group_rank_by_logical_rank: tuple[int, ...] | None = None,
) -> _SbhdLayoutRedistributionPlan:
    """Build the SBHD all-to-all-v plan for one rank of a CP layout conversion."""
    if cp_size < 1:
        raise ValueError(f"cp_size must be positive, got {cp_size}")
    if tp_size < 1:
        raise ValueError(f"tp_size must be positive, got {tp_size}")
    if not 0 <= cp_rank < cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}")
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"tp_rank must be in [0, {tp_size}), got {tp_rank}")

    group_size = cp_size * tp_size
    if group_rank_by_logical_rank is None:
        group_rank_by_logical_rank = tuple(range(group_size))
    if sorted(group_rank_by_logical_rank) != list(range(group_size)):
        raise ValueError("group_rank_by_logical_rank must be a permutation of the group ranks")

    source_ids = _local_sbhd_segment_ids(
        layout=source_layout, cp_size=cp_size, cp_rank=cp_rank, tp_size=tp_size, tp_rank=tp_rank
    )
    target_ids = _local_sbhd_segment_ids(
        layout=target_layout, cp_size=cp_size, cp_rank=cp_rank, tp_size=tp_size, tp_rank=tp_rank
    )

    def destination_group_rank(segment_id: int) -> int:
        destination_cp_rank, destination_tp_rank = _sbhd_segment_owner(
            segment_id=segment_id, layout=target_layout, cp_size=cp_size, tp_size=tp_size
        )
        destination_logical_rank = destination_cp_rank * tp_size + destination_tp_rank
        return group_rank_by_logical_rank[destination_logical_rank]

    send_entries = sorted(
        (destination_group_rank(segment_id=segment_id), slot)
        for slot, segment_id in enumerate(source_ids)
    )
    send_slots = tuple(slot for _, slot in send_entries)
    input_segment_counts = tuple(
        sum(destination == rank for destination, _ in send_entries) for rank in range(group_size)
    )

    received_ids = []
    output_segment_counts = []
    source_logical_ranks = sorted(
        range(group_size), key=lambda logical_rank: group_rank_by_logical_rank[logical_rank]
    )
    for source_logical_rank in source_logical_ranks:
        source_cp_rank, source_tp_rank = divmod(source_logical_rank, tp_size)
        rank_source_ids = _local_sbhd_segment_ids(
            layout=source_layout,
            cp_size=cp_size,
            cp_rank=source_cp_rank,
            tp_size=tp_size,
            tp_rank=source_tp_rank,
        )
        ids_from_source = [
            segment_id
            for segment_id in rank_source_ids
            if _sbhd_segment_owner(
                segment_id=segment_id, layout=target_layout, cp_size=cp_size, tp_size=tp_size
            )
            == (cp_rank, tp_rank)
        ]
        received_ids.extend(ids_from_source)
        output_segment_counts.append(len(ids_from_source))

    if sorted(received_ids) != sorted(target_ids):
        raise RuntimeError(
            f"Invalid {source_layout}-to-{target_layout} SBHD redistribution plan for "
            f"CP rank {cp_rank}, TP rank {tp_rank}: received {received_ids}, "
            f"expected {target_ids}"
        )
    receive_permutation = tuple(received_ids.index(segment_id) for segment_id in target_ids)

    return _SbhdLayoutRedistributionPlan(
        send_slots=send_slots,
        input_segment_counts=input_segment_counts,
        output_segment_counts=tuple(output_segment_counts),
        receive_permutation=receive_permutation,
    )


def _redistribute_sbhd_layout(
    input_: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
    source_layout: CpPartitionMode,
    target_layout: CpPartitionMode,
    sequence_parallel: bool,
    tp_group: Optional[torch.distributed.ProcessGroup],
    tp_cp_group: Optional[torch.distributed.ProcessGroup],
) -> torch.Tensor:
    """Redistribute local SBHD sequence segments with a differentiable all-to-all-v."""
    cp_size = cp_group.size()
    if cp_size == 1 or source_layout == target_layout:
        return input_

    cp_rank = cp_group.rank()
    tp_size, tp_rank = 1, 0
    communication_group = cp_group
    group_rank_by_logical_rank = None
    if sequence_parallel and tp_group is not None and tp_group.size() > 1:
        if tp_cp_group is None:
            raise ValueError(
                "tp_cp_group is required for direct sequence-parallel SBHD layout conversion"
            )
        tp_size, tp_rank = tp_group.size(), tp_group.rank()
        communication_group = tp_cp_group
        group_rank_by_logical_rank = _get_sbhd_group_rank_by_logical_rank(
            cp_group=cp_group, tp_group=tp_group, tp_cp_group=tp_cp_group
        )

    plan = _build_sbhd_layout_redistribution_plan(
        source_layout=source_layout,
        target_layout=target_layout,
        cp_size=cp_size,
        cp_rank=cp_rank,
        tp_size=tp_size,
        tp_rank=tp_rank,
        group_rank_by_logical_rank=group_rank_by_logical_rank,
    )

    input_contiguous = input_.contiguous()
    local_seq_len = input_contiguous.shape[0]
    local_segment_count = _sbhd_segments_per_rank(tp_size=tp_size)
    if local_seq_len % local_segment_count != 0:
        raise ValueError(
            "SBHD CP layout conversion requires the sequence length local to each TP×CP rank to "
            f"be divisible by {local_segment_count}, got {local_seq_len}"
        )
    segment_len = local_seq_len // local_segment_count
    segment_shape = (local_segment_count, segment_len, *input_contiguous.shape[1:])
    segments = input_contiguous.reshape(segment_shape)

    if plan.send_slots == tuple(range(local_segment_count)):
        send_buffer = input_contiguous
    else:
        send_buffer = segments.flip(0).reshape(input_contiguous.shape)
    input_split_sizes = [count * segment_len for count in plan.input_segment_counts]
    output_split_sizes = [count * segment_len for count in plan.output_segment_counts]
    conversion_name = f"{source_layout}_to_{target_layout}"
    with nvtx_range(f"cp_layout/sbhd/all_to_all/{conversion_name}"):
        received = all_to_all(
            group=communication_group,
            input_=send_buffer,
            output_split_sizes_=output_split_sizes,
            input_split_sizes=input_split_sizes,
        )

    received_segments = received.reshape(segment_shape)
    if plan.receive_permutation == tuple(range(local_segment_count)):
        output = received
    else:
        output = received_segments.flip(0).reshape(input_contiguous.shape)
    return output.contiguous()
