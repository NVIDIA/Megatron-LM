# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Context-parallel sequence-layout conversion."""

from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Literal

import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import all_to_all

CPLayout = Literal["contiguous", "zigzag"]


@dataclass(frozen=True)
class _LayoutRedistributionPlan:
    """Rank-local all-to-all plan expressed in sequence-segment counts."""

    send_slots: tuple[int, ...]
    input_segment_counts: tuple[int, ...]
    output_segment_counts: tuple[int, ...]
    receive_permutation: tuple[int, ...]


@dataclass(frozen=True, eq=False)
class THDCPLayoutPlan:
    """Rank-local all-to-all-v plan for packed sequence layout conversion.

    Forward fields describe contiguous-to-zigzag conversion. Reverse fields describe the inverse
    conversion. The plan is built once per packed batch and reused around every attention layer.
    """

    contiguous_local_token_count: int
    zigzag_local_token_count: int
    cu_seqlens_padded: torch.Tensor
    max_seqlen_padded: int
    pad_between_seqs: bool
    forward_send_indices: torch.Tensor
    forward_receive_positions: torch.Tensor
    forward_input_split_sizes: tuple[int, ...]
    forward_output_split_sizes: tuple[int, ...]
    reverse_send_indices: torch.Tensor
    reverse_receive_indices: torch.Tensor


@dataclass(frozen=True)
class _LayoutParallelContext:
    """Process-group coordinates used by a layout redistribution."""

    cp_size: int
    cp_rank: int
    tp_size: int
    tp_rank: int
    communication_group: torch.distributed.ProcessGroup
    group_rank_by_logical_rank: tuple[int, ...]

    @property
    def group_size(self) -> int:
        """Return the number of ranks participating in the redistribution."""
        return self.cp_size * self.tp_size

    @property
    def logical_rank(self) -> int:
        """Return this rank's CP-major, TP-minor logical rank."""
        return self.cp_rank * self.tp_size + self.tp_rank


def _segments_per_rank(tp_size: int) -> int:
    """Return two segments for CP-only conversion and one for even-TP SP conversion."""
    if tp_size == 1:
        return 2
    if tp_size % 2 != 0:
        raise ValueError(
            "Sequence-parallel CP layout conversion requires an even tensor-parallel size, "
            f"got {tp_size}"
        )
    return 1


def _local_segment_ids(
    layout: CPLayout, cp_size: int, cp_rank: int, tp_size: int = 1, tp_rank: int = 0
) -> tuple[int, ...]:
    """Return the atomic sequence segments owned by one TP x CP rank."""
    segments_per_rank = _segments_per_rank(tp_size)
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
def _segment_owner(
    segment_id: int, layout: CPLayout, cp_size: int, tp_size: int
) -> tuple[int, int]:
    for cp_rank in range(cp_size):
        for tp_rank in range(tp_size):
            if segment_id in _local_segment_ids(
                layout, cp_size, cp_rank, tp_size=tp_size, tp_rank=tp_rank
            ):
                return cp_rank, tp_rank
    raise ValueError(
        f"Segment {segment_id} is not present in the {layout} layout for "
        f"{cp_size=} and {tp_size=}"
    )


@lru_cache(maxsize=None)
def _build_group_rank_by_logical_rank(
    cp_global_ranks: tuple[int, ...],
    tp_global_ranks: tuple[int, ...],
    tp_cp_global_ranks: tuple[int, ...],
    current_global_rank: int,
) -> tuple[int, ...]:
    """Map logical ``cp_rank * tp_size + tp_rank`` coordinates to group ranks."""
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


def _get_group_rank_by_logical_rank(
    cp_group: torch.distributed.ProcessGroup,
    tp_group: torch.distributed.ProcessGroup,
    tp_cp_group: torch.distributed.ProcessGroup,
) -> tuple[int, ...]:
    return _build_group_rank_by_logical_rank(
        cp_global_ranks=tuple(torch.distributed.get_process_group_ranks(cp_group)),
        tp_global_ranks=tuple(torch.distributed.get_process_group_ranks(tp_group)),
        tp_cp_global_ranks=tuple(torch.distributed.get_process_group_ranks(tp_cp_group)),
        current_global_rank=torch.distributed.get_rank(),
    )


def _get_layout_parallel_context(
    cp_group: torch.distributed.ProcessGroup,
    sequence_parallel: bool,
    tp_group: torch.distributed.ProcessGroup | None,
    tp_cp_group: torch.distributed.ProcessGroup | None,
) -> _LayoutParallelContext:
    """Resolve the rank coordinates and process group for a layout redistribution."""
    cp_size, cp_rank = cp_group.size(), cp_group.rank()
    tp_size, tp_rank = 1, 0
    communication_group = cp_group
    group_rank_by_logical_rank = tuple(range(cp_size))

    if sequence_parallel and tp_group is None:
        raise ValueError("tp_group is required for sequence-parallel CP layout conversion")
    if sequence_parallel and tp_group.size() > 1:
        if tp_cp_group is None:
            raise ValueError("tp_cp_group is required for sequence-parallel CP layout conversion")
        tp_size, tp_rank = tp_group.size(), tp_group.rank()
        communication_group = tp_cp_group
        group_rank_by_logical_rank = _get_group_rank_by_logical_rank(
            cp_group, tp_group, tp_cp_group
        )

    return _LayoutParallelContext(
        cp_size=cp_size,
        cp_rank=cp_rank,
        tp_size=tp_size,
        tp_rank=tp_rank,
        communication_group=communication_group,
        group_rank_by_logical_rank=group_rank_by_logical_rank,
    )


@lru_cache(maxsize=None)
def _build_layout_redistribution_plan(
    source_layout: CPLayout,
    target_layout: CPLayout,
    cp_size: int,
    cp_rank: int,
    tp_size: int = 1,
    tp_rank: int = 0,
    group_rank_by_logical_rank: tuple[int, ...] | None = None,
) -> _LayoutRedistributionPlan:
    """Build the all-to-all-v plan for one rank of a CP layout conversion."""
    group_size = cp_size * tp_size
    if group_rank_by_logical_rank is None:
        group_rank_by_logical_rank = tuple(range(group_size))
    if sorted(group_rank_by_logical_rank) != list(range(group_size)):
        raise ValueError("group_rank_by_logical_rank must be a permutation of the group ranks")

    source_ids = _local_segment_ids(
        source_layout, cp_size, cp_rank, tp_size=tp_size, tp_rank=tp_rank
    )
    target_ids = _local_segment_ids(
        target_layout, cp_size, cp_rank, tp_size=tp_size, tp_rank=tp_rank
    )

    def destination_group_rank(segment_id: int) -> int:
        destination_cp_rank, destination_tp_rank = _segment_owner(
            segment_id, target_layout, cp_size, tp_size
        )
        destination_logical_rank = destination_cp_rank * tp_size + destination_tp_rank
        return group_rank_by_logical_rank[destination_logical_rank]

    send_entries = sorted(
        (destination_group_rank(segment_id), slot) for slot, segment_id in enumerate(source_ids)
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
        rank_source_ids = _local_segment_ids(
            source_layout, cp_size, source_cp_rank, tp_size=tp_size, tp_rank=source_tp_rank
        )
        ids_from_source = [
            segment_id
            for segment_id in rank_source_ids
            if _segment_owner(segment_id, target_layout, cp_size, tp_size) == (cp_rank, tp_rank)
        ]
        received_ids.extend(ids_from_source)
        output_segment_counts.append(len(ids_from_source))

    if sorted(received_ids) != sorted(target_ids):
        raise RuntimeError(
            f"Invalid {source_layout}-to-{target_layout} CP redistribution plan for "
            f"CP rank {cp_rank}, TP rank {tp_rank}: received {received_ids}, "
            f"expected {target_ids}"
        )
    receive_permutation = tuple(received_ids.index(segment_id) for segment_id in target_ids)

    return _LayoutRedistributionPlan(
        send_slots=send_slots,
        input_segment_counts=input_segment_counts,
        output_segment_counts=tuple(output_segment_counts),
        receive_permutation=receive_permutation,
    )


def _build_thd_cp_layout_plan_from_rank_order_indices(
    rank_order_indices: torch.Tensor,
    source_token_count: int,
    cu_seqlens_padded: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    tp_size: int = 1,
    tp_rank: int = 0,
    group_rank_by_logical_rank: tuple[int, ...] | None = None,
    pad_between_seqs: bool | torch.Tensor = False,
) -> THDCPLayoutPlan:
    """Build a packed layout plan from source indices grouped by target logical rank.

    ``rank_order_indices`` contains one entry for every padded attention position. Non-negative
    entries identify positions in the contiguous input; ``-1`` entries are temporary padding.
    """
    if rank_order_indices.ndim != 1:
        raise ValueError("rank_order_indices must be one-dimensional")
    group_size = cp_size * tp_size
    target_token_count = rank_order_indices.numel()
    if source_token_count <= 0 or source_token_count % group_size != 0:
        raise ValueError(
            "The contiguous token count must be positive and divisible by CP * TP, got "
            f"{source_token_count}, CP size {cp_size}, and TP size {tp_size}"
        )
    if target_token_count == 0 or target_token_count % group_size != 0:
        raise ValueError(
            "The padded attention token count must be positive and divisible by CP * TP, got "
            f"{target_token_count}, CP size {cp_size}, and TP size {tp_size}"
        )
    if group_rank_by_logical_rank is None:
        group_rank_by_logical_rank = tuple(range(group_size))
    if sorted(group_rank_by_logical_rank) != list(range(group_size)):
        raise ValueError("group_rank_by_logical_rank must be a permutation of the group ranks")

    contiguous_local_token_count = source_token_count // group_size
    zigzag_local_token_count = target_token_count // group_size
    logical_rank = cp_rank * tp_size + tp_rank
    rank_order_indices = rank_order_indices.to(dtype=torch.int64)
    target_indices_by_logical_rank = rank_order_indices.view(group_size, zigzag_local_token_count)

    # Map every source token to its padded target position and destination rank.
    target_positions = torch.arange(
        target_token_count, dtype=torch.int64, device=rank_order_indices.device
    )
    source_indices = torch.where(rank_order_indices >= 0, rank_order_indices, source_token_count)
    rank_order_position_by_global_index = rank_order_indices.new_empty(source_token_count + 1)
    rank_order_position_by_global_index.scatter_(0, source_indices, target_positions)
    rank_order_position_by_global_index = rank_order_position_by_global_index[:-1]
    source_start = logical_rank * contiguous_local_token_count
    source_stop = source_start + contiguous_local_token_count
    destination_logical_ranks = torch.div(
        rank_order_position_by_global_index[source_start:source_stop],
        zigzag_local_token_count,
        rounding_mode="floor",
    )
    logical_to_group = torch.tensor(
        group_rank_by_logical_rank, dtype=torch.int64, device=rank_order_indices.device
    )
    destination_group_ranks = logical_to_group.index_select(0, destination_logical_ranks)
    forward_send_indices = torch.argsort(destination_group_ranks, stable=True)
    forward_input_split_sizes = torch.zeros(
        group_size, dtype=torch.int64, device=rank_order_indices.device
    ).index_add_(0, destination_group_ranks, torch.ones_like(destination_group_ranks))

    # all_to_all_single concatenates messages in source group-rank order. Model that order and
    # scatter the received values into their padded attention positions.
    target_global_indices = target_indices_by_logical_rank[logical_rank]
    target_valid_positions = torch.nonzero(target_global_indices >= 0, as_tuple=False).flatten()
    target_global_indices = target_global_indices.index_select(0, target_valid_positions)
    source_logical_ranks = torch.div(
        target_global_indices, contiguous_local_token_count, rounding_mode="floor"
    )
    source_group_ranks = logical_to_group.index_select(0, source_logical_ranks)
    received_order_key = source_group_ranks * source_token_count + target_global_indices
    received_target_positions = torch.argsort(received_order_key)
    forward_receive_positions = target_valid_positions.index_select(0, received_target_positions)
    forward_output_split_sizes = torch.zeros(
        group_size, dtype=torch.int64, device=rank_order_indices.device
    ).index_add_(0, source_group_ranks, torch.ones_like(source_group_ranks))
    if not isinstance(pad_between_seqs, torch.Tensor):
        pad_between_seqs = torch.tensor(
            pad_between_seqs, dtype=torch.bool, device=rank_order_indices.device
        )
    plan_metadata = torch.cat(
        (
            forward_input_split_sizes,
            forward_output_split_sizes,
            (cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]).max().to(torch.int64).view(1),
            pad_between_seqs.to(dtype=torch.int64, device=rank_order_indices.device).view(1),
        )
    ).tolist()
    forward_input_split_sizes = tuple(plan_metadata[:group_size])
    forward_output_split_sizes = tuple(plan_metadata[group_size : 2 * group_size])
    max_seqlen_padded, pad_between_seqs = plan_metadata[-2:]

    return THDCPLayoutPlan(
        contiguous_local_token_count=contiguous_local_token_count,
        zigzag_local_token_count=zigzag_local_token_count,
        cu_seqlens_padded=cu_seqlens_padded,
        max_seqlen_padded=max_seqlen_padded,
        pad_between_seqs=bool(pad_between_seqs),
        forward_send_indices=forward_send_indices,
        forward_receive_positions=forward_receive_positions,
        forward_input_split_sizes=forward_input_split_sizes,
        forward_output_split_sizes=forward_output_split_sizes,
        reverse_send_indices=forward_receive_positions,
        reverse_receive_indices=torch.argsort(forward_send_indices),
    )


def _build_thd_rank_order_indices(
    cu_seqlens: torch.Tensor,
    cu_seqlens_padded: torch.Tensor | None,
    cp_size: int,
    tp_size: int,
    expected_source_token_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build padded dual-chunk attention order without Transformer Engine helpers."""
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be one-dimensional with at least two entries")
    if cu_seqlens.dtype != torch.int32:
        raise ValueError(f"cu_seqlens must use torch.int32, got {cu_seqlens.dtype}")
    if cu_seqlens_padded is not None:
        if cu_seqlens_padded.shape != cu_seqlens.shape:
            raise ValueError("cu_seqlens and cu_seqlens_padded must have the same shape")
        if cu_seqlens_padded.dtype != torch.int32:
            raise ValueError(
                f"cu_seqlens_padded must use torch.int32, got {cu_seqlens_padded.dtype}"
            )
        if cu_seqlens_padded.device != cu_seqlens.device:
            raise ValueError("cu_seqlens and cu_seqlens_padded must be on the same device")
        source_cu_seqlens = cu_seqlens_padded
    else:
        source_cu_seqlens = cu_seqlens

    source_lengths = source_cu_seqlens[1:] - source_cu_seqlens[:-1]

    # The token counts are needed on the host to validate and allocate the route tensors.
    alignment = 2 * cp_size
    target_lengths = (
        torch.div(source_lengths + alignment - 1, alignment, rounding_mode="floor") * alignment
    )
    source_token_count, target_token_count = torch.stack(
        (source_cu_seqlens[-1].to(torch.int64), target_lengths.sum().to(torch.int64))
    ).tolist()
    if (
        expected_source_token_count is not None
        and source_token_count != expected_source_token_count
    ):
        raise ValueError(
            "The packed token count does not match the physical cumulative sequence lengths, got "
            f"{expected_source_token_count} and {source_token_count}"
        )

    # CP attention needs two equal chunks per CP rank. Pad each sequence to that granularity,
    # then add only enough padding to the last sequence to split the batch evenly over TP.
    local_target_token_count = target_token_count // cp_size
    tp_padding = (-local_target_token_count) % tp_size
    if tp_padding:
        target_lengths = target_lengths.clone()
        target_lengths[-1] += tp_padding * cp_size
        target_token_count += tp_padding * cp_size
    target_cu_seqlens = torch.cat(
        (torch.zeros_like(cu_seqlens[:1]), target_lengths.cumsum(dim=0, dtype=torch.int32))
    )

    target_positions = torch.arange(target_token_count, dtype=torch.int64, device=cu_seqlens.device)
    sequence_ids = torch.searchsorted(target_cu_seqlens[1:], target_positions, right=True)
    target_sequence_starts = target_cu_seqlens.index_select(0, sequence_ids).to(torch.int64)
    offsets = target_positions - target_sequence_starts
    padded_lengths = target_lengths.index_select(0, sequence_ids).to(torch.int64)
    chunk_lengths = torch.div(padded_lengths, 2 * cp_size, rounding_mode="floor")
    chunk_ids = torch.div(offsets, chunk_lengths, rounding_mode="floor")
    offsets_in_chunk = torch.remainder(offsets, chunk_lengths)

    destination_cp_ranks = torch.minimum(chunk_ids, 2 * cp_size - chunk_ids - 1)
    chunk_slots = (chunk_ids >= cp_size).to(torch.int64)
    target_local_positions = (
        torch.div(target_sequence_starts, cp_size, rounding_mode="floor")
        + chunk_slots * chunk_lengths
        + offsets_in_chunk
    )
    rank_order_positions = (
        destination_cp_ranks * (target_token_count // cp_size) + target_local_positions
    )

    source_lengths_by_position = source_lengths.index_select(0, sequence_ids).to(torch.int64)
    source_sequence_starts = source_cu_seqlens.index_select(0, sequence_ids).to(torch.int64)
    source_indices = torch.where(
        offsets < source_lengths_by_position,
        source_sequence_starts + offsets,
        torch.full_like(offsets, -1),
    )
    rank_order_indices = torch.empty_like(source_indices)
    rank_order_indices.scatter_(0, rank_order_positions, source_indices)
    return rank_order_indices, target_cu_seqlens


def build_thd_cp_layout_plan(
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    cp_group: torch.distributed.ProcessGroup,
    cu_seqlens_padded: torch.Tensor | None = None,
    sequence_parallel: bool = False,
    tp_group: torch.distributed.ProcessGroup | None = None,
    tp_cp_group: torch.distributed.ProcessGroup | None = None,
) -> THDCPLayoutPlan:
    """Build a reusable packed sequence layout plan for attention.

    Args:
        cu_seqlens: Global cumulative actual sequence lengths in ``torch.int32``.
        total_tokens: Global token count in the contiguous residual stream.
        cp_group: Context-parallel process group.
        cu_seqlens_padded: Optional physical offsets already present in the input.
        sequence_parallel: Whether the residual stream is also sharded over TP ranks.
        tp_group: Tensor-parallel process group, required with sequence parallelism.
        tp_cp_group: Combined TP x CP process group, required when TP size is greater than one.

    Returns:
        A rank-local plan reusable for both directions of the layout conversion.
    """
    context = _get_layout_parallel_context(cp_group, sequence_parallel, tp_group, tp_cp_group)
    if total_tokens % context.group_size != 0:
        raise ValueError(
            "The packed token count must be divisible by CP * TP, got "
            f"{total_tokens}, CP size {context.cp_size}, and TP size {context.tp_size}"
        )

    rank_order_indices, target_cu_seqlens_padded = _build_thd_rank_order_indices(
        cu_seqlens,
        cu_seqlens_padded,
        context.cp_size,
        context.tp_size,
        expected_source_token_count=total_tokens,
    )
    pad_between_seqs = torch.any(cu_seqlens != target_cu_seqlens_padded)
    return _build_thd_cp_layout_plan_from_rank_order_indices(
        rank_order_indices,
        source_token_count=total_tokens,
        cu_seqlens_padded=target_cu_seqlens_padded,
        cp_size=context.cp_size,
        cp_rank=context.cp_rank,
        tp_size=context.tp_size,
        tp_rank=context.tp_rank,
        group_rank_by_logical_rank=context.group_rank_by_logical_rank,
        pad_between_seqs=pad_between_seqs,
    )


def _redistribute_layout(
    input_: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
    source_layout: CPLayout,
    target_layout: CPLayout,
    sequence_parallel: bool,
    tp_group: torch.distributed.ProcessGroup | None,
    tp_cp_group: torch.distributed.ProcessGroup | None,
) -> torch.Tensor:
    """Redistribute local sequence segments with a differentiable all-to-all-v."""
    if cp_group.size() == 1 or source_layout == target_layout:
        return input_
    context = _get_layout_parallel_context(cp_group, sequence_parallel, tp_group, tp_cp_group)

    plan = _build_layout_redistribution_plan(
        source_layout=source_layout,
        target_layout=target_layout,
        cp_size=context.cp_size,
        cp_rank=context.cp_rank,
        tp_size=context.tp_size,
        tp_rank=context.tp_rank,
        group_rank_by_logical_rank=context.group_rank_by_logical_rank,
    )

    input_contiguous = input_.contiguous()
    local_seq_len = input_contiguous.shape[0]
    local_segment_count = _segments_per_rank(context.tp_size)
    if local_seq_len % local_segment_count != 0:
        raise ValueError(
            "CP layout conversion requires the sequence length local to each TP x CP rank to be "
            f"divisible by {local_segment_count}, got {local_seq_len}"
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
    received = all_to_all(
        context.communication_group,
        send_buffer,
        output_split_sizes_=output_split_sizes,
        input_split_sizes=input_split_sizes,
    )

    received_segments = received.reshape(segment_shape)
    if plan.receive_permutation == tuple(range(local_segment_count)):
        output = received
    else:
        output = received_segments.flip(0).reshape(input_contiguous.shape)
    return output.contiguous()


def _redistribute_thd_layout(
    input_: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
    plan: THDCPLayoutPlan,
    source_layout: CPLayout,
    target_layout: CPLayout,
    sequence_parallel: bool,
    tp_group: torch.distributed.ProcessGroup | None,
    tp_cp_group: torch.distributed.ProcessGroup | None,
) -> torch.Tensor:
    """Apply a packed sequence layout plan with a differentiable all-to-all-v."""
    if source_layout == target_layout:
        return input_
    context = _get_layout_parallel_context(cp_group, sequence_parallel, tp_group, tp_cp_group)
    expected_token_count = (
        plan.contiguous_local_token_count
        if source_layout == "contiguous"
        else plan.zigzag_local_token_count
    )
    if input_.shape[0] != expected_token_count:
        raise ValueError(
            "The local packed token count does not match the THD layout plan, got "
            f"{input_.shape[0]} and {expected_token_count}"
        )
    if len(plan.forward_input_split_sizes) != context.communication_group.size():
        raise ValueError("The THD layout plan was built for a different process group size")
    if plan.forward_send_indices.device != input_.device:
        raise ValueError("The THD layout plan and input must be on the same device")

    if source_layout == "contiguous" and target_layout == "zigzag":
        send_indices = plan.forward_send_indices
        input_split_sizes = plan.forward_input_split_sizes
        output_split_sizes = plan.forward_output_split_sizes
    elif source_layout == "zigzag" and target_layout == "contiguous":
        send_indices = plan.reverse_send_indices
        input_split_sizes = plan.forward_output_split_sizes
        output_split_sizes = plan.forward_input_split_sizes
    else:
        raise ValueError(f"Unsupported THD layout conversion: {source_layout} to {target_layout}")

    send_buffer = input_.index_select(0, send_indices)
    received = all_to_all(
        context.communication_group,
        send_buffer,
        output_split_sizes_=list(output_split_sizes),
        input_split_sizes=list(input_split_sizes),
    )
    if source_layout == "contiguous":
        output = received.new_zeros((plan.zigzag_local_token_count, *received.shape[1:]))
        output.index_copy_(0, plan.forward_receive_positions, received)
        return output
    return received.index_select(0, plan.reverse_receive_indices)


def contiguous_to_zigzag(
    input_: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
    sequence_parallel: bool = False,
    tp_group: torch.distributed.ProcessGroup | None = None,
    tp_cp_group: torch.distributed.ProcessGroup | None = None,
    thd_plan: THDCPLayoutPlan | None = None,
) -> torch.Tensor:
    """Convert contiguous CP sequence shards to Megatron's zigzag attention layout."""
    if thd_plan is not None:
        return _redistribute_thd_layout(
            input_,
            cp_group,
            thd_plan,
            "contiguous",
            "zigzag",
            sequence_parallel,
            tp_group,
            tp_cp_group,
        )
    return _redistribute_layout(
        input_, cp_group, "contiguous", "zigzag", sequence_parallel, tp_group, tp_cp_group
    )


def zigzag_to_contiguous(
    input_: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
    sequence_parallel: bool = False,
    tp_group: torch.distributed.ProcessGroup | None = None,
    tp_cp_group: torch.distributed.ProcessGroup | None = None,
    thd_plan: THDCPLayoutPlan | None = None,
) -> torch.Tensor:
    """Convert Megatron's zigzag attention shards back to contiguous CP shards."""
    if thd_plan is not None:
        return _redistribute_thd_layout(
            input_,
            cp_group,
            thd_plan,
            "zigzag",
            "contiguous",
            sequence_parallel,
            tp_group,
            tp_cp_group,
        )
    return _redistribute_layout(
        input_, cp_group, "zigzag", "contiguous", sequence_parallel, tp_group, tp_cp_group
    )


@dataclass(eq=False)
class ContextParallelLayoutManager:
    """Manage CP layout transitions across a sequence of layers."""

    layer_layouts: tuple[CPLayout, ...]
    boundary_layout: CPLayout
    sequence_parallel: bool
    cp_group: torch.distributed.ProcessGroup
    tp_group: torch.distributed.ProcessGroup | None
    tp_cp_group: torch.distributed.ProcessGroup | None
    requires_conversion: bool = field(init=False)

    def __post_init__(self) -> None:
        """Determine whether the layer sequence needs layout conversion."""
        self.requires_conversion = self.cp_group.size() > 1 and any(
            layout != self.boundary_layout for layout in self.layer_layouts
        )

    def _convert_cp_layout(
        self,
        hidden_states: torch.Tensor,
        source_layout: CPLayout,
        target_layout: CPLayout,
        thd_plan: THDCPLayoutPlan | None = None,
    ) -> torch.Tensor:
        if not self.requires_conversion or source_layout == target_layout:
            return hidden_states
        if source_layout == "contiguous" and target_layout == "zigzag":
            return contiguous_to_zigzag(
                hidden_states,
                self.cp_group,
                self.sequence_parallel,
                self.tp_group,
                self.tp_cp_group,
                thd_plan,
            )
        if source_layout == "zigzag" and target_layout == "contiguous":
            return zigzag_to_contiguous(
                hidden_states,
                self.cp_group,
                self.sequence_parallel,
                self.tp_group,
                self.tp_cp_group,
                thd_plan,
            )
        raise ValueError(f"Unsupported CP layout conversion: {source_layout} to {target_layout}")

    def prepare_layer_input(
        self, layer_index: int, hidden_states: torch.Tensor, thd_plan: THDCPLayoutPlan | None = None
    ) -> torch.Tensor:
        """Convert when a layer requires a different layout than its predecessor."""
        source_layout = (
            self.boundary_layout if layer_index == 0 else self.layer_layouts[layer_index - 1]
        )
        return self._convert_cp_layout(
            hidden_states, source_layout, self.layer_layouts[layer_index], thd_plan
        )

    def finalize_layer_output(
        self, layer_index: int, hidden_states: torch.Tensor, thd_plan: THDCPLayoutPlan | None = None
    ) -> torch.Tensor:
        """Restore the boundary layout after the final layer."""
        if layer_index != len(self.layer_layouts) - 1:
            return hidden_states
        return self._convert_cp_layout(
            hidden_states, self.layer_layouts[layer_index], self.boundary_layout, thd_plan
        )

    def build_packed_zigzag_layout(
        self, packed_seq_params: PackedSeqParams
    ) -> tuple[THDCPLayoutPlan, PackedSeqParams]:
        """Build one THD conversion plan and its zigzag-layout metadata."""
        if packed_seq_params.qkv_format != "thd":
            raise ValueError(
                "Packed CP layout conversion requires packed_seq_params.qkv_format='thd'"
            )

        cu_seqlens = packed_seq_params.cu_seqlens_q
        if cu_seqlens is None:
            raise ValueError("Packed CP layout conversion requires actual query lengths")
        if packed_seq_params.cu_seqlens_kv is not cu_seqlens:
            raise ValueError("Packed CP layout conversion requires shared Q/KV sequence metadata")

        cu_seqlens_padded = packed_seq_params.cu_seqlens_q_padded
        if packed_seq_params.cu_seqlens_kv_padded is not cu_seqlens_padded:
            raise ValueError("Packed CP layout conversion requires shared Q/KV padding metadata")

        total_tokens = packed_seq_params.total_tokens
        if total_tokens is None:
            raise ValueError("Packed CP layout conversion requires total_tokens")
        plan = build_thd_cp_layout_plan(
            cu_seqlens,
            total_tokens,
            self.cp_group,
            cu_seqlens_padded=cu_seqlens_padded,
            sequence_parallel=self.sequence_parallel,
            tp_group=self.tp_group,
            tp_cp_group=self.tp_cp_group,
        )
        zigzag_packed_seq_params = replace(
            packed_seq_params,
            cu_seqlens_q_padded=plan.cu_seqlens_padded,
            cu_seqlens_kv_padded=plan.cu_seqlens_padded,
            max_seqlen_q=plan.max_seqlen_padded,
            max_seqlen_kv=plan.max_seqlen_padded,
            total_tokens=None,
            seq_idx=None,
            pad_between_seqs=plan.pad_between_seqs,
        )
        return plan, zigzag_packed_seq_params

    def build_forward_state(
        self, packed_seq_params: PackedSeqParams | None
    ) -> "ContextParallelLayoutState | None":
        """Build the layout state for one forward pass."""
        if not self.requires_conversion:
            return None

        thd_plan = None
        zigzag_packed_seq_params = packed_seq_params
        if packed_seq_params is not None:
            thd_plan, zigzag_packed_seq_params = self.build_packed_zigzag_layout(packed_seq_params)

        return ContextParallelLayoutState(
            manager=self,
            thd_plan=thd_plan,
            contiguous_packed_seq_params=packed_seq_params,
            zigzag_packed_seq_params=zigzag_packed_seq_params,
        )


@dataclass(eq=False)
class ContextParallelLayoutState:
    """Per-forward state for CP layout conversion."""

    manager: ContextParallelLayoutManager
    thd_plan: THDCPLayoutPlan | None
    contiguous_packed_seq_params: PackedSeqParams | None
    zigzag_packed_seq_params: PackedSeqParams | None

    def prepare_layer(
        self, layer_index: int, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, PackedSeqParams | None]:
        """Prepare a layer's input and matching packed metadata."""
        hidden_states = self.manager.prepare_layer_input(layer_index, hidden_states, self.thd_plan)
        packed_seq_params = (
            self.zigzag_packed_seq_params
            if self.manager.layer_layouts[layer_index] == "zigzag"
            else self.contiguous_packed_seq_params
        )
        return hidden_states, packed_seq_params

    def finalize_layer(self, layer_index: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """Finalize a layer's output layout."""
        return self.manager.finalize_layer_output(layer_index, hidden_states, self.thd_plan)
