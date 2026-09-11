# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Common chunkwise context-parallel interfaces for linear attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

import torch


@dataclass(frozen=True)
class CPForwardUnpackedSummary:
    """One shard's unpacked ``S_out = M S_in + delta_S`` summary."""

    transition: torch.Tensor
    state_update: torch.Tensor


@dataclass(frozen=True)
class CPForwardPackedSummary:
    """One shard's packed ``S_out = M S_in + delta_S`` summary."""

    packed: torch.Tensor


CPForwardSummary = CPForwardUnpackedSummary | CPForwardPackedSummary


@dataclass(frozen=True)
class CPBackwardUnpackedSummary:
    """One shard's unpacked ``dS_in = M^T dS_out + gamma`` summary."""

    transition: torch.Tensor
    local_state_grad: torch.Tensor


@dataclass(frozen=True)
class CPBackwardPackedSummary:
    """One shard's packed ``dS_in = M^T dS_out + gamma`` summary."""

    packed: torch.Tensor


CPBackwardSummary = CPBackwardUnpackedSummary | CPBackwardPackedSummary


@dataclass(frozen=True)
class CPSavedContext:
    """Backend state saved from forward for backward.

    ``metadata`` must not contain tensors; tensor state belongs in ``tensors`` so that the
    variant's autograd adapter can save it with ``ctx.save_for_backward``.
    """

    tensors: tuple[torch.Tensor, ...]
    metadata: object | None = None


@dataclass(frozen=True)
class PackedSequenceCPMetadata:
    """Rank-local metadata for chunkwise CP over packed sequences.

    Args:
        local_seq_idx: Per-token sequence IDs for the local contiguous shard.
        local_cu_seqlens: Sequence boundaries within the local contiguous shard.
        preceding_rank_start: Inclusive start of the forward-summary prefix.
        following_rank_stop: Exclusive end of the backward-summary suffix.
    """

    local_seq_idx: torch.Tensor
    local_cu_seqlens: torch.Tensor
    preceding_rank_start: int
    following_rank_stop: int


CPInputT = TypeVar("CPInputT")
LocalContextT = TypeVar("LocalContextT")
BackwardContextT = TypeVar("BackwardContextT")
CPGradientsT = TypeVar("CPGradientsT")


class LinearAttentionCPBackend(Protocol[CPInputT, LocalContextT, BackwardContextT, CPGradientsT]):
    """Local kernel interface for one contiguous chunkwise-CP shard.

    Megatron owns all process-group operations and passes only selected summary slices to these
    methods.
    """

    def cp_forward_prepare(self, inputs: CPInputT) -> tuple[CPForwardSummary, LocalContextT]:
        """Compute the local state summary with zero incoming state."""
        ...

    def cp_forward_apply(
        self, local_context: LocalContextT, preceding_summaries: CPForwardSummary
    ) -> tuple[torch.Tensor, CPSavedContext]:
        """Compose the preceding summaries and compute the local output."""
        ...

    def cp_backward_prepare(
        self, output_grad: torch.Tensor, saved_context: CPSavedContext
    ) -> tuple[CPBackwardSummary, BackwardContextT]:
        """Compute local ``gamma`` with zero outgoing-state gradient."""
        ...

    def cp_backward_apply(
        self, backward_context: BackwardContextT, following_summaries: CPBackwardSummary
    ) -> CPGradientsT:
        """Compose the reverse-causal suffix and finish the local input gradients."""
        ...


@dataclass(frozen=True)
class CPForwardResult:
    """Results retained by a variant-specific autograd adapter after CP forward."""

    output: torch.Tensor
    saved_context: CPSavedContext


def chunkwise_cp_forward(
    backend: LinearAttentionCPBackend[CPInputT, LocalContextT, BackwardContextT, CPGradientsT],
    inputs: CPInputT,
    cp_group: torch.distributed.ProcessGroup,
    preceding_slice: slice,
) -> CPForwardResult:
    """Gather local summaries and apply the selected causal prefix."""
    local_summary, local_context = backend.cp_forward_prepare(inputs=inputs)
    gathered_summary = _all_gather_forward_summary(local_summary, cp_group)
    preceding_summaries = _slice_forward_summary(gathered_summary, preceding_slice)
    output, saved_context = backend.cp_forward_apply(
        local_context=local_context, preceding_summaries=preceding_summaries
    )

    return CPForwardResult(output=output, saved_context=saved_context)


def chunkwise_cp_backward(
    backend: LinearAttentionCPBackend[CPInputT, LocalContextT, BackwardContextT, CPGradientsT],
    output_grad: torch.Tensor,
    saved_context: CPSavedContext,
    cp_group: torch.distributed.ProcessGroup,
    following_slice: slice,
) -> CPGradientsT:
    """Gather local adjoint summaries and apply the selected reverse-causal suffix."""
    local_summary, backward_context = backend.cp_backward_prepare(
        output_grad=output_grad, saved_context=saved_context
    )
    gathered_summary = _all_gather_backward_summary(local_summary, cp_group)
    following_summaries = _slice_backward_summary(gathered_summary, following_slice)
    return backend.cp_backward_apply(
        backward_context=backward_context, following_summaries=following_summaries
    )


def _all_gather_forward_summary(
    local_summary: CPForwardSummary, cp_group: torch.distributed.ProcessGroup
) -> CPForwardSummary:
    if isinstance(local_summary, CPForwardPackedSummary):
        return CPForwardPackedSummary(_all_gather_tensor(local_summary.packed, cp_group))

    transition, state_update = _all_gather_unpacked_summary(
        local_summary.transition, local_summary.state_update, cp_group
    )
    return CPForwardUnpackedSummary(transition=transition, state_update=state_update)


def _all_gather_backward_summary(
    local_summary: CPBackwardSummary, cp_group: torch.distributed.ProcessGroup
) -> CPBackwardSummary:
    if isinstance(local_summary, CPBackwardPackedSummary):
        return CPBackwardPackedSummary(_all_gather_tensor(local_summary.packed, cp_group))

    transition, local_state_grad = _all_gather_unpacked_summary(
        local_summary.transition, local_summary.local_state_grad, cp_group
    )
    return CPBackwardUnpackedSummary(transition=transition, local_state_grad=local_state_grad)


def _slice_forward_summary(summary: CPForwardSummary, rank_slice: slice) -> CPForwardSummary:
    if isinstance(summary, CPForwardPackedSummary):
        return CPForwardPackedSummary(summary.packed[rank_slice])
    return CPForwardUnpackedSummary(
        transition=summary.transition[rank_slice], state_update=summary.state_update[rank_slice]
    )


def _slice_backward_summary(summary: CPBackwardSummary, rank_slice: slice) -> CPBackwardSummary:
    if isinstance(summary, CPBackwardPackedSummary):
        return CPBackwardPackedSummary(summary.packed[rank_slice])
    return CPBackwardUnpackedSummary(
        transition=summary.transition[rank_slice],
        local_state_grad=summary.local_state_grad[rank_slice],
    )


def build_packed_sequence_cp_metadata(
    global_seq_idx: torch.Tensor, cp_rank: int, cp_size: int
) -> PackedSequenceCPMetadata:
    """Build rank-local packed-sequence metadata for chunkwise CP.

    Args:
        global_seq_idx: Nondecreasing global per-token sequence IDs in ``[1, T]`` layout.
        cp_rank: This rank's position in causal order.
        cp_size: Number of contiguous CP shards.

    Returns:
        The local sequence IDs, local sequence boundaries, and summary slice bounds for this rank.
    """
    if global_seq_idx.ndim != 2 or global_seq_idx.shape[0] != 1:
        raise ValueError("Chunkwise CP packed sequences require global_seq_idx with shape [1, T]")
    if global_seq_idx.dtype != torch.int32:
        raise ValueError(f"global_seq_idx must have dtype torch.int32, got {global_seq_idx.dtype}")
    if global_seq_idx.shape[1] % cp_size != 0:
        raise ValueError(
            "The packed token count must be divisible by the CP size, got "
            f"{global_seq_idx.shape[1]} and {cp_size}"
        )
    if global_seq_idx.shape[1] == 0:
        raise ValueError("Packed chunkwise CP input must contain at least one token")

    local_sequence_length = global_seq_idx.shape[1] // cp_size
    shard_start = cp_rank * local_sequence_length
    shard_stop = shard_start + local_sequence_length
    sequence_ids = global_seq_idx[0]
    local_seq_idx = global_seq_idx[:, shard_start:shard_stop]
    first_sequence_id = local_seq_idx[0, 0]
    last_sequence_id = local_seq_idx[0, -1]
    first_token = torch.searchsorted(sequence_ids, first_sequence_id, right=False)
    last_token = torch.searchsorted(sequence_ids, last_sequence_id, right=True) - 1
    first_token_index, last_token_index = torch.stack((first_token, last_token)).tolist()

    sequence_change = local_seq_idx[:, 1:] != local_seq_idx[:, :-1]
    sequence_starts = torch.nonzero(sequence_change[0], as_tuple=False).flatten() + 1
    local_cu_seqlens = torch.cat(
        (
            sequence_starts.new_zeros(1),
            sequence_starts,
            sequence_starts.new_tensor([local_sequence_length]),
        )
    ).to(torch.int32)
    return PackedSequenceCPMetadata(
        local_seq_idx=local_seq_idx,
        local_cu_seqlens=local_cu_seqlens,
        preceding_rank_start=first_token_index // local_sequence_length,
        following_rank_stop=last_token_index // local_sequence_length + 1,
    )


def _all_gather_unpacked_summary(
    transition: torch.Tensor, state_update: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-gather an unpacked state summary without packing its tensors."""
    _validate_summary_tensor(transition)
    _validate_summary_tensor(state_update)
    if transition.device != state_update.device:
        raise ValueError("All tensors in a CP summary must be on the same device")

    with torch.distributed._coalescing_manager(group=cp_group, device=transition.device):
        gathered_transition = _all_gather_tensor(transition, cp_group)
        gathered_state_update = _all_gather_tensor(state_update, cp_group)
    return gathered_transition, gathered_state_update


def _all_gather_tensor(
    local_tensor: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    _validate_summary_tensor(local_tensor)

    output = torch.empty(
        (cp_group.size(), *local_tensor.shape), dtype=local_tensor.dtype, device=local_tensor.device
    )
    torch.distributed.all_gather_into_tensor(output, local_tensor, group=cp_group)
    return output


def _validate_summary_tensor(tensor: torch.Tensor) -> None:
    """Validate a tensor communicated as part of a CP summary."""
    if tensor.dtype != torch.float32:
        raise ValueError("CP summary tensors must use torch.float32")
    if not tensor.is_contiguous():
        raise ValueError("CP summary tensors must be contiguous")
