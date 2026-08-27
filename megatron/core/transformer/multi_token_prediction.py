# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import warnings
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import torch
from torch import Tensor

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping, replace_prefix_for_sharding
from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.backends import BackendSpecProvider, LocalSpecProvider
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.utils import is_vp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import (
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.tensor_parallel.inference_layers import (
    inference_all_gather_from_tensor_model_parallel_region,
    is_inference_column_parallel_linear,
)
from megatron.core.transformer.enums import AttnMaskType, LayerType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.torch_norm import LayerNormBuilder
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.core.utils import (
    get_pg_rank,
    get_pg_size,
    is_torch_min_version,
    make_tp_sharded_tensor_for_checkpoint,
    make_viewless_tensor,
)

if TYPE_CHECKING:
    from megatron.core.models.hybrid.hybrid_block import HybridStackSubmodules

if is_torch_min_version("1.13.0"):
    dist_all_gather_func = torch.distributed.all_gather_into_tensor
else:
    dist_all_gather_func = torch.distributed._all_gather_base

SUPPORTED_ATTN_MASK = [
    AttnMaskType.padding,
    AttnMaskType.causal,
    AttnMaskType.no_mask,
    AttnMaskType.padding_causal,
]

if HAVE_TE:
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
else:
    TESpecProvider = None

from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout

_HIDDEN_STATE_MIXING_RNG_TRACKER_NAME = 'mtp-hsm-rng'
_HIDDEN_STATE_MIXING_RNG_SEED_OFFSET = 1 << 40


def _initialize_hidden_state_mixing_rng_tracker(
    dp_group: Optional[torch.distributed.ProcessGroup],
) -> Optional[str]:
    """Create one checkpointable hidden-state-mixing RNG stream per DP replica."""
    rng_tracker = tensor_parallel.get_cuda_rng_tracker()
    if not rng_tracker.is_initialized():
        return None
    if _HIDDEN_STATE_MIXING_RNG_TRACKER_NAME not in rng_tracker.get_states():
        seed = (
            torch.cuda.initial_seed() + _HIDDEN_STATE_MIXING_RNG_SEED_OFFSET + get_pg_rank(dp_group)
        ) % (2**63 - 1)
        rng_tracker.add(_HIDDEN_STATE_MIXING_RNG_TRACKER_NAME, seed)
    return _HIDDEN_STATE_MIXING_RNG_TRACKER_NAME


def _mix_hidden_state_history(
    older_hidden_states: Tensor,
    newest_hidden_state: Tensor,
    *,
    sequence_parallel: bool = False,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    rng_tracker_name: Optional[str] = None,
) -> Tensor:
    """Select one accumulated hidden state independently for every token.

    Draw one reproducible mask for this DP replica's full logical sequence, then give
    each SP/CP token owner a disjoint slice. TP ranks with replicated sequence positions
    continue to use the same selections.
    """
    assert older_hidden_states.size(0) > 0, "Hidden State Mixing requires an older state."
    num_older_states = older_hidden_states.size(0)
    num_states = num_older_states + 1

    sequence_length, batch_size, hidden_size = newest_hidden_state.shape
    sequence_parallel_size = get_pg_size(tp_group) if sequence_parallel else 1
    sequence_parallel_rank = get_pg_rank(tp_group) if sequence_parallel else 0
    context_parallel_size = get_pg_size(cp_group)
    context_parallel_rank = get_pg_rank(cp_group)
    sequence_owner_count = sequence_parallel_size * context_parallel_size
    sequence_owner_rank = context_parallel_rank * sequence_parallel_size + sequence_parallel_rank

    rng_tracker = tensor_parallel.get_cuda_rng_tracker()
    rng_context = (
        rng_tracker.fork(rng_tracker_name or tensor_parallel.get_data_parallel_rng_tracker_name())
        if rng_tracker.is_initialized()
        else nullcontext()
    )
    with rng_context:
        all_indices = torch.randint(
            num_states,
            (1, sequence_owner_count * sequence_length, batch_size, 1),
            device=newest_hidden_state.device,
        )
    owner_start = sequence_owner_rank * sequence_length
    indices = all_indices[:, owner_start : owner_start + sequence_length]

    # roll_tensor zeroes positions without a local continuation; use the newest state there.
    selected_is_newest = indices.eq(num_older_states)
    older_indices = indices.clamp_max(num_older_states - 1)
    invalid_locations = older_hidden_states.eq(0).all(dim=-1, keepdim=True)
    selected_is_invalid = torch.gather(invalid_locations, dim=0, index=older_indices)
    use_newest = selected_is_newest | selected_is_invalid
    selected_older = torch.gather(
        older_hidden_states, dim=0, index=older_indices.expand(-1, -1, -1, hidden_size)
    )
    return torch.where(
        use_newest.expand_as(selected_older), newest_hidden_state.unsqueeze(0), selected_older
    ).squeeze(0)


def tie_word_embeddings_state_dict(
    sharded_state_dict: ShardedStateDict,
    word_emb_weight: Tensor,
    word_emb_weight_key: str,
    tp_group: torch.distributed.ProcessGroup,
    dp_cp_group: torch.distributed.ProcessGroup,
) -> None:
    """tie the embedding of the mtp processing stage in a given sharded state dict.

    Args:
        sharded_state_dict (ShardedStateDict): state dict with the weight to tie.
        word_emb_weight (Tensor): weight of the word embedding.
        word_emb_weight_key (str): key of the word embedding in the sharded state dict.
        tp_group (torch.distributed.ProcessGroup): The tensor parallel group
        dp_cp_group (torch.distributed.ProcessGroup): The dp-cp comm group

    Returns: None, acts in-place
    """
    mtp_word_emb_replica_id = (
        1,  # copy of embedding in pre processing stage
        0,
        get_pg_rank(dp_cp_group),
    )
    assert word_emb_weight_key in sharded_state_dict
    del sharded_state_dict[word_emb_weight_key]
    sharded_state_dict[word_emb_weight_key] = make_tp_sharded_tensor_for_checkpoint(
        tensor=word_emb_weight,
        key=word_emb_weight_key,
        replica_id=mtp_word_emb_replica_id,
        allow_shape_mismatch=True,
        tp_group=tp_group,
        dp_cp_group=dp_cp_group,
    )


def tie_output_layer_state_dict(
    sharded_state_dict: ShardedStateDict,
    output_layer_weight: Tensor,
    output_layer_weight_key: str,
    tp_group: torch.distributed.ProcessGroup,
    dp_cp_group: torch.distributed.ProcessGroup,
) -> None:
    """tie the output layer of the mtp processing stage in a given sharded state dict.

    Args:
        sharded_state_dict (ShardedStateDict): state dict with the weight to tie.
        output_layer_weight (Tensor): weight of the output layer.
        output_layer_weight_key (str): key of the output layer in the sharded state dict.
        tp_group (torch.distributed.ProcessGroup): The tensor parallel group
        dp_cp_group (torch.distributed.ProcessGroup): The dp-cp comm group

    Returns: None, acts in-place
    """
    mtp_output_layer_replica_id = (
        1,  # copy of output layer in post processing stage
        0,
        get_pg_rank(dp_cp_group),
    )
    assert output_layer_weight_key in sharded_state_dict
    del sharded_state_dict[output_layer_weight_key]
    sharded_state_dict[output_layer_weight_key] = make_tp_sharded_tensor_for_checkpoint(
        tensor=output_layer_weight,
        key=output_layer_weight_key,
        replica_id=mtp_output_layer_replica_id,
        allow_shape_mismatch=True,
        tp_group=tp_group,
        dp_cp_group=dp_cp_group,
    )


def roll_tensor(tensor, shifts=-1, dims=-1, cp_group=None, packed_seq_params=None, return_sum=True):
    """Roll the tensor input along the sequence dimension with Context Parallelism (CP) support.

    This function extends the original roll_tensor to support Context Parallelism, which allows
    MTP to work with CP > 1. When CP is enabled, the sequence dimension is split across CP ranks,
    and tensor rolling requires communication between adjacent CP ranks to properly handle the
    boundary conditions.

    For CP=1 (default behavior): Uses standard torch.roll with zero padding
    For CP>1: Splits tensor into chunks, performs rolling within each chunk, then exchanges
    boundary elements between adjacent CP ranks to maintain sequence continuity.

    For packed sequences: Respects sequence boundaries when rolling to avoid mixing tokens
    from different sequences.

    Args:
        tensor (Tensor): The input tensor to roll. If None, returns (None, None).
        shifts (int): The shift of the tensor (typically -1 for MTP).
        dims (int): The dimension to roll (typically -1 for sequence dimension).
        cp_group (ProcessGroup): The context parallelism process group. If None or size=1,
                               falls back to standard rolling behavior.
        packed_seq_params (PackedSeqParams): Parameters for packed sequence processing.
                                            If provided, respects sequence boundaries.
        return_sum (bool): Whether to calculate and return the rolled tensor sum.
                           Defaults to True.
    Returns:
        tuple: (rolled_tensor, sum_of_rolled_tensor). The sum is None when disabled.
    """
    if tensor is None:
        return None, None

    # Handle packed sequences cases
    if packed_seq_params is not None:
        return _roll_tensor_packed_seq(
            tensor, shifts, dims, packed_seq_params, cp_group, return_sum=return_sum
        )

    # Standard rolling behavior when CP is not enabled (cp_group is None or size=1)
    if cp_group is None or cp_group.size() == 1:
        rolled_tensor = torch.roll(tensor, shifts=shifts, dims=dims)
        rolled_tensor.select(dims, shifts).fill_(0)
        rolled_sum = rolled_tensor.sum() if return_sum else None
        return rolled_tensor, rolled_sum

    # CP-enabled rolling: Split tensor into chunks and handle boundary communication
    # This matches the batch splitting logic in get_batch_on_this_cp_rank() function
    tensor_list = tensor.chunk(2, dim=dims)
    rolled_tensor_list = []
    for i in range(len(tensor_list)):
        rolled_tensor_list.append(torch.roll(tensor_list[i], shifts=shifts, dims=dims))

    # Prepare tensors for communication between CP ranks
    # Each CP rank needs to send boundary elements to adjacent ranks
    tensor_send_list = []
    tensor_recv_list = []
    for i in range(len(rolled_tensor_list)):
        tensor_send_list.append(rolled_tensor_list[i].select(dims, shifts).contiguous())
        empty_tensor = torch.empty(
            tensor_send_list[i].shape,
            dtype=tensor_send_list[i].dtype,
            device=torch.cuda.current_device(),
        )
        tensor_recv_list.append(empty_tensor)

    # Get the global rank of next and prev process in the cp group
    global_ranks = torch.distributed.get_process_group_ranks(group=cp_group)
    local_rank = torch.distributed.get_rank(group=cp_group)
    next_rank = global_ranks[(local_rank + 1) % len(global_ranks)]
    prev_rank = global_ranks[(local_rank - 1) % len(global_ranks)]

    # Start send and recv ops
    ops = []
    if local_rank != 0:
        req_send_first_part = torch.distributed.isend(tensor=tensor_send_list[0], dst=prev_rank)
        ops.append(req_send_first_part)
        req_recv_second_part = torch.distributed.irecv(tensor=tensor_recv_list[1], src=prev_rank)
        ops.append(req_recv_second_part)
    else:
        # Inserted elements are set to be 0.0.
        tensor_recv_list[1] = 0
    if local_rank != len(global_ranks) - 1:
        req_recv_first_part = torch.distributed.irecv(tensor=tensor_recv_list[0], src=next_rank)
        ops.append(req_recv_first_part)
        req_send_second_part = torch.distributed.isend(tensor=tensor_send_list[1], dst=next_rank)
        ops.append(req_send_second_part)
    else:
        # For the last CP rank, the removed elements of second part go into the first part
        tensor_recv_list[0] = tensor_send_list[1]

    # Wait for all communication operations to complete
    for op in ops:
        op.wait()

    # Splicing: Replace boundary elements with received elements from adjacent ranks
    # This ensures proper sequence continuity across CP boundaries
    index = [slice(None)] * rolled_tensor_list[0].dim()
    index[dims] = shifts
    for i in range(len(rolled_tensor_list)):
        rolled_tensor_list[i][tuple(index)] = tensor_recv_list[i]

    # Concatenate the processed chunks back into a single tensor
    rolled_tensor = torch.cat(rolled_tensor_list, dim=dims)

    rolled_sum = rolled_tensor.sum() if return_sum else None
    return rolled_tensor, rolled_sum


def _roll_tensor_packed_seq(
    tensor, shifts, dims, packed_seq_params, cp_group=None, return_sum=True
):
    """Roll tensor with packed sequence support.
    This function handles rolling for packed sequences by respecting sequence boundaries
    """

    # Notice: This is a naive implementation to test the correctness,
    # a better solution will only sync the boundary tokens once.
    assert (
        dims == -1 or dims == tensor.dim() - 1
    ), "Packed sequence roll only supports the last dimension."
    assert shifts == -1, "Packed sequence roll only supports a single-token left shift."
    cu_seqlens = packed_seq_params.cu_seqlens_q
    assert cu_seqlens is not None, "Packed sequence parameters must provide cu_seqlens_q."

    rolled_tensor = tensor.clone()

    cp_size = cp_group.size() if cp_group is not None else 1
    if cp_size == 1:
        # CP disabled: roll each packed sequence independently within its boundaries
        for i in range(len(cu_seqlens) - 1):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i + 1]
            seq_slice = tensor[..., start_idx:end_idx]
            rolled_seq = torch.roll(seq_slice, shifts=shifts, dims=dims)
            # Zero out the last position(s) that would cross sequence boundaries
            rolled_seq[..., shifts:] = 0
            rolled_tensor[..., start_idx:end_idx] = rolled_seq
        rolled_sum = rolled_tensor.sum() if return_sum else None
        return rolled_tensor, rolled_sum

    # CP enabled: each rank owns two chunks per sequence (front and mirrored tail).
    local_rank = torch.distributed.get_rank(group=cp_group)
    global_ranks = torch.distributed.get_process_group_ranks(group=cp_group)
    next_rank = global_ranks[(local_rank + 1) % cp_size]
    prev_rank = global_ranks[(local_rank - 1) % cp_size]

    # Iterate over each sequence individually
    for i in range(len(cu_seqlens) - 1):
        start_idx = cu_seqlens[i]
        end_idx = cu_seqlens[i + 1]

        # the idx has been multiplied by cp_size, need to divide it by cp_size to get the local idx
        local_start_idx = start_idx // cp_size
        local_end_idx = end_idx // cp_size

        # Skip empty sequences - this can happen when a sequence is very short and
        # after dividing by cp_size, the local slice has zero length
        local_seq_len = local_end_idx - local_start_idx
        if local_seq_len == 0:
            continue

        tensor_slice = rolled_tensor[..., local_start_idx:local_end_idx].clone()

        # The following code is very similar as the code in roll_tensor function
        local_chunks = tensor_slice.chunk(2, dim=dims)
        rolled_chunks = [torch.roll(chunk, shifts=shifts, dims=dims) for chunk in local_chunks]

        tensor_send_list = []
        tensor_recv_list = []
        for chunk in rolled_chunks:
            # Skip empty chunks that can occur when the sequence slice is very small
            if chunk.size(dims) == 0:
                tensor_send_list.append(
                    torch.empty(chunk.shape[:-1], dtype=chunk.dtype, device=chunk.device)
                )
                tensor_recv_list.append(
                    torch.empty(chunk.shape[:-1], dtype=chunk.dtype, device=chunk.device)
                )
                continue
            boundary = chunk.select(dims, shifts).contiguous().clone()
            tensor_send_list.append(boundary)
            tensor_recv_list.append(torch.empty_like(boundary))

        ops = []
        if local_rank != 0:
            ops.append(torch.distributed.isend(tensor=tensor_send_list[0], dst=prev_rank))
            ops.append(torch.distributed.irecv(tensor=tensor_recv_list[1], src=prev_rank))
        else:
            tensor_recv_list[1].zero_()

        if local_rank != cp_size - 1:
            ops.append(torch.distributed.irecv(tensor=tensor_recv_list[0], src=next_rank))
            ops.append(torch.distributed.isend(tensor=tensor_send_list[1], dst=next_rank))
        else:
            tensor_recv_list[0].copy_(tensor_send_list[1])

        for op in ops:
            op.wait()

        index = [slice(None)] * rolled_chunks[0].dim()
        index[dims] = shifts
        for chunk, recv in zip(rolled_chunks, tensor_recv_list):
            # Skip empty chunks
            if chunk.size(dims) == 0:
                continue
            chunk[tuple(index)] = recv

        seq_result = torch.cat(rolled_chunks, dim=dims)

        # update the rolled tensor
        rolled_tensor[..., local_start_idx:local_end_idx] = seq_result

    rolled_sum = rolled_tensor.sum() if return_sum else None
    return rolled_tensor, rolled_sum


def _packed_seq_params_for_local_hsm_roll(
    packed_seq_params: PackedSeqParams,
    local_seq_length: int,
    cp_group: Optional[torch.distributed.ProcessGroup],
    tp_group: Optional[torch.distributed.ProcessGroup],
) -> Optional[PackedSeqParams]:
    """Re-express packed document boundaries in this HSM roll's local frame.

    ``cu_seqlens`` is global, while a sequence-parallel rank holds a 1/tp slice of the
    context-parallel-local sequence. ``_roll_tensor_packed_seq`` maps global to local by
    dividing by ``cp_size`` alone -- a scale with no offset, so it is correct only for
    the rank whose slice starts at zero. Dividing and then subtracting where this slice
    starts fixes that: documents keep their order and stay contiguous in CP-local space,
    so their intersections with a contiguous slice tile it exactly. Documents outside
    the slice collapse to zero length, which the roll already skips.

    The padded boundaries define physical CP ownership, while the unpadded boundaries
    identify valid-token ends. Both become local roll boundaries; the padded variants
    and ``total_tokens`` are then cleared because their global coordinates no longer match.

    Returns:
        Translated params, or None when the boundaries cannot be translated, in which
        case the caller should pass the originals through unchanged.
    """
    cu_seqlens = packed_seq_params.cu_seqlens_q
    if cu_seqlens is None:
        return None
    padded = packed_seq_params.cu_seqlens_q_padded
    if padded is not None and padded is not cu_seqlens:
        cp_size = get_pg_size(cp_group)
        cp_rank = get_pg_rank(cp_group)
        boundaries = [padded.new_zeros(())]
        local_offset = padded.new_zeros(())
        for doc_idx in range(len(cu_seqlens) - 1):
            padded_length = padded[doc_idx + 1] - padded[doc_idx]
            valid_length = cu_seqlens[doc_idx + 1] - cu_seqlens[doc_idx]
            chunk_length = torch.div(padded_length, 2 * cp_size, rounding_mode='floor')
            if cp_rank == cp_size - 1:
                owned_length = 2 * chunk_length
                valid = torch.minimum(
                    (valid_length - cp_rank * chunk_length).clamp_min(0), owned_length
                )
                boundaries.extend([local_offset + valid, local_offset + owned_length])
                local_offset = local_offset + owned_length
                continue
            starts = (cp_rank * chunk_length, (2 * cp_size - cp_rank - 1) * chunk_length)
            for start in starts:
                valid = torch.minimum((valid_length - start).clamp_min(0), chunk_length)
                boundaries.extend([local_offset + valid, local_offset + chunk_length])
                local_offset = local_offset + chunk_length
        cp_local = torch.stack(boundaries).unique_consecutive()
        add_zigzag_midpoints = False
    else:
        cp_size = get_pg_size(cp_group)
        cp_local = (
            torch.div(cu_seqlens, cp_size, rounding_mode='floor') if cp_size > 1 else cu_seqlens
        )
        add_zigzag_midpoints = cp_size > 1 and get_pg_rank(cp_group) != cp_size - 1

    window_start = get_pg_rank(tp_group) * local_seq_length
    # A document's local slots are its two zigzag chunks, which are adjacent locally but
    # usually far apart globally, so the midpoint between them is a boundary too: the
    # token after the front chunk's last one lives on another rank, not in the next local
    # slot. Splitting there is what roll_tensor's CP branch used chunk(2) for.
    #
    # The exception is the last CP rank. The zigzag hands rank r chunks r and
    # 2 * cp_size - 1 - r, which for r == cp_size - 1 are chunks cp_size - 1 and cp_size:
    # neighbours. Its local piece really is contiguous, and splitting it there would
    # blank a slot whose continuation is sitting right next to it.
    if add_zigzag_midpoints:
        midpoints = torch.div(cp_local[:-1] + cp_local[1:], 2, rounding_mode='floor')
        cp_local = torch.cat(
            [torch.stack([cp_local[:-1], midpoints], dim=1).flatten(), cp_local[-1:]]
        )
    shard_local = cp_local.clamp(window_start, window_start + local_seq_length) - window_start
    return replace(
        packed_seq_params,
        cu_seqlens_q=shard_local,
        cu_seqlens_kv=shard_local,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv_padded=None,
        total_tokens=None,
        seq_idx=None,
    )


class MTPLossLoggingHelper:
    """Helper class for logging MTP losses and acceptance rates."""

    tracker = {}

    @staticmethod
    def save_metrics_to_tracker(
        loss: torch.Tensor,
        correct: torch.Tensor,
        total: torch.Tensor,
        layer_number: int,
        num_layers: int,
        reduce_group: torch.distributed.ProcessGroup = None,
        avg_group: torch.distributed.ProcessGroup = None,
    ):
        """Save the mtp metrics (loss, correct, total) for logging.

        Args:
            loss (torch.Tensor): The normalized loss value for this MTP layer.
            correct (torch.Tensor): Number of correct predictions.
            total (torch.Tensor): Total number of predictions.
            layer_number (int): Layer index of the loss.
            num_layers (int): The number of total layers.
            reduce_group (torch.distributed.ProcessGroup): The group for reducing the loss.
            avg_group (torch.distributed.ProcessGroup): The group for averaging the loss.
        """
        # Skip mtp loss logging if layer_number is None.
        if layer_number is None:
            return

        tracker = MTPLossLoggingHelper.tracker
        if "loss_values" not in tracker:
            tracker["loss_values"] = torch.zeros(num_layers, device=torch.cuda.current_device())
        if "correct_values" not in tracker:
            tracker["correct_values"] = torch.zeros(num_layers, device=torch.cuda.current_device())
        if "total_values" not in tracker:
            tracker["total_values"] = torch.zeros(num_layers, device=torch.cuda.current_device())

        tracker["loss_values"][layer_number] += loss.detach()
        tracker["correct_values"][layer_number] += correct.detach()
        tracker["total_values"][layer_number] += total.detach()
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    @staticmethod
    def clean_metrics_in_tracker():
        """Clear the mtp metrics."""
        tracker = MTPLossLoggingHelper.tracker
        if "loss_values" in tracker:
            tracker["loss_values"].zero_()
        if "correct_values" in tracker:
            tracker["correct_values"].zero_()
        if "total_values" in tracker:
            tracker["total_values"].zero_()
        tracker["reduce_group"] = None
        tracker["avg_group"] = None

    @staticmethod
    def reduce_metrics_in_tracker():
        """Collect and reduce the mtp metrics across ranks."""
        tracker = MTPLossLoggingHelper.tracker
        if "loss_values" not in tracker:
            return

        loss_values = tracker["loss_values"]
        if tracker.get('reduce_group') is not None:
            torch.distributed.all_reduce(loss_values, group=tracker.get('reduce_group'))
        if tracker.get('avg_group') is not None:
            torch.distributed.all_reduce(
                loss_values, group=tracker['avg_group'], op=torch.distributed.ReduceOp.AVG
            )

        for key in ["correct_values", "total_values"]:
            if key not in tracker:
                continue
            values = tracker[key]
            if tracker.get('reduce_group') is not None:
                torch.distributed.all_reduce(values, group=tracker.get('reduce_group'))
            if tracker.get('avg_group') is not None:
                torch.distributed.all_reduce(
                    values, group=tracker['avg_group'], op=torch.distributed.ReduceOp.SUM
                )

    @staticmethod
    def track_mtp_metrics(loss_scale, iteration, writer, wandb_writer=None, total_loss_dict=None):
        """Track the Multi-Token Prediction (MTP) metrics for logging."""
        MTPLossLoggingHelper.reduce_metrics_in_tracker()
        tracker = MTPLossLoggingHelper.tracker
        if "loss_values" not in tracker:
            return

        mtp_losses = tracker["loss_values"] * loss_scale
        mtp_corrects = tracker.get("correct_values", torch.zeros_like(mtp_losses))
        mtp_totals = tracker.get("total_values", torch.ones_like(mtp_losses))

        # Process-local logging state; cumulative rates intentionally reset after restart/resume.
        if (
            "cumulative_correct_values" not in tracker
            or tracker["cumulative_correct_values"].shape != mtp_corrects.shape
        ):
            tracker["cumulative_correct_values"] = torch.zeros_like(mtp_corrects)
        if (
            "cumulative_total_values" not in tracker
            or tracker["cumulative_total_values"].shape != mtp_totals.shape
        ):
            tracker["cumulative_total_values"] = torch.zeros_like(mtp_totals)

        tracker["cumulative_correct_values"] += mtp_corrects
        tracker["cumulative_total_values"] += mtp_totals
        mtp_cumulative_corrects = tracker["cumulative_correct_values"]
        mtp_cumulative_totals = tracker["cumulative_total_values"]

        mtp_num_layers = mtp_losses.shape[0]
        for i in range(mtp_num_layers):
            loss_name = f"mtp_{i+1} loss"
            step_acc_name = f"mtp_{i+1}_acceptance_rate"
            cum_acc_name = f"mtp_{i+1}_cumulative_acceptance_rate"

            loss = mtp_losses[i]
            # Empty masks can leave no valid MTP positions, so clamp denominators to avoid NaNs.
            step_rate = (mtp_corrects[i] / torch.clamp(mtp_totals[i], min=1)) * 100.0
            cum_rate = (
                mtp_cumulative_corrects[i] / torch.clamp(mtp_cumulative_totals[i], min=1)
            ) * 100.0

            if total_loss_dict is not None:
                total_loss_dict[loss_name] = (
                    total_loss_dict.get(loss_name, torch.zeros_like(loss)) + loss
                )

            if writer is not None:
                writer.add_scalar(loss_name, loss, iteration)
                writer.add_scalar(step_acc_name, step_rate, iteration)
                writer.add_scalar(cum_acc_name, cum_rate, iteration)
            if wandb_writer is not None:
                wandb_writer.log({f"{loss_name}": loss}, iteration)
                wandb_writer.log({f"{step_acc_name}": step_rate}, iteration)
                wandb_writer.log({f"{cum_acc_name}": cum_rate}, iteration)

        MTPLossLoggingHelper.clean_metrics_in_tracker()


def _mtp_logits_are_vocab_sharded(
    output_layer: Callable, runtime_gather_output: Optional[bool]
) -> bool:
    """Return whether MTP logits are still vocab-sharded across tensor-parallel ranks."""
    if runtime_gather_output is not None:
        return not runtime_gather_output
    return not getattr(output_layer, "gather_output", False)


def _vocab_parallel_argmax(
    vocab_parallel_logits: Tensor, tp_group: torch.distributed.ProcessGroup, tp_size: int
) -> Tensor:
    """Return global argmax ids from logits sharded across the vocab dimension."""
    vocab_shard_size = vocab_parallel_logits.size(-1)
    local_max_vals, local_argmax = vocab_parallel_logits.max(dim=-1)  # [s, b], [s, b]

    gathered_max_vals = [torch.empty_like(local_max_vals) for _ in range(tp_size)]
    gathered_argmax = [torch.empty_like(local_argmax) for _ in range(tp_size)]
    torch.distributed.all_gather(gathered_max_vals, local_max_vals, group=tp_group)
    torch.distributed.all_gather(gathered_argmax, local_argmax, group=tp_group)

    stacked_max_vals = torch.stack(gathered_max_vals, dim=0)
    stacked_argmax = torch.stack(gathered_argmax, dim=0)
    winning_rank = stacked_max_vals.argmax(dim=0)  # [s, b]
    winning_local_argmax = torch.gather(stacked_argmax, 0, winning_rank.unsqueeze(0)).squeeze(
        0
    )  # [s, b]
    return winning_rank * vocab_shard_size + winning_local_argmax  # [s, b]


def _compute_mtp_acceptance_counts(
    mtp_logits: Tensor,
    mtp_labels: Tensor,
    loss_mask: Tensor,
    output_layer: Callable,
    runtime_gather_output: Optional[bool],
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> tuple[Tensor, Tensor]:
    """Compute MTP acceptance correct/total counts."""
    with torch.no_grad():
        logits_are_vocab_sharded = _mtp_logits_are_vocab_sharded(
            output_layer, runtime_gather_output
        )
        if (
            tp_group is None
            and logits_are_vocab_sharded
            and parallel_state.is_initialized()
            and parallel_state.get_tensor_model_parallel_world_size() > 1
        ):
            raise ValueError(
                "tp_group must be provided when computing MTP acceptance counts "
                "from vocab-sharded logits under tensor model parallelism."
            )
        tp_size = torch.distributed.get_world_size(group=tp_group) if tp_group is not None else 1

        # Apply TP rank offsets only when logits are vocab-sharded; gathered logits already
        # contain global vocab ids in their last dimension.
        if tp_group is not None and tp_size > 1 and logits_are_vocab_sharded:
            preds = _vocab_parallel_argmax(mtp_logits, tp_group, tp_size)
        else:
            preds = torch.argmax(mtp_logits, dim=-1)  # [s, b]

        labels_match = mtp_labels.transpose(0, 1).contiguous()  # [b, s] => [s, b]
        mask_match = loss_mask.transpose(0, 1).contiguous()  # [b, s] => [s, b]
        valid_positions = mask_match.bool()
        correct = ((preds == labels_match) & valid_positions).sum().float()
        total = valid_positions.sum().float()

    return correct, total


@dataclass
class MultiTokenPredictionLayerSubmodules:
    """
    Dataclass for specifying the submodules of a MultiTokenPrediction module.

    Args:
        hnorm: Specification or instance of the hidden states normalization to be applied.
        enorm: Specification or instance of the embedding normalization to be applied.
        eh_proj (Union[ModuleSpec, type]): Specification or instance of the
            linear projection to be applied.
        mtp_model_layer (Union[ModuleSpec, type]): Specification
            or instance of the transformer or mamba block to be applied.
    """

    enorm: LayerNormBuilder
    hnorm: LayerNormBuilder
    # TODO(nschank): Move this back below transformer_layer once eh_proj and transformer_layer have
    # their defaults removed.
    layer_norm: LayerNormBuilder

    eh_proj: Union[ModuleSpec, type] = None
    mtp_model_layer: Union[ModuleSpec, type] = None


def get_mtp_layer_spec(
    mtp_model_layer_spec: ModuleSpec, use_transformer_engine: bool
) -> ModuleSpec:
    """Get the MTP layer spec.

    Returns:
        ModuleSpec: Module specification with TE modules
    """
    return get_mtp_layer_spec_for_backend(
        mtp_model_layer_spec,
        backend=TESpecProvider() if use_transformer_engine else LocalSpecProvider(),
    )


def get_mtp_layer_spec_for_backend(
    mtp_model_layer_spec: ModuleSpec, backend: BackendSpecProvider
) -> ModuleSpec:
    """Get the MTP layer spec.

    Returns:
        ModuleSpec: Module specification with modules from the backend.
    """
    column_parallel_linear_impl: type = backend.column_parallel_linear()
    layer_norm_impl = backend.layer_norm()
    mtp_layer_spec = ModuleSpec(
        module=MultiTokenPredictionLayer,
        submodules=MultiTokenPredictionLayerSubmodules(
            enorm=layer_norm_impl,
            hnorm=layer_norm_impl,
            eh_proj=column_parallel_linear_impl,
            mtp_model_layer=mtp_model_layer_spec,
            layer_norm=layer_norm_impl,
        ),
    )
    return mtp_layer_spec


def mtp_on_this_rank(
    layout: PipelineParallelLayerLayout = None,
    mtp_num_layers: Optional[int] = None,
    ignore_virtual: Optional[bool] = True,
    vp_stage: Optional[int] = None,
    pp_group: Optional[torch.distributed.ProcessGroup] = None,
    vp_size: Optional[int] = None,
) -> bool:
    """
    Check if there is MTP on the current rank.

    Behavior:
        - If a custom pipeline model parallel layout is provided:
            - If virtual pipeline parallelism is enabled (and `ignore_virtual` is False), checks
              whether any MTP layers are present on this (pp_rank, vp_stage) pair.
            - Otherwise, checks all virtual pipeline ranks of the current pipeline rank. Returns
              True if any virtual sub-rank includes at least one MTP layer.
        - If no custom layout is provided, assumes all MTP layers (if any) are placed on the last
          pipeline stage. The function returns True only on the last pipeline stage.
    """
    mtp_on_this_rank = False
    if pp_group is not None:
        pp_rank = get_pg_rank(pp_group)
        pp_size = get_pg_size(pp_group)
    else:
        # Compatibility fallback for callers that have not migrated to an explicit PP group.
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        pp_size = None
    if vp_size is None and layout is not None:
        vp_size = layout.virtual_pipeline_model_parallel_size
    elif vp_size is None and not ignore_virtual:
        vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

    if layout is not None:
        # with custom PP layout, we support put MTP layers on any pipeline stage
        if not ignore_virtual and vp_size not in (None, 1):
            assert vp_stage is not None, "vp_stage must be passed if virtual pipeline is enabled"
            num_layers_to_build = layout.layout[pp_rank][vp_stage].count(LayerType.mtp)
            mtp_on_this_rank = num_layers_to_build > 0
        else:
            for vpp_rank in range(len(layout.layout[pp_rank])):
                num_layers_to_build = layout.layout[pp_rank][vpp_rank].count(LayerType.mtp)
                if num_layers_to_build > 0:
                    mtp_on_this_rank = True
                    break
    else:
        # without custom PP layout, we only support put all of MTP layers on the last pipeline stage
        if mtp_num_layers is not None:
            if pp_size is None:
                # Compatibility fallback for callers without explicit pipeline metadata.
                pp_size = parallel_state.get_pipeline_model_parallel_world_size()
            mtp_on_this_rank = pp_rank == pp_size - 1
            if mtp_on_this_rank and not ignore_virtual and vp_size not in (None, 1):
                assert (
                    vp_stage is not None
                ), "vp_stage must be passed if virtual pipeline is enabled"
                mtp_on_this_rank = vp_stage == vp_size - 1
        else:
            mtp_on_this_rank = False
    return mtp_on_this_rank


def get_mtp_ranks(pp_ranks: List[int], config: TransformerConfig) -> List[int]:
    """Get the ranks of the MTP layers."""
    mtp_ranks = set()
    if config.mtp_num_layers is None:
        return []
    if config.pipeline_model_parallel_layout is None:
        return [pp_ranks[-1]]
    layout = config.pipeline_model_parallel_layout.layout
    for pp_rank in range(len(layout)):
        for vpp_rank in range(len(layout[pp_rank])):
            num_layers_to_build = layout[pp_rank][vpp_rank].count(LayerType.mtp)
            if num_layers_to_build:
                mtp_ranks.add(pp_ranks[pp_rank])
    return list(mtp_ranks)


def get_mtp_layer_offset(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> int:
    """Get the offset of the MTP layer."""
    if config.pipeline_model_parallel_size > 1:
        if config.pipeline_model_parallel_layout:
            if pp_rank is None:
                # Compatibility fallback for callers without explicit pipeline metadata.
                pp_rank = parallel_state.get_pipeline_model_parallel_rank()
            layout = config.pipeline_model_parallel_layout
            if layout.virtual_pipeline_model_parallel_size > 1:
                assert (
                    vp_stage is not None
                ), "vp_stage must be passed if virtual pipeline is enabled"
            else:
                vp_stage = 0
            offset = sum(
                layout.layout[previous_pp_rank][previous_vp_stage].count(LayerType.mtp)
                for previous_vp_stage in range(vp_stage + 1)
                for previous_pp_rank in range(
                    layout.pipeline_model_parallel_size if previous_vp_stage < vp_stage else pp_rank
                )
            )
        else:
            offset = 0
    else:
        offset = 0
    return offset


def get_mtp_num_layers_to_build(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> int:
    """Get the number of MTP layers to build."""
    if pp_rank is None:
        # Compatibility fallback for callers that have not migrated to explicit PP ranks.
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()

    if config.pipeline_model_parallel_layout is not None:
        # If we have a custom PP layout, get the number of mtp layers in the layout array.
        layout = config.pipeline_model_parallel_layout
        if layout.virtual_pipeline_model_parallel_size > 1:
            assert vp_stage is not None, "vp_stage must be passed if virtual pipeline is enabled"
        else:
            vp_stage = 0
        num_layers_to_build = layout.layout[pp_rank][vp_stage].count(LayerType.mtp)
        assert num_layers_to_build == config.mtp_num_layers or num_layers_to_build == 0, (
            f"Currently, we only support put all of MTP layers on the last pipeline stage, "
            f"so the number of MTP layers to build ({num_layers_to_build}) must match "
            f"mtp_num_layers ({config.mtp_num_layers}) or be 0."
        )
    else:
        vp_size = config.virtual_pipeline_model_parallel_size
        if vp_size not in (None, 1):
            assert vp_stage is not None, "vp_stage must be passed if virtual pipeline is enabled"
        is_last_vp_stage = vp_size in (None, 1) or vp_stage == vp_size - 1
        is_last_pp_stage = pp_rank == config.pipeline_model_parallel_size - 1
        num_layers_to_build = (
            config.mtp_num_layers if is_last_pp_stage and is_last_vp_stage else 0
        ) or 0
    return num_layers_to_build


class MTPLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for mtp loss."""

    main_loss_backward_scale: torch.Tensor = torch.tensor(1.0)

    @staticmethod
    def forward(ctx, output: torch.Tensor, mtp_loss: torch.Tensor):
        """Preserve the mtp by storing it in the context to avoid garbage collection.

        Args:
            output (torch.Tensor): The output tensor.
            mtp_loss (torch.Tensor): The mtp loss tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        ctx.save_for_backward(mtp_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for mtp loss..

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled mtp loss
                                               gradient.
        """
        (mtp_loss,) = ctx.saved_tensors
        mtp_loss_backward_scale = MTPLossAutoScaler.main_loss_backward_scale
        scaled_mtp_loss_grad = torch.ones_like(mtp_loss) * mtp_loss_backward_scale
        return grad_output, scaled_mtp_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """set the scale of the mtp loss.

        Args:
            scale (torch.Tensor): The scale value to set. Please ensure that the scale passed in
                                  matches the scale of the main_loss.
        """
        MTPLossAutoScaler.main_loss_backward_scale = scale


def process_mtp_loss(
    hidden_states: Tensor,
    labels: Tensor,
    loss_mask: Optional[Tensor],
    output_layer: Callable,
    output_weight: Optional[Tensor],
    runtime_gather_output: Optional[bool],
    is_training: bool,
    compute_language_model_loss: Callable,
    config: TransformerConfig,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
    scale_logits_fn: Optional[Callable[[Tensor], Tensor]] = None,
    input_ids: Optional[Tensor] = None,
    mtp_input_mask: Optional[Tensor] = None,
    metric_avg_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Tensor:
    """Process Multi-Token Prediction (MTP) loss computation.

    This is a standalone function that handles MTP loss computation. It's used on the
    post_process rank to split concatenated hidden states and compute MTP losses.

    Args:
        hidden_states (Tensor): Hidden states tensor (concatenated with MTP outputs).
        labels (Tensor): Ground truth labels.
        loss_mask (Optional[Tensor]): Mask for loss computation. If None, uses all ones.
        output_layer (Callable): Output layer method to compute logits.
        output_weight (Optional[Tensor]): Optional output weight for shared embeddings.
        runtime_gather_output (Optional[bool]): Whether to gather output at runtime.
        is_training (bool): Whether the model is in training mode.
        compute_language_model_loss (Callable): Method to compute language model loss.
        config (TransformerConfig): Model configuration containing mtp_num_layers etc.
        cp_group (Optional[ProcessGroup]): Context parallelism process group.
        tp_group (Optional[ProcessGroup]): Tensor parallelism process group.
        packed_seq_params (Optional[PackedSeqParams]): Packed sequence parameters.
        scale_logits_fn (Optional[Callable[[Tensor], Tensor]]): Optional function to
            scale logits before loss computation (e.g., MuP output scaling).
        input_ids (Optional[Tensor]): Input token IDs. Used to derive labels when
            ``labels`` is None (e.g. RL training), by rolling left to match the SFT
            label convention (``label[i] = input_id[i + 1]``). Ignored when ``labels``
            is provided.
        mtp_input_mask (Optional[Tensor]): Boolean mask over tokens that are valid as
            additional MTP conditioning inputs. The mask accumulates across prediction
            steps so a path stays masked after it reaches an invalid token.
        metric_avg_group (Optional[ProcessGroup]): Group used to average MTP logging metrics.

    Returns:
        Tensor: Updated hidden states after MTP loss processing (first chunk only).
    """
    hidden_states_list = torch.chunk(hidden_states, 1 + config.mtp_num_layers, dim=0)
    hidden_states = hidden_states_list[0]

    # When labels are not provided (e.g. RL training), derive them from input_ids by
    # rolling left so that label[i] = input_id[i + 1], matching the SFT label format.
    derived_labels_from_input_ids = False
    if labels is None:
        if input_ids is None:
            return hidden_states
        labels, _ = roll_tensor(
            input_ids,
            shifts=-1,
            dims=-1,
            cp_group=cp_group,
            packed_seq_params=packed_seq_params,
            return_sum=False,
        )
        derived_labels_from_input_ids = True

    if config.mtp_detach_heads:
        if output_weight is not None:
            output_weight = output_weight.detach()
        else:
            output_weight = output_layer.weight.detach()

    mtp_labels = labels.clone()
    if loss_mask is None:
        loss_mask = torch.ones_like(mtp_labels)
    if derived_labels_from_input_ids:
        # input_ids has no real token beyond the sequence window, so the last rolled-in
        # label is fabricated (zeroed). Roll loss_mask in lockstep with the
        # input_ids -> labels shift so that boundary position is masked.
        loss_mask, _ = roll_tensor(
            loss_mask,
            shifts=-1,
            dims=-1,
            cp_group=cp_group,
            packed_seq_params=packed_seq_params,
            return_sum=False,
        )

    # Store the original number of tokens before rolling for proper normalization
    # when calculate_per_token_loss is enabled. This ensures MTP gradients are
    # correctly scaled relative to the main loss gradients in finalize_model_grads.
    original_num_tokens = loss_mask.sum()

    cumulative_mtp_input_mask = None
    rolled_num_tokens = original_num_tokens
    if mtp_input_mask is not None:
        assert mtp_input_mask.shape == loss_mask.shape, (
            f"mtp_input_mask shape {mtp_input_mask.shape} must match "
            f"loss_mask shape {loss_mask.shape}"
        )
        mtp_input_mask = mtp_input_mask.to(dtype=torch.bool)

    for mtp_layer_number in range(config.mtp_num_layers):
        mtp_logits, _ = output_layer(
            hidden_states_list[mtp_layer_number + 1],
            weight=output_weight,
            runtime_gather_output=runtime_gather_output,
        )
        if scale_logits_fn is not None:
            mtp_logits = scale_logits_fn(mtp_logits)
        mtp_labels, _ = roll_tensor(
            mtp_labels,
            shifts=-1,
            dims=-1,
            cp_group=cp_group,
            packed_seq_params=packed_seq_params,
            return_sum=False,
        )

        if mtp_input_mask is not None:
            # Each MTP step consumes one additional token. Accumulate validity so
            # one invalid conditioning token also masks every later step on that path.
            mask_metadata = torch.cat((loss_mask, mtp_input_mask.to(dtype=loss_mask.dtype)), dim=0)
            mask_metadata, _ = roll_tensor(
                mask_metadata,
                shifts=-1,
                dims=-1,
                cp_group=cp_group,
                packed_seq_params=packed_seq_params,
                return_sum=False,
            )
            loss_mask, mtp_input_mask = mask_metadata.chunk(2, dim=0)
            mtp_input_mask = mtp_input_mask.to(dtype=torch.bool)
            if cumulative_mtp_input_mask is None:
                cumulative_mtp_input_mask = mtp_input_mask
            else:
                cumulative_mtp_input_mask = cumulative_mtp_input_mask & mtp_input_mask
            layer_loss_mask = loss_mask * cumulative_mtp_input_mask
            num_tokens = layer_loss_mask.sum()
        else:
            loss_mask, rolled_num_tokens = roll_tensor(
                loss_mask,
                shifts=-1,
                dims=-1,
                cp_group=cp_group,
                packed_seq_params=packed_seq_params,
            )
            layer_loss_mask = loss_mask
            # roll_tensor already computed this reduction. Preserve the legacy
            # no-mask fast path for all non-multimodal MTP callers.
            num_tokens = rolled_num_tokens

        mtp_loss = compute_language_model_loss(mtp_labels, mtp_logits)

        mtp_loss = layer_loss_mask * mtp_loss

        if is_training:
            mtp_loss_for_log = (
                torch.sum(mtp_loss) * (num_tokens > 0).to(mtp_loss.dtype)
            ) / num_tokens.clamp(min=1)
            correct, total = _compute_mtp_acceptance_counts(
                mtp_logits,
                mtp_labels,
                layer_loss_mask,
                output_layer,
                runtime_gather_output,
                tp_group,
            )

            if metric_avg_group is None:
                # Compatibility fallback for callers that have not migrated to explicit groups.
                metric_avg_group = parallel_state.get_data_parallel_group(
                    with_context_parallel=True
                )
            MTPLossLoggingHelper.save_metrics_to_tracker(
                mtp_loss_for_log,
                correct,
                total,
                mtp_layer_number,
                config.mtp_num_layers,
                avg_group=metric_avg_group,
            )
        mtp_loss_scale = config.mtp_loss_scaling_factor / config.mtp_num_layers
        if config.calculate_per_token_loss:
            # This uses local counts; exact parity for packed or uneven valid-token
            # distributions across DP/CP ranks would require all-reduced counts.
            # When calculate_per_token_loss is enabled, finalize_model_grads will
            # divide all gradients by total_num_tokens (from main loss).
            # However, MTP has fewer valid tokens due to rolling. To ensure correct
            # per-token gradient weighting, we normalize by the rolled token count
            # and re-scale by the original token count.
            # Avoid division by zero
            num_tokens_safe = torch.clamp(num_tokens, min=1)
            mtp_loss_normalized = (
                mtp_loss_scale * mtp_loss * (original_num_tokens / num_tokens_safe)
            )
            hidden_states = MTPLossAutoScaler.apply(hidden_states, mtp_loss_normalized)
        else:
            safe_num_tokens = num_tokens.clamp(min=1)
            hidden_states = MTPLossAutoScaler.apply(
                hidden_states, mtp_loss_scale * mtp_loss / safe_num_tokens
            )

    return hidden_states


class MultiTokenPredictionLayer(MegatronModule):
    """The implementation for Multi-Token Prediction (MTP) which extends
    the prediction scope to multiple future tokens at each position.

    This MTP implementation sequentially predict additional tokens and keep the complete
    causal chain at each prediction depth, by using D sequential modules to predict
    D additional tokens.

    The k-th MTP module consists of a shared embedding layer, a projection matrix,
    a Transformer block, and a shared output head.

    For the i-th input token at the (k - 1)-th prediction depth, we first combine
    the representation of the i-th token and the embedding of the (i + K)-th token with
    the linear projection. The combined serves as the input of the Transformer block at
    the k-th depth to produce the output representation.

    For more information, refer to DeepSeek-V3 Technical Report
    https://arxiv.org/pdf/2412.19437.pdf
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MultiTokenPredictionLayerSubmodules,
        layer_number: int = 1,
        vp_stage: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        # For hybrid path - pattern and submodules to build inner layers directly
        mtp_layer_pattern: Optional[str] = None,
        hybrid_submodules: Optional[HybridStackSubmodules] = None,
        mamba_submodules: Optional[HybridStackSubmodules] = None,
        name: str | None = None,
    ):
        """
        Args:
            name (str | None): module instance name passed top-down from its paranet module
        """
        super().__init__(config=config)
        if mamba_submodules is not None:
            if hybrid_submodules is not None:
                raise ValueError(
                    "Cannot specify both hybrid_submodules and mamba_submodules. "
                    "mamba_submodules has been deprecated; use hybrid_submodules instead."
                )
            warnings.warn(
                "mamba_submodules has been deprecated. Use hybrid_submodules instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            hybrid_submodules = mamba_submodules
        self.sequence_parallel = config.sequence_parallel
        self.submodules = submodules
        self.layer_number = layer_number + get_mtp_layer_offset(
            self.config, vp_stage, pp_rank=pg_collection.pp.rank()
        )
        self.vp_stage = vp_stage
        self.cp_group = pg_collection.cp
        self.tp_group = pg_collection.tp if pg_collection is not None else None
        self.mtp_layer_pattern = mtp_layer_pattern

        # Validate attention mask type if using transformer-based inner layers
        if self.submodules.mtp_model_layer is not None and hasattr(
            self.submodules.mtp_model_layer, 'submodules'
        ):
            from megatron.core.models.hybrid.hybrid_block import HybridStackSubmodules
            from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules

            layer_submodules = None
            if isinstance(self.submodules.mtp_model_layer.submodules, HybridStackSubmodules):
                attention_layer_spec = self.submodules.mtp_model_layer.submodules.attention_layer
                if hasattr(attention_layer_spec, 'submodules'):
                    assert isinstance(attention_layer_spec.submodules, TransformerLayerSubmodules)
                    layer_submodules = attention_layer_spec.submodules
            elif isinstance(self.submodules.mtp_model_layer.submodules, TransformerLayerSubmodules):
                layer_submodules = self.submodules.mtp_model_layer.submodules
            else:
                raise ValueError(
                    "Unsupported mtp_model_layer submodules type for attention mask validation."
                )
            if layer_submodules:
                self_attention_spec = layer_submodules.self_attention
                attn_mask_type = self_attention_spec.params.get('attn_mask_type', '')
                assert attn_mask_type in SUPPORTED_ATTN_MASK, (
                    f"Multi-Token Prediction (MTP) is not yet supported with "
                    f"{attn_mask_type} attention mask type. "
                    f"The supported attention mask types are {SUPPORTED_ATTN_MASK}."
                )

        self.enorm = self.submodules.enorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        self.hnorm = self.submodules.hnorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # For the linear projection at the (k - 1)-th MTP layer, the input is the concatenation
        # of the i-th token's hidden states and the (i + K)-th token's decoder input,
        # so the input's shape is [s, b, 2*h].
        # The output will be send to the following transformer layer,
        # so the output's shape should be [s, b, h].
        self.eh_proj = build_module(
            self.submodules.eh_proj,
            self.config.hidden_size * 2,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="mtp_eh_proj",
            tp_group=pg_collection.tp if pg_collection is not None else None,
            name=(name + ".eh_proj") if name is not None else None,
        )
        # eh_proj's input all-gather reuses the shared "tp" symmetric buffer right
        # after the preceding layer's all-gather (the fused rs-add-norm-ag terminates
        # with one, as does the previous MTP step's output all-gather), so it must
        # barrier before overwriting. Only the inference-optimized linear implements
        # this all-gather; other eh_proj impls have no such buffer to guard.
        if is_inference_column_parallel_linear(self.eh_proj):
            self.eh_proj.set_barrier_before_all_gather(True)

        # Build inner layers: two possible paths
        # 1. Hybrid path: use HybridStack for hybrid pattern support
        # 2. GPT path: single TransformerLayer
        if mtp_layer_pattern is not None and hybrid_submodules is not None:
            from megatron.core.models.hybrid.hybrid_block import HybridStack
            from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers

            self.mtp_model_layer = HybridStack(
                config=self.config,
                submodules=hybrid_submodules,
                layer_type_list=validate_segment_layers(mtp_layer_pattern),
                pp_layer_offset=0,
                pre_process=True,  # Always receives input from eh_proj
                post_layer_norm=False,  # MTP has its own final_layernorm
                post_process=True,  # MTP layer is self-contained
                pg_collection=pg_collection,
                is_mtp_layer=True,
                name=(name + ".mtp_model_layer") if name is not None else None,
            )
        elif self.config.mtp_num_layers is not None:
            # GPT path: Uses the transformer block spec for MTP layer
            # MTP inner layers use their own layer numbering (self.layer_number = 1, 2, etc.)
            # rather than continuing from decoder layer numbers. This is consistent with the
            # Mamba path and ensures proper aux loss tracking in router.py.
            self.mtp_model_layer = build_module(
                self.submodules.mtp_model_layer,
                config=self.config,
                vp_stage=self.vp_stage,
                layer_number=self.layer_number,
                is_mtp_layer=True,
                pg_collection=pg_collection,
                name=(name + ".mtp_model_layer") if name is not None else None,
            )

        # The MTP inner block's first all-gather reuses the same "tp" symmetric buffer
        # that _concat_embeddings' output all-gather just wrote, with no reduce-scatter in
        # between, so it must barrier before overwriting. Later all-gathers in the inner
        # block are each preceded by a reduce-scatter and need no barrier. modules() yields
        # in forward order, so the first inference column-parallel linear is that all-gather.
        if self.mtp_layer_pattern is not None:
            # Hybrid path: HybridStack of layers.
            first_inner_layer = self.mtp_model_layer.layers[0]
        else:
            # GPT path: single TransformerLayer.
            first_inner_layer = self.mtp_model_layer

        for module in first_inner_layer.modules():
            if is_inference_column_parallel_linear(module):
                module.set_barrier_before_all_gather(True)
                break

        self.final_layernorm = self.submodules.layer_norm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.offload_context = nullcontext()

    def _get_embeddings(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        embedding: Callable,
        hidden_states: torch.Tensor,
        packed_seq_params: Optional[PackedSeqParams] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mtp_input_mask: Optional[torch.Tensor] = None,
    ):
        """
        Preprocesses input data for the Multi-Token Prediction (MTP) layers.

        This function computes the decoder input and sends updated input_ids and position_ids to
        the next layer.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            position_ids (torch.Tensor): The position IDs corresponding to the input tokens.
            embedding (Callable): The embedding module
                from gpt model to compute the decoder input.
            hidden_states (torch.Tensor): hidden states tensor of shape [s, b, h] where s is the
                sequence length, b is the batch size, and h is the hidden size.
            packed_seq_params (PackedSeqParams): Parameters for packed sequence processing.
            mtp_input_mask (torch.Tensor, optional): Mask of conditioning tokens backed by
                regular token embeddings. Shape: [b, s].
        """
        # Calc logits for the current Multi-Token Prediction (MTP) layers.
        if mtp_input_mask is None:
            input_ids, _ = roll_tensor(
                input_ids,
                shifts=-1,
                dims=-1,
                cp_group=self.cp_group,
                packed_seq_params=packed_seq_params,
                return_sum=False,
            )
        else:
            assert mtp_input_mask.shape == input_ids.shape, (
                f"mtp_input_mask shape {mtp_input_mask.shape} must match "
                f"input_ids shape {input_ids.shape}"
            )
            # Roll IDs and validity together so CP performs one boundary exchange.
            token_metadata = torch.cat((input_ids, mtp_input_mask.to(dtype=input_ids.dtype)), dim=0)
            token_metadata, _ = roll_tensor(
                token_metadata,
                shifts=-1,
                dims=-1,
                cp_group=self.cp_group,
                packed_seq_params=packed_seq_params,
                return_sum=False,
            )
            input_ids, mtp_input_mask = token_metadata.chunk(2, dim=0)
            mtp_input_mask = mtp_input_mask.to(dtype=torch.bool)
        position_ids, _ = roll_tensor(
            position_ids,
            shifts=-1,
            dims=-1,
            cp_group=self.cp_group,
            packed_seq_params=packed_seq_params,
            return_sum=False,
        )
        if padding_mask is not None:
            padding_mask, _ = roll_tensor(
                padding_mask,
                shifts=-1,
                dims=-1,
                cp_group=self.cp_group,
                packed_seq_params=packed_seq_params,
                return_sum=False,
            )
        # embedding
        decoder_input = embedding(input_ids=input_ids, position_ids=position_ids)

        if mtp_input_mask is not None:
            # Keep invalid placeholder values in the forward pass, but prevent
            # later causal positions from updating their shared embedding rows.
            valid_decoder_input = mtp_input_mask.transpose(0, 1).unsqueeze(-1)
            decoder_input = torch.where(valid_decoder_input, decoder_input, decoder_input.detach())

        # Mirror the scatter in the model's own forward (see hybrid_model.py:
        # "the embedding skips SP scatter for models whose outer wrapper
        # scatters instead"). Multimodal LMs build LanguageModelEmbedding with
        # scatter_to_sequence_parallel=False so they can insert media into a
        # full-length embedding before scattering. The MTP block calls the
        # embedding directly and so must apply the same scatter, otherwise
        # decoder_input stays full-length while the backbone hidden_states
        # arrive sequence-parallel sharded and _concat_embeddings fails.
        if self.config.sequence_parallel and not getattr(
            embedding, "scatter_to_sequence_parallel", True
        ):
            decoder_input = scatter_to_sequence_parallel_region(decoder_input, group=self.tp_group)

        if self.config.mtp_detach_heads:
            decoder_input = decoder_input.detach()

        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
        # make_viewless_tensor no-ops when hidden_states is not a view (_base is None),
        # which happens after detach() with mtp_detach_heads. Activation
        # checkpointing (CheckpointFunction.apply) requires at least one input tensor
        # with requires_grad=True to produce a differentiable output, so we ensure it
        # here to maintain gradient flow to MTP layer parameters.
        if not hidden_states.requires_grad:
            hidden_states.requires_grad_(True)

        return (input_ids, position_ids, padding_mask, mtp_input_mask, decoder_input, hidden_states)

    def _concat_embeddings(self, hidden_states: torch.Tensor, decoder_input: torch.Tensor):
        """
        Concatenate the tokens before sending to transformer layer.
        """
        decoder_input = apply_module(self.enorm)(decoder_input)
        decoder_input = make_viewless_tensor(inp=decoder_input, requires_grad=True, keep_graph=True)
        hidden_states = apply_module(self.hnorm)(hidden_states)
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
        # At the (k - 1)-th MTP module, concatenates the i-th token's hidden_states
        # and the (i + K)-th token's embedding, and combine them with linear projection.
        hidden_states = torch.cat((decoder_input, hidden_states), -1)
        hidden_states, _ = self.eh_proj(hidden_states)
        # For tensor parallel we need to gather the tensor across the model-parallel
        # ranks after the linear projection.
        if InferenceMode.is_active():
            # This all-gather immediately follows eh_proj's input all-gather on the
            # same symmetric buffer (only eh_proj's local matmul runs in between), so
            # it must barrier before overwriting the buffer's previous contents.
            hidden_states = inference_all_gather_from_tensor_model_parallel_region(
                hidden_states, self.tp_group, self.config, barrier_before=True
            )
        else:
            hidden_states = gather_from_tensor_model_parallel_region(
                hidden_states, group=self.tp_group
            )
        # For sequence parallel, scatter after linear_fc and before transformer layer.
        if self.sequence_parallel:
            hidden_states = scatter_to_sequence_parallel_region(hidden_states, group=self.tp_group)
        return hidden_states

    def _proj_and_transformer_layer(
        self,
        hidden_states: Tensor,
        decoder_input: Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Concatenates embeddings with hidden states and then applies transformer layer forward.
        """
        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # Unlike transformer_block.py which needs to support mixed-precision in
        # different layers,currently MTP only use global fp8 context.
        if self.config.fp8:
            fp8_context = get_fp8_context(self.config)
            transformer_layer_fp8_context = get_fp8_context(self.config)
        else:
            fp8_context = nullcontext()
            transformer_layer_fp8_context = nullcontext()

        # TODO: currently ignoring FP4 in MTP layers because we need more numerical validation
        with rng_context:
            with fp8_context:
                hidden_states = self._concat_embeddings(hidden_states, decoder_input)

            # Use a separate fp8 context for the transformer layer. This is to ensure that when the
            # transformer layer is cudagraphed, the FP8GlobalStateManager.is_first_fp8_module() is
            # True so that the fp8 weight caching can be triggered correctly.
            with transformer_layer_fp8_context:
                if self.mtp_layer_pattern is not None:
                    hidden_states = self.mtp_model_layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        padding_mask=padding_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        inference_context=inference_params,
                        packed_seq_params=packed_seq_params,
                    )
                else:
                    # GPT path: single TransformerLayer
                    hidden_states, _ = self.mtp_model_layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        rotary_pos_cos=rotary_pos_cos,
                        rotary_pos_sin=rotary_pos_sin,
                        attention_bias=attention_bias,
                        inference_params=inference_params,
                        packed_seq_params=packed_seq_params,
                        sequence_len_offset=sequence_len_offset,
                        padding_mask=padding_mask,
                    )

        hidden_states = self._postprocess(hidden_states)

        return hidden_states

    def _postprocess(self, hidden_states: torch.Tensor):
        """
        Postprocesses the output of the transformer layers.
        """

        # Layer norm before shared head layer.
        hidden_states = apply_module(self.final_layernorm)(hidden_states)
        # TENorm produces a "viewed" tensor. This will result in schedule.py's
        # deallocate_output_tensor() throwing an error, so a viewless tensor is
        # created to prevent this.
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        return hidden_states

    def forward_single_position(
        self,
        hidden_states: Tensor,
        next_token_ids: Tensor,
        position_ids: Tensor,
        embedding: Callable,
        attention_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward for single positions without roll_tensor (speculative decoding).

        Unlike the regular forward which rolls input_ids to get the next token's
        embedding, this method directly takes the correct next_token_ids. This is
        used in speculative decoding where the correct next token is known after
        verification.

        Args:
            hidden_states (Tensor): Hidden states at positions of interest [N, B, H].
            next_token_ids (Tensor): The correct next token IDs [B, N].
            position_ids (Tensor): Position IDs for the next tokens [B, N].
            embedding (Callable): The embedding module.

        Returns:
            Tensor: MTP hidden states [N, B, H].
        """
        decoder_input = embedding(input_ids=next_token_ids, position_ids=position_ids)
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=False, keep_graph=False
        )
        hidden_states = self._proj_and_transformer_layer(
            hidden_states=hidden_states,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        return hidden_states

    def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        decoder_input: Tensor,
        attention_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
    ):
        """Forward through ``_proj_and_transformer_layer`` with activation
        recomputation.

        Mirrors ``transformer_block._checkpointed_forward``:

        * Non-tensor objects (``attention_bias``, ``inference_params``,
          ``packed_seq_params``) are captured by the ``custom_forward``
          closure; only tensor / ``None`` arguments flow positionally
          through the underlying checkpoint primitive. This is required
          by both backends: ``tensor_parallel.checkpoint`` because its
          ``save_for_backward`` only accepts tensors and ``None``, and
          ``te_checkpoint`` because its reentrant implementation only
          tracks positional tensor inputs as checkpoint inputs (kwarg
          tensors are not represented in the recompute backward path).
        * Quantized recipes (fp8, fp4) route through ``te_checkpoint``;
          everything else uses ``tensor_parallel.checkpoint``.
        * Only ``fp8 + delayed scaling`` needs an outer quantization
          context entered before ``te_checkpoint``; see the
          ``outer_quantization_context`` block below.
        """

        def custom_forward(
            hidden_states,
            decoder_input,
            attention_mask,
            padding_mask,
            context,
            context_mask,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        ):
            return self._proj_and_transformer_layer(
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )

        # Decide the outer quantization context, matching
        # ``transformer_block._checkpointed_forward``. Only ``fp8 + delayed
        # scaling`` needs an active context at the ``te_checkpoint`` entry
        # point: TE's ``_CheckpointFunction.forward`` samples
        # ``FP8GlobalStateManager.is_fp8_enabled()`` there to gate the
        # phase-1 amax-buffer stash that phase-2 backward looks up via
        # ``global_fp8_buffer_pos_fwd_recompute``. With fp8 only entered
        # *inside* ``_proj_and_transformer_layer``, TE samples fp8 as off,
        # phase-1 skips the stash, and phase-2 raises ``KeyError``.
        # Non-delayed fp8 recipes (MXFP8BlockScaling, Float8CurrentScaling)
        # and fp4 (NVFP4BlockScaling) treat the stash/lookup as a noop, so
        # the inner context entered inside ``_proj_and_transformer_layer``
        # is sufficient.
        if self.config.fp8 and self.config.fp8_recipe == Fp8Recipe.delayed:
            outer_quantization_context = get_fp8_context(self.config)
        else:
            outer_quantization_context = nullcontext()

        def checkpoint_handler():
            """Determines whether to use the `te_checkpoint` or `tensor_parallel.checkpoint`"""
            # fp4 quantization is internally implemented via TE's
            # ``fp8_autocast`` (see ``fp4_utils.get_fp4_context``), so
            # quantized recompute on either fp8 or fp4 must go through
            # ``te_checkpoint``. Matches ``transformer_block``'s policy.
            if self.config.fp8 or self.config.fp4:
                from megatron.core.extensions.transformer_engine import te_checkpoint

                return te_checkpoint(
                    custom_forward,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                    decoder_input,
                    attention_mask,
                    padding_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    sequence_len_offset,
                )
            else:
                # tensor_parallel.checkpoint stashes args via autograd's
                # ``save_for_backward``, which only accepts tensors and ``None``.
                # Pass tensor / ``None`` args positionally and capture the
                # non-tensor objects (``attention_bias``, ``inference_params``,
                # ``packed_seq_params``) via the ``custom_forward`` closure.
                return tensor_parallel.checkpoint(
                    custom_forward,
                    self.config.distribute_saved_activations,
                    hidden_states,
                    decoder_input,
                    attention_mask,
                    padding_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    sequence_len_offset,
                )

        if self.config.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            assert (
                self.config.recompute_num_layers == 1
            ), "recompute_num_layers must be 1 for MTP recompute"
            with outer_quantization_context:
                outputs = checkpoint_handler()
        elif self.config.recompute_method == 'block':
            # TODO: implement block-based recompute for MTP
            warnings.warn(
                "recompute_method == 'block' is not supported for MTP yet." " Skipping recompute."
            )
            outputs = self._proj_and_transformer_layer(
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
        else:
            raise ValueError("Invalid activation recompute method.")

        return outputs

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        hidden_states: Tensor,
        attention_mask: Tensor,
        padding_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        embedding=None,
        mtp_input_mask: Optional[Tensor] = None,
    ):
        """
        Execute the forward pass through the Multi-Token Prediction (MTP) layer.

        Args:
            input_ids (Tensor): Input token IDs .
            position_ids (Tensor): Positional IDs of the input tokens.
            hidden_states (Tensor): Hidden states tensor of shape [s, b, h] where s is the
                sequence length, b is the batch size, and h is the hidden size.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention, if applicable.
            context_mask (Tensor, optional): Mask for cross-attention context, if applicable.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Tensor, optional): Cosine component of rotary positional embeddings.
            rotary_pos_sin (Tensor, optional): Sine component of rotary positional embeddings.
            sequence_len_offset (Tensor, optional): Offset for sequence length, if applicable.
            embedding (Callable): The embedding module from gpt model to compute the decoder input.
            mtp_input_mask (Tensor, optional): Mask of valid MTP conditioning tokens.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """
        assert context is None, "multi token prediction + cross attention is not yet supported."
        input_ids, position_ids, padding_mask, mtp_input_mask, decoder_input, hidden_states = (
            self._get_embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                padding_mask=padding_mask,
                embedding=embedding,
                hidden_states=hidden_states,
                packed_seq_params=packed_seq_params,
                mtp_input_mask=mtp_input_mask,
            )
        )

        if self.config.recompute_granularity == 'full' and self.training:
            hidden_states = self._checkpointed_forward(
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
        else:
            hidden_states = self._proj_and_transformer_layer(
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )

        return hidden_states, input_ids, position_ids, padding_mask, mtp_input_mask

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the multi token prediction layer.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (Optional[dict], optional): Additional metadata for sharding.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the multi
            token prediction layer.
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)

        # Backward compatibility: GPT MTP checkpoints were saved with the submodule
        # named 'transformer_layer'. Remap checkpoint keys so old checkpoints load
        # correctly. Mamba MTP models keep 'mtp_model_layer' as their native format
        # since no older checkpoints exist for them.
        if self.mtp_layer_pattern is None:
            apply_prefix_mapping(
                sharded_state_dict, {f'{prefix}mtp_model_layer.': f'{prefix}transformer_layer.'}
            )

        return sharded_state_dict


@dataclass
class MultiTokenPredictionBlockSubmodules:
    """
    Dataclass for specifying the submodules of a multi token prediction block.

    This class defines the structure for configuring the layers, allowing for
    flexible and customizable architecture designs.

    Args:
        layer_specs (List[ModuleSpec], optional): A list of module specifications for
            the layers within the multi token prediction block. Each specification typically
            defines a complete multi token prediction layer (e.g., shared embedding,
            projection matrix, transformer block, shared output head).
    """

    layer_specs: Optional[List[ModuleSpec]] = None


def _get_mtp_block_submodules(
    config: TransformerConfig, spec: Union[MultiTokenPredictionBlockSubmodules, ModuleSpec]
) -> MultiTokenPredictionBlockSubmodules:
    """
    Retrieve or construct MultiTokenPredictionBlockSubmodules based on the provided specification.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        spec (Union[MultiTokenPredictionBlockSubmodules, ModuleSpec]): Specification for the
            multi token prediction block submodules.
            Can be either a MultiTokenPredictionBlockSubmodules instance or a ModuleSpec.

    Returns:
        MultiTokenPredictionBlockSubmodules: The submodules for the multi token prediction block.
    """

    # Transformer block submodules.
    if isinstance(spec, MultiTokenPredictionBlockSubmodules):
        return spec
    elif isinstance(spec, ModuleSpec):
        if issubclass(spec.module, MultiTokenPredictionBlock):
            return spec.submodules
        else:
            raise Exception(f"specialize for {spec.module.__name__}.")
    else:
        raise Exception(f"specialize for {type(spec).__name__}.")


class MultiTokenPredictionBlock(MegatronModule):
    """The implementation for Multi-Token Prediction (MTP) which extends
    the prediction scope to multiple future tokens at each position.

    This MTP implementation sequentially predict additional tokens and keep the complete
    causal chain at each prediction depth, by using D sequential modules to predict
    D additional tokens.

    The k-th MTP module consists of a shared embedding layer, a projection matrix,
    a Transformer block, and a shared output head.

    For the i-th input token at the (k - 1)-th prediction depth, we first combine
    the representation of the i-th token and the embedding of the (i + K)-th token with
    the linear projection. The combined serves as the input of the Transformer block at
    the k-th depth to produce the output representation.

    When `mtp_use_repeated_layer=True` in config, instead of creating N separate MTP layers,
    only 1 layer is created and applied mtp_num_layers times.

    For more information, please refer to DeepSeek-V3 Technical Report
    https://arxiv.org/pdf/2412.19437.pdf
    """

    def __init__(
        self,
        config: TransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        vp_stage: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        # New: For hybrid path with unified pattern syntax
        mtp_layer_pattern: Optional[str] = None,
        mtp_num_depths: int = 0,
        hybrid_submodules: Optional["HybridStackSubmodules"] = None,
        mamba_submodules: Optional["HybridStackSubmodules"] = None,
        name: str | None = None,
    ):
        """
        Args:
            name (str | None): module instance name passed top-down from its paranet module
        """
        super().__init__(config=config)
        if mamba_submodules is not None:
            if hybrid_submodules is not None:
                raise ValueError(
                    "Cannot specify both hybrid_submodules and mamba_submodules. "
                    "mamba_submodules has been deprecated; use hybrid_submodules instead."
                )
            warnings.warn(
                "mamba_submodules has been deprecated. Use hybrid_submodules instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            hybrid_submodules = mamba_submodules
        self.submodules = _get_mtp_block_submodules(config, spec)
        self.mtp_loss_scaling_factor = config.mtp_loss_scaling_factor
        self.vp_stage = vp_stage
        self.mtp_layer_pattern = mtp_layer_pattern
        self.mtp_num_depths = mtp_num_depths
        self.hybrid_submodules = hybrid_submodules
        self.mtp_use_repeated_layer = self.config.mtp_use_repeated_layer
        self.name = name

        vp_size = config.virtual_pipeline_model_parallel_size
        assert is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size), (
            f"MTP layers must be placed on the last virtual pipeline stage. "
            f"Got vp_stage={vp_stage} with vp_size={vp_size}. "
            f"Placing MTP layers on different VPP stages is not currently supported."
        )

        # Initialize Context Parallelism (CP) support for MTP
        # This enables MTP to work with CP > 1 by providing the CP process group
        # to the roll_tensor function for proper boundary communication
        if pg_collection is None:
            # Use default MPU process groups if not provided
            required_pgs = ['cp', 'tp', 'pp'] + (['dp'] if self.config.mtp_hsm else [])
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=required_pgs)
        else:
            # Ensure the provided process groups include TP, CP, and PP.
            for group_name in ('tp', 'cp', 'pp'):
                assert (
                    getattr(pg_collection, group_name, None) is not None
                ), f"MultiTokenPredictionBlock pg_collection must have {group_name} process group"
            if self.config.mtp_hsm:
                assert hasattr(
                    pg_collection, 'dp'
                ), "MultiTokenPredictionBlock with HSM requires a dp process group"

        self._build_layers(pg_collection)
        assert len(self.layers) > 0, "MultiTokenPredictionBlock must have at least one layer."
        self.cp_group = pg_collection.cp
        self.tp_group = pg_collection.tp
        self.pp_rank = pg_collection.pp.rank()
        self.dp_group = pg_collection.dp if self.config.mtp_hsm else None
        self.hidden_state_mixing_rng_tracker_name = (
            _initialize_hidden_state_mixing_rng_tracker(self.dp_group)
            if self.config.mtp_hsm
            else None
        )
        self.sequence_parallel = config.sequence_parallel

        if self.config.mtp_detach_heads:
            # Tag MTP params so the optimizer can clip their gradients separately.
            for param in self.parameters():
                param.grad_norm_group = 'mtp'

    def _build_layers(self, pg_collection):
        # Determine number of depths to build
        if self.mtp_num_depths > 0:
            num_depths = self.mtp_num_depths
        else:
            num_depths = self.config.mtp_num_layers or len(self.submodules.layer_specs)

        def build_layer_legacy(layer_spec, layer_number):
            """Build layer using legacy spec-based approach."""
            fp8_init_context = get_fp8_context(self.config, is_init=True)
            with fp8_init_context:
                module = build_module(
                    layer_spec,
                    config=self.config,
                    layer_number=layer_number,
                    vp_stage=self.vp_stage,
                    pg_collection=pg_collection,
                    mtp_layer_pattern=self.mtp_layer_pattern,
                    name=(self.name + f".layers.{layer_number}") if self.name is not None else None,
                )
            return module

        def build_layer_with_pattern(
            layer_spec, layer_number, mtp_layer_pattern, hybrid_submodules
        ):
            """Build layer using pattern-based approach (new Mamba path)."""
            fp8_init_context = get_fp8_context(self.config, is_init=True)
            with fp8_init_context:
                module = build_module(
                    layer_spec,
                    config=self.config,
                    layer_number=layer_number,
                    vp_stage=self.vp_stage,
                    pg_collection=pg_collection,
                    mtp_layer_pattern=mtp_layer_pattern,
                    hybrid_submodules=hybrid_submodules,
                    name=(self.name + f".layers.{layer_number}") if self.name is not None else None,
                )
            return module

        # New Mamba path: use mtp_layer_pattern and hybrid_submodules
        if self.mtp_layer_pattern is not None and self.hybrid_submodules is not None:
            if self.mtp_use_repeated_layer:
                # Shared/repeated layer: build one layer, use it for all depths
                layer_spec = self.submodules.layer_specs[0]
                shared_layer = build_layer_with_pattern(
                    layer_spec,
                    layer_number=1,
                    mtp_layer_pattern=self.mtp_layer_pattern,
                    hybrid_submodules=self.hybrid_submodules,
                )
                self.layers = torch.nn.ModuleList([shared_layer])
            else:
                # Non-shared: each depth gets its own layers
                self.layers = torch.nn.ModuleList(
                    [
                        build_layer_with_pattern(
                            self.submodules.layer_specs[
                                min(i, len(self.submodules.layer_specs) - 1)
                            ],
                            layer_number=i + 1,
                            mtp_layer_pattern=self.mtp_layer_pattern,
                            hybrid_submodules=self.hybrid_submodules,
                        )
                        for i in range(num_depths)
                    ]
                )
        elif self.mtp_use_repeated_layer:
            # Legacy repeated layer mode
            if len(self.submodules.layer_specs) != 1:
                warnings.warn(
                    "Repeated MTP mode expects exactly 1 layer spec, got "
                    f"{len(self.submodules.layer_specs)} instead. "
                    f"The first layer will be applied {self.config.mtp_num_layers} times."
                )
            self.layers = torch.nn.ModuleList(
                [build_layer_legacy(self.submodules.layer_specs[0], layer_number=1)]
            )
        else:
            # Legacy mode: build from layer_specs
            self.layers = torch.nn.ModuleList(
                [
                    build_layer_legacy(layer_spec, i + 1)
                    for i, layer_spec in enumerate(self.submodules.layer_specs)
                ]
            )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        hidden_states: Tensor,
        attention_mask: Tensor,
        padding_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        extra_block_kwargs: Optional[dict] = None,
        embedding=None,
        mtp_input_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Perform the forward pass through all of the MTP modules.

        Args:
            hidden_states (Tensor): Hidden states for input token with the shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            mtp_input_mask (Tensor, optional): Mask of valid MTP conditioning tokens.

        Returns:
            (Tensor): The mtp loss tensor of shape [b, s].
        """
        # get hidden states from previous mtp stages
        offset = get_mtp_layer_offset(self.config, self.vp_stage, pp_rank=self.pp_rank)
        hidden_states_list = list(torch.chunk(hidden_states, 1 + offset, dim=0))
        hidden_states = hidden_states_list[offset]

        if self.config.mtp_detach_heads:
            hidden_states = hidden_states.detach()

        hidden_state_mixing_enabled = self.config.mtp_hsm and self.training
        if hidden_state_mixing_enabled:
            hidden_state_history = [hidden_states]

        for iteration in range(self.config.mtp_num_layers):
            layer_idx = 0 if self.mtp_use_repeated_layer else iteration

            # Older HSM entries predict earlier targets than the newest entry. Roll
            # them once per depth so all candidates correspond to the same target.
            if hidden_state_mixing_enabled and len(hidden_state_history) > 1:
                entries_to_roll = hidden_state_history[:-1]
                newest_entry = hidden_state_history[-1]
                num_entries = len(entries_to_roll)
                sequence_length, batch_size, hidden_size = entries_to_roll[0].shape
                stacked = torch.stack(entries_to_roll, dim=0)
                flattened = stacked.permute(0, 2, 3, 1).reshape(
                    num_entries * batch_size, hidden_size, sequence_length
                )
                # Under sequence parallelism this rank holds a 1/tp slice of its CP
                # chunks, not the chunk pair roll_tensor's CP branch assumes, so that
                # branch's neighbour exchange fills the boundary slots with tokens from
                # unrelated positions -- and a plausible-looking hidden state is one
                # the mixing step cannot recognise as invalid. Withholding cp_group takes the
                # contiguous path, which zeroes the slot that has no local continuation
                # so the mix falls back to the newest entry there instead.
                sequence_parallel_size = get_pg_size(self.tp_group) if self.sequence_parallel else 1
                # Document boundaries are global too, so they need the same treatment:
                # translated into this shard's frame, the withheld cp_group leaves the
                # roll on its cp_size == 1 path, which then rolls each document's local
                # piece within its own bounds and zeroes the seam.
                roll_packed_seq_params = packed_seq_params
                use_local_packed_roll = sequence_parallel_size > 1
                if packed_seq_params is not None:
                    padded_cu_seqlens = packed_seq_params.cu_seqlens_q_padded
                    genuinely_padded = (
                        padded_cu_seqlens is not None
                        and padded_cu_seqlens is not packed_seq_params.cu_seqlens_q
                    )
                    use_local_packed_roll = use_local_packed_roll or genuinely_padded
                if use_local_packed_roll and packed_seq_params is not None:
                    shard_params = _packed_seq_params_for_local_hsm_roll(
                        packed_seq_params,
                        local_seq_length=sequence_length,
                        cp_group=self.cp_group,
                        tp_group=self.tp_group if self.sequence_parallel else None,
                    )
                    if shard_params is not None:
                        roll_packed_seq_params = shard_params
                rolled, _ = roll_tensor(
                    flattened,
                    shifts=-1,
                    dims=-1,
                    cp_group=None if use_local_packed_roll else self.cp_group,
                    packed_seq_params=roll_packed_seq_params,
                    return_sum=False,
                )
                rolled_older_hidden_states = rolled.reshape(
                    num_entries, batch_size, hidden_size, sequence_length
                ).permute(0, 3, 1, 2)
                hidden_states_input = _mix_hidden_state_history(
                    rolled_older_hidden_states,
                    newest_entry,
                    sequence_parallel=self.sequence_parallel,
                    tp_group=self.tp_group,
                    cp_group=self.cp_group,
                    rng_tracker_name=self.hidden_state_mixing_rng_tracker_name,
                )
                hidden_state_history = list(rolled_older_hidden_states.unbind(0)) + [newest_entry]
            else:
                hidden_states_input = hidden_states

            hidden_states, input_ids, position_ids, padding_mask, mtp_input_mask = self.layers[
                layer_idx
            ](
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states_input,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                embedding=embedding,
                mtp_input_mask=mtp_input_mask,
                **(extra_block_kwargs or {}),
            )

            if hidden_state_mixing_enabled:
                hidden_state_history.append(hidden_states)

            # append the output hidden states of the current mtp layer
            # to the hidden_states_list
            hidden_states_list.append(hidden_states)

        # concat the hidden states of all mtp layers
        hidden_states = torch.cat(hidden_states_list, dim=0)
        return hidden_states

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the multi token prediction module.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (Optional[dict], optional): Additional metadata for sharding.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the multi
            token prediction module.
        """
        sharded_state_dict = {}
        layer_prefix = f'{prefix}layers.'
        for layer in self.layers:
            offset = get_mtp_layer_offset(self.config, self.vp_stage, pp_rank=self.pp_rank)
            sharded_prefix = f'{layer_prefix}{layer.layer_number - 1}.'

            state_dict_prefix = f'{layer_prefix}{layer.layer_number - 1 - offset}.'
            sharded_pp_offset = []
            layer_sharded_state_dict = layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )
            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)
            sharded_state_dict.update(layer_sharded_state_dict)
        return sharded_state_dict
