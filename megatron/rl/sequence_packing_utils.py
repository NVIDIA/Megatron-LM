# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch
import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass, field
from megatron.core.utils import log_single_rank
from megatron.training.global_vars import get_args, get_tokenizer
from megatron.training.utils import get_nvtx_range
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core import mpu
import logging
import typing
from megatron.core.num_microbatches_calculator import (
        get_num_microbatches,
        reconfigure_num_microbatches_calculator,
    )

logger = logging.getLogger(__name__)


@dataclass
class PackingInfo:
    """Information about how sequences are packed into bins.
    
    Attributes:
        bin_seq_indices: List where each element contains the global sequence indices in that bin
        seq_starts: Dict mapping bin index to list of start positions for each sequence in that bin
        seq_lengths: List of all original sequence lengths (indexed by global sequence index)
        seq_to_bin_idx: List mapping each global sequence index to its bin index
        packing_algo: Algorithm used for distributing bins ('fifo' or 'round-robin')
    """
    bin_seq_indices: List[List[int]]
    seq_starts: Dict[int, List[int]]
    seq_lengths: List[int]
    seq_to_bin_idx: List[Optional[int]]
    packing_algo: typing.Literal['fifo', 'round-robin']


@dataclass
class PackingContext:
    """Context containing all information needed for sequence packing during training.
    
    Attributes:
        bin_size: Maximum size of each bin (in tokens)
        packer: 'SequencePacker' instance used for packing
        packing_info: PackingInfo object with bin assignments and metadata
        original_generation_masks: Generation masks for all sequences before packing
        original_trajs: All trajectories before packing
        packed_trajs: Packed trajectories tensor [num_bins, bin_size]
        packed_position_ids: Position IDs for packed sequences [num_bins, bin_size]
        packed_attention_mask: Attention mask for packed sequences [num_bins, 1, bin_size, bin_size]
        packed_loss_mask: Loss mask for packed sequences [num_bins, bin_size]
        original_inference_logprobs: Inference logprobs for all sequences before packing (optional)
        bin_advantages: List of advantage tensors for each bin
        cached_packed_seq_params: Pre-computed PackedSeqParams for each bin
    """
    bin_size: int
    packer: 'SequencePacker'
    packing_info: PackingInfo
    original_generation_masks: torch.Tensor
    original_trajs: torch.Tensor
    packed_trajs: torch.Tensor
    packed_position_ids: torch.Tensor
    packed_attention_mask: torch.Tensor
    packed_loss_mask: torch.Tensor
    original_inference_logprobs: Optional[torch.Tensor] = None
    bin_advantages: List[torch.Tensor] = field(default_factory=list)
    cached_packed_seq_params: List[Optional[PackedSeqParams]] = field(default_factory=list)


def load_packed_data_by_index(bin_idx: int, packing_context: PackingContext, logprobs_is_correction: bool):
    """Load packed data by index.

    Args:
        bin_idx: Index of the bin to load.
    """
    args = get_args()
    # Get packing context (should always be available in packed mode)
    idx = slice(bin_idx, bin_idx + 1)

    # Get cached PackedSeqParams for proper attention masking in Transformer Engine
    # These were pre-computed in prepare_data_for_update to avoid repeated tensor allocations
    packed_seq_params = packing_context.cached_packed_seq_params[bin_idx]

    # Extract packed data for this bin (already on GPU)
    tokens = packing_context.packed_trajs[idx]
    position_ids = packing_context.packed_position_ids[idx]

    # Check if we have old_logprobs and ref_logprobs as attributes
    # These are set after logprobs computation, so they may not exist during initial forward pass
    old_logprobs = getattr(packing_context, 'old_logprobs', None)
    if old_logprobs is not None:
        old_logprobs = old_logprobs[idx]
    
    ref_logprobs = getattr(packing_context, 'ref_logprobs', None)
    if ref_logprobs is not None:
        ref_logprobs = ref_logprobs[idx]
        
    # Slice from position 1 because logprobs predict the next token, so they are
    # shifted by 1 relative to the input tokens (logprobs has shape [batch, seq_len-1])
    loss_mask = packing_context.packed_loss_mask[idx, 1:]

    # Get sequence-level data for this bin
    packing_info = packing_context.packing_info
    seq_starts = packing_info.seq_starts[bin_idx]
    seq_indices = packing_info.bin_seq_indices[bin_idx]

    # Handle empty bins (used for padding to ensure all ranks have same iterations)
    if not seq_indices:
        seq_lengths = []
        advantages = torch.tensor([], device='cuda')
    else:
        seq_lengths = [packing_info.seq_lengths[idx] for idx in seq_indices]
        advantages = packing_context.bin_advantages[bin_idx]

    # Extract packed inference_logprobs if available
    packed_inference_logprobs = getattr(packing_context, 'packed_inference_logprobs', None)
    if packed_inference_logprobs is not None and logprobs_is_correction:
        inference_logprobs = packed_inference_logprobs[idx]
    else:
        inference_logprobs = None

    return (
        tokens,
        advantages,
        old_logprobs,
        loss_mask,
        position_ids,
        ref_logprobs,
        inference_logprobs,
        seq_starts,
        seq_lengths,
        seq_indices,
        packed_seq_params,
    )


def log_packing_efficiency(packing_context: PackingContext):
    # Log packing efficiency (for this rank's bins)
    packing_info = packing_context.packing_info
    packed_trajs = packing_context.packed_trajs
    my_bin_seq_indices = packing_info.bin_seq_indices
    num_bins = len(packing_info.bin_seq_indices)
    total_tokens = sum(packing_info.seq_lengths)  # All sequences
    my_sequences = sum(len(indices) for indices in my_bin_seq_indices)
    my_tokens = sum(
        packing_info.seq_lengths[idx]
        for indices in my_bin_seq_indices
        for idx in indices
    )
    total_capacity = packed_trajs.shape[0] * packed_trajs.shape[1]
    packing_efficiency = my_tokens / total_capacity if total_capacity > 0 else 0
    avg_seq_length = total_tokens / len(packing_info.seq_lengths)
    rank = mpu.get_expert_data_parallel_rank()
    expert_data_parallel_world_size = mpu.get_expert_data_parallel_world_size()

    log_single_rank(logger, logging.INFO, f"[Sequence Packing] Statistics:")
    log_single_rank(
        logger,
        logging.INFO,
        f"[Sequence Packing]  - Total sequences: {len(packing_info.seq_lengths)}",
    )
    log_single_rank(
        logger, logging.INFO, f"[Sequence Packing]  - Total bins: {num_bins}"
    )
    log_single_rank(
        logger,
        logging.INFO,
        f"[Sequence Packing]  - Bin size: {packed_trajs.shape[1]} tokens",
    )
    log_single_rank(
        logger,
        logging.INFO,
        f"[Sequence Packing]  - Average sequence length: {avg_seq_length:.1f} tokens",
    )
    log_single_rank(
        logger,
        logging.INFO,
        f"[Sequence Packing]  - This rank: {my_sequences} sequences in {packed_trajs.shape[0]} bins",
    )
    log_single_rank(
        logger,
        logging.INFO,
        f"[Sequence Packing]  - Packing efficiency: {packing_efficiency:.1%} ({my_tokens:,} / {total_capacity:,} tokens)",
    )

    # Add detailed per-rank sequence distribution analysis
    if torch.distributed.is_initialized():
        # Gather sequence counts from all ranks
        seq_counts_per_bin = [len(indices) for indices in my_bin_seq_indices]
        non_empty_bins = [c for c in seq_counts_per_bin if c > 0]

        # Create tensor with rank statistics
        rank_stats = torch.tensor(
            [
                float(rank),
                float(len(my_bin_seq_indices)),  # total bins
                float(len(non_empty_bins)),  # non-empty bins
                float(my_sequences),  # total sequences
                (
                    float(min(non_empty_bins)) if non_empty_bins else 0.0
                ),  # min sequences per bin
                (
                    float(max(non_empty_bins)) if non_empty_bins else 0.0
                ),  # max sequences per bin
                (
                    float(my_sequences / len(non_empty_bins)) if non_empty_bins else 0.0
                ),  # avg sequences per non-empty bin
            ],
            device='cuda',
        )

        # Gather from all ranks
        world_size = mpu.get_data_parallel_world_size()
        all_rank_stats = [torch.zeros_like(rank_stats) for _ in range(world_size)]
        torch.distributed.all_gather(
            all_rank_stats, rank_stats, group=mpu.get_data_parallel_group()
        )

        # Print detailed statistics for each rank
        if rank == 0:
            log_single_rank(
                logger,
                logging.INFO,
                f"[Sequence Packing] Per-rank distribution ({packing_info.packing_algo} algorithm):",
            )
            log_single_rank(
                logger,
                logging.INFO,
                "[Sequence Packing]  Rank | Total Bins | Non-empty | Sequences | Min/Bin | Max/Bin | Avg/Bin",
            )
            log_single_rank(
                logger,
                logging.INFO,
                "[Sequence Packing]  -----|------------|-----------|-----------|---------|---------|--------",
            )
            for stats in all_rank_stats:
                r = int(stats[0].item())
                total_bins = int(stats[1].item())
                non_empty = int(stats[2].item())
                sequences = int(stats[3].item())
                min_seq = int(stats[4].item())
                max_seq = int(stats[5].item())
                avg_seq = stats[6].item()
                log_single_rank(
                    logger,
                    logging.INFO,
                    f"[Sequence Packing]   {r:3d} | {total_bins:10d} | {non_empty:9d} | {sequences:9d} | {min_seq:7d} | {max_seq:7d} | {avg_seq:6.1f}",
                )

            # Also show first few bins for rank 0 as example
            log_single_rank(
                logger,
                logging.INFO,
                f"[Sequence Packing]  Example (Rank 0 first 10 bins): {seq_counts_per_bin[:10]}",
            )

            # Show the improvement from round-robin
            total_seqs_all_ranks = sum(int(stats[3].item()) for stats in all_rank_stats)
            avg_seqs_per_rank = total_seqs_all_ranks / world_size
            max_deviation = max(
                abs(int(stats[3].item()) - avg_seqs_per_rank)
                for stats in all_rank_stats
            )
            log_single_rank(
                logger,
                logging.INFO,
                f"[Sequence Packing]  Round-robin distribution quality:",
            )
            log_single_rank(
                logger,
                logging.INFO,
                f"[Sequence Packing]  - Average sequences per rank: {avg_seqs_per_rank:.1f}",
            )
            log_single_rank(
                logger,
                logging.INFO,
                f"[Sequence Packing]  - Max deviation from average: {max_deviation:.0f} sequences ({max_deviation/avg_seqs_per_rank*100:.1f}%)",
            )

def get_actual_sequence_lengths(sequences: torch.Tensor, pad_token: int) -> List[int]:
    """Get actual sequence lengths for pre-padded sequences.

    Args:
        sequences: Tensor of shape [batch_size, seq_len] with pre-padded sequences
        pad_token: The padding token ID

    Returns:
        List of actual sequence lengths (excluding padding)
    """
    if len(sequences.shape) != 2:
        raise ValueError(f"Expected 2D tensor, got shape {sequences.shape}")

    actual_lengths = []

    # Find actual length of each sequence by locating where padding starts
    for seq in sequences:
        # Find the last non-padding token
        non_pad_mask = seq != pad_token
        if non_pad_mask.any():
            # Get the position of the last non-padding token
            actual_length = non_pad_mask.nonzero(as_tuple=True)[0][-1].item() + 1
        else:
            actual_length = 0  # All padding
        actual_lengths.append(actual_length)

    return actual_lengths


def create_empty_bins(
    num_empty_bins : int,
    bin_size : int,
    packed_trajs : torch.Tensor,
    packed_position_ids : torch.Tensor,
    packed_loss_mask : torch.Tensor,
    packed_attention_mask : torch.Tensor,
    tokenizer,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """Create empty bins for padding to ensure all ranks have the same number of bins.

    Args:
        num_empty_bins: Number of empty bins to create
        bin_size: Size of each bin
        packed_trajs: Packed trajectories tensor (for dtype/device reference)
        packed_position_ids: Packed position IDs tensor (for dtype/device reference)
        packed_loss_mask: Packed loss mask tensor (for dtype/device reference)
        packed_attention_mask: Packed attention mask tensor (can be None)
        tokenizer: Tokenizer for pad token

    Returns:
        Tuple of (empty_trajs, empty_position_ids, empty_loss_mask, empty_attention_mask, empty_packing_info_entries)
    """
    device = packed_trajs.device

    # Create empty bins with proper shape
    empty_bins = []
    empty_position_ids_list = []
    empty_loss_mask_list = []
    empty_attention_mask_list = []
    empty_packing_info_entries = []

    for i in range(num_empty_bins):
        # Trajectories filled with pad tokens
        empty_bin = torch.full(
            (1, bin_size), tokenizer.pad, dtype=packed_trajs.dtype, device=device
        )
        empty_bins.append(empty_bin)

        # Zero position IDs
        empty_pos_ids = torch.zeros(1, bin_size, dtype=packed_position_ids.dtype, device=device)
        empty_position_ids_list.append(empty_pos_ids)

        # Zero loss mask (so no loss contribution)
        empty_loss = torch.zeros(1, bin_size, dtype=packed_loss_mask.dtype, device=device)
        empty_loss_mask_list.append(empty_loss)

        # Zero attention mask if needed
        if packed_attention_mask is not None:
            # Attention mask is always 4D: [num_bins, 1, bin_size, bin_size]
            empty_attn = torch.zeros(
                1, 1, bin_size, bin_size, dtype=packed_attention_mask.dtype, device=device
            )
            empty_attention_mask_list.append(empty_attn)

        # Empty packing info entries
        empty_packing_info_entries.append(
            {
                'bin_seq_indices': [],  # No sequences in empty bin
                'seq_starts': [],  # No sequence starts
            }
        )

    # Concatenate all empty bins
    if num_empty_bins > 0:
        empty_trajs = torch.cat(empty_bins, dim=0)
        empty_position_ids = torch.cat(empty_position_ids_list, dim=0)
        empty_loss_mask = torch.cat(empty_loss_mask_list, dim=0)
        empty_attention_mask = (
            torch.cat(empty_attention_mask_list, dim=0)
            if packed_attention_mask is not None
            else None
        )
    else:
        empty_trajs = None
        empty_position_ids = None
        empty_loss_mask = None
        empty_attention_mask = None

    return (
        empty_trajs,
        empty_position_ids,
        empty_loss_mask,
        empty_attention_mask,
        empty_packing_info_entries,
    )

def get_default_packed_seq_params(seq_length: int, device: torch.device) -> PackedSeqParams:
    """Create a default PackedSeqParams that acts as no-op for a single sequence.

    This ensures CUDA graph signature consistency when packed_seq_params
    would otherwise be None. A single sequence spanning the full length
    means no actual packing boundaries

    Args:
        seq_length: The sequence length 
        device: Device to create tensors on.

    Returns:
        PackedSeqParams configured as a single unpacked sequence.
    """
    # Single sequence spanning the full length = no actual packing
    cu_seqlens = torch.tensor([0, seq_length], dtype=torch.int32, device=device)

    return PackedSeqParams(
        qkv_format='thd',
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv_padded=None,
        max_seqlen_q=seq_length,
        max_seqlen_kv=seq_length,
    )

def create_packed_seq_params(packing_context: PackingContext):
    cached_packed_seq_params = []
    packing_info = packing_context.packing_info
    bin_size = packing_context.bin_size
    device = packing_context.packed_trajs.device
    for bin_idx in range(len(packing_context.packed_trajs)):
        params = create_packed_seq_params_for_bin(
            packing_info=packing_info,
            bin_idx=bin_idx,
            bin_size=bin_size,
            device=device,
        )
        cached_packed_seq_params.append(params)
    return cached_packed_seq_params

def create_packed_seq_params_for_bin(
    packing_info: PackingInfo, bin_idx: int, bin_size: int, device: torch.device
) -> Optional[PackedSeqParams]:
    """Create PackedSeqParams for a single bin to enable proper attention masking in TE.

    When using Transformer Engine with sequence packing, we need to provide cu_seqlens
    (cumulative sequence lengths) so that TE knows the boundaries between sequences
    within a packed bin. This prevents attention leakage between unrelated sequences.

    Args:
        packing_info: PackingInfo object containing packing metadata from SequencePacker
        bin_idx: Index of the bin to create params for
        bin_size: Size of the bin (padded sequence length)
        device: Device to create tensors on

    Returns:
        PackedSeqParams with cu_seqlens set for proper attention masking (or None if empty)
    """
    seq_indices = packing_info.bin_seq_indices[bin_idx]

    # Handle empty bins (padding bins with no sequences)
    if not seq_indices:
        return None

    # Get actual sequence lengths for sequences in this bin
    seq_lengths_in_bin = [packing_info.seq_lengths[idx] for idx in seq_indices]

    # Build cumulative sequence lengths for actual sequences
    # cu_seqlens should be [0, len(seq1), len(seq1)+len(seq2), ..., total_actual_len]
    cu_seqlens_list = np.append(np.cumsum([0] + seq_lengths_in_bin), bin_size)

    cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=device)

    # Pad cu_seqlens to bin_size by repeating the last value (creates zero-length ghost sequences)
    # This ensures a fixed tensor size for CUDA graph compatibility
    if len(cu_seqlens) < bin_size:
        out = cu_seqlens.new_full((bin_size,), bin_size)
        out[:len(cu_seqlens)] = cu_seqlens
        cu_seqlens = out

    max_seqlen = bin_size

    return PackedSeqParams(
        qkv_format='thd',
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv_padded=None,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )


def pack_inference_logprobs(
    inference_logprobs: List[torch.Tensor],
    packing_info: PackingInfo,
    generation_masks: torch.Tensor,
    bin_size: int,
) -> torch.Tensor:
    """Pack inference logprobs into bins aligned with packed sequences.

    Args:
        inference_logprobs: List of inference logprobs tensors for each sequence
        packing_info: PackingInfo object containing bin assignments and sequence positions
        generation_masks: Tensor indicating which tokens were generated
        bin_size: Size of each bin

    Returns:
        Packed inference logprobs tensor of shape [num_bins, bin_size - 1]
    """
    num_bins = len(packing_info.bin_seq_indices)

    # Create packed inference logprobs tensor (logprobs are 1 token shorter than sequences)
    packed_inference_logprobs = torch.zeros(
        (num_bins, bin_size - 1), dtype=torch.float32, device='cpu'
    )

    # Create mapping from global sequence index to local bin index
    # This is needed because seq_to_bin_idx uses global bin indices,
    # but after distribution each rank only has a subset of bins
    seq_to_local_bin = {}
    for local_bin_idx, seq_indices in enumerate(packing_info.bin_seq_indices):
        for seq_idx in seq_indices:
            seq_to_local_bin[seq_idx] = local_bin_idx

    # Align and pack inference logprobs based on generation masks
    for seq_idx in range(len(inference_logprobs)):
        if seq_idx not in seq_to_local_bin:
            continue  # Skip sequences not on this rank

        local_bin_idx = seq_to_local_bin[seq_idx]

        # Get the position of this sequence within the bin
        seq_positions = packing_info.bin_seq_indices[local_bin_idx]
        seq_pos_in_bin = seq_positions.index(seq_idx)
        seq_start = packing_info.seq_starts[local_bin_idx][seq_pos_in_bin]

        # Get generation mask for this sequence to find where generation starts
        gen_mask = generation_masks[seq_idx]
        # Find first generation token (accounting for the shift in get_logprobs)
        first_gen_idx = gen_mask.int().argmax().item() - 1

        # Get the inference logprobs for this sequence
        if isinstance(inference_logprobs[seq_idx], torch.Tensor):
            seq_inf_logprobs = inference_logprobs[seq_idx]
        else:
            continue  # Skip if no inference logprobs

        # Calculate where to place inference logprobs in the packed tensor
        # The inference logprobs start at the first generated token position
        pack_start = seq_start + first_gen_idx
        pack_end = min(
            pack_start + len(seq_inf_logprobs), seq_start + packing_info.seq_lengths[seq_idx] - 1
        )
        actual_len = pack_end - pack_start

        if actual_len > 0 and pack_end <= bin_size - 1:
            packed_inference_logprobs[local_bin_idx, pack_start:pack_end] = seq_inf_logprobs[
                :actual_len
            ]

    return packed_inference_logprobs


def compute_packed_inference_logprobs_stats(
    old_logprobs: torch.Tensor,
    packed_inference_logprobs: torch.Tensor,
    packed_loss_mask: torch.Tensor,
    group_stats: Any,
) -> None:
    """Compute statistics for packed inference logprobs for logging purposes.

    Compares packed inference logprobs with old logprobs using the packed loss mask
    to identify valid positions. Updates group_stats with computed metrics.

    Args:
        old_logprobs: Old logprobs tensor in packed format [num_bins, seq_len-1]
        packed_inference_logprobs: Packed inference logprobs [num_bins, seq_len-1]
        packed_loss_mask: Loss mask indicating valid positions [num_bins, seq_len]
        group_stats: Statistics object to update with computed metrics
    """
    # Lazy import to avoid circular dependency (rl_utils imports from this module)
    from megatron.rl.rl_utils import update_inference_logprobs_group_stats

    # Ensure all tensors are on the same device (CPU for stats computation)
    old_logprobs = old_logprobs.cpu()
    packed_inference_logprobs = packed_inference_logprobs.cpu()
    packed_loss_mask = packed_loss_mask.cpu()

    # Use packed_loss_mask to identify valid positions for stats (shift by 1 for logprobs)
    mask = packed_loss_mask[:, 1:].bool()

    # Ensure shapes match
    if mask.shape != old_logprobs.shape:
        return

    # Update group statistics using common helper
    update_inference_logprobs_group_stats(
        old_logprobs=old_logprobs,
        inference_logprobs=packed_inference_logprobs,
        mask=mask,
        group_stats=group_stats,
    )


class SequencePacker:
    """Packs multiple sequences into bins to minimize padding and improve GPU utilization."""

    def __init__(self, bin_size: int, pad_token: int, max_sequences_per_bin: int = 16):
        self.bin_size = bin_size
        self.pad_token = pad_token
        self.max_sequences_per_bin = max_sequences_per_bin

    def pack_sequences(
        self, trajs: torch.Tensor, generation_masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, PackingInfo]:
        """Pack sequences into bins using a greedy first-fit algorithm."""
        # Convert trajectories to list for packing
        sequences = [trajs[i] for i in range(trajs.shape[0])]

        sequences_tensor = torch.stack(sequences)

        seq_lengths = get_actual_sequence_lengths(sequences_tensor, self.pad_token)

        # Trim sequences to actual lengths
        sequences = [sequences_tensor[i, :length] for i, length in enumerate(seq_lengths)]

        sorted_indices = sorted(range(len(sequences)), key=lambda i: seq_lengths[i], reverse=True)

        bins = []
        bin_seq_indices = []  # Track which sequences are in each bin
        current_bin = []
        current_bin_indices = []
        current_bin_length = 0

        # Pack sequences into bins
        sequences_per_bin = []
        for idx in sorted_indices:
            seq = sequences[idx]
            seq_len = len(seq)

            if (
                current_bin_length + seq_len <= self.bin_size
                and len(current_bin) < self.max_sequences_per_bin
            ):
                current_bin.append(seq)
                current_bin_indices.append(idx)
                current_bin_length += seq_len
            else:
                # Start a new bin
                if current_bin:
                    bins.append(current_bin)
                    bin_seq_indices.append(current_bin_indices)
                    sequences_per_bin.append(len(current_bin))
                current_bin = [seq]
                current_bin_indices = [idx]
                current_bin_length = seq_len

        # Don't forget the last bin
        if current_bin:
            bins.append(current_bin)
            bin_seq_indices.append(current_bin_indices)
            sequences_per_bin.append(len(current_bin))

        # Create packed tensors
        num_bins = len(bins)
        device = sequences[0].device
        dtype = sequences[0].dtype

        # Log packing distribution
        if sequences_per_bin:
            avg_seqs_per_bin = sum(sequences_per_bin) / len(sequences_per_bin)
            min_seqs = min(sequences_per_bin)
            max_seqs = max(sequences_per_bin)
            log_single_rank(
                logger,
                logging.INFO,
                (
                    f"[SequencePacker] Packing distribution: {num_bins} bins, "
                    f"avg {avg_seqs_per_bin:.1f} seqs/bin, "
                    f"min {min_seqs}, max {max_seqs} seqs/bin "
                    f"(limit: {self.max_sequences_per_bin})"
                ),
            )
            # Store for later use
            self.last_avg_seqs_per_bin = avg_seqs_per_bin

        packed_sequences = torch.full(
            (num_bins, self.bin_size), self.pad_token, dtype=dtype, device=device
        )
        position_ids = torch.zeros(
            (num_bins, self.bin_size), dtype=torch.long, device=device, requires_grad=False
        )
        attention_mask = torch.zeros(
            (num_bins, 1, self.bin_size, self.bin_size), dtype=torch.bool, device=device
        )
        loss_mask = torch.zeros((num_bins, self.bin_size), dtype=torch.float, device=device)

        # Track packing information for unpacking later
        seq_starts_dict: Dict[int, List[int]] = {}
        seq_to_bin_idx: List[Optional[int]] = [None] * len(sequences)

        # Build seq_to_bin_idx mapping
        for bin_idx, seq_indices in enumerate(bin_seq_indices):
            for seq_idx in seq_indices:
                seq_to_bin_idx[seq_idx] = bin_idx

        # Fill bins
        for bin_idx, (bin_seqs, seq_indices) in enumerate(zip(bins, bin_seq_indices)):
            seq_starts = []
            current_pos = 0

            for seq_idx, seq in enumerate(bin_seqs):
                start = current_pos
                end = start + len(seq)
                seq_starts.append(start)
                current_pos = end

                # Pack sequence
                packed_sequences[bin_idx, start:end] = seq

                # Position IDs reset for each sequence
                position_ids[bin_idx, start:end] = torch.arange(
                    len(seq), device=device, requires_grad=False
                )

                # Causal attention mask within each sequence
                seq_len = end - start
                attention_mask[bin_idx, 0, start:end, start:end] = torch.tril(
                    torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
                )

                # Loss mask (excluding padding)
                loss_mask[bin_idx, start:end] = 1.0

                # Apply generation mask if provided
                if generation_masks is not None:
                    orig_idx = seq_indices[seq_idx]
                    gen_mask = generation_masks[orig_idx][
                        : len(seq)
                    ]  # Truncate to actual seq length
                    loss_mask[bin_idx, start:end] *= gen_mask.float()

            seq_starts.append(current_pos)
            seq_starts_dict[bin_idx] = seq_starts

        # Note: We'll store the actual padded length later when we know it
        # (it depends on the original trajectories passed to pack_sequences)

        # Invert attention mask, before inversion: (True = attend, False = mask)
        attention_mask = ~attention_mask

        # Create the PackingInfo dataclass
        packing_info = PackingInfo(
            bin_seq_indices=bin_seq_indices,
            seq_starts=seq_starts_dict,
            seq_lengths=seq_lengths,
            seq_to_bin_idx=seq_to_bin_idx,
            packing_algo='fifo'
        )

        seq_per_bin = [len(indices) for indices in packing_info.bin_seq_indices]
        log_single_rank(
            logger, logging.DEBUG, (f"Initial packing output (before distribution):")
        )
        log_single_rank(
            logger,
            logging.DEBUG,
            f"  - Total bins created: {len(packing_info.bin_seq_indices)}",
        )
        log_single_rank(
            logger, logging.DEBUG, f"  - Total sequences packed: {sum(seq_per_bin)}"
        )
        log_single_rank(
            logger,
            logging.DEBUG,
            f"  - Sequences per bin: min={min(seq_per_bin)}, max={max(seq_per_bin)}, avg={sum(seq_per_bin)/len(seq_per_bin):.1f}",
        )
        log_single_rank(logger, logging.DEBUG, f"  - First 20 bins: {seq_per_bin[:20]}")

        return packed_sequences, position_ids, attention_mask, loss_mask, packing_info

def distribute_packed_bins(
    packed_trajs: torch.Tensor,
    packed_position_ids: torch.Tensor,
    packed_attention_mask: torch.Tensor,
    packed_loss_mask: torch.Tensor,
    packing_info: PackingInfo,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, PackingInfo]:
    """Distribute packed bins across the data parallel ranks."""
    rank = mpu.get_expert_data_parallel_rank()
    world_size = mpu.get_expert_data_parallel_world_size()
    tokenizer = get_tokenizer()

    # Distribute packed bins across data parallel ranks
    num_bins, bin_size = packed_trajs.shape
    packing_algo = packing_info.packing_algo

    if packing_algo == 'round-robin':
        # Round-robin assignment: rank i gets bins [i, i+world_size, i+2*world_size, ...]
        my_bin_indices = list(range(rank, num_bins, world_size))
    else:  # fifo (default)
        world_size = world_size if world_size > 0 else 1
        # FIFO assignment: divide bins sequentially across ranks
        bins_per_rank = num_bins // world_size
        extra_bins = num_bins % world_size

        # Calculate start and end indices for this rank
        if rank < extra_bins:
            # Ranks with extra bins
            start_idx = rank * (bins_per_rank + 1)
            end_idx = start_idx + bins_per_rank + 1
        else:
            # Ranks without extra bins
            start_idx = rank * bins_per_rank + extra_bins
            end_idx = start_idx + bins_per_rank

        my_bin_indices = list(range(start_idx, end_idx))

    # Calculate the maximum bins any rank has (for synchronization)
    max_bins_per_rank = (num_bins + world_size - 1) // world_size

    # Extract this rank's bins
    my_packed_trajs = []
    my_packed_position_ids = []
    my_packed_attention_mask = []
    my_packed_loss_mask = []
    my_bin_seq_indices = []
    my_seq_starts = {}


    # Build the local data from the global indices
    for new_idx, old_idx in enumerate(my_bin_indices):
        my_packed_trajs.append(packed_trajs[old_idx])
        my_packed_position_ids.append(packed_position_ids[old_idx])
        if packed_attention_mask is not None:
            my_packed_attention_mask.append(packed_attention_mask[old_idx])
        my_packed_loss_mask.append(packed_loss_mask[old_idx])
        my_bin_seq_indices.append(packing_info.bin_seq_indices[old_idx])
        my_seq_starts[new_idx] = packing_info.seq_starts[old_idx]

    # Stack the selected bins
    packed_trajs = (
        torch.stack(my_packed_trajs)
        if my_packed_trajs
        else torch.empty(
            0,
            packed_trajs.shape[1],
            dtype=packed_trajs.dtype,
            device=packed_trajs.device,
        )
    )
    packed_position_ids = (
        torch.stack(my_packed_position_ids)
        if my_packed_position_ids
        else torch.empty(
            0,
            packed_position_ids.shape[1],
            dtype=packed_position_ids.dtype,
            device=packed_position_ids.device,
        )
    )
    packed_attention_mask = (
        torch.stack(my_packed_attention_mask) if my_packed_attention_mask else None
    )
    packed_loss_mask = (
        torch.stack(my_packed_loss_mask)
        if my_packed_loss_mask
        else torch.empty(
            0,
            packed_loss_mask.shape[1],
            dtype=packed_loss_mask.dtype,
            device=packed_loss_mask.device,
        )
    )

    # Debug: Check what we're extracting
    log_single_rank(logger, logging.DEBUG, (f"Rank 0 {packing_algo} bin assignment:"))
    log_single_rank(
        logger, logging.DEBUG, f"  - Total bins before distribution: {num_bins}"
    )
    log_single_rank(
        logger,
        logging.DEBUG,
        f"  - Bins assigned to rank 0: {my_bin_indices[:10]}... (showing first 10)",
    )
    log_single_rank(
        logger,
        logging.DEBUG,
        f"  - Number of bins for this rank: {len(my_bin_indices)}",
    )
    log_single_rank(
        logger,
        logging.DEBUG,
        f"  - Length of my_bin_seq_indices: {len(my_bin_seq_indices)}",
    )
    if len(my_bin_seq_indices) > 0:
        log_single_rank(
            logger,
            logging.DEBUG,
            f"  - Sequences in first 5 bins: {[len(indices) for indices in my_bin_seq_indices[:5]]}",
        )

    # Create updated packing info for this rank
    new_packing_info = PackingInfo(
        bin_seq_indices=my_bin_seq_indices,
        seq_starts=my_seq_starts,
        seq_lengths=packing_info.seq_lengths,  # Keep all sequence lengths
        seq_to_bin_idx=packing_info.seq_to_bin_idx,  # Keep mapping
        packing_algo=packing_algo,
    )

    # Add empty bins if this rank has fewer than max_bins_per_rank
    current_bins = len(my_bin_indices)
    if current_bins < max_bins_per_rank:
        num_empty_bins = max_bins_per_rank - current_bins

        # Create empty bins using the helper function
        (
            empty_trajs,
            empty_position_ids,
            empty_loss_mask,
            empty_attention_mask,
            empty_packing_entries,
        ) = create_empty_bins(
            num_empty_bins,
            bin_size,
            packed_trajs,
            packed_position_ids,
            packed_loss_mask,
            packed_attention_mask,
            tokenizer,
        )

        # Append empty bins to packed tensors
        packed_trajs = torch.cat([packed_trajs, empty_trajs], dim=0)
        packed_position_ids = torch.cat(
            [packed_position_ids, empty_position_ids], dim=0
        )
        packed_loss_mask = torch.cat([packed_loss_mask, empty_loss_mask], dim=0)

        if packed_attention_mask is not None and empty_attention_mask is not None:
            packed_attention_mask = torch.cat(
                [packed_attention_mask, empty_attention_mask], dim=0
            )

        # Add empty entries to packing_info
        for i, entry in enumerate(empty_packing_entries):
            bin_idx = current_bins + i
            new_packing_info.bin_seq_indices.append(entry['bin_seq_indices'])
            new_packing_info.seq_starts[bin_idx] = entry['seq_starts']

    return packed_trajs, packed_position_ids, packed_attention_mask, packed_loss_mask, new_packing_info


def pack_all_trajectories(trajs, generation_masks, inference_logprobs, global_advantages, bin_size, max_sequences_per_bin, packing_algo):
    tokenizer = get_tokenizer()
    expert_data_parallel_world_size = mpu.get_expert_data_parallel_world_size()
    nvtx_range = get_nvtx_range()

    with nvtx_range("regather_trajectories", time=True):
        # Regather trajectories from all ranks for packing
        trajs = trajs.cuda()
        trajs_list = [torch.empty_like(trajs) for _ in range(expert_data_parallel_world_size)]
        torch.distributed.all_gather(
            trajs_list, trajs, group=mpu.get_expert_data_parallel_group()
        )
        trajs = torch.cat(trajs_list, dim=0)

        # Gather all generation masks
        generation_masks = generation_masks.cuda()
        masks_list = [torch.empty_like(generation_masks) for _ in range(expert_data_parallel_world_size)]
        torch.distributed.all_gather(
            masks_list, generation_masks, group=mpu.get_expert_data_parallel_group()
        )
        generation_masks = torch.cat(masks_list, dim=0)

        # Gather inference logprobs if present
        if inference_logprobs is not None:
            inference_logprobs = inference_logprobs.cuda()
            logprobs_list = [torch.empty_like(inference_logprobs) for _ in range(expert_data_parallel_world_size)]
            torch.distributed.all_gather(
                logprobs_list, inference_logprobs, group=mpu.get_expert_data_parallel_group()
            )
            inference_logprobs = torch.cat(logprobs_list, dim=0)

    with nvtx_range("pack_sequences", time=True):
        # Create packer with max sequences per bin limit to prevent extreme imbalance
        packer = SequencePacker(
            bin_size=bin_size,
            pad_token=tokenizer.pad,
            max_sequences_per_bin=max_sequences_per_bin,
        )

        # Pack sequences with generation masks
        (
            packed_trajs,
            packed_position_ids,
            packed_attention_mask,
            packed_loss_mask,
            packing_info,
        ) = packer.pack_sequences(trajs, generation_masks)
        packing_info.packing_algo = packing_algo

        # Distribute packed bins across the data parallel ranks
        (
            packed_trajs,
            packed_position_ids,
            packed_attention_mask,
            packed_loss_mask,
            packing_info,
        ) = distribute_packed_bins(
            packed_trajs,
            packed_position_ids,
            packed_attention_mask,
            packed_loss_mask,
            packing_info,
        )

    # Create bin_advantages list
    bin_advantages = []
    for seq_indices in packing_info.bin_seq_indices:
        if seq_indices:
            bin_advantages.append(global_advantages[seq_indices])
        else:
            bin_advantages.append(
                torch.tensor([], dtype=global_advantages.dtype, device=global_advantages.device)
            )

    # Pre-compute all PackedSeqParams for all bins ONCE to avoid repeated
    # tensor allocations that cause CUDA memory fragmentation and periodic spikes
    # Create a temporary packing context to pass to create_packed_seq_params
    cached_packed_seq_params = [
        create_packed_seq_params_for_bin(
                packing_info=packing_info,
                bin_idx=bin_idx,
                bin_size=bin_size,
                device=packed_trajs.device,
            ) for bin_idx in range(len(packed_trajs))
    ]

    # Create the final PackingContext
    packing_context = PackingContext(
        bin_size=bin_size,
        packer=packer,
        packing_info=packing_info,
        original_generation_masks=generation_masks,
        original_trajs=trajs,
        packed_trajs=packed_trajs,
        packed_position_ids=packed_position_ids,
        packed_attention_mask=packed_attention_mask,
        packed_loss_mask=packed_loss_mask,
        original_inference_logprobs=inference_logprobs,
        bin_advantages=bin_advantages,
        cached_packed_seq_params=cached_packed_seq_params,
    )

    log_packing_efficiency(packing_context)

    return packing_context


def get_microbatch_dataloader(packing_context: PackingContext) -> Tuple[DataLoader, int]:
    args = get_args()
    num_bins_this_rank = len(packing_context.packed_trajs)
    dp_world_size = mpu.get_data_parallel_world_size()

    # Ratio of collected sequences to the global batch size
    pct_of_sequences_per_batch = len(packing_context.packing_info.seq_lengths) / args.global_batch_size

    # Ceiling division means we will reuse some bins
    # If we did floor we would leave some behind
    local_bins_per_step = math.ceil(pct_of_sequences_per_batch * num_bins_this_rank)
    effective_global_batch_size = local_bins_per_step * dp_world_size

    # Store packing plan in runtime state for the training loop to use
    optimizer_steps = -(-num_bins_this_rank // local_bins_per_step)

    old_num_microbatches = get_num_microbatches()

    reconfigure_num_microbatches_calculator(
        rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
        rampup_batch_size=args.rampup_batch_size,
        global_batch_size=effective_global_batch_size,
        micro_batch_size=args.micro_batch_size,
        data_parallel_size=dp_world_size,
        decrease_batch_size_if_needed=args.decrease_batch_size_if_needed,
    )

    new_num_microbatches = get_num_microbatches()

    log_single_rank(
        logger, logging.INFO, f"[Sequence Packing] Multi-step training plan:"
    )
    log_single_rank(
        logger,
        logging.INFO,
        f"[Sequence Packing]  - Target sequences per step: {args.global_batch_size}",
    )
    log_single_rank(
        logger,
        logging.INFO,
        f"[Sequence Packing]  - Bins per rank per step: {pct_of_sequences_per_batch}*{num_bins_this_rank}={local_bins_per_step}",
    )
    log_single_rank(
        logger,
        logging.INFO,
        f"[Sequence Packing]  - Total optimizer steps: {optimizer_steps}",
    )
    log_single_rank(
        logger,
        logging.INFO,
        f"[Sequence Packing]  - Microbatches per step: {new_num_microbatches} (was {old_num_microbatches})",
    )

    bin_seq_indices = packing_context.packing_info.bin_seq_indices
    for step in range(min(3, optimizer_steps)):
        start_idx = step * local_bins_per_step
        end_idx = min(start_idx + local_bins_per_step, num_bins_this_rank)
        step_bins = end_idx - start_idx

        actual_seqs = sum(
            len(bin_seq_indices[bin_idx])
            for bin_idx in range(start_idx, end_idx)
            if bin_idx < len(bin_seq_indices)
        )
        est_global_seqs = actual_seqs * dp_world_size
        log_single_rank(
            logger,
            logging.INFO,
            f"[Sequence Packing]  - Step {step + 1}: {step_bins} bins, ~{est_global_seqs} sequences globally",
        )

    if optimizer_steps > 3:
        log_single_rank(logger, logging.INFO, f"  - ... ({optimizer_steps - 3} more steps)")

    bin_indices = torch.arange(num_bins_this_rank)
    dataset = TensorDataset(bin_indices)
    loader = DataLoader(dataset, batch_size=args.micro_batch_size, shuffle=False, collate_fn=lambda x: x[0], drop_last=True)
    return loader, optimizer_steps

def update_sequence_packing_metrics(args):
    """Update bin tracking for sequence packing mode."""
    if args.rl_use_sequence_packing:
        bin_count = (
            mpu.get_data_parallel_world_size() * args.micro_batch_size * get_num_microbatches()
        )
        args.consumed_train_bins += bin_count


def get_sequence_packing_log_info(args):
    """Get logging information for sequence packing mode."""
    if args.consumed_train_bins > 0:
        return f' consumed bins: {args.consumed_train_bins:12d} |'
    return ''


def get_sequence_packing_tensorboard_metrics(args):
    """Get tensorboard metrics for sequence packing mode."""
    metrics = {}
    if args.consumed_train_bins > 0:
        bin_batch_size = (
            mpu.get_data_parallel_world_size() * args.micro_batch_size * get_num_microbatches()
        )
        metrics['bin-batch-size'] = bin_batch_size
        metrics['consumed-bins'] = args.consumed_train_bins
    return metrics
