# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc

# Keep this to make the env registered.
import itertools
import logging
import math
import pickle
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from wandb import wandb_run

from megatron.core import mpu
from megatron.core.datasets.megatron_tokenizer import MegatronLegacyTokenizer
from megatron.core.inference.utils import get_event_loop
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.optimizer import MegatronOptimizer
from megatron.core.parallel_state import get_tensor_model_parallel_src_rank
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.core.transformer.cuda_graphs import _CudagraphGlobalRecord
from megatron.core.transformer.utils import toggle_cuda_graphs
from megatron.rl.agent.api import (
    EvaluationRequest,
    EvaluationResponse,
    GroupedRolloutRequest,
    RewardEvaluationResult,
    Rollout,
    TokenRollout,
)
from megatron.rl.agent.weighted_multi_task import WeightedMultiTask
from megatron.rl.inference.megatron import MegatronChatLocal, MegatronLocal
from megatron.rl.logging import LOG_DIR as lang_rl_log_dir
from megatron.rl.logging import log as lang_rl_log
from megatron.rl.server.inference.inference_interface_server import InferenceInterfaceServer
from megatron.training.global_vars import (
    get_args,
    get_tensorboard_writer,
    get_timers,
    get_tokenizer,
    get_wandb_writer,
)
from megatron.training.tokenizer.tokenizer import CustomTikTokenizer, _HuggingFaceTokenizer
from megatron.training.utils import get_ltor_masks_and_position_ids, get_nvtx_range, print_rank_0

logger = logging.getLogger(__name__)

# Global variable to store packing context for forward_step
_GLOBAL_PACKING_CONTEXT = None

GroupedRollouts = list[list[TokenRollout | Rollout]]


@dataclass(slots=True)
class RolloutStats:
    mean_reward: float
    mean_sim: None | float
    mean_length: float
    mean_length_std: float
    max_length: float
    min_length: float
    reward_means: list[float]
    reward_stds: list[float]
    rewards: list[float]
    min_piold_to_inf_prob: None | float
    max_piold_to_inf_prob: None | float
    mean_piold_to_inf_prob: None | float
    min_inf_train_prob_abs_diff: None | float
    max_inf_train_prob_abs_diff: None | float
    mean_inf_train_prob_abs_diff: None | float
    advantages: None | list[list[float]]
    min_inf_prob: None | float
    max_inf_prob: None | float
    mean_inf_prob: None | float


# Runtime state container for RL-specific data that shouldn't be checkpointed
class RLRuntimeState:
    """Container for seq packing runtime state that is rank-specific"""

    def __init__(self):
        self.sequence_packing_plan = None
        self.sequence_packing_metadata = None
        self.packing_context = None
        self.sequences_this_iteration_on_rank = 0
        self.latest_batch_num_sequences = 0

    def reset_iteration_counters(self):
        """Reset per-iteration counters."""
        self.sequences_this_iteration_on_rank = 0

    def increment_sequences(self, count):
        """Increment the sequence counter."""
        self.sequences_this_iteration_on_rank += count
        self.latest_batch_num_sequences = count


# Global runtime state instance
_rl_runtime_state = RLRuntimeState()


def get_rl_runtime_state():
    """Get the global RL runtime state."""
    return _rl_runtime_state


def create_empty_bins(
    num_empty_bins,
    bin_size,
    packed_trajs,
    packed_position_ids,
    packed_loss_mask,
    packed_attention_mask,
    tokenizer,
):
    """Create empty bins for padding to ensure all ranks have the same number of bins.

    Args:
        num_empty_bins: Number of empty bins to create
        bin_size: Size of each bin
        packed_trajs: Packed trajectories tensor (for dtype/device reference)
        packed_position_ids: Packed position IDs tensor
        packed_loss_mask: Packed loss mask tensor
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


def pack_inference_logprobs(
    inference_logprobs: List[torch.Tensor],
    packing_info: Dict[str, Any],
    generation_masks: torch.Tensor,
    bin_size: int,
) -> torch.Tensor:
    """Pack inference logprobs into bins aligned with packed sequences.

    Args:
        inference_logprobs: List of inference logprobs tensors for each sequence
        packing_info: Dictionary containing bin assignments and sequence positions
        generation_masks: Tensor indicating which tokens were generated
        bin_size: Size of each bin

    Returns:
        Packed inference logprobs tensor of shape [num_bins, bin_size - 1]
    """
    num_bins = len(packing_info['bin_seq_indices'])

    # Create packed inference logprobs tensor (logprobs are 1 token shorter than sequences)
    packed_inference_logprobs = torch.zeros(
        (num_bins, bin_size - 1), dtype=torch.float32, device='cpu'
    )

    # Create mapping from global sequence index to local bin index
    # This is needed because seq_to_bin_idx uses global bin indices,
    # but after distribution each rank only has a subset of bins
    seq_to_local_bin = {}
    for local_bin_idx, seq_indices in enumerate(packing_info['bin_seq_indices']):
        for seq_idx in seq_indices:
            seq_to_local_bin[seq_idx] = local_bin_idx

    # Align and pack inference logprobs based on generation masks
    for seq_idx in range(len(inference_logprobs)):
        if seq_idx not in seq_to_local_bin:
            continue  # Skip sequences not on this rank

        local_bin_idx = seq_to_local_bin[seq_idx]

        # Get the position of this sequence within the bin
        seq_positions = packing_info['bin_seq_indices'][local_bin_idx]
        seq_pos_in_bin = seq_positions.index(seq_idx)
        seq_start = packing_info['seq_starts'][local_bin_idx][seq_pos_in_bin]

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
            pack_start + len(seq_inf_logprobs), seq_start + packing_info['seq_lengths'][seq_idx] - 1
        )
        actual_len = pack_end - pack_start

        if actual_len > 0 and pack_end <= bin_size - 1:
            packed_inference_logprobs[local_bin_idx, pack_start:pack_end] = seq_inf_logprobs[
                :actual_len
            ]

    return packed_inference_logprobs


def align_unpacked_inference_logprobs(
    inference_logprobs: List[torch.Tensor],
    old_logprobs_for_data: torch.Tensor,
    generation_masks: torch.Tensor,
    group_stats: Any,
) -> torch.Tensor:
    """Align inference logprobs with old_logprobs for unpacked sequences and compute statistics.

    Args:
        inference_logprobs: List of inference logprobs tensors for each sequence
        old_logprobs_for_data: Template tensor with correct shape for alignment
        generation_masks: Tensor indicating which tokens were generated
        group_stats: Statistics object to update with computed metrics

    Returns:
        Aligned inference logprobs tensor
    """
    # Get first occurrence of a generation token
    # In get_logprobs() we chop off the first token -> the generation mask is shifted by one
    gen_masks_for_alignment = generation_masks
    first_gen_tok = gen_masks_for_alignment.int().argmax(dim=1) - 1

    # Align inference logprobs with old_logprobs
    # Note: We use old_logprobs_for_data as template since it has correct shape
    padded_inference_logprobs = old_logprobs_for_data.clone()

    # We need to align old_logprobs and inference logprobs as the latter are only for generations
    for i, inf_logprobs in enumerate(inference_logprobs):
        first_gen_idx = first_gen_tok[i]
        # We subtract -1 here because we append eod token on the train side, and we do not
        # get it from the inference. For the eod token, we reuse old_logprobs value.
        end_idx = min(first_gen_idx + len(inf_logprobs), padded_inference_logprobs.shape[1])
        actual_len = end_idx - first_gen_idx
        if actual_len > 0:
            padded_inference_logprobs[i, first_gen_idx:end_idx] = inf_logprobs[:actual_len]

    # Create truncated mask for statistics
    if old_logprobs_for_data.shape[1] + 1 < gen_masks_for_alignment.shape[1]:
        gen_masks_for_alignment = gen_masks_for_alignment[:, : old_logprobs_for_data.shape[1] + 1]

    truncated_mask = gen_masks_for_alignment[:, 1:].bool()

    # Final safety check
    if truncated_mask.shape != old_logprobs_for_data.shape:
        if truncated_mask.shape[1] > old_logprobs_for_data.shape[1]:
            truncated_mask = truncated_mask[:, : old_logprobs_for_data.shape[1]]
        elif truncated_mask.shape[1] < old_logprobs_for_data.shape[1]:
            pad_size = old_logprobs_for_data.shape[1] - truncated_mask.shape[1]
            truncated_mask = torch.nn.functional.pad(truncated_mask, (0, pad_size), value=False)

    # Compute statistics
    n_elems = truncated_mask.sum()

    ratios = (old_logprobs_for_data - padded_inference_logprobs).exp()[truncated_mask]
    abs_diffs = (old_logprobs_for_data.exp() - padded_inference_logprobs.exp()).abs()[
        truncated_mask
    ]

    # Two probability values cannot be more than 1.0 apart
    assert all(abs_diffs <= 1.0)

    # Update group statistics
    group_stats.min_piold_to_inf_prob = ratios.min().item()
    group_stats.max_piold_to_inf_prob = ratios.max().item()
    group_stats.mean_piold_to_inf_prob = (ratios.sum() / n_elems).item()
    group_stats.min_inf_train_prob_abs_diff = abs_diffs.min().item()
    group_stats.max_inf_train_prob_abs_diff = abs_diffs.max().item()
    group_stats.mean_inf_train_prob_abs_diff = (abs_diffs.sum() / n_elems).item()

    # Compute inference probability statistics
    inf_probs = padded_inference_logprobs.exp()[truncated_mask]
    group_stats.min_inf_prob = inf_probs.min().item()
    group_stats.max_inf_prob = inf_probs.max().item()
    group_stats.mean_inf_prob = inf_probs.mean().item()

    return padded_inference_logprobs


class SequencePacker:
    """Packs multiple sequences into bins to minimize padding and improve GPU utilization."""

    def __init__(self, bin_size: int, pad_token: int, max_sequences_per_bin: int = 16):
        self.bin_size = bin_size
        self.pad_token = pad_token
        self.max_sequences_per_bin = max_sequences_per_bin

    def pack_sequences(
        self, sequences: List[torch.Tensor], generation_masks: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Pack sequences into bins using a greedy first-fit algorithm."""
        sequences_tensor = torch.stack(sequences)

        seq_lengths = get_actual_sequence_lengths(sequences_tensor, self.pad_token)

        # Trim sequences to actual lengths
        sequences = [sequences_tensor[i, :length] for i, length in enumerate(seq_lengths)]

        sorted_indices = sorted(range(len(sequences)), key=lambda i: seq_lengths[i], reverse=True)

        args = get_args()
        # Check that sequences can fit in bins
        # TODO(jalbericiola): this should probably be moved to the arguments file
        assert (
            args.seq_length <= self.bin_size
        ), f"seq_length ({args.seq_length}) must be <= bin_size ({self.bin_size})"

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
            print_rank_0(
                f"[SequencePacker] Packing distribution: {num_bins} bins, "
                f"avg {avg_seqs_per_bin:.1f} seqs/bin, "
                f"min {min_seqs}, max {max_seqs} seqs/bin "
                f"(limit: {self.max_sequences_per_bin})"
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
        packing_info = {
            'bin_seq_indices': bin_seq_indices,  # Which original sequences are in each bin
            'seq_starts': [],  # Start position of each sequence within its bin
            'seq_lengths': seq_lengths,  # Original sequence lengths
            'seq_to_bin_idx': [None] * len(sequences),  # Map from sequence index to bin index
        }

        # Build seq_to_bin_idx mapping
        for bin_idx, seq_indices in enumerate(bin_seq_indices):
            for seq_idx in seq_indices:
                packing_info['seq_to_bin_idx'][seq_idx] = bin_idx

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

            packing_info['seq_starts'].append(seq_starts)

        # Add bin_start_positions - a dict mapping bin_idx to list of start positions for each sequence in that bin
        bin_start_positions = {}
        for bin_idx in range(num_bins):
            bin_start_positions[bin_idx] = packing_info['seq_starts'][bin_idx]
        packing_info['bin_start_positions'] = bin_start_positions

        # Note: We'll store the actual padded length later when we know it
        # (it depends on the original trajectories passed to pack_sequences)

        # Invert attention mask, before inversion: (True = attend, False = mask)
        attention_mask = ~attention_mask

        return packed_sequences, position_ids, attention_mask, loss_mask, packing_info


def get_agent(args):
    """Get an agent based on environment configuration.

    If args.langrl_env_config is provided, uses weighted environment selection.
    Otherwise falls back to legacy single environment selection.
    """
    with open(args.langrl_env_config, 'r') as f:
        config = yaml.safe_load(f)

    return WeightedMultiTask.from_config(config)


_INFERENCE_INTERFACE = None


def get_inference_interface(args, loop, model):
    global _INFERENCE_INTERFACE
    if _INFERENCE_INTERFACE is None:
        rank = torch.distributed.get_rank()
        if rank == 0 and args.langrl_external_server:
            if args.langrl_inference_server_type == 'inplace_megatron':
                _INFERENCE_INTERFACE = loop.run_until_complete(
                    InferenceInterfaceServer.launch(MegatronLocal, model=model[0])
                )
            elif args.langrl_inference_server_type == 'inplace_megatron_chat':
                _INFERENCE_INTERFACE = loop.run_until_complete(
                    InferenceInterfaceServer.launch(
                        MegatronChatLocal,
                        model=model[0],
                        conversation_template=args.langrl_inference_server_conversation_template,
                    )
                )
            else:
                raise ValueError(f"Unknown inference_server_type {args.inference_server_type}")
        else:
            if args.langrl_inference_server_type == 'inplace_megatron':
                _INFERENCE_INTERFACE = loop.run_until_complete(MegatronLocal.launch(model[0]))
            elif args.langrl_inference_server_type == 'inplace_megatron_chat':
                _INFERENCE_INTERFACE = loop.run_until_complete(
                    MegatronChatLocal.launch(
                        model[0],
                        conversation_template=args.langrl_inference_server_conversation_template,
                    )
                )
            else:
                raise ValueError(
                    f"Unknown inference_server_type {args.langrl_inference_server_type}"
                )
    return _INFERENCE_INTERFACE


_ROLLOUT_GENERATOR = None


def get_rollout_generator(args, inference_interface, n_prompts, samples_per_group):
    global _ROLLOUT_GENERATOR
    if not args.rl_partial_rollouts or _ROLLOUT_GENERATOR is None:
        agent = get_agent(args)
        # Collect Rollouts
        request = GroupedRolloutRequest(
            num_groups=-1 if args.rl_partial_rollouts else n_prompts,
            rollouts_per_group=samples_per_group,
            inference_interface=inference_interface,
            generation_args={
                'temperature': args.grpo_default_temperature,
                'max_tokens': args.seq_length,
                'top_p': args.grpo_default_top_p,
            },
            filter_groups_with_same_reward=args.grpo_filter_groups_with_same_reward,
        )
        _ROLLOUT_GENERATOR = agent.get_grouped_rollouts(request)
    return _ROLLOUT_GENERATOR


def get_environment_rollouts(
    model: LanguageModule, optimizer: MegatronOptimizer, n_prompts: int, samples_per_group: int
):
    """Sample environment rollouts from an LLM.

    Args:
        model: Model to sample from.
        n_prompts: Number of prompts to sample for across *all* data parallel workers.
        samples_per_group: Amount of trajectories per prompt.

    Returns:
        GroupedRollouts object which is a nested list with each element being a list of rollouts of a group.
    """
    args = get_args()
    nvtx_range = get_nvtx_range()

    assert (
        n_prompts % mpu.get_expert_data_parallel_world_size() == 0
    ), "n_prompts must be divisible by data_parallel_world_size"

    with nvtx_range("rollout-collection"):
        loop = get_event_loop()
        with megatron_rl_inference_mode(
            model,
            optimizer,
            args.enable_cuda_graph,
            args.rl_reset_cuda_graphs,
            args.rl_offload_optimizer_during_inference,
            args.rl_offload_kv_cache_during_training,
            args.rl_remove_kv_cache_during_training,
        ) as inference_interface:

            with nvtx_range("inference-setup"):
                # Asyncronously run inference and rollout collection
                rollout_generator = get_rollout_generator(
                    args, inference_interface, n_prompts, samples_per_group
                )

            # NOTE(jbarker): we need to double check this when using PP>1
            rank = torch.distributed.get_rank()
            with nvtx_range("collect-rollouts"):
                if rank == 0:
                    print(f"Collecting rollouts on rank {rank}, Iteration {args.curr_iteration}...")
                    rollouts = [
                        loop.run_until_complete(anext(rollout_generator)) for _ in range(n_prompts)
                    ]
                    if not args.rl_partial_rollouts:
                        while True:
                            try:
                                loop.run_until_complete(anext(rollout_generator))
                                assert False, "Unexpected group left in generator."
                            except StopAsyncIteration:
                                break
                else:
                    # Just set up space to collect the rollouts
                    rollouts = [[None for _ in range(samples_per_group)] for _ in range(n_prompts)]

        with nvtx_range("sync-rollouts"):
            # Wait for Rollouts to be collected
            # TODO(jbarker): double check why this isn't causing rank 0 memory allocations
            torch.distributed.broadcast_object_list(rollouts, src=0)
        print(f"Got rollouts on rank {rank}")

    if lang_rl_log_dir and rank == get_tensor_model_parallel_src_rank():
        with open(
            lang_rl_log_dir
            + f'/rollouts_rank{rank}_iteration{args.curr_iteration}_'
            + f'{Path(args.langrl_env_config).stem}.pkl',
            'wb',
        ) as f:
            pickle.dump(rollouts, f)

    return rollouts


def selective_log_softmax(logits, index):
    """Taken from: https://github.com/huggingface/trl/blob/26d86757a7c7e24e397ea44f57ecce6031dfac01/trl/trainer/utils.py#L1659.

    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = (
            selected_logits - logsumexp_values
        )  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = torch.nn.functional.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(
                -1
            )
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


def get_logprobs(model, tokens, position_ids, attention_mask, no_grad=False):
    """Get sequence logprobs from their token ids.

    Args:
        model: model to predict with.
        tokens: inputs for which we want to get logprobs.
        position_ids: position ids that come with tokens.
        attention_mask: attention mask that comes with tokens.

    Returns:
        Logprobs of input sequences.

    """
    nvtx_range = get_nvtx_range()

    with nvtx_range("get-logprobs", time=False):

        with nvtx_range("forward-pass", time=False):
            # TODO(vitalyk): use fp16/bf16 as a function argument. Do not use args.
            args = get_args()
            # This is a hack to fix megatron's behaviour when flash-decode affects the training code flow.
            flash_decode = model.config.flash_decode
            model.config.flash_decode = False
            with torch.no_grad() if no_grad else nullcontext():
                logits = model(
                    tokens,
                    position_ids,
                    attention_mask,
                    runtime_gather_output=True,
                    fp32_output=not (args.fp16 or args.bf16),
                )
            model.config.flash_decode = flash_decode
            # We do not need logprobs for the n+1 token.
        with nvtx_range("log-softmax", time=False):
            logprobs = selective_log_softmax(logits[:, :-1, :], tokens[:, 1:])

    return logprobs


def compute_group_stats(
    rollouts: GroupedRollouts, tokenizer: MegatronLegacyTokenizer
) -> RolloutStats:
    """Add group-based rollout stats for logging.

    Args:
        rollouts: Rollouts to generate the stats for. Each inner list is a group (as in GRPO group), i.e. all rollouts are for the same prompt.
        tokenizer: Tokenizer to tokenize the rollouts in case they are raw strings.

    Returns:
       RolloutStats object containing all the stats.
    """
    args = get_args()
    # TODO (rkirby) Maybe do some of this after the tensor building
    group_reward_means = []
    group_reward_stds = []
    group_length_means = []
    group_length_stds = []
    group_length_maxs = []
    group_length_mins = []
    group_rollout_similarities = []
    for group in rollouts:
        group_rewards = []
        group_lengths = []
        for rollout in group:
            if isinstance(rollout, TokenRollout):
                lang_rl_log(
                    f"Rollout: [{rollout.env_id}] [{rollout.reward} : {len(rollout.trajectory)} tokens] {tokenizer.detokenize(rollout.trajectory)}"
                )
                assert (len(rollout.trajectory) == args.seq_length) or (
                    rollout.trajectory[-1] == tokenizer.eod
                ), f"Rollout is not the correct length: {len(rollout.trajectory)} {rollout.trajectory[-1]}\n{tokenizer.detokenize(rollout.trajectory)}"
            else:
                lang_rl_log(
                    f"Rollout: [{rollout.env_id}] [{rollout.reward} : {len(rollout.trajectory)} chars] {rollout.trajectory}"
                )
            group_rewards.append(rollout.reward)
            group_lengths.append(len(rollout.trajectory))
        if args.rl_calculate_intra_group_similarity:
            # We can probably compute this outside, but in case we switch to different group sizes for different envs, let's keep it here.
            combos = itertools.combinations(range(len(group)), 2)
            # For every pair (excluding ourselves), check the sequence similarity and log.
            # Use this to track the diversity of generated rollouts within a group.
            intra_group_sim = np.mean(
                list(
                    map(
                        lambda idx_pair: SequenceMatcher(
                            None, group[idx_pair[0]].trajectory, group[idx_pair[1]].trajectory
                        ).ratio(),
                        combos,
                    )
                )
            )
            group_rollout_similarities.append(intra_group_sim)
        else:
            group_rollout_similarities = None

        group_length_maxs.append(max(group_lengths))
        group_length_mins.append(min(group_lengths))
        group_reward_means.append(np.mean(group_rewards))
        group_reward_stds.append(np.std(group_rewards))
        group_length_means.append(np.mean(group_lengths))
        # https://arxiv.org/abs/2504.21233 reports that lens variants hurts.
        # Let's track this.
        group_length_stds.append(np.std(group_lengths))
    stats = RolloutStats(
        mean_reward=np.mean(group_reward_means),
        mean_sim=np.mean(group_rollout_similarities) if group_rollout_similarities else None,
        mean_length=np.mean(group_length_means),
        mean_length_std=np.mean(group_length_stds),
        max_length=np.max(group_length_maxs),
        min_length=np.min(group_length_mins),
        reward_means=group_reward_means,
        reward_stds=group_reward_stds,
        min_piold_to_inf_prob=None,
        max_piold_to_inf_prob=None,
        mean_piold_to_inf_prob=None,
        min_inf_train_prob_abs_diff=None,
        max_inf_train_prob_abs_diff=None,
        mean_inf_train_prob_abs_diff=None,
        min_inf_prob=None,
        max_inf_prob=None,
        mean_inf_prob=None,
        rewards=None,  # We will fill those in later in prepare_data_for_update.
        advantages=None,  # We will fill those in later in prepare_data_for_update.
    )
    return stats


def maybe_log_training_metrics(
    group_stats: RolloutStats,
    current_iteration: int,
    tokenizer: MegatronLegacyTokenizer,
    example_group: list[TokenRollout | Rollout],
    wandb_writer: wandb_run.Run | None = None,
    tb_writer: SummaryWriter | None = None,
):
    """Log training metrics if writers are available.

    Args:
        group_stats: RolloutStats object to pass to writers.
        current_iteration: Current training iteration.
        tokenizer: Tokenizer to untokenize trajectories for logging.
        example_group: A list of rollouts of one group to log examples of trajectories.
        wandb_writer: W&B writer object.
        tb_writer:  Tensorboard writer object.
    """
    if wandb_writer:
        group_table = wandb_writer.Table(
            columns=['group_means', 'group_stds'],
            data=list(zip(group_stats.reward_means, group_stats.reward_stds)),
        )
        rollout_table = wandb_writer.Table(
            columns=['reward'], data=[[r] for r in group_stats.rewards]
        )
        advantages = wandb_writer.Table(
            columns=['advantages'], data=[[x] for x in group_stats.advantages]
        )
        wandb_writer.log(
            {
                **{
                    'group_means_hist': wandb_writer.plot.histogram(
                        group_table, 'group_means', 'Group Means'
                    ),
                    'group_stds_hist': wandb_writer.plot.histogram(
                        group_table, 'group_stds', 'Group STDs'
                    ),
                    'rewards_hist': wandb_writer.plot.histogram(
                        rollout_table, 'reward', 'All Rewards'
                    ),
                    'mean_length': group_stats.mean_length,
                    'mean_length_std': group_stats.mean_length_std,
                    'max_length': group_stats.max_length,
                    'min_length': group_stats.min_length,
                    'mean_reward': group_stats.mean_reward,
                    'mean_advantage': np.mean(group_stats.advantages),
                    'advantages_hist': wandb_writer.plot.histogram(
                        advantages, 'advantages', 'Advantages'
                    ),
                    'min_piold_to_inf_prob': group_stats.min_piold_to_inf_prob,
                    'max_piold_to_inf_prob': group_stats.max_piold_to_inf_prob,
                    'mean_piold_to_inf_prob': group_stats.mean_piold_to_inf_prob,
                    'min_inf_train_prob_abs_diff': group_stats.min_inf_train_prob_abs_diff,
                    'max_inf_train_prob_abs_diff': group_stats.max_inf_train_prob_abs_diff,
                    'mean_inf_train_prob_abs_diff': group_stats.mean_inf_train_prob_abs_diff,
                    'min_inf_prob': group_stats.min_inf_prob,
                    'max_inf_prob': group_stats.max_inf_prob,
                    'mean_inf_prob': group_stats.mean_inf_prob,
                    # For now only log the first group
                    'rollouts': wandb_writer.Table(
                        columns=['Trajectories', 'Tokens', 'Rewards'],
                        rows=[
                            [
                                (
                                    tokenizer.detokenize(r.trajectory)
                                    if isinstance(r, TokenRollout)
                                    else r.trajectory
                                ),
                                r.trajectory,
                                r.reward,
                            ]
                            for r in example_group
                        ],
                    ),
                },
                **(
                    {'mean_intra_group_similarity': group_stats.mean_sim}
                    if group_stats.mean_sim
                    else {}
                ),
            },
            step=current_iteration,
        )
    if tb_writer:
        tb_writer.add_scalar('mean_reward', group_stats.mean_reward, current_iteration)


def prepare_trajectories(
    rollouts: GroupedRollouts, tokenizer: MegatronLegacyTokenizer, seq_length: int
):
    """Pad trajectories and extract the generation masks.

    Args:
        rollouts: Rollouts to extract trajectories from.
        tokenizer: Tokenizer to get the padding token and potentially tokenize.
        seq_length:  Maximum sequence length to pad to.

    Returns:
        Trajectories and their generation masks.

    Raises:
        ValueError:
    """
    # Track counts for each environment ID
    env_id_counts = Counter()

    DEFAULT_PAD_TOKENS = ['<|finetune_right_pad_id|>']

    if isinstance(tokenizer, _HuggingFaceTokenizer):
        if not tokenizer.pad:
            for pad_token in DEFAULT_PAD_TOKENS:
                if pad_token in tokenizer.vocab:
                    print_rank_0(f"Updating tokenizer pad token to {pad_token}")
                    tokenizer._tokenizer.pad_token_id = tokenizer.vocab[pad_token]
                    break
            else:
                raise ValueError("No pad token found in tokenizer vocabulary")
    elif isinstance(tokenizer, CustomTikTokenizer):
        assert "<SPECIAL_233>" in tokenizer.vocab, "Pad token is NOT in the tokenizer"
        tokenizer._pad_id = tokenizer.vocab["<SPECIAL_233>"]

    print_rank_0(
        f"Tokenizer vocab size: {tokenizer.vocab_size}\n"
        f"Tokenizer PAD: '{tokenizer.detokenize([tokenizer.pad])} ({tokenizer.pad})'\n"
        f"Tokenizer EOD: '{tokenizer.detokenize([tokenizer.eod])} ({tokenizer.eod})'"
    )

    trajs = []
    generation_masks = []
    inference_logprobs = []
    for group in rollouts:
        for rollout in group:
            generation_mask = rollout.generation_mask if isinstance(rollout, TokenRollout) else None

            trajectory = (
                rollout.trajectory.copy()
                if isinstance(rollout, TokenRollout)
                else tokenizer.tokenize(rollout.trajectory)
            )
            inf_logprobs = rollout.logprobs

            length = len(trajectory)
            assert length <= seq_length, "Rollout too long, how did this happen?"
            if len(trajectory) < seq_length:
                assert (
                    trajectory[-1] == tokenizer.eod
                ), "Trajectories under a seq_length limit should have eod token at the end."

            if length < seq_length:
                trajectory.extend([tokenizer.pad] * (seq_length - length))
                if generation_mask:
                    generation_mask.extend([False] * (seq_length - length))
            trajs.append(trajectory)
            generation_masks.append(generation_mask)

            if inf_logprobs is not None:
                inf_logprobs_tensor = torch.Tensor(inf_logprobs)
                # Don't pad individual logprobs here - padding happens later if needed
                inference_logprobs.append(inf_logprobs_tensor)
            else:
                inference_logprobs.append(None)

            env_id = rollout.env_id
            env_id_counts[env_id] += 1

    print(
        "Rollout counts:"
        + "".join([f"\n\t{env_id}: {count}" for env_id, count in env_id_counts.items()])
    )

    generation_masks = torch.tensor(generation_masks, dtype=torch.bool, device='cpu')
    trajs = torch.tensor(trajs, device='cpu')

    args = get_args()
    # Only process if we have inference_logprobs
    if inference_logprobs and any(lp is not None for lp in inference_logprobs):
        if args.use_sequence_packing:
            # For sequence packing, we need to pad all logprobs to the same size
            padded_logprobs = []
            for logprobs in inference_logprobs:
                if logprobs is not None:
                    if len(logprobs) < seq_length:
                        # Pad with zeros (these positions will be masked anyway)
                        padding_size = seq_length - len(logprobs)
                        padded = torch.nn.functional.pad(logprobs, (0, padding_size), value=0.0)
                        padded_logprobs.append(padded)
                    else:
                        padded_logprobs.append(logprobs)
                else:
                    # Create zero tensor for None logprobs
                    padded_logprobs.append(torch.zeros(seq_length))
            inference_logprobs = torch.stack(padded_logprobs)
        else:
            # For non-packing mode, keep as list of tensors (unpadded)
            # This preserves the original behavior where each sequence can have different lengths
            pass
    else:
        inference_logprobs = None

    # Some sanity checks regarding the tokenization
    assert (
        tokenizer.bos is None or (trajs[:, 0] == tokenizer.bos).all()
    ), "First token should be bos"
    assert (
        tokenizer.bos is None or (trajs[:, 1] != tokenizer.bos).all()
    ), "Second token should not be bos"
    assert (
        (trajs * generation_masks.int() == tokenizer.eod).sum(axis=1) <= 1
    ).all(), "Only one eod per trajectory in generated tokens."
    # TODO(rkirby):
    # We should avoid the tokenizer pad token being the same as the eod token for proper loss masking,
    # But now the deepseek tokenizer has the pad token set to eod, we need to handle this.
    # assert (tokenizer.pad != tokenizer.eod), "Pad and eod should be different"
    return trajs, generation_masks, inference_logprobs


def prepare_packed_trajectories(
    all_rollouts: GroupedRollouts, tokenizer: MegatronLegacyTokenizer, args
):
    """Prepare trajectories for sequence packing mode with distributed processing.
    Distributes trajectory preparation across ranks, then gathers results for packing.

    Args:
        all_rollouts: All rollouts to process.
        tokenizer: Tokenizer to get the padding token and potentially tokenize.
        args: Arguments containing seq_length and distributed settings.

    Returns:
        Trajectories, generation masks, and inference logprobs (all gathered from all ranks).
    """
    world_size = mpu.get_expert_data_parallel_world_size()
    # For packing, distribute trajectory preparation across ranks
    # Each rank prepares a portion, then we gather for packing
    total_rollouts = len(all_rollouts)
    rollouts_per_rank = total_rollouts // (world_size if world_size > 0 else 1)
    rank = mpu.get_expert_data_parallel_rank()

    # Each rank prepares its portion
    start_idx = rank * rollouts_per_rank
    end_idx = (
        start_idx + rollouts_per_rank
        if rank < mpu.get_expert_data_parallel_world_size() - 1
        else total_rollouts
    )
    my_rollouts = all_rollouts[start_idx:end_idx]

    # Prepare this rank's portion
    my_trajs, my_generation_masks, my_inference_logprobs = prepare_trajectories(
        my_rollouts, tokenizer, args.seq_length
    )

    # Move to GPU for all_gather operation
    # Note: prepare_trajectories already returns tensors, not lists

    my_trajs = my_trajs.cuda()
    my_generation_masks = my_generation_masks.cuda()
    if my_inference_logprobs is not None:
        my_inference_logprobs = my_inference_logprobs.cuda()

    # All-gather trajectories from all ranks
    # This is more efficient than having all ranks process all sequences
    if world_size > 1:
        # Gather all trajectories
        trajs_list = [torch.empty_like(my_trajs) for _ in range(world_size)]
        torch.distributed.all_gather(
            trajs_list, my_trajs, group=mpu.get_expert_data_parallel_group()
        )
        trajs = torch.cat(trajs_list, dim=0)

        # Gather all generation masks
        masks_list = [torch.empty_like(my_generation_masks) for _ in range(world_size)]
        torch.distributed.all_gather(
            masks_list, my_generation_masks, group=mpu.get_expert_data_parallel_group()
        )
        generation_masks = torch.cat(masks_list, dim=0)

        # Gather inference logprobs if present
        if my_inference_logprobs is not None:
            logprobs_list = [torch.empty_like(my_inference_logprobs) for _ in range(world_size)]
            torch.distributed.all_gather(
                logprobs_list, my_inference_logprobs, group=mpu.get_expert_data_parallel_group()
            )
            inference_logprobs = torch.cat(logprobs_list, dim=0)
        else:
            inference_logprobs = None
    else:
        # Single process case (testing)
        trajs = my_trajs
        generation_masks = my_generation_masks
        inference_logprobs = my_inference_logprobs

    return trajs, generation_masks, inference_logprobs


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


def prepare_data_for_update(
    model: list[LanguageModule],
    ref_state_dict: Dict[str, Any],
    rollouts: GroupedRollouts,
    tokenizer: MegatronLegacyTokenizer,
) -> RerunDataIterator:
    """Extract data for the update from raw rollouts.

    Args:
        model: Current policy as the zero-eth element.
        ref_state_dict: Reference policy state dict.
        rollouts: Rollouts to extract the data from.
        tokenizer: Tokenizer to pad/tokenize data.

    Returns:
        Cycled iterator over dataset batches. In GRPO we might want to go over the same data multiple times.
    """
    args = get_args()
    timers = get_timers()
    wandb_writer = get_wandb_writer()
    tb_writer = get_tensorboard_writer()
    nvtx_range = get_nvtx_range()
    model = model[0]

    with nvtx_range("prepare-data-for-update"):
        with nvtx_range("compute-group-stats"):
            # These are computed on all rollouts for reporting purposes
            group_stats = compute_group_stats(rollouts, tokenizer)
            rewards = np.array([[rollout.reward for rollout in group] for group in rollouts])
            group_stats.rewards = rewards.flatten().tolist()
            group_stats.advantages = (
                (
                    (rewards - rewards.mean(axis=1, keepdims=True))
                    / (1e-4 + rewards.std(axis=1, keepdims=True))
                )
                .flatten()
                .tolist()
            )

        all_rollouts = rollouts

        # Now split the rollouts across the data parallel ranks for training
        # This needs to be done at this point because we are about to calculate logprobs
        if (expert_data_parallel_world_size := mpu.get_expert_data_parallel_world_size()) > 0:
            data_split_size = len(rollouts) // expert_data_parallel_world_size
            data_split_range = (
                mpu.get_expert_data_parallel_rank() * data_split_size,
                (mpu.get_expert_data_parallel_rank() + 1) * data_split_size,
            )
            rollouts = rollouts[data_split_range[0] : data_split_range[1]]

        # [g, group_size]
        # Making an assumption that all groups are of the same size!
        # For packing mode, use all rollouts to compute rewards
        rollouts_for_rewards = all_rollouts if args.use_sequence_packing else rollouts
        rewards = torch.tensor(
            [[rollout.reward for rollout in group] for group in rollouts_for_rewards], device='cpu'
        )

        # We flatten them for logging.
        with nvtx_range("prepare_trajectories"):
            if args.use_sequence_packing:
                trajs, generation_masks, inference_logprobs = prepare_packed_trajectories(
                    all_rollouts, tokenizer, args
                )
            else:
                trajs, generation_masks, inference_logprobs = prepare_trajectories(
                    rollouts, tokenizer, args.seq_length
                )
        # Store reference to original data (no clone needed since we don't modify in-place)
        original_trajs = trajs

        # Sequence packing or standard processing
        packing_context = {}  # Store all packing-related data

        if args.use_sequence_packing:
            with nvtx_range("sequence_packing"):
                timers('sequence-packing-overhead', log_level=1).start()

                bin_size = args.sequence_packing_bin_size

                # Create packer with max sequences per bin limit to prevent extreme imbalance
                max_sequences_per_bin = getattr(args, 'sequence_packing_max_sequences_per_bin', 100)
                packer = SequencePacker(
                    bin_size=bin_size,
                    pad_token=tokenizer.pad,
                    max_sequences_per_bin=max_sequences_per_bin,
                )
                packing_context['packer'] = packer  # Store for reuse

                # Convert trajectories to list for packing
                traj_list = [trajs[i] for i in range(trajs.shape[0])]

                # Pack sequences with generation masks
                (
                    packed_trajs,
                    packed_position_ids,
                    packed_attention_mask,
                    packed_loss_mask,
                    packing_info,
                ) = packer.pack_sequences(traj_list, generation_masks)

                rank = mpu.get_expert_data_parallel_rank()
                # Debug: Check packing output
                if rank == 0:
                    seq_per_bin = [len(indices) for indices in packing_info['bin_seq_indices']]
                    print(f"\nDEBUG: Initial packing output (before distribution):")
                    print(f"  - Total bins created: {len(packing_info['bin_seq_indices'])}")
                    print(f"  - Total sequences packed: {sum(seq_per_bin)}")
                    print(
                        f"  - Sequences per bin: min={min(seq_per_bin)}, max={max(seq_per_bin)}, avg={sum(seq_per_bin)/len(seq_per_bin):.1f}"
                    )
                    print(f"  - First 20 bins: {seq_per_bin[:20]}")

                # Store packing info for later unpacking
                # Also store the actual padded length in packing_info for the unpacker
                packing_info['original_padded_length'] = original_trajs.shape[1]

                # Distribute packed bins across data parallel ranks
                num_bins = packed_trajs.shape[0]
                world_size = mpu.get_expert_data_parallel_world_size()

                # Choose distribution algorithm based on args.sequence_packing_algo
                packing_algo = getattr(args, 'sequence_packing_algo', 'fifo')

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
                my_packed_inference_logprobs = []
                my_bin_seq_indices = []
                my_seq_starts = {}

                # Check if we have packed inference logprobs
                has_packed_inference_logprobs = 'packed_inference_logprobs' in packing_context

                for new_idx, old_idx in enumerate(my_bin_indices):
                    my_packed_trajs.append(packed_trajs[old_idx])
                    my_packed_position_ids.append(packed_position_ids[old_idx])
                    if packed_attention_mask is not None:
                        my_packed_attention_mask.append(packed_attention_mask[old_idx])
                    my_packed_loss_mask.append(packed_loss_mask[old_idx])
                    if has_packed_inference_logprobs:
                        my_packed_inference_logprobs.append(
                            packing_context['packed_inference_logprobs'][old_idx]
                        )
                    my_bin_seq_indices.append(packing_info['bin_seq_indices'][old_idx])
                    my_seq_starts[new_idx] = packing_info['seq_starts'][old_idx]

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

                # Stack the packed inference logprobs if available
                if has_packed_inference_logprobs and my_packed_inference_logprobs:
                    packed_inference_logprobs = torch.stack(my_packed_inference_logprobs)
                    packing_context['packed_inference_logprobs'] = packed_inference_logprobs
                elif has_packed_inference_logprobs:
                    # Create empty tensor if no bins for this rank
                    packed_inference_logprobs = torch.empty(
                        0,
                        bin_size - 1,
                        dtype=packing_context['packed_inference_logprobs'].dtype,
                        device=packing_context['packed_inference_logprobs'].device,
                    )
                    packing_context['packed_inference_logprobs'] = packed_inference_logprobs

                # Debug: Check what we're extracting
                if rank == 0:
                    print(f"\nDEBUG: Rank 0 {packing_algo} bin assignment:")
                    print(f"  - Total bins before distribution: {num_bins}")
                    print(
                        f"  - Bins assigned to rank 0: {my_bin_indices[:10]}... (showing first 10)"
                    )
                    print(f"  - Number of bins for this rank: {len(my_bin_indices)}")
                    print(f"  - Length of my_bin_seq_indices: {len(my_bin_seq_indices)}")
                    if len(my_bin_seq_indices) > 0:
                        print(
                            f"  - Sequences in first 5 bins: {[len(indices) for indices in my_bin_seq_indices[:5]]}"
                        )

                # Create updated packing info for this rank
                packing_info = {
                    'bin_seq_indices': my_bin_seq_indices,
                    'seq_starts': my_seq_starts,
                    'seq_lengths': packing_info['seq_lengths'],  # Keep all sequence lengths
                    'seq_to_bin_idx': packing_info['seq_to_bin_idx'],  # Keep mapping
                    'original_padded_length': packing_info['original_padded_length'],
                }

                # Add empty bins if this rank has fewer than max_bins_per_rank
                current_bins = len(my_bin_indices)
                if current_bins < max_bins_per_rank:
                    num_empty_bins = max_bins_per_rank - current_bins

                    # Create empty bins using the helper function
                    bin_size = packed_trajs.shape[1]
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

                    # Create empty inference logprobs if needed
                    if has_packed_inference_logprobs:
                        empty_inference_logprobs = torch.zeros(
                            (num_empty_bins, bin_size - 1),
                            dtype=packed_inference_logprobs.dtype,
                            device=packed_inference_logprobs.device,
                        )
                        packed_inference_logprobs = torch.cat(
                            [packed_inference_logprobs, empty_inference_logprobs], dim=0
                        )
                        packing_context['packed_inference_logprobs'] = packed_inference_logprobs

                    # Add empty entries to packing_info
                    for i, entry in enumerate(empty_packing_entries):
                        bin_idx = current_bins + i
                        packing_info['bin_seq_indices'].append(entry['bin_seq_indices'])
                        packing_info['seq_starts'][bin_idx] = entry['seq_starts']

                packing_context['packing_info'] = packing_info
                packing_context['original_generation_masks'] = generation_masks
                packing_context['original_trajs'] = original_trajs
                # Move packed tensors to GPU once to avoid CPU-GPU transfers every iteration
                packing_context['packed_trajs'] = packed_trajs.cuda()
                packing_context['packed_position_ids'] = packed_position_ids.cuda()
                packing_context['packed_attention_mask'] = (
                    packed_attention_mask.cuda() if packed_attention_mask is not None else None
                )
                packing_context['packed_loss_mask'] = packed_loss_mask.cuda()

                # Store the original padding positions for correct unpacking
                # The loss_mask will be based on original_trajs, so we need to preserve that pattern
                packing_context['original_padding_positions'] = original_trajs == tokenizer.pad

                # Store my_bin_seq_indices for later use
                packing_context['my_bin_seq_indices'] = my_bin_seq_indices

                # Log packing efficiency (for this rank's bins)
                total_tokens = sum(packing_info['seq_lengths'])  # All sequences
                my_sequences = sum(len(indices) for indices in my_bin_seq_indices)
                my_tokens = sum(
                    packing_info['seq_lengths'][idx]
                    for indices in my_bin_seq_indices
                    for idx in indices
                )
                total_capacity = packed_trajs.shape[0] * packed_trajs.shape[1]
                packing_efficiency = my_tokens / total_capacity if total_capacity > 0 else 0
                avg_seq_length = total_tokens / len(packing_info['seq_lengths'])

                # Store global average sequences per bin in packing context
                if num_bins > 0:
                    global_avg_seqs_per_bin = len(packing_info['seq_lengths']) / num_bins
                else:
                    global_avg_seqs_per_bin = 1  # Default to 1 if no bins
                packing_context['global_avg_seqs_per_bin'] = global_avg_seqs_per_bin

                print_rank_0(f"\n[Sequence Packing] Statistics:")
                print_rank_0(f"  - Total sequences: {len(packing_info['seq_lengths'])}")
                print_rank_0(f"  - Total bins: {num_bins}")
                print_rank_0(f"  - Bin size: {packed_trajs.shape[1]} tokens")
                print_rank_0(f"  - Average sequence length: {avg_seq_length:.1f} tokens")
                print_rank_0(f"  - Average sequences per bin: {global_avg_seqs_per_bin:.1f}")
                print_rank_0(
                    f"  - This rank: {my_sequences} sequences in {packed_trajs.shape[0]} bins"
                )
                print_rank_0(
                    f"  - Packing efficiency: {packing_efficiency:.1%} ({my_tokens:,} / {total_capacity:,} tokens)"
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
                        print(
                            f"\n[Sequence Packing] Per-rank distribution ({packing_algo} algorithm):"
                        )
                        print(
                            "  Rank | Total Bins | Non-empty | Sequences | Min/Bin | Max/Bin | Avg/Bin"
                        )
                        print(
                            "  -----|------------|-----------|-----------|---------|---------|--------"
                        )
                        for stats in all_rank_stats:
                            r = int(stats[0].item())
                            total_bins = int(stats[1].item())
                            non_empty = int(stats[2].item())
                            sequences = int(stats[3].item())
                            min_seq = int(stats[4].item())
                            max_seq = int(stats[5].item())
                            avg_seq = stats[6].item()
                            print(
                                f"   {r:3d} | {total_bins:10d} | {non_empty:9d} | {sequences:9d} | {min_seq:7d} | {max_seq:7d} | {avg_seq:6.1f}"
                            )

                        # Also show first few bins for rank 0 as example
                        print(f"\n  Example (Rank 0 first 10 bins): {seq_counts_per_bin[:10]}")

                        # Show the improvement from round-robin
                        total_seqs_all_ranks = sum(int(stats[3].item()) for stats in all_rank_stats)
                        avg_seqs_per_rank = total_seqs_all_ranks / world_size
                        max_deviation = max(
                            abs(int(stats[3].item()) - avg_seqs_per_rank)
                            for stats in all_rank_stats
                        )
                        print(f"\n  Round-robin distribution quality:")
                        print(f"  - Average sequences per rank: {avg_seqs_per_rank:.1f}")
                        print(
                            f"  - Max deviation from average: {max_deviation:.0f} sequences ({max_deviation/avg_seqs_per_rank*100:.1f}%)"
                        )

                # Update data for packed computation
                trajs = packed_trajs
                position_ids = packed_position_ids
                attention_mask = packed_attention_mask

                timers('sequence-packing-overhead').stop()

        # Always compute standard masks for the original data (we'll need them later)
        with nvtx_range("get_ltor_masks_and_position_ids"):
            original_attention_mask, original_loss_mask, original_position_ids = (
                get_ltor_masks_and_position_ids(
                    original_trajs,
                    tokenizer.eod,
                    tokenizer.pad,
                    args.reset_position_ids,
                    args.reset_attention_mask,
                    eod_mask_loss=False,
                    pad_mask_loss=True,
                )
            )
            original_loss_mask[~generation_masks] = 0.0

        if not args.use_sequence_packing:
            # Use original masks if not packing
            attention_mask = original_attention_mask
            loss_mask = original_loss_mask
            position_ids = original_position_ids

        with torch.no_grad(), nvtx_range("compute_logprobs"):
            timers('compute-logprobs', log_level=0).start()
            # Before we can update the model, we need to get the logprobs for the \pi_{old} model.
            # Use packed sequences if packing is enabled for performance benefits
            if args.use_sequence_packing and 'packed_trajs' in packing_context:
                compute_trajs = packing_context['packed_trajs']
                compute_position_ids = packing_context['packed_position_ids']
                compute_attention_mask = packing_context['packed_attention_mask']
                use_packed_computation = True
            else:
                compute_trajs = original_trajs
                compute_position_ids = original_position_ids
                compute_attention_mask = original_attention_mask
                use_packed_computation = False

        with nvtx_range("create-logprobs-dataloader"):
            data_iter = DataLoader(
                TensorDataset(compute_trajs, compute_position_ids), batch_size=args.micro_batch_size
            )
            old_logprobs = []

            # Compute logprobs
            for batch_idx, (b_trajs, b_posids) in enumerate(data_iter):
                # Get attention mask slice
                if compute_attention_mask is not None:
                    start_idx = batch_idx * args.micro_batch_size
                    end_idx = min(
                        start_idx + args.micro_batch_size, compute_attention_mask.shape[0]
                    )
                    b_attn_mask = compute_attention_mask[start_idx:end_idx].cuda()
                else:
                    b_attn_mask = None

                logprobs = get_logprobs(
                    model, b_trajs.cuda(), b_posids.cuda(), b_attn_mask, no_grad=True
                )
                old_logprobs.append(logprobs.detach().cpu())

            old_logprobs = torch.concat(old_logprobs, dim=0)

            # Handle packed vs unpacked logprobs
            if use_packed_computation and 'packing_info' in packing_context:
                # Store packed logprobs on GPU for forward_step
                packing_context['old_logprobs'] = old_logprobs.cuda()
                # Keep old_logprobs as None for the data loading path
                old_logprobs_for_data = None
            else:
                # In unpacked mode, we need to unpack if we computed on packed data
                old_logprobs_for_data = old_logprobs

            timers('compute-logprobs').stop()

        # Inference logprobs 2 tokens shorter than old_logprobs.
        # One token difference is because we remove the first one in get_logprobs(), the other one is eod padding, if I got it correct. The difference should be one token if we are cut by the sequence length.

        # Handle inference logprobs alignment (skip if using sequence packing)
        if (
            inference_logprobs is not None
            and args.rl_inference_logprobs_is_correction
            and not args.use_sequence_packing
        ):
            inference_logprobs = align_unpacked_inference_logprobs(
                inference_logprobs=inference_logprobs,
                old_logprobs_for_data=old_logprobs_for_data,
                generation_masks=generation_masks,
                group_stats=group_stats,
            )
        else:
            if not args.use_sequence_packing:
                # Keep inference_logprobs as None instead of zeros
                inference_logprobs = None
            # For sequence packing, inference_logprobs will be handled separately

        # Handle packing of inference_logprobs for sequence packing mode
        if (
            args.use_sequence_packing
            and inference_logprobs is not None
            and args.rl_inference_logprobs_is_correction
        ):
            with nvtx_range("pack-inference-logprobs"):
                # Pack the inference logprobs using the helper function
                packed_inference_logprobs = pack_inference_logprobs(
                    inference_logprobs=inference_logprobs,
                    packing_info=packing_context['packing_info'],
                    generation_masks=generation_masks,
                    bin_size=args.sequence_packing_bin_size,
                )

                # Store packed inference logprobs in packing context
                packing_context['packed_inference_logprobs'] = packed_inference_logprobs.cuda()
                packing_context['has_inference_logprobs'] = True

        # TODO(vitalyk): add a test for prepare_data_for_update.

        with torch.no_grad(), nvtx_range("compute_ref_logprobs"):
            # We need to load the ref model state dict and compute the logprobs for the ref model
            cur_st_dict = {
                k: (v.cpu() if v is not None else v) for k, v in model.state_dict().items()
            }
            model.load_state_dict(ref_state_dict)
            ref_logprobs = []

            # Compute reference logprobs
            for batch_idx, (b_trajs, b_posids) in enumerate(data_iter):
                # Get attention mask slice
                if compute_attention_mask is not None:
                    start_idx = batch_idx * args.micro_batch_size
                    end_idx = min(
                        start_idx + args.micro_batch_size, compute_attention_mask.shape[0]
                    )
                    b_attn_mask = compute_attention_mask[start_idx:end_idx].cuda()
                else:
                    b_attn_mask = None

                logprobs = get_logprobs(
                    model, b_trajs.cuda(), b_posids.cuda(), b_attn_mask, no_grad=True
                )
                ref_logprobs.append(logprobs.detach().cpu())

            ref_logprobs = torch.concat(ref_logprobs, dim=0)

            # Handle packed vs unpacked logprobs
            if use_packed_computation and 'packing_info' in packing_context:
                # Store packed logprobs on GPU for forward_step
                packing_context['ref_logprobs'] = ref_logprobs.cuda()
                # Keep ref_logprobs as None for the data loading path
                # since we won't use TensorDataset in packed mode
                ref_logprobs_for_data = None
            else:
                # In unpacked mode, use the computed logprobs directly
                ref_logprobs_for_data = ref_logprobs
            # logprobs are [b, seq, h] now.
            model.load_state_dict(cur_st_dict)

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        # model.train()

        with nvtx_range("prepare_advantages"):
            timers('prepare-advantages', log_level=0).start()

            advantages = (rewards - rewards.mean(axis=1, keepdim=True)) / (
                1e-4 + rewards.std(axis=1, keepdim=True)
            )

            # Flatten advantages for training and move to GPU
            advantages = advantages.view(-1).cuda()

            timers('prepare-advantages').stop()
        with nvtx_range("create_dataloader"):
            if args.use_sequence_packing:
                # Store packing context in runtime state for forward_step
                runtime_state = get_rl_runtime_state()
                runtime_state.packing_context = packing_context

                packing_info = packing_context['packing_info']
                packing_context['bin_advantages'] = []
                for bin_idx, seq_indices in enumerate(packing_info['bin_seq_indices']):
                    if seq_indices:
                        packing_context['bin_advantages'].append(advantages[seq_indices])
                    else:
                        packing_context['bin_advantages'].append(
                            torch.tensor([], dtype=advantages.dtype, device=advantages.device)
                        )

                num_bins_this_rank = len(packing_context['packed_trajs'])
                bin_indices = torch.arange(num_bins_this_rank)

                my_bin_seq_indices = packing_context.get('my_bin_seq_indices', [])

                my_sequences = sum(len(indices) for indices in my_bin_seq_indices)

                actual_seqs_per_bin_this_rank = (
                    my_sequences / num_bins_this_rank if num_bins_this_rank > 0 else 1
                )
                global_avg_seqs_per_bin = max(
                    1, packing_context.get('global_avg_seqs_per_bin', actual_seqs_per_bin_this_rank)
                )

                target_sequences_per_step = args.global_batch_size
                dp_world_size = max(1, mpu.get_data_parallel_world_size())

                total_bins_needed = max(
                    1, math.ceil(target_sequences_per_step / global_avg_seqs_per_bin)
                )

                # Ensure divisibility by dp_world_size
                if total_bins_needed % dp_world_size != 0:
                    total_bins_needed = ((total_bins_needed // dp_world_size) + 1) * dp_world_size

                bins_per_rank_per_step = total_bins_needed // dp_world_size
                bins_per_rank_per_step = min(bins_per_rank_per_step, num_bins_this_rank)

                # Synchronize across ranks - all ranks must process same number of bins
                bins_per_rank_tensor = torch.tensor(
                    [bins_per_rank_per_step], dtype=torch.long, device='cuda'
                )
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(
                        bins_per_rank_tensor,
                        op=torch.distributed.ReduceOp.MIN,
                        group=mpu.get_data_parallel_group(),
                    )
                bins_per_rank_per_step = int(bins_per_rank_tensor.item())

                effective_global_batch_size = bins_per_rank_per_step * dp_world_size

                total_steps = len(bin_indices) // bins_per_rank_per_step + (
                    1 if len(bin_indices) % bins_per_rank_per_step else 0
                )

                # Store packing plan in runtime state for the training loop to use
                runtime_state = get_rl_runtime_state()
                runtime_state.sequence_packing_plan = {
                    'bin_indices': bin_indices,
                    'bins_per_rank_per_step': bins_per_rank_per_step,
                    'total_steps': total_steps,
                    'current_step': 0,
                    'packing_context': packing_context,
                }

                runtime_state.sequence_packing_metadata = {
                    'num_bins': num_bins_this_rank,
                    'num_bins_this_rank': num_bins_this_rank,
                    'num_sequences': len(packing_info['seq_lengths']),
                    'avg_seqs_per_bin': global_avg_seqs_per_bin,
                    'avg_seqs_per_bin_this_rank': actual_seqs_per_bin_this_rank,
                }

                if args.micro_batch_size != 1:
                    print_rank_0(
                        f"WARNING: micro_batch_size={args.micro_batch_size} but sequence packing expects 1. Using 1."
                    )
                micro_batch_size = 1

                from megatron.core.num_microbatches_calculator import (
                    get_num_microbatches,
                    reconfigure_num_microbatches_calculator,
                )

                old_num_microbatches = get_num_microbatches()

                reconfigure_num_microbatches_calculator(
                    rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                    rampup_batch_size=args.rampup_batch_size,
                    global_batch_size=effective_global_batch_size,
                    micro_batch_size=micro_batch_size,
                    data_parallel_size=dp_world_size,
                    decrease_batch_size_if_needed=args.decrease_batch_size_if_needed,
                )

                new_num_microbatches = get_num_microbatches()

                print_rank_0(f"\n[Sequence Packing] Multi-step training plan:")
                print_rank_0(f"  - Target sequences per step: {target_sequences_per_step}")
                print_rank_0(f"  - Bins per rank per step: {bins_per_rank_per_step}")
                print_rank_0(
                    f"  - Estimated sequences per step: ~{int(effective_global_batch_size * global_avg_seqs_per_bin)}"
                )
                print_rank_0(f"  - Total optimizer steps: {total_steps}")
                print_rank_0(
                    f"  - Microbatches per step: {new_num_microbatches} (was {old_num_microbatches})"
                )

                for step in range(min(3, total_steps)):
                    start_idx = step * bins_per_rank_per_step
                    end_idx = min(start_idx + bins_per_rank_per_step, num_bins_this_rank)
                    step_bins = end_idx - start_idx

                    actual_seqs = sum(
                        len(my_bin_seq_indices[bin_idx])
                        for bin_idx in range(start_idx, end_idx)
                        if bin_idx < len(my_bin_seq_indices)
                    )
                    est_global_seqs = actual_seqs * dp_world_size
                    print_rank_0(
                        f"  - Step {step + 1}: {step_bins} bins, ~{est_global_seqs} sequences globally"
                    )

                if total_steps > 3:
                    print_rank_0(f"  - ... ({total_steps - 3} more steps)")

                start_idx = 0
                end_idx = min(bins_per_rank_per_step, num_bins_this_rank)
                step_bin_indices = bin_indices[start_idx:end_idx]
                dataset = TensorDataset(step_bin_indices)
                loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
            else:
                runtime_state = get_rl_runtime_state()
                runtime_state.packing_context = None
                dataset_tensors = [
                    original_trajs,
                    advantages,
                    old_logprobs_for_data,
                    original_loss_mask,
                    original_position_ids,
                    ref_logprobs_for_data,
                ]
                if args.rl_inference_logprobs_is_correction:
                    if inference_logprobs is not None:
                        dataset_tensors.append(inference_logprobs)
                    else:
                        # Create dummy tensor matching the batch size only if correction is enabled
                        dataset_tensors.append(torch.zeros_like(old_logprobs_for_data))
                else:
                    # If correction is not enabled, always append zeros
                    dataset_tensors.append(torch.zeros_like(old_logprobs_for_data))

                data = TensorDataset(*dataset_tensors)
                loader = DataLoader(data, batch_size=args.micro_batch_size)

        with nvtx_range("log-wandb-tb"):
            maybe_log_training_metrics(
                group_stats=group_stats,
                current_iteration=args.curr_iteration,
                tokenizer=tokenizer,
                example_group=rollouts[0],
                wandb_writer=wandb_writer,
                tb_writer=tb_writer,
            )

    return RerunDataIterator(itertools.cycle(loader))


def get_rollout_data_iterator(
    model: LanguageModule,
    optimizer: MegatronOptimizer,
    iteration: int,
    ref_state_dict: Dict[str, torch.Tensor],
) -> RerunDataIterator:

    args = get_args()
    tokenizer = get_tokenizer()

    buffered_rollouts = get_environment_rollouts(
        model, optimizer, args.grpo_prompts_per_step, args.grpo_group_size
    )
    buffered_rollouts = prepare_data_for_update(model, ref_state_dict, buffered_rollouts, tokenizer)

    return buffered_rollouts


def setup_grpo_data_iterator(
    model: LanguageModule,
    optimizer: MegatronOptimizer,
    iteration: int,
    ref_state_dict: Dict[str, torch.Tensor],
    buffered_rollouts: RerunDataIterator | None = None,
) -> RerunDataIterator:
    """
    Set up the data iterator for GRPO training.

    Args:
        model: The language model
        optimizer: The Megatron optimizer
        iteration: Current training iteration
        ref_state_dict: Reference model state dict for GRPO
        buffered_rollouts: Previously collected rollouts (if any)

    Returns:
        RerunDataIterator for the current training step
    """
    args = get_args()

    # We collect new rollouts when we've gone over the collected data 'grpo_iterations' times.
    if (
        iteration
        % (args.grpo_iterations * ((args.grpo_samples_per_iteration) // args.global_batch_size))
        == 0
    ):
        buffered_rollouts = get_rollout_data_iterator(model, optimizer, iteration, ref_state_dict)

        # Reset packing step counter when new rollouts are collected
        runtime_state = get_rl_runtime_state()
        if runtime_state.sequence_packing_plan is not None:
            runtime_state.sequence_packing_plan['current_step'] = 0

    # Handle sequence packing: update the data loader for the current optimizer step
    runtime_state = get_rl_runtime_state()
    if runtime_state.sequence_packing_plan is not None:
        plan = runtime_state.sequence_packing_plan
        if plan['current_step'] < plan['total_steps']:
            # Create loader for current chunk of bins
            start_idx = plan['current_step'] * plan['bins_per_rank_per_step']
            end_idx = min(start_idx + plan['bins_per_rank_per_step'], len(plan['bin_indices']))
            step_bin_indices = plan['bin_indices'][start_idx:end_idx]

            dataset = TensorDataset(step_bin_indices)
            loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
            train_data_iterator = RerunDataIterator(itertools.cycle(loader))

            # Advance to next step for next iteration
            plan['current_step'] += 1

            # Log which bins we're processing
            my_bin_seq_indices = plan['packing_context'].get('my_bin_seq_indices', [])
            step_sequences = sum(
                len(my_bin_seq_indices[bin_idx.item()])
                for bin_idx in step_bin_indices
                if bin_idx.item() < len(my_bin_seq_indices)
            )
            # Estimate global sequences for this step
            est_global_sequences = step_sequences * mpu.get_data_parallel_world_size()
            print_rank_0(
                f"[Sequence Packing] Optimizer step {plan['current_step']}/{plan['total_steps']}: "
                f"processing {len(step_bin_indices)} bins (~{est_global_sequences} sequences globally)"
            )

            runtime_state.reset_iteration_counters()

        else:
            print_rank_0(f"[Sequence Packing] All bins processed, waiting for new rollouts")
            train_data_iterator = buffered_rollouts
    else:
        train_data_iterator = buffered_rollouts

    return train_data_iterator


def evaluate_and_print_results_rl(
    data_iterator: Iterator[TensorDataset],
    model: list[LanguageModule],
    optimizer: MegatronOptimizer,
    iteration: int,
    write_to_tensorboard: bool = True,
):
    """Helper function to evaluate and dump results on screen.

    Args:
        data_iterator: Iterator over batches of evaluation dataset.
        model: Model to evaluate with.
        iteration: Current training iteration.
        write_to_tensorboard: Dumpt stuff to tensorboard or not.
    """
    args = get_args()

    # TODO(vitalyk): I do not track eval loss as in training. We probably should.
    # megatron-lm uses forward_step_func to do the above.

    # Use context manager to temporarily disable sequence parallelism for evaluation

    with torch.no_grad():
        with megatron_rl_inference_mode(
            model,
            optimizer,
            args.enable_cuda_graph,
            args.rl_reset_cuda_graphs,
            args.rl_offload_optimizer_during_inference,
            args.rl_offload_kv_cache_during_training,
            args.rl_remove_kv_cache_during_training,
        ) as inference_interface:

            loop = get_event_loop()

            rank = torch.distributed.get_rank()
            if rank == 0:
                print(f"Collecting evaluation results on rank {rank}...")
                agent = get_agent(args)
                request = EvaluationRequest(
                    inference_interface=inference_interface,
                    num_prompts=args.rl_prompts_per_eval,
                    validation=True,
                    rank_info=None,
                    generation_args={
                        'temperature': args.grpo_default_temperature,
                        'max_tokens': args.seq_length,
                        'top_p': args.grpo_default_top_p,
                    },
                )
                evaluation_responses = loop.run_until_complete(agent.run_evaluation(request))
                if not isinstance(evaluation_responses, list):
                    evaluation_responses = [evaluation_responses]
            else:
                evaluation_responses = None

        dp_eval_results: list[None | list[EvaluationResponse]] = [
            None for _ in range(args.world_size)
        ]
        dist.gather_object(
            evaluation_responses,
            dp_eval_results if dist.get_rank() == (args.world_size - 1) else None,
            dst=args.world_size - 1,
        )

        if dist.get_rank() == args.world_size - 1:
            dp_eval_results = [x for x in dp_eval_results if x is not None]
            # TODO(rkirby): maybe factor this out into a function?
            eval_metrics = defaultdict(list)
            for responses in dp_eval_results:
                for response in responses:
                    if response is None:
                        continue
                    for k, v in response.metrics().items():
                        eval_metrics[f"{response.env_id}_eval_mean_{k}"].extend(v)
                    for result in response.results:
                        if isinstance(result, RewardEvaluationResult):
                            try:
                                lang_rl_log(
                                    f"Evaluation: [{response.env_id}] [{result.reward}] {result.prompt} {result.response}"
                                )
                            except Exception as e:
                                lang_rl_log(f"Error: {e}")
                                lang_rl_log(f"Result: {result}")
            print(
                "Collected metrics:"
                + "".join([f"\n\t{k} count: {len(v)}" for k, v in eval_metrics.items()])
            )
            eval_metrics = {k: np.mean(v) for k, v in eval_metrics.items()}
            if write_to_tensorboard:
                tb_writer = get_tensorboard_writer()
                if tb_writer:
                    for k, v in eval_metrics.items():
                        tb_writer.add_scalar(k, v, iteration)
            wandb_writer = get_wandb_writer()
            if wandb_writer:
                wandb_writer.log(eval_metrics, step=iteration)
            print(
                "Evaluation results:"
                + "".join([f"\n\t{k}: {v:0.4f}" for k, v in eval_metrics.items()])
            )
            if lang_rl_log_dir:
                with open(
                    lang_rl_log_dir
                    + f'/eval_rank{rank}_iteration{args.curr_iteration}_'
                    + f'{Path(args.langrl_env_config).stem}.pkl',
                    'wb',
                ) as f:
                    pickle.dump(dp_eval_results, f)


def calculate_grpo_loss(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clamp_eps_lower: float,
    clamp_eps_upper: float,
    kl_beta: float,
    entropy_weight: float,
    inference_logprobs: torch.Tensor | None = None,
    is_truncation_coef: float | None = None,
    seq_starts: list | None = None,
    seq_lengths: list | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get GRPO loss, the kl term of the loss and the pi/pi_{old} ratios.

    Args:
        current_logprobs: pi logprobs, [batch, seq] for unpacked or [1, bin_size] for packed.
        old_logprobs: pi_{old} logprobs, [batch, seq] for unpacked or [1, bin_size] for packed.
        ref_logprobs: pi_{ref} logprobs, [batch, seq] for unpacked or [1, bin_size] for packed.
        advantages: advantages tensor, [batch,] for unpacked or [num_sequences_in_bin,] for packed.
        clamp_eps_lower: eps to clamp ratios from below.
        clamp_eps_upper: eps to clamp ratios from above, if vanilla GRPO, this should be equal to clamp_eps_lower.
        kl_beta: weight for the KL penalty term measuring the distance between pi and pi_{ref}.
        entropy_weight: weight for the entropy term.
        inference_logprobs: pi_{old} logprobs calculated by the inference engine.
            If not None, importance sampling correction will be applied.
        is_truncation_coef: importance sampling truncation coefficient. Will be applied if it is not None and inference_logprobs are present.
        seq_starts: (optional) For packed sequences: start positions of each sequence in the bin.
        seq_lengths: (optional) For packed sequences: original lengths of each sequence.

    Returns:
        total per-token GRPO loss [batch, seq] or [1, bin_size],
        kl_term of the loss [batch, seq] or [1, bin_size],
        pi/pi_{old} ratios [batch, seq] or [1, bin_size],
        entropy_term of the loss [batch, seq] or [1, bin_size],
        truncated_from_above [batch, seq] or [1, bin_size] (whether we clamped the ratios or not),
        truncated_from_below [batch, seq] or [1, bin_size] (whether we clamped the ratios or not).
    """
    # Ensure shapes match before computation
    if current_logprobs.shape != old_logprobs.shape:
        print_rank_0(
            f"WARNING: Shape mismatch - current_logprobs: {current_logprobs.shape}, old_logprobs: {old_logprobs.shape}"
        )

    ratios = (current_logprobs - old_logprobs).exp()
    clamped_ratios = ratios.clamp(1 - clamp_eps_lower, 1 + clamp_eps_upper)
    truncated_from_above = torch.gt(ratios, 1 + clamp_eps_upper)
    truncated_from_below = torch.lt(ratios, 1 - clamp_eps_lower)

    # Handle advantages based on whether this is packed or unpacked
    if seq_starts is not None and seq_lengths is not None:
        # Packed sequences: map each sequence's advantage to its tokens
        bin_size = current_logprobs.shape[1]
        packed_advantages = torch.zeros(
            (1, bin_size), device=current_logprobs.device, dtype=current_logprobs.dtype
        )

        for seq_idx, (start, seq_len) in enumerate(zip(seq_starts, seq_lengths)):
            # Logprobs are 1 token shorter than sequences
            end = min(start + seq_len - 1, bin_size)
            if end > start:
                packed_advantages[0, start:end] = advantages[seq_idx].item()

        advantages = packed_advantages
    else:
        # Unpacked sequences: broadcast single advantage per sequence
        # Reshape to [batch, 1] to match logprobs shape [batch, seq]
        advantages = advantages.view(-1, 1)

    ref_diff = ref_logprobs - current_logprobs
    kl_term = ref_diff.exp() - ref_diff - 1
    entropy_term = current_logprobs.exp() * current_logprobs

    is_weights = torch.tensor(1.0, dtype=old_logprobs.dtype).to(old_logprobs.device)
    if inference_logprobs is not None:
        is_weights = (old_logprobs - inference_logprobs).exp()
        if is_truncation_coef is not None:
            is_weights = torch.min(
                is_weights,
                torch.tensor(is_truncation_coef, dtype=old_logprobs.dtype).to(old_logprobs.device),
            )

    loss = (
        -is_weights * torch.min(ratios * advantages, clamped_ratios * advantages)
        + kl_beta * kl_term
        + entropy_weight * entropy_term
    )

    return loss, kl_term, ratios, entropy_term, truncated_from_above, truncated_from_below


@contextmanager
def megatron_rl_inference_mode(
    model: list[LanguageModule],
    optimizer: MegatronOptimizer,
    enable_cuda_graph: bool,
    reset_cuda_graphs: bool,
    offload_optimizer_during_inference: bool,
    offload_kv_cache_during_training: bool,
    remove_kv_cache_during_training: bool,
):
    """Manage the model inference context when collecting rollouts.

    Args:
        model: model to prepare.
        optimizer: optimizer used to train the model.
        enable_cuda_graph: use cuda graphs or not.
        reset_cuda_graphs: rebuild cuda graphs for each inference stage or not.
        offload_optimizer_during_inference: move optimizer to cpu during inference or not.
        offload_kv_cache_during_training: manually offload kv cache to host before training or not.
        remove_kv_cache_during_training: manually remove kv cache before training or not.

    Yields:
        None: this context manager does not return a value.

    """
    args = get_args()
    loop = get_event_loop()
    nvtx_range = get_nvtx_range()

    print(f"[{dist.get_rank()}:DP] Entering inference mode")

    # If we get a lower precision wrapper, we go one object deeper.
    lang_module = model[0].module.module if hasattr(model[0].module, "module") else model[0].module

    lang_module.eval()

    rotary_module = getattr(lang_module, "rotary_pos_emb", None)
    # Vanilla RotaryEmbedding module has lru_cache decorator which breaks RL training
    # as it tries to reuse frequences tensors cached in inference mode.
    has_lru_cache = rotary_module is not None and hasattr(rotary_module.forward, "cache_parameters")
    if has_lru_cache:
        rotary_module.forward.cache_clear()

    with torch.no_grad():

        if offload_optimizer_during_inference:
            with nvtx_range("offload-optimizer-before-inference"):
                optimizer.offload_to_cpu()

        if enable_cuda_graph:
            toggle_cuda_graphs(lang_module, True, reset_cuda_graphs=reset_cuda_graphs)

        inference_interface = get_inference_interface(args, loop, model)

        with nvtx_range("onload-kv-cache-before-inference"):
            if offload_kv_cache_during_training:
                assert (
                    reset_cuda_graphs
                ), "reset_cuda_graphs must be True when offloading kv cache during training"
                print(
                    f"[{dist.get_rank()}:DP] Restoring kv cache ({inference_interface._coordinator.engine.context.memory_buffer.numel() / 1024**3:.2f} GB) to GPU"
                )
                kv_cache = inference_interface._coordinator.engine.context.memory_buffer
                inference_interface._coordinator.engine.context.memory_buffer = kv_cache.cuda()
            elif remove_kv_cache_during_training:
                if inference_interface._coordinator.engine.context.memory_buffer is None:
                    inference_interface._coordinator.engine.context.build_memory_buffer()

        if enable_cuda_graph and not _CudagraphGlobalRecord.cudagraph_created:
            with nvtx_range("wait-for-decode-only"):
                while not inference_interface._coordinator.engine.context.is_decode_only():
                    active_requests, finished_requests, step_time = loop.run_until_complete(
                        inference_interface._coordinator.engine.async_step()
                    )
            with nvtx_range("build-cuda-graphs"):
                inference_interface._coordinator.engine.build_cuda_graphs(reset_context=False)

        inference_interface.resume()

        yield inference_interface

        with nvtx_range("suspend-engine"):
            loop.run_until_complete(inference_interface.suspend())

        with nvtx_range("offload-kv-cache-after-inference"):
            if offload_kv_cache_during_training:
                kv_cache = inference_interface._coordinator.engine.context.memory_buffer
                print(
                    f"[{dist.get_rank()}:DP] Offloading kv cache ({kv_cache.numel() * kv_cache.element_size() / 1024**3:.2f} GB) to CPU"
                )
                inference_interface._coordinator.engine.context.memory_buffer = kv_cache.cpu()
            elif remove_kv_cache_during_training:
                inference_interface._coordinator.engine.context.memory_buffer = None

        if enable_cuda_graph:
            toggle_cuda_graphs(lang_module, False, reset_cuda_graphs=reset_cuda_graphs)

        if offload_optimizer_during_inference:
            with nvtx_range("onload-optimizer-after-inference"):
                optimizer.restore_from_cpu()

        lang_module.train()

        if has_lru_cache:
            rotary_module.forward.cache_clear()

        print(f"[{dist.get_rank()}:DP] Exiting inference mode")


def get_iteration_sequence_count(args):
    """Get the total number of sequences processed in this iteration across all ranks."""
    runtime_state = get_rl_runtime_state()
    sequences_tensor = torch.tensor(
        runtime_state.sequences_this_iteration_on_rank, device='cuda', dtype=torch.long
    )
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(sequences_tensor, group=mpu.get_data_parallel_group())
    return int(sequences_tensor.item())


def update_sequence_packing_metrics(args):
    """Update bin tracking for sequence packing mode."""
    if args.use_sequence_packing:
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
