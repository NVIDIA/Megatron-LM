# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc

import copy
from functools import partial
# Keep this to make the env registered.
import itertools
import math
import logging
import json
import os
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional 

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.num_microbatches_calculator import reconfigure_num_microbatches_calculator
from megatron.core.optimizer import MegatronOptimizer
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.pipeline_parallel.utils import is_pp_last_stage, get_pp_last_rank
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.core.tokenizers import MegatronTokenizer
from megatron.core.tokenizers.text.libraries.huggingface_tokenizer import HuggingFaceTokenizer
from megatron.core.transformer.cuda_graphs import _CudagraphGlobalRecord
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.utils import (
    toggle_cuda_graphs,
    transition_moe_cudagraphs,
)
from megatron.core.inference.utils import set_decode_expert_padding
from megatron.core.resharding.refit import swap_model_weights
from megatron.core.inference.unified_memory import (
    advise_managed_module_parameters_preferred_location,
    prefetch_managed_module_parameters,
)
from megatron.core.inference.utils import device_memory_summary
from megatron.core.utils import get_asyncio_loop, log_single_rank
from megatron.rl.sequence_packing_utils import (
    get_microbatch_dataloader,
    pack_inference_logprobs,
    compute_packed_inference_logprobs_stats,
    pack_all_trajectories,
    load_packed_data_by_index,
    get_sequence_packing_tensorboard_metrics,
    get_sequence_packing_log_info,
    get_default_packed_seq_params,
    get_packing_actual_tokens,
    get_packing_compute_tokens,
    get_packing_efficiency,
    get_packing_avg_seq_length,
    update_microbatch_calculator,
)
from megatron.rl.agent.api import (
    EvaluationRequest,
    EvaluationResponse,
    GroupedRolloutRequest,
    GroupedRollouts,
    RewardEvaluationResult,
    Rollout,
    RolloutGroup,
    Rollouts,
    TokenRollout,
)
from megatron.rl.agent.weighted_multi_task import WeightedMultiTask
from megatron.rl.inference.megatron import MegatronLocal
from megatron.rl.logging import LOG_DIR as lang_rl_log_dir
from megatron.rl.logging import log as lang_rl_log
from megatron.rl.server.inference.inference_interface_server import InferenceInterfaceServer
from megatron.training.global_vars import (
    get_args,
    get_tensorboard_writer,
    get_tokenizer,
    get_wandb_writer,
)
from megatron.training.utils import (
    get_ltor_masks_and_position_ids,
    get_nvtx_range,
    print_rank_0,
)
from megatron.core.utils import get_pg_rank, get_pg_size, get_attr_wrapped_model, unwrap_model
from megatron.core.process_groups_config import ProcessGroupCollection
from wandb import wandb_run
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    is_batch_invariant_mode_enabled,
)

from megatron.core.inference.contexts.dynamic_context import HAVE_TORCH_MEMORY_SAVER
if HAVE_TORCH_MEMORY_SAVER:
    from torch_memory_saver import torch_memory_saver

logger = logging.getLogger(__name__)

# Global variable to store packing context for forward_step
_GLOBAL_PACKING_CONTEXT = None


# Track whether the inference model is currently paused (offloaded to CPU).
# Model starts on GPU after creation and is used immediately, so starts as False.
_INFERENCE_MODEL_IS_PAUSED = False


def _torch_saver_swap_inference_model(*, to_cpu: bool) -> None:
    """Swap RL inference model weights between CPU and GPU using torch_memory_saver.

    Uses torch_memory_saver.pause()/resume() to transfer inference model weights
    that were allocated within a torch_memory_saver.region() context.

    Args:
        to_cpu: If True, move weights to CPU (pause). If False, restore weights to GPU (resume).
    """
    global _INFERENCE_MODEL_IS_PAUSED

    if not HAVE_TORCH_MEMORY_SAVER:
        raise RuntimeError(
            "torch_memory_saver is required for inference model offloading when not using UVM. "
            "Please install it: pip install torch_memory_saver "
            "(see https://github.com/fzyzcjy/torch_memory_saver)"
        )

    tag = "rl_inference_model"
    if to_cpu:
        if not _INFERENCE_MODEL_IS_PAUSED:
            print_rank_0(f"torch_memory_saver: pausing {tag}, before: {device_memory_summary()}")
            torch_memory_saver.pause(tag)
            _INFERENCE_MODEL_IS_PAUSED = True
            print_rank_0(f"torch_memory_saver: paused  {tag}, after:  {device_memory_summary()}")
    else:
        if _INFERENCE_MODEL_IS_PAUSED:
            print_rank_0(f"torch_memory_saver: resuming {tag}, before: {device_memory_summary()}")
            torch_memory_saver.resume(tag)
            _INFERENCE_MODEL_IS_PAUSED = False
            print_rank_0(f"torch_memory_saver: resumed  {tag}, after:  {device_memory_summary()}")


def _maybe_prefetch_separate_inference_model_weights(model_core, *, to_cpu: bool) -> None:
    """Prefetch RL *separate inference model* weights to CPU/GPU.

    Supports two modes:
    1. UVM-based offloading (when --rl-inference-model-unified-memory-level=1)
    2. torch_memory_saver-based offloading (when offloading is enabled but UVM is not)

    Gated by user args; this assumes the separate inference model was allocated
    with UVM or torch_memory_saver when enabled.
    """
    args = get_args()
    if not args.rl_offload_inference_model_weights_when_idle:
        return

    # Check for torch_memory_saver path (when offloading is enabled but UVM is not)
    if args.rl_inference_model_unified_memory_level != 1:
        _torch_saver_swap_inference_model(to_cpu=to_cpu)
        return

    # UVM-based path (when UVM level is 1)
    device = -1 if to_cpu else int(torch.cuda.current_device())
    # Note: include_buffers=False because buffers created with explicit device= in register_buffer()
    # are not allocated via the UVM mempool and will fail UVM operations. Only parameters are UVM-allocated.
    advise_managed_module_parameters_preferred_location(model_core, device=device, include_buffers=False)
    nbytes = prefetch_managed_module_parameters(model_core, device=device, include_buffers=False)
    # Ensure pages are resident before we enter CUDA-graph capture / inference, or before training continues.
    torch.cuda.synchronize()

    if to_cpu:
        print_rank_0(f"[Rank 0] offloaded {nbytes / 1024**2:.2f} MB of separate RL inference model weights to CPU (other ranks may vary)")
    else:
        print_rank_0(f"[Rank 0] prefetched {nbytes / 1024**2:.2f} MB of separate RL inference model weights to GPU (other ranks may vary)")


def verify_model_weights_swap(
    train_model: LanguageModule,
    inference_model: LanguageModule,
    seq_len: int = 8,
    batch_size: int = 2,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> None:
    """Verify that the inference model produces the same forward pass outputs
    as the training model after the weights have been swapped.

    This function should be called after swap_model_weights to ensure the weight
    transfer was successful. It runs a forward pass on both models and asserts
    the outputs match.  This is meant for debugging purposes only.

    Args:
        train_model: The training model (source of weights).
        inference_model: The inference model (target of weights).
        seq_len: Sequence length for test input.
        batch_size: Batch size for test input.
        atol: Absolute tolerance for comparing outputs.
        rtol: Relative tolerance for comparing outputs.

    Raises:
        AssertionError: If forward pass outputs do not match within tolerance.
    """
    args = get_args()

    # Unwrap models to get the core module
    train_lm = train_model[0] if isinstance(train_model, (list, tuple)) else train_model
    inf_lm = inference_model[0] if isinstance(inference_model, (list, tuple)) else inference_model

    train_core = unwrap_model(train_lm)
    inf_core = unwrap_model(inf_lm)

    actual_vocab_size = getattr(args, 'padded_vocab_size', 128256)
    actual_seq_len = min(seq_len, getattr(args, 'seq_length', seq_len))
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    # Generate deterministic test input - same across ALL ranks
    torch.manual_seed(1234)
    test_tokens = torch.randint(
        low=0, high=actual_vocab_size, size=(batch_size, actual_seq_len),
        device=device, dtype=torch.long
    )
    test_position_ids = (
        torch.arange(actual_seq_len, device=device, dtype=torch.long)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    test_attention_mask = torch.ones(
        (batch_size, 1, actual_seq_len, actual_seq_len), device=device, dtype=torch.bool
    )

    # Save and restore training state
    train_was_training = train_core.training
    inf_was_training = inf_core.training

    train_core.eval()
    inf_core.eval()

    try:
        with torch.no_grad():
            train_output = train_lm(
                test_tokens, test_position_ids, test_attention_mask,
                runtime_gather_output=True
            )

            inf_output = inf_lm(
                test_tokens, test_position_ids, test_attention_mask,
                runtime_gather_output=True
            )

        # Only check on ranks that have output (last PP stage)
        if train_output is not None and inf_output is not None:
            assert train_output.shape == inf_output.shape, (
                f"Output shape mismatch: train={train_output.shape}, infer={inf_output.shape}"
            )
            
            max_diff = (train_output - inf_output).abs().max().item()
            assert torch.allclose(train_output, inf_output, atol=atol, rtol=rtol), (
                f"Forward pass outputs do not match: max_diff={max_diff:.6e}, atol={atol}, rtol={rtol}"
            )

    finally:
        # Restore training state
        if train_was_training:
            train_core.train()
        if inf_was_training:
            inf_core.train()



@dataclass(slots=True)
class RolloutStats:
    rewards: list[list[float]] # inner list is for a group
    env_ids: list[str] # same length as len(rewards)
    turn_lens: list[list[int]] # token lengths of turns, grouped.
    traj_lens: list[list[int]] # all turns comprise one trajectory.
    num_turns: None | list[list[int]] # num_turns per traj
    advantages: None | list[list[float]]
    min_piold_to_inf_prob: None | float
    max_piold_to_inf_prob: None | float
    mean_piold_to_inf_prob: None | float
    min_inf_train_prob_abs_diff: None | float
    max_inf_train_prob_abs_diff: None | float
    mean_inf_train_prob_abs_diff: None | float
    min_inf_prob: None | float
    max_inf_prob: None | float
    mean_inf_prob: None | float
    # Per-rollout policy/kv-cache epoch summaries, grouped (one inner list per
    # group). Token-weighted; lag is current_iteration - epoch at log time.
    policy_first_epoch: list[list[int]]
    policy_avg_epoch: list[list[float]]
    policy_last_epoch: list[list[int]]
    kv_first_epoch: list[list[int]]
    kv_avg_epoch: list[list[float]]
    kv_last_epoch: list[list[int]]
    completed_epochs: list[list[int]]
    num_evictions: list[list[int]]
    # Per-rollout identity (grouped like rewards), used to label the rollout table.
    rollout_env_ids: list[list[str]]
    problem_ids: list[list[str | None]]


# Runtime state container for RL-specific data that shouldn't be checkpointed
class RLRuntimeState:
    """Container for runtime state that is not checkpointed, tracking state between rollout collections"""

    def __init__(self):
        self.packing_context = None
        self.last_collection_iteration = 0
        self.sequences_this_iteration_on_rank = 0
        self.latest_batch_num_sequences = 0
        # Derived throughput metrics (set by log_rl_throughput_metrics, read by RLProfiler).
        # Per-GPU variants are available via methods that divide by world_size.
        self.world_size = None
        # batch_size * seq_length / time: nominal throughput based on batch configuration
        self.tokens_per_sec = None
        # Total tokens in packed bins across all DP ranks / time: what the GPU actually processes.
        self.compute_tokens_per_sec = None
        # Real non-padding tokens across all DP ranks / time: true useful throughput.
        self.actual_tokens_per_sec = None
        # Fraction of bin capacity filled with real tokens (actual / total capacity)
        self.packing_efficiency = None

    def reset_iteration_counters(self, iteration):
        """Reset per-iteration counters."""
        self.sequences_this_iteration_on_rank = 0
        self.last_collection_iteration = iteration
        self.tokens_per_sec = None
        self.compute_tokens_per_sec = None
        self.actual_tokens_per_sec = None
        self.packing_efficiency = None

    def increment_sequences(self, count):
        """Increment the sequence counter."""
        self.sequences_this_iteration_on_rank += count
        self.latest_batch_num_sequences = count


# Global runtime state instance
_rl_runtime_state = RLRuntimeState()


def get_rl_runtime_state():
    """Get the global RL runtime state."""
    return _rl_runtime_state


def log_rl_throughput_metrics(args, batch_size, elapsed_time_per_iteration, iteration, wandb_writer):
    """Compute, log, and store RL token throughput metrics.

    Returns a string fragment to append to the training log line.
    Also logs metrics to wandb and stores them on RLRuntimeState for
    downstream consumers (e.g. RLProfiler).
    """
    log_string = ''
    tokens_per_sec = None
    tokens_per_sec_per_gpu = None
    compute_tokens_per_sec = None
    compute_tokens_per_sec_per_gpu = None
    actual_tokens_per_sec = None
    actual_tokens_per_sec_per_gpu = None
    packing_efficiency = None

    if args.seq_length > 0:
        tokens_per_iteration = batch_size * args.seq_length
        tokens_per_sec = tokens_per_iteration / elapsed_time_per_iteration
        tokens_per_sec_per_gpu = tokens_per_sec / args.world_size

        # For sequence packing, break down into compute vs actual tokens
        if args.rl_use_sequence_packing:
            runtime_state = get_rl_runtime_state()
            if runtime_state.packing_context is not None:
                dp_world_size = mpu.get_data_parallel_world_size()

                compute_tokens = get_packing_compute_tokens(runtime_state.packing_context)
                all_ranks_compute_tokens = compute_tokens * dp_world_size
                compute_tokens_per_sec = all_ranks_compute_tokens / elapsed_time_per_iteration
                compute_tokens_per_sec_per_gpu = compute_tokens_per_sec / args.world_size

                actual_tokens = get_packing_actual_tokens(runtime_state.packing_context)
                all_ranks_actual_tokens = actual_tokens * dp_world_size
                actual_tokens_per_sec = all_ranks_actual_tokens / elapsed_time_per_iteration
                actual_tokens_per_sec_per_gpu = actual_tokens_per_sec / args.world_size

                packing_efficiency = get_packing_efficiency(runtime_state.packing_context)

        # Add tokens/sec to log string
        log_string += f' toks/s: {tokens_per_sec:.0f} |'
        log_string += f' toks/s/gpu: {tokens_per_sec_per_gpu:.0f} |'
        if compute_tokens_per_sec is not None:
            log_string += f' compute_toks/s: {compute_tokens_per_sec:.0f} |'
            log_string += f' compute_toks/s/gpu: {compute_tokens_per_sec_per_gpu:.0f} |'
        if actual_tokens_per_sec is not None:
            log_string += f' actual_toks/s: {actual_tokens_per_sec:.0f} |'
            log_string += f' actual_toks/s/gpu: {actual_tokens_per_sec_per_gpu:.0f} |'
            log_string += f' packing_eff: {packing_efficiency:.1%} |'

    # Log throughput metrics to wandb
    if wandb_writer is not None:
        if tokens_per_sec is not None:
            wandb_writer.log({
                'throughput/tokens_per_sec': tokens_per_sec,
                'throughput/tokens_per_sec_per_gpu': tokens_per_sec_per_gpu,
            }, iteration)
        if compute_tokens_per_sec is not None:
            wandb_writer.log({
                'throughput/compute_tokens_per_sec': compute_tokens_per_sec,
                'throughput/compute_tokens_per_sec_per_gpu': compute_tokens_per_sec_per_gpu,
            }, iteration)
        if actual_tokens_per_sec is not None:
            wandb_writer.log({
                'throughput/actual_tokens_per_sec': actual_tokens_per_sec,
                'throughput/actual_tokens_per_sec_per_gpu': actual_tokens_per_sec_per_gpu,
                'throughput/packing_efficiency': packing_efficiency,
            }, iteration)

    # Store derived throughput metrics on RLRuntimeState so that
    # downstream consumers (e.g. RLProfiler) can read them.
    # Per-GPU values are derived via methods on RLRuntimeState.
    runtime_state = get_rl_runtime_state()
    runtime_state.world_size = args.world_size
    runtime_state.tokens_per_sec = tokens_per_sec
    runtime_state.compute_tokens_per_sec = compute_tokens_per_sec
    runtime_state.actual_tokens_per_sec = actual_tokens_per_sec
    runtime_state.packing_efficiency = packing_efficiency

    # Log average sequence length. With packing this shows real sequence
    # lengths; without packing it equals seq_length as a baseline.
    packing_ctx = runtime_state.packing_context
    if args.rl_use_sequence_packing and packing_ctx is not None:
        avg_seq_length = get_packing_avg_seq_length(packing_ctx)
        log_string += f' avg_seq_len: {avg_seq_length:.1f} |'
        if wandb_writer is not None:
            wandb_writer.log({'throughput/avg_seq_length': avg_seq_length}, iteration)
    elif args.log_throughput:
        log_string += f' avg_seq_len: {args.seq_length} |'

    return log_string


def update_inference_logprobs_group_stats(
    old_logprobs: torch.Tensor,
    inference_logprobs: torch.Tensor,
    mask: torch.Tensor,
    group_stats: Any,
) -> None:
    """Update group statistics with inference/train logprobs comparison metrics.

    This is the common statistics computation used by both packed and unpacked cases.

    Args:
        old_logprobs: Old logprobs tensor (train side)
        inference_logprobs: Inference logprobs tensor (aligned to match old_logprobs shape)
        mask: Boolean mask indicating valid positions for statistics
        group_stats: Statistics object to update with computed metrics
    """
    n_elems = mask.sum()
    if n_elems > 0:
        ratios = (old_logprobs - inference_logprobs).exp()[mask]
        abs_diffs = (old_logprobs.exp() - inference_logprobs.exp()).abs()[mask]

        group_stats.min_piold_to_inf_prob = ratios.min().item()
        group_stats.max_piold_to_inf_prob = ratios.max().item()
        group_stats.mean_piold_to_inf_prob = (ratios.sum() / n_elems).item()
        group_stats.min_inf_train_prob_abs_diff = abs_diffs.min().item()
        group_stats.max_inf_train_prob_abs_diff = abs_diffs.max().item()
        group_stats.mean_inf_train_prob_abs_diff = (abs_diffs.sum() / n_elems).item()

        inf_probs = inference_logprobs.exp()[mask]
        group_stats.min_inf_prob = inf_probs.min().item()
        group_stats.max_inf_prob = inf_probs.max().item()
        group_stats.mean_inf_prob = inf_probs.mean().item()


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
        first_gen_idx = int(first_gen_tok[i])
        if first_gen_idx < 0:
            # No generation tokens (dropped/placeholder rollout) — nothing to align.
            continue
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

    # Sanity check: Two probability values cannot be more than 1.0 apart
    abs_diffs = (old_logprobs_for_data.exp() - padded_inference_logprobs.exp()).abs()[truncated_mask]
    assert all(abs_diffs <= 1.0)

    # Update group statistics using common helper
    update_inference_logprobs_group_stats(
        old_logprobs=old_logprobs_for_data,
        inference_logprobs=padded_inference_logprobs,
        mask=truncated_mask,
        group_stats=group_stats,
    )

    return padded_inference_logprobs


def get_agent(args, parallel_generation_tasks: int | None = None):
    """Get an agent based on environment configuration.

    If args.langrl_env_config is provided, uses weighted environment selection.
    Otherwise falls back to legacy single environment selection.
    """
    with open(args.langrl_env_config, 'r') as f:
        config = yaml.safe_load(f)

    return WeightedMultiTask.from_config(
        config,
        parallel_generation_tasks=parallel_generation_tasks,
    )


_INFERENCE_INTERFACE = None


def get_inference_interface(args, loop, model):
    global _INFERENCE_INTERFACE
    if _INFERENCE_INTERFACE is None:
        inference_model = model[0]

        # Speculative rollout: back the inference engine with a fast early-exit
        # draft model that shares weights with the full model and exits after the
        # first `rl_speculative_exit_layer` transformer blocks.  The training
        # forward pass (get_logprobs) always uses the full model, so no IS
        # correction is required.
        exit_layer = getattr(args, 'rl_speculative_exit_layer', None)
        if exit_layer is not None:
            from megatron.rl.inference.draft_model import EarlyExitGPTModel
            full_gpt = unwrap_model(inference_model)
            total_layers = len(full_gpt.decoder.layers)
            if not (1 <= exit_layer < total_layers):
                raise ValueError(
                    f"--rl-speculative-exit-layer={exit_layer} must satisfy "
                    f"1 <= exit_layer < {total_layers} (total transformer layers)."
                )
            inference_model = EarlyExitGPTModel(full_gpt, exit_layer)
            log_single_rank(
                logger, logging.INFO,
                f"[Speculative Rollout] Inference engine uses "
                f"EarlyExitGPTModel (exit_layer={exit_layer}/{total_layers}). "
                f"Draft generates {args.rl_speculative_oversample_factor}x rollouts; "
                f"selection strategy: {args.rl_speculative_selection_strategy}."
            )

        _INFERENCE_INTERFACE = loop.run_until_complete(
            MegatronLocal.launch(
                inference_model,
                host='0.0.0.0',
                port=8294,
                verbose=args.inference_text_gen_server_logging)
        )
    return _INFERENCE_INTERFACE


_ROLLOUT_GENERATOR = None


async def _speculative_select_generator(base_generator, k, strategy):
    """Async generator wrapper that down-selects oversampled rollout groups.

    Consumes groups of ``k * oversample_factor`` rollouts from ``base_generator``
    and yields groups of exactly ``k`` rollouts, selected by ``strategy``.
    Skips falsy groups (``None`` or empty), which can be emitted by upstream
    filters (e.g. ``filter_groups_with_same_reward``).
    """
    from megatron.rl.agent.speculative_mixin import select_rollouts, SelectionStrategy
    strat = SelectionStrategy(strategy)
    async for group in base_generator:
        if not group:
            yield group
            continue
        yield select_rollouts(list(group), k, strat)


def get_rollout_generator(args, inference_interface, n_prompts, samples_per_group):
    global _ROLLOUT_GENERATOR
    if not (streaming := args.rl_partial_rollouts) or _ROLLOUT_GENERATOR is None:
        agent = get_agent(
            args,
            parallel_generation_tasks=args.rl_parallel_generation_tasks if streaming else n_prompts,
        )

        # When speculative rollout is enabled, inflate rollouts_per_group so the
        # draft engine generates oversample_factor * samples_per_group candidates.
        # A thin async wrapper then down-selects to the original samples_per_group
        # before the rollouts reach the training pipeline.
        exit_layer = getattr(args, 'rl_speculative_exit_layer', None)
        effective_rollouts_per_group = (
            samples_per_group * args.rl_speculative_oversample_factor
            if exit_layer is not None
            else samples_per_group
        )


        request = GroupedRolloutRequest(
            num_groups=args.rl_generation_batch_size if streaming else n_prompts,
            streaming=streaming,
            rollouts_per_group=effective_rollouts_per_group,
            inference_interface=inference_interface,
            generation_args={
                'temperature': args.rl_default_temperature,
                'max_tokens': args.inference_max_seq_length,
                'top_p': args.rl_default_top_p,
                'top_k': args.rl_default_top_k,
            },
            filter_groups_with_same_reward=args.grpo_filter_groups_with_same_reward,
            enforce_order=args.rl_enforce_generation_order,
        )
        base_gen = agent.get_grouped_rollouts(request)

        if exit_layer is not None:
            _ROLLOUT_GENERATOR = _speculative_select_generator(
                base_gen,
                k=samples_per_group,
                strategy=args.rl_speculative_selection_strategy,
            )
        else:
            _ROLLOUT_GENERATOR = base_gen
    return _ROLLOUT_GENERATOR


def get_environment_rollouts(
    model: LanguageModule, inference_model: LanguageModule, optimizer: MegatronOptimizer, n_prompts: int, samples_per_group: int
):
    """Sample environment rollouts from an LLM.

    Args:
        model: Model to sample from.
        inference_model: Inference model to use for inference.
        n_prompts: Number of prompts to sample for across *all* data parallel workers.
        samples_per_group: Amount of trajectories per prompt.

    Returns:
        GroupedRollouts object which is a nested list with each element being a list of rollouts of a group.
    """
    args = get_args()
    nvtx_range = get_nvtx_range()

    if args.rl_offload_optimizer_during_inference:
        with nvtx_range("rl/offload-optimizer-before-inference", time=True):
            if not args.rl_training_cuda_graphs:
                with nvtx_range("rl/offload/grad-buffers", time=True):
                    model[0].offload_grad_buffers()
            else:
                logger.warning(
                    "Gradient buffers will not be offloaded when training cudagraphs are enabled!")
            with nvtx_range("rl/offload/optimizer-state", time=True):
                optimizer.offload_to_cpu()

    # If we have separate training and inference models we to refit weights from the training model to the inference model.
    has_separate_inference_model = inference_model is not None
    if has_separate_inference_model:
        # If the separate inference model weights were prefetched to CPU while idle, bring them
        # back to GPU before refit/copy and before any CUDA-graph'd inference.
        with nvtx_range("rl/prefetch-weights-to-gpu", time=True):
            inf_core = unwrap_model(inference_model[0])
            _maybe_prefetch_separate_inference_model_weights(inf_core, to_cpu=False)
        swap_model_weights(model, inference_model, args.refit_method)
        if args.rl_verify_model_weights_swap:
            verify_model_weights_swap(
                train_model=model,
                inference_model=inference_model,
                atol=.1,
                rtol=5e-4,
            )
    else:
        inference_model = model

    inference_pg_collection = get_attr_wrapped_model(inference_model[0], "pg_collection")
    pg_size = get_pg_size(inference_pg_collection.ep)
    assert (n_prompts % pg_size == 0), f"{n_prompts=} must be divisible by {pg_size=}"

    with nvtx_range("rl/rollout-collection", time=True):
        loop = get_asyncio_loop()
        with megatron_rl_inference_mode(
            inference_model,
            optimizer,
            args.cuda_graph_impl,
            False, # offload optimizer during rollout collection is handled above
            training_model=model if has_separate_inference_model else None,
        ) as inference_interface:

            with nvtx_range("rl/inference-setup", time=True):
                # Asyncronously run inference and rollout collection
                rollout_generator = get_rollout_generator(
                    args, inference_interface, n_prompts, samples_per_group
                )

            # NOTE(jbarker): we need to double check this when using PP>1
            rank = torch.distributed.get_rank()
            with nvtx_range("rl/collect-rollouts", time=True):
                if rank == 0:
                    log_single_rank(
                        logger,
                        logging.INFO,
                        f"Collecting rollouts, Iteration {args.curr_iteration}...",
                    )
                    rollouts = [
                        loop.run_until_complete(anext(rollout_generator)) for _ in range(n_prompts)
                    ]
                    # In deterministic mode, sort rollouts by problem_id for consistent ordering
                    # regardless of completion order due to system timing jitter.
                    if torch.are_deterministic_algorithms_enabled():
                        rollouts.sort(key=lambda group: group[0].problem_id if group and group[0].problem_id else "")
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

        with nvtx_range("rl/sync-rollouts", time=True):
            # Wait for Rollouts to be collected
            # TODO(jbarker): double check why this isn't causing rank 0 memory allocations
            torch.distributed.broadcast_object_list(rollouts, src=0)
        logger.debug(f"Got rollouts on rank {rank}")

    if lang_rl_log_dir and rank == get_pg_rank(inference_pg_collection.tp):
        with open(
            lang_rl_log_dir
            + f'/rollouts_rank{rank}_iteration{args.curr_iteration}_'
            + f'{Path(args.langrl_env_config).stem}.json',
            'w',
        ) as f:
            json.dump([[r.model_dump() for r in group] for group in rollouts], f)

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
    use_bik_logsoftmax = is_batch_invariant_mode_enabled()
    if logits.dtype in [torch.float32, torch.float64] and not use_bik_logsoftmax:
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


_VP_BIK_WARNED = False


def _vp_chunk_tokens(partition_vocab_size: int, target_bytes: int = 512 * 2**20) -> int:
    """Tokens per chunk so one fp32 [chunk, partition_vocab] buffer stays ~target_bytes."""
    return max(1, target_bytes // (partition_vocab_size * 4))


class _VocabParallelSelectiveLogProbs(torch.autograd.Function):
    """`selective_log_softmax` over vocab-parallel (TP-sharded) logits.

    Never materializes full-vocab logits, an fp32 logits copy, or a saved
    softmax: the statistics (max, sum-exp, target logit) are computed in fp32
    over sequence chunks and combined across the TP group with one MAX and one
    SUM all-reduce; the backward recomputes the softmax chunk-by-chunk from the
    (already-live) input logits. At seq 131072 / CP 4 / TP 2 / vocab 131072
    this replaces the gathered path's ~24 GiB of logit-head transients and
    saved tensors per rank (8 GiB gathered logits + 8 GiB saved log_softmax
    output + 8 GiB backward transients) with the 4 GiB sharded logits,
    ~0.5 GiB chunk buffers, and the unavoidable 4 GiB grad-of-logits.
    """

    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, tp_group, chunk_tokens):
        """See `vocab_parallel_selective_log_softmax` for the contract."""
        orig_shape = target.shape
        partition_vocab_size = vocab_parallel_logits.size(-1)
        logits_2d = vocab_parallel_logits.reshape(-1, partition_vocab_size)
        target_1d = target.reshape(-1)

        if tp_group is not None and torch.distributed.is_initialized():
            tp_rank = torch.distributed.get_rank(tp_group)
            tp_world = torch.distributed.get_world_size(tp_group)
        else:
            tp_rank, tp_world = 0, 1

        # Same partition convention as the vocab-parallel output layer
        # (VocabUtility.vocab_range_from_per_partition_vocab_size).
        vocab_start = tp_rank * partition_vocab_size
        vocab_end = vocab_start + partition_vocab_size

        target_mask = (target_1d < vocab_start) | (target_1d >= vocab_end)
        masked_target = (target_1d - vocab_start).masked_fill(target_mask, 0)

        logits_max = logits_2d.amax(dim=-1).float()
        if tp_world > 1:
            torch.distributed.all_reduce(
                logits_max, op=torch.distributed.ReduceOp.MAX, group=tp_group
            )

        n = logits_2d.size(0)
        chunk = chunk_tokens or _vp_chunk_tokens(partition_vocab_size)
        # Row 0: logit[target] - max (nonzero only on the rank owning the target id).
        # Row 1: sum(exp(logits - max)) over the local vocab partition.
        stats = torch.empty(2, n, dtype=torch.float32, device=logits_2d.device)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            # copy=True: `.float()` would alias (and the in-place ops corrupt)
            # the saved logits when the input is already fp32.
            shifted = logits_2d[s:e].to(torch.float32, copy=True)
            shifted.sub_(logits_max[s:e].unsqueeze(-1))
            stats[0, s:e] = shifted.gather(-1, masked_target[s:e].unsqueeze(-1)).squeeze(-1)
            stats[1, s:e] = shifted.exp_().sum(dim=-1)
        stats[0].masked_fill_(target_mask, 0.0)
        if tp_world > 1:
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM, group=tp_group)

        # log p(target) = (logit[target] - max) - log(sum(exp(logits - max)))
        logprobs = stats[0] - stats[1].log()

        ctx.save_for_backward(
            vocab_parallel_logits, logits_max, stats[1].clone(), target_mask, masked_target
        )
        ctx.vp_chunk = chunk
        return logprobs.view(orig_shape).to(vocab_parallel_logits.dtype)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        """d log p(target) / d logit_j = (1 if j == target else 0) - softmax_j."""
        vocab_parallel_logits, logits_max, sum_exp, target_mask, masked_target = ctx.saved_tensors
        partition_vocab_size = vocab_parallel_logits.size(-1)
        logits_2d = vocab_parallel_logits.reshape(-1, partition_vocab_size)
        n = logits_2d.size(0)

        grad_out_1d = grad_output.reshape(-1).float()
        grad_input = torch.empty_like(vocab_parallel_logits)
        grad_2d = grad_input.reshape(-1, partition_vocab_size)

        chunk = ctx.vp_chunk
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            go = grad_out_1d[s:e]
            # grad_j = go * delta_{j==target} - go * softmax_j, with
            # softmax_j = exp(logit_j - max) / sum_exp (sum_exp is TP-global).
            # Fold -go/sum_exp into one row-scale so the chunk sees a single
            # multiply after exp.
            sm = logits_2d[s:e].to(torch.float32, copy=True)
            sm.sub_(logits_max[s:e].unsqueeze(-1)).exp_()
            sm.mul_((go / sum_exp[s:e]).neg_().unsqueeze(-1))
            # One-hot term: only the rank owning the target id contributes.
            # Vectorized with a fixed-size arange + masked grad (no
            # torch.nonzero -> no device-host sync, CUDA-graph-capture-safe).
            arange = torch.arange(e - s, device=sm.device)
            sm[arange, masked_target[s:e]] += go * (~target_mask[s:e]).to(go.dtype)
            grad_2d[s:e] = sm.to(grad_input.dtype)
        return grad_input, None, None, None


def vocab_parallel_selective_log_softmax(vocab_parallel_logits, index, tp_group=None, chunk_tokens=None):
    """Vocab-parallel, sequence-chunked drop-in for `selective_log_softmax`.

    Equivalent to all-gathering the TP-sharded logits to the full vocabulary
    and calling ``selective_log_softmax(full_logits, index)``, but never
    materializes the gathered tensor or a saved softmax. Statistics are
    computed in fp32 regardless of the input dtype, so bf16 inputs get
    slightly *more* accurate logprobs than the bf16 log_softmax fallback in
    `selective_log_softmax`. Chunking is deterministic (fixed sequential
    accumulation order); `is_batch_invariant_mode_enabled` is not consulted.

    Args:
        vocab_parallel_logits: [..., vocab/tp] logits sharded over ``tp_group``
            with the output-layer convention (rank r owns rows
            [r * Vp, (r + 1) * Vp)). Pass unsharded logits with
            ``tp_group=None`` (or a world-size-1 group).
        index: [...] target token ids in the FULL vocabulary.
        tp_group: process group over which the vocab dimension is sharded.
        chunk_tokens: tokens per fp32 chunk; default sizes chunks to ~512 MiB.

    Note:
        A sequence-sliced view (e.g. ``logits[:, :-1, :]``) stays reshape-safe
        (no copy) only when the leading batch dim is 1 — always true in RL
        training (micro-batch-size 1). With batch > 1 the internal
        ``reshape`` silently copies the sliced logits (correct, but costs the
        memory the function exists to save).

    Returns:
        Log-probabilities with the same shape as ``index`` and the same dtype
        as ``vocab_parallel_logits``.
    """
    return _VocabParallelSelectiveLogProbs.apply(vocab_parallel_logits, index, tp_group, chunk_tokens)


def _zigzag_slice(x: torch.Tensor, cp_size: int, cp_rank: int) -> torch.Tensor:
    """Pick chunks ``cp_rank`` and ``2*cp_size - cp_rank - 1`` after viewing
    the sequence dim as ``2*cp_size`` equal chunks, then concatenate.

    Mirrors ``get_batch_on_this_cp_rank`` in ``megatron/core/utils.py`` — the
    canonical Megatron-LM load-balanced (zigzag) CP layout that TE ring
    attention, RoPE-CP, and Mamba-CP all assume.
    """
    seq_len = x.shape[1]
    chunk_size = seq_len // (2 * cp_size)
    # [B, S, ...] -> [B, 2*CP, S/(2*CP), ...]
    x = x.view(x.shape[0], 2 * cp_size, chunk_size, *x.shape[2:])
    index = torch.tensor(
        [cp_rank, 2 * cp_size - cp_rank - 1], dtype=torch.int64, device=x.device
    )
    x = x.index_select(1, index)
    # [B, 2, S/(2*CP), ...] -> [B, S/CP, ...]
    return x.reshape(x.shape[0], -1, *x.shape[3:]).contiguous()


def _scatter_for_context_parallel(
    tokens: torch.Tensor,
    position_ids: torch.Tensor,
    packed_seq_params: 'PackedSeqParams',
    cp_size: int,
) -> tuple:
    """Prepare local inputs for one context-parallel rank using the
    canonical Megatron-LM zigzag (load-balanced) CP layout.

    Each CP rank receives a NON-CONTIGUOUS pair of chunks
    ``(chunk_r, chunk_{2*cp_size-r-1})`` after a 2*cp_size partition of the
    sequence dim — matching what TE ring attention (``cp_comm_type=p2p``),
    Megatron's RoPE-CP slicer, and the Mamba CP layer all expect. A plain
    contiguous slice produces silently-wrong attention / SSM outputs (see
    cp_zigzag_scatter_bug memory).

    Labels are pre-shifted globally then zigzag-sliced the same way, so each
    rank's local labels stay aligned with its local tokens with no cross-rank
    communication. The final boundary label is a dummy (always loss-masked).

    Args:
        tokens:          Full token tensor  [batch, seq_len].
        position_ids:    Full position-id tensor  [batch, seq_len].
        packed_seq_params: PackedSeqParams for the full bin.  A shallow copy is
                         returned with the ``cp_group`` and ``local_cp_size``
                         fields set so Transformer Engine uses ring attention.
        cp_size:         Context-parallel world size.

    Returns:
        (local_tokens, local_position_ids, cp_packed_seq_params, local_labels)
        where every tensor has sequence length ``seq_len // cp_size``.
    """
    cp_rank  = mpu.get_context_parallel_rank()
    cp_group = mpu.get_context_parallel_group()

    seq_len = tokens.shape[1]
    assert seq_len % (2 * cp_size) == 0, (
        f"Sequence length {seq_len} must be divisible by 2*context_parallel_size "
        f"({2 * cp_size}) for the zigzag CP layout."
    )

    # Pre-shifted labels: tokens_shifted[i] = tokens[i+1] for i < S-1, and
    # tokens_shifted[S-1] = tokens[S-1] (dummy boundary, always loss-masked).
    # Zigzag-slicing this together with tokens keeps labels aligned per-rank.
    tokens_shifted = torch.cat([tokens[:, 1:], tokens[:, -1:]], dim=1)

    local_tokens       = _zigzag_slice(tokens,          cp_size, cp_rank)
    local_position_ids = _zigzag_slice(position_ids,    cp_size, cp_rank)
    local_labels       = _zigzag_slice(tokens_shifted,  cp_size, cp_rank)

    # Shallow-copy so we do not mutate the caller's object.
    cp_packed_seq_params = copy.copy(packed_seq_params)
    cp_packed_seq_params.cp_group      = cp_group
    cp_packed_seq_params.local_cp_size = cp_size
    # THD CP path needs cu_seqlens_*_padded; fall back to the unpadded variant
    # when the caller didn't supply one (matches get_thd_batch_on_this_cp_rank).
    if cp_packed_seq_params.cu_seqlens_q_padded is None:
        cp_packed_seq_params.cu_seqlens_q_padded  = cp_packed_seq_params.cu_seqlens_q
    if cp_packed_seq_params.cu_seqlens_kv_padded is None:
        cp_packed_seq_params.cu_seqlens_kv_padded = cp_packed_seq_params.cu_seqlens_kv

    return local_tokens, local_position_ids, cp_packed_seq_params, local_labels


def _gather_logprobs_context_parallel(
    local_logprobs: torch.Tensor,
    no_grad: bool,
) -> torch.Tensor:
    """All-gather per-rank logprobs and invert the zigzag scatter.

    Each rank holds ``[batch, 2*chunk]`` logprobs corresponding to chunks
    ``(r, 2*cp_size-r-1)`` of the global sequence. After all-gather we split
    each rank's slice into halves and place them in their global chunk slots
    to reconstruct ``[batch, seq_len]``, then drop the final dummy position
    appended by ``_scatter_for_context_parallel`` → ``[batch, seq_len - 1]``.

    Uses ``torch.distributed.nn.functional.all_gather`` on the training path so
    that gradients flow back through the gather (reduce-scatter) to each rank's
    local forward pass; ``cat`` in zigzag order routes each chunk's gradient
    back to its source rank.

    Args:
        local_logprobs: Local logprob tensor  [batch, 2*chunk].
        no_grad:        True when called in inference/reference-logprob mode.

    Returns:
        Full logprob tensor  [batch, seq_len - 1].
    """
    cp_group = mpu.get_context_parallel_group()
    cp_size  = mpu.get_context_parallel_world_size()

    if no_grad:
        gathered = [torch.empty_like(local_logprobs) for _ in range(cp_size)]
        torch.distributed.all_gather(gathered, local_logprobs, group=cp_group)
    else:
        # Differentiable all-gather: backward is a reduce-scatter that routes
        # each rank's gradient slice back to the correct local forward pass.
        gathered = torch.distributed.nn.functional.all_gather(local_logprobs, group=cp_group)

    # Each rank's gather result contains [chunk_r | chunk_{2*CP-r-1}]; split
    # and place in global chunk order.
    chunk_size = local_logprobs.shape[1] // 2
    chunks = [None] * (2 * cp_size)
    for r in range(cp_size):
        chunks[r]                    = gathered[r][:, :chunk_size]
        chunks[2 * cp_size - r - 1]  = gathered[r][:, chunk_size:]

    # cat → [batch, seq_len]; drop the dummy boundary position → [batch, seq_len-1].
    return torch.cat(chunks, dim=1)[:, :-1]


def get_logprobs(model, tokens, position_ids, no_grad=False, sequence_packing=False, packed_seq_params=None):
    """Get sequence logprobs from their token ids.

    Args:
        model: model to predict with.
        tokens: inputs for which we want to get logprobs.
        position_ids: position ids that come with tokens.
        attention_mask: attention mask that comes with tokens.
        no_grad: whether to run in no_grad mode.
        packed_seq_params: Optional PackedSeqParams for sequence packing with TE.
            When provided with qkv_format='thd', the input tokens are sliced to
            remove padding before the forward pass, and outputs are padded back.
        packed_seq_len: Optional length of the packed sequence (excluding padding).
            Required when packed_seq_params is provided to avoid CPU-GPU synchronization.

    Returns:
        Logprobs of input sequences  [batch, seq_len - 1].

        With context parallelism (cp_size > 1) each rank runs the forward pass
        on its ``seq_len // cp_size`` token slice.  Logprobs are all-gathered
        after the log-softmax so the returned tensor always has the full
        sequence length, matching the cp_size == 1 interface exactly.
    """

    args = get_args()
    # Ensure packed_seq_params is always provided for CUDA graph signature consistency.
    # When sequence_packing is enabled, construct from packing config (max_sequences_per_bin).
    # When sequence_packing is disabled, construct a single-sequence default so the CUDA
    # graph signature matches the training forward_step in train_rl.py.
    # This is necessary because reference logprobs steps will reuse the training forward graph.
    if packed_seq_params is None:
        if sequence_packing:
            packed_seq_params = get_default_packed_seq_params(
                seq_length=tokens.shape[1],
                max_sequences_per_bin=args.rl_sequence_packing_max_sequences_per_bin,
                device=tokens.device,
            )
        else:
            cu_seqlens = torch.tensor([0, tokens.shape[1]], dtype=torch.int32, device=tokens.device)
            packed_seq_params = PackedSeqParams(
                qkv_format='thd',
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=tokens.shape[1],
                max_seqlen_kv=tokens.shape[1],
                total_tokens=tokens.shape[1],
            )

    cp_size    = mpu.get_context_parallel_world_size()
    nvtx_range = get_nvtx_range()

    with nvtx_range("rl/get-logprobs", time=True):
        with nvtx_range("rl/forward-pass", time=True):
            # TODO(vitalyk): use fp16/bf16 as a function argument. Do not use args.

            attention_mask_for_forward = None

            # This is a hack to fix megatron's behaviour when flash-decode affects the training code flow.
            flash_decode = model.config.flash_decode
            model.config.flash_decode = False
            fp32_output = not (args.fp16 or args.bf16)

            # Vocab-parallel logprobs: keep the output-layer logits TP-sharded
            # ([*, vocab/tp]) and compute token logprobs without ever gathering
            # the full vocabulary. The gathered path materializes
            # [seq/cp, vocab] logits (8 GiB at seq 131072 / CP4 / vocab 128k in
            # bf16) plus an equal-sized saved log_softmax output and backward
            # transients — the exact 8 GiB allocations that OOM the GRPO
            # train-step backward.
            use_vp_logprobs = bool(getattr(args, 'rl_vocab_parallel_logprobs', False))
            if use_vp_logprobs and is_batch_invariant_mode_enabled():
                # The vp path does not route through the batch-invariant
                # kernels that --batch-invariant-mode patches in (its sum-exp
                # reduction is shape-dependent), so honor the mode by falling
                # back to the gathered log-softmax path.
                global _VP_BIK_WARNED
                if not _VP_BIK_WARNED:
                    log_single_rank(
                        logger,
                        logging.WARNING,
                        "--rl-vocab-parallel-logprobs is not batch-invariant; "
                        "using the gathered logprob path because "
                        "--batch-invariant-mode is enabled.",
                    )
                    _VP_BIK_WARNED = True
                use_vp_logprobs = False

            if cp_size > 1:
                # Scatter: each rank processes seq_len // cp_size tokens.
                local_tokens, local_position_ids, cp_packed_seq_params, local_labels = (
                    _scatter_for_context_parallel(tokens, position_ids, packed_seq_params, cp_size)
                )
                with torch.no_grad() if no_grad else nullcontext():
                    logits_or_hidden_states = model(
                        local_tokens,
                        local_position_ids,
                        attention_mask_for_forward,
                        packed_seq_params=cp_packed_seq_params,
                        runtime_gather_output=not use_vp_logprobs,
                        fp32_output=fp32_output,
                    )
            else:
                with torch.no_grad() if no_grad else nullcontext():
                    logits_or_hidden_states = model(
                        tokens,
                        position_ids,
                        attention_mask_for_forward,
                        packed_seq_params=packed_seq_params,
                        runtime_gather_output=not use_vp_logprobs,
                        fp32_output=fp32_output,
                    )

            model.config.flash_decode = flash_decode

        pg_collection = get_attr_wrapped_model(model, "pg_collection")
        pp_group = pg_collection.pp

        if not is_pp_last_stage(pp_group):
            return logits_or_hidden_states

        logits = logits_or_hidden_states
        tp_group = pg_collection.tp if use_vp_logprobs else None
        with nvtx_range("rl/log-softmax", time=True):
            if cp_size > 1:
                # Compute local logprobs then gather the full sequence.
                if use_vp_logprobs:
                    local_logprobs = vocab_parallel_selective_log_softmax(
                        logits, local_labels, tp_group
                    )
                else:
                    local_logprobs = selective_log_softmax(logits, local_labels)
                logprobs = _gather_logprobs_context_parallel(local_logprobs, no_grad)
            else:
                # We do not need logprobs for the n+1 token.
                if use_vp_logprobs:
                    logprobs = vocab_parallel_selective_log_softmax(
                        logits[:, :-1, :], tokens[:, 1:], tp_group
                    )
                else:
                    logprobs = selective_log_softmax(logits[:, :-1, :], tokens[:, 1:])
        return logprobs


def calculate_grpo_advantages(rewards: list[list[float]], num_turns: list[list[int]]) -> np.ndarray:
    """Calculate GRPO advantages from rewards/num_turns.

    For multiturn rollouts, the logic is a bit more involved.
    # For training, we'll be turning each turn into a trajectory with the same reward
    # within a trajectory, e.g. if [[a,b],[c,d,e]] trajectory has reward 1.0, we will
    # get [a,b] with 1.0 and [c,d,e] with 1.0 when doing updates.
    """

    rewards = np.array(rewards)

    num_turns = np.array(num_turns)
    # Each outer dimension of num_turns is a group. Sum of those gives total num_turns per group.
    # Let's use this to calculate advantage.
    # mean/std should be repeated based on group lens
    group_turns = num_turns.sum(axis=-1)
    reward_means = rewards.mean(axis=1, keepdims=True).repeat(group_turns)
    reward_stds = rewards.std(axis=1, keepdims=True).repeat(group_turns)

    # rewards are originally [g, group_size]
    # Making an assumption that all groups are of the same size!
    # @vitalyk: this will go away when we start sending env-based sample reqs.
    rewards = rewards.flatten().repeat(num_turns.flatten())

    return ((rewards - reward_means) / (1e-4 + reward_stds)).tolist()


def expand_epoch_segments(
    per_turn_boundaries: list[list[tuple[int, int]]],
    per_turn_token_count: list[int],
) -> list[tuple[int, int]]:
    """Expand run-length-encoded epoch boundaries into per-token (epoch, count) segments.

    Each turn's ``policy_epoch`` / ``kv_cache_epoch`` is a sparse list of
    ``(start_token_index, epoch)`` boundaries: a tuple ``(i, e)`` means "tokens
    from index ``i`` until the next boundary were generated under epoch ``e``".
    The last boundary of a turn runs to that turn's ``token_count``. Segments are
    concatenated across turns in order, so the first segment covers the rollout's
    first token and the last segment covers its last token.
    """
    segments: list[tuple[int, int]] = []
    for boundaries, total_len in zip(per_turn_boundaries, per_turn_token_count):
        if not boundaries:
            continue
        for idx, (start, epoch) in enumerate(boundaries):
            end = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else total_len
            count = end - start
            if count > 0:
                segments.append((epoch, count))
    return segments


def rollout_epoch_summary(
    per_turn_boundaries: list[list[tuple[int, int]]],
    per_turn_token_count: list[int],
) -> tuple[int, float, int] | None:
    """First, token-weighted-average, and last *epoch* of a rollout across all turns.

    Returns ``None`` if the rollout has no tokens. The token-weighted average is a
    true per-token average (weighting each epoch by how many tokens it covers),
    unlike a per-boundary mean. Lag is obtained later as ``current_iteration -
    epoch``, so ``first`` -> oldest-token lag and ``last`` -> newest-token lag.
    """
    segments = expand_epoch_segments(per_turn_boundaries, per_turn_token_count)
    if not segments:
        return None
    first_epoch = segments[0][0]
    last_epoch = segments[-1][0]
    total = sum(count for _, count in segments)
    avg_epoch = sum(epoch * count for epoch, count in segments) / total
    return first_epoch, avg_epoch, last_epoch


def get_rl_logging_dir(args) -> str:
    """Base directory for RL logging artifacts (staleness dumps, plots).

    Uses --rl-logging-dir when set, else $LANGRL_LOG_DIR/rl_logging, else ./rl_logging.
    """
    if args.rl_logging_dir:
        return args.rl_logging_dir
    env_dir = os.environ.get("LANGRL_LOG_DIR")
    if env_dir:
        return os.path.join(env_dir, "rl_logging")
    return os.path.join(".", "rl_logging")


def _turn_prompt_length(rollout, turn_idx: int) -> Optional[int]:
    """Best-effort number of prompt tokens for a turn (so plots can mark prompt/gen)."""
    if isinstance(rollout, TokenRollout):
        gen_mask = rollout.generation_mask[turn_idx] if rollout.generation_mask else None
        if gen_mask is None:
            return None
        for i, is_gen in enumerate(gen_mask):
            if is_gen:
                return i
        return len(gen_mask)
    if rollout.prompt_length is not None and turn_idx < len(rollout.prompt_length):
        return rollout.prompt_length[turn_idx]
    return None


# Lazily-opened JSONL file handle for the per-run staleness dump (rank 0 only).
_STALENESS_DUMP_FILE = None


def dump_staleness_data(
    rollouts: GroupedRollouts, group_stats: RolloutStats, current_iteration: int
) -> None:
    """Append a focused, labeled per-rollout staleness/length record (rank 0, opt-in).

    One JSON object per rollout per line, tagged by iteration/batch_id/group/env so
    rl_profiling can render per-batch/env/group/rollout views offline. Stores per-rollout
    reward + GRPO advantage, the run-length-encoded epoch boundaries and per-turn token
    counts, not token ids.
    """
    args = get_args()
    if not args.rl_log_staleness_data:
        return
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    global _STALENESS_DUMP_FILE
    if _STALENESS_DUMP_FILE is None:
        out_dir = Path(get_rl_logging_dir(args)) / "staleness"
        out_dir.mkdir(parents=True, exist_ok=True)
        run_id = os.environ.get("LANGRL_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
        _STALENESS_DUMP_FILE = open(out_dir / f"staleness_{run_id}.jsonl", "a")
        logger.info(f"[rl-logging] Writing staleness data to {out_dir}")

    # Per-rollout GRPO advantage (authoritative, from group_stats). `advantages` is a
    # flat per-turn list in group-major order; a rollout's turns share one advantage,
    # so take the first turn's value per rollout.
    adv_flat = group_stats.advantages
    per_group_adv = []
    _t = 0
    for group_num_turns in group_stats.num_turns:
        adv_row = []
        for nt in group_num_turns:
            adv_row.append(float(adv_flat[_t]))
            _t += nt
        per_group_adv.append(adv_row)

    for group_idx, group in enumerate(rollouts):
        group_rewards = group_stats.rewards[group_idx]
        group_reward_mean = float(np.mean(group_rewards))
        group_reward_std = float(np.std(group_rewards))
        # All-equal rewards => ~zero advantage => no gradient (the GRPO group-filter target).
        group_degenerate = group_reward_std < 1e-6
        for rollout_idx, rollout in enumerate(group):
            turns = []
            for turn_idx, (pe, kve) in enumerate(
                zip(rollout.policy_epoch, rollout.kv_cache_epoch)
            ):
                evicts = (
                    int(rollout.num_evictions[turn_idx])
                    if turn_idx < len(rollout.num_evictions)
                    else 0
                )
                turns.append({
                    "token_count": len(rollout.trajectory[turn_idx]),
                    "prompt_length": _turn_prompt_length(rollout, turn_idx),
                    "policy_epoch": [[int(s), int(e)] for s, e in pe],
                    "kv_cache_epoch": [[int(s), int(e)] for s, e in kve],
                    "num_evictions": evicts,
                })
            reward = (
                [float(x) for x in rollout.reward]
                if isinstance(rollout.reward, list)
                else float(rollout.reward)
            )
            record = {
                "iteration": int(current_iteration),
                "batch_id": int(group.batch_id),
                "index_in_batch": int(group.index_in_batch),
                "group_idx": int(group_idx),
                "rollout_idx": int(rollout_idx),
                "env_id": rollout.env_id,
                "problem_id": rollout.problem_id,
                "reward": reward,
                "advantage": per_group_adv[group_idx][rollout_idx],
                "group_reward_mean": group_reward_mean,
                "group_reward_std": group_reward_std,
                "group_degenerate": group_degenerate,
                "traj_len": int(sum(len(t) for t in rollout.trajectory)),
                "num_turns": len(rollout.trajectory),
                "num_evictions": int(sum(rollout.num_evictions)),
                "turns": turns,
            }
            _STALENESS_DUMP_FILE.write(json.dumps(record) + "\n")
    _STALENESS_DUMP_FILE.flush()


def compute_group_stats(
    rollouts: GroupedRollouts, tokenizer: MegatronTokenizer, seq_len: int,
) -> RolloutStats:
    """Add group-based rollout stats for logging.

    Args:
        rollouts: Rollouts to generate the stats for. Each inner list is a group (as in GRPO group), i.e. all rollouts are for the same prompt.
        tokenizer: Tokenizer to tokenize the rollouts in case they are raw strings.
        seq_len: Maximum sequence length.

    Returns:
       RolloutStats object containing all the stats.
    """
    # TODO (rkirby) Maybe do some of this after the tensor building
    group_reward_means = []
    group_reward_stds = []
    turn_lens = []
    traj_lens = []
    rewards = []
    env_ids = []
    group_reward_ids = []
    num_turns = [] # num_turns per traj
    all_policy_first = []
    all_policy_avg = []
    all_policy_last = []
    all_kv_first = []
    all_kv_avg = []
    all_kv_last = []
    all_completed_epochs = []
    all_num_evictions = []
    all_rollout_env_ids = []
    all_problem_ids = []
    for group in rollouts:
        group_rewards = []
        group_traj_lengths = []
        group_turn_lengths = []
        group_num_turns = []
        group_policy_first = []
        group_policy_avg = []
        group_policy_last = []
        group_kv_first = []
        group_kv_avg = []
        group_kv_last = []
        group_completed_epochs = []
        group_num_evictions = []
        group_rollout_env_ids = []
        group_problem_ids = []
        for rollout in group:
            if isinstance(rollout, TokenRollout):
                for turn_traj in rollout.trajectory:
                    detokenized_traj = tokenizer.detokenize(turn_traj)
                    lang_rl_log(
                        f"Rollout: [{rollout.env_id}] [{rollout.reward} : {len(rollout.trajectory)} tokens] {detokenized_traj}"
                    )
                    # Multi-turn agents can terminate a turn on a tool-call boundary,
                    # which is neither tokenizer.eod (11) nor a hit-seq_len truncation.
                    # The downstream packing/loss code only requires len <= seq_len;
                    # the strict EOD/full-length check was a single-turn assumption.
                    # TODO(vitalyk): tighten this with a per-agent terminator set if
                    # we want to keep some sanity check on multi-turn boundaries.
                    assert len(turn_traj) <= seq_len, (
                        f"Rollout too long: {len(turn_traj)} > {seq_len} "
                        f"(last token {turn_traj[-1]})\n{detokenized_traj}"
                    )
            else:
                lang_rl_log(
                    f"Rollout: [{rollout.env_id}] [{rollout.reward} : {len(rollout.trajectory)} chars] {rollout.trajectory}"
                )
            group_num_turns.append(len(rollout.trajectory))
            group_rewards.append(rollout.reward)
            roll_turn_lens = [len(t) for t in rollout.trajectory]
            group_turn_lengths.extend(roll_turn_lens)
            group_traj_lengths.append(sum(roll_turn_lens))
            assert rollout.policy_epoch, "Rollout has no policy_epoch data"
            assert rollout.kv_cache_epoch, "Rollout has no kv_cache_epoch data"
            # Token-weighted first/avg/last epoch from the per-turn RLE boundaries.
            policy_summary = rollout_epoch_summary(rollout.policy_epoch, roll_turn_lens)
            kv_summary = rollout_epoch_summary(rollout.kv_cache_epoch, roll_turn_lens)
            assert (
                policy_summary is not None and kv_summary is not None
            ), "Rollout has no tokens to summarize epochs over"
            group_policy_first.append(policy_summary[0])
            group_policy_avg.append(policy_summary[1])
            group_policy_last.append(policy_summary[2])
            group_kv_first.append(kv_summary[0])
            group_kv_avg.append(kv_summary[1])
            group_kv_last.append(kv_summary[2])
            group_completed_epochs.extend(turn[-1][1] for turn in rollout.policy_epoch)
            group_num_evictions.append(sum(rollout.num_evictions))
            group_rollout_env_ids.append(rollout.env_id)
            group_problem_ids.append(rollout.problem_id)
        all_policy_first.append(group_policy_first)
        all_policy_avg.append(group_policy_avg)
        all_policy_last.append(group_policy_last)
        all_kv_first.append(group_kv_first)
        all_kv_avg.append(group_kv_avg)
        all_kv_last.append(group_kv_last)
        all_completed_epochs.append(group_completed_epochs)
        all_num_evictions.append(group_num_evictions)
        all_rollout_env_ids.append(group_rollout_env_ids)
        all_problem_ids.append(group_problem_ids)
        traj_lens.append(group_traj_lengths)
        # Guard against an all-placeholder group (every sub-request failed, e.g.
        # workplace_assistant prompts already exceeding seq_length at turn 1).
        # turn_lens drives min/max/mean stats downstream; max([]) crashes.
        turn_lens.append(group_turn_lengths or [0])
        env_ids.append(group[0].env_id) # All rollouts in a group share the env_id by design.
        rewards.append(group_rewards)
        # https://arxiv.org/abs/2504.21233 reports that lens variance hurts.
        # Let's track this.
        num_turns.append(group_num_turns)

    stats = RolloutStats(
        traj_lens=traj_lens,
        turn_lens=turn_lens,
        rewards=rewards,
        # --------
        # Everything above is per-group, i.e. it is a list of lists,
        # with the inner list being the group data.
        env_ids=env_ids,
        num_turns=num_turns,
        advantages=calculate_grpo_advantages(rewards, num_turns),
        min_piold_to_inf_prob=None,
        max_piold_to_inf_prob=None,
        mean_piold_to_inf_prob=None,
        min_inf_train_prob_abs_diff=None,
        max_inf_train_prob_abs_diff=None,
        mean_inf_train_prob_abs_diff=None,
        min_inf_prob=None,
        max_inf_prob=None,
        mean_inf_prob=None,
        policy_first_epoch=all_policy_first,
        policy_avg_epoch=all_policy_avg,
        policy_last_epoch=all_policy_last,
        kv_first_epoch=all_kv_first,
        kv_avg_epoch=all_kv_avg,
        kv_last_epoch=all_kv_last,
        completed_epochs=all_completed_epochs,
        num_evictions=all_num_evictions,
        rollout_env_ids=all_rollout_env_ids,
        problem_ids=all_problem_ids,
    )
    return stats



def prep_wandb_metrics(
        wandb_writer: wandb_run.Run,
        traj_lens: List[List[int]],
        turn_lens: List[List[int]],
        rewards: List[List[float]],
        num_turns: List[List[int]],
        advantages: List[float],
        policy_first_epoch: List[List[int]],
        policy_avg_epoch: List[List[float]],
        policy_last_epoch: List[List[int]],
        kv_first_epoch: List[List[int]],
        kv_avg_epoch: List[List[float]],
        kv_last_epoch: List[List[int]],
        completed_epochs: List[List[int]],
        num_evictions: List[List[int]],
        env_ids: List[List[str]],
        problem_ids: List[List[str | None]],
        current_iteration: int,
        example_group: list[TokenRollout | Rollout] | None = None,
        tokenizer: MegatronTokenizer | None = None,
    ):

    """Make a wandb-parseable dictionary of metrics for logging.

    Staleness (a.k.a. lag) is ``current_iteration - epoch`` and is reported in a
    consistent ``staleness/{policy|kv_cache}/{first|avg|last}/...`` scheme. The
    ``first``/``avg``/``last`` epochs are per-rollout summaries (the ``avg`` is
    token-weighted, computed upstream in ``compute_group_stats``).

    Args:
        wandb_writer: Wandb run to log to.
        traj_lens: Grouped list of trajectory lengths.
        turn_lens: Grouped list of turn lengths.
        rewards: Grouped list of rewards.
        num_turns: Grouped list of number of turns in the trajectories.
        advantages: Flattened list of advantages.
        policy_first_epoch: Grouped per-rollout first-token policy epoch.
        policy_avg_epoch: Grouped per-rollout token-weighted average policy epoch.
        policy_last_epoch: Grouped per-rollout last-token policy epoch.
        kv_first_epoch: Grouped per-rollout first-token KV-cache epoch.
        kv_avg_epoch: Grouped per-rollout token-weighted average KV-cache epoch.
        kv_last_epoch: Grouped per-rollout last-token KV-cache epoch.
        completed_epochs: Grouped list of per-turn max policy epoch stamps.
        num_evictions: Grouped list of per-rollout number of evictions.
        env_ids: Grouped per-rollout environment ids (labels the rollout table).
        problem_ids: Grouped per-rollout problem/prompt ids (labels the rollout table).
        current_iteration: Current training iteration.
        example_group: A list of rollouts of one group to log examples of trajectories.
        tokenizer: Tokenizer to untokenize trajectories for logging.
    """

    def _flat(grouped):
        return [x for g in grouped for x in g]

    def _lag(grouped_epochs):
        return [current_iteration - e for g in grouped_epochs for e in g]

    def _dist(prefix, values, title, native_hist=True):
        """Scalars + a Table-backed histogram chart for a 1-D list of values; also a
        native wandb.Histogram (stacks into an over-time heatmap) when native_hist.
        The Table is always kept, so the raw values stay queryable."""
        if not values:
            return {}
        arr = np.asarray(values, dtype=float)
        out = {
            f'{prefix}/mean': float(arr.mean()),
            f'{prefix}/min': float(arr.min()),
            f'{prefix}/max': float(arr.max()),
            f'{prefix}/p50': float(np.percentile(arr, 50)),
            f'{prefix}/p90': float(np.percentile(arr, 90)),
            f'{prefix}/p99': float(np.percentile(arr, 99)),
            f'{prefix}_hist': wandb_writer.plot.histogram(
                wandb_writer.Table(columns=['value'], data=[[v] for v in values]),
                'value', title,
            ),
        }
        if native_hist:
            out[f'{prefix}/histogram'] = wandb_writer.Histogram(values)
        return out

    rewards_flat = _flat(rewards)
    traj_lens_flat = _flat(traj_lens)
    turn_lens_flat = _flat(turn_lens)
    evictions_flat = _flat(num_evictions)
    env_ids_flat = _flat(env_ids)
    problem_ids_flat = _flat(problem_ids)

    # Lag = current_iteration - epoch, per rollout, by category and source.
    policy_first = _lag(policy_first_epoch)
    policy_avg = _lag(policy_avg_epoch)
    policy_last = _lag(policy_last_epoch)
    kv_first = _lag(kv_first_epoch)
    kv_avg = _lag(kv_avg_epoch)
    kv_last = _lag(kv_last_epoch)

    group_table = wandb_writer.Table(
        columns=['group_means', 'group_stds'],
        data=[[np.mean(g), np.std(g)] for g in rewards],
    )

    metrics = {
        'group_means_hist': wandb_writer.plot.histogram(group_table, 'group_means', 'Group Means'),
        'group_stds_hist': wandb_writer.plot.histogram(group_table, 'group_stds', 'Group STDs'),
        'rewards_hist': wandb_writer.plot.histogram(
            wandb_writer.Table(columns=['reward'], data=[[r] for r in rewards_flat]),
            'reward', 'All Rewards',
        ),
        'advantages_hist': wandb_writer.plot.histogram(
            wandb_writer.Table(columns=['advantages'], data=[[x] for x in advantages]),
            'advantages', 'Advantages',
        ),
        # One row per rollout: identity (env/problem), reward, length, evictions,
        # and first/avg/last lag.
        'rollout_table': wandb_writer.Table(
            columns=[
                'env_id', 'problem_id',
                'reward', 'traj_length', 'num_evictions',
                'policy_first', 'policy_avg', 'policy_last',
                'kv_cache_first', 'kv_cache_avg', 'kv_cache_last',
            ],
            data=list(zip(
                env_ids_flat, problem_ids_flat,
                rewards_flat, traj_lens_flat, evictions_flat,
                policy_first, policy_avg, policy_last,
                kv_first, kv_avg, kv_last,
            )),
        ),
        'mean_turn_length': np.mean([np.mean(g) for g in turn_lens]),
        'mean_turn_length_std': np.mean([np.std(g) for g in turn_lens]),
        'max_turn_length': max([max(g) for g in turn_lens]),
        'min_turn_length': min([min(g) for g in turn_lens]),
        'mean_traj_length': np.mean([np.mean(g) for g in traj_lens]),
        'mean_traj_length_std': np.mean([np.std(g) for g in traj_lens]),
        'max_traj_length': max([max(g) for g in traj_lens]),
        'min_traj_length': min([min(g) for g in traj_lens]),
        'mean_num_turns': np.mean([np.mean(g) for g in num_turns]),
        'max_num_turns': max([max(g) for g in num_turns]),
        'min_num_turns': min([min(g) for g in num_turns]),
        'mean_reward': np.mean([np.mean(g) for g in rewards]),
        'mean_advantage': np.mean(advantages),
        'nonzero_groups_ratio': np.count_nonzero(advantages) / len(advantages),
        'total_eviction_count': sum(evictions_flat),
        'max_num_evictions': max(evictions_flat) if evictions_flat else 0,
        'mean_completion_gap': np.mean(
            [current_iteration - s for g in completed_epochs for s in g]
        ),
    }

    # Staleness distributions: staleness/{policy|kv_cache}/{first|avg|last}/...
    metrics.update(_dist('staleness/policy/first', policy_first, 'Policy lag (first token)'))
    metrics.update(_dist('staleness/policy/avg', policy_avg, 'Policy lag (avg token)'))
    metrics.update(_dist('staleness/policy/last', policy_last, 'Policy lag (last token)'))
    metrics.update(_dist('staleness/kv_cache/first', kv_first, 'KV-cache lag (first token)'))
    metrics.update(_dist('staleness/kv_cache/avg', kv_avg, 'KV-cache lag (avg token)'))
    metrics.update(_dist('staleness/kv_cache/last', kv_last, 'KV-cache lag (last token)'))

    # Rollout-length and eviction distributions.
    metrics.update(_dist('length/traj', traj_lens_flat, 'Trajectory lengths'))
    metrics.update(_dist('length/turn', turn_lens_flat, 'Turn lengths'))
    # Evictions: keep the Table-backed chart only (no native histogram, per request).
    metrics.update(
        _dist('evictions/per_rollout', evictions_flat, 'Evictions per rollout', native_hist=False)
    )
    if example_group:
        if tokenizer is None:
            raise ValueError("If you provide an example group to log, you need to provide a tokenizer too.")
        metrics['rollouts'] = wandb_writer.Table(
            columns=['Trajectories', 'Tokens', 'Rewards'],
            rows=[
                [
                    tokenizer.detokenize(turn) if isinstance(r, TokenRollout) else turn,
                    r.trajectory,
                    r.reward,
                ]
                for r in example_group for turn in r.trajectory
            ],
        )
    return metrics


def maybe_log_training_metrics(
    group_stats: RolloutStats,
    current_iteration: int,
    tokenizer: MegatronTokenizer,
    example_groups: dict[str, list[TokenRollout | Rollout]],
):
    """Log training metrics if writers are available.

    Args:
        group_stats: RolloutStats object to pass to writers.
        current_iteration: Current training iteration.
        tokenizer: Tokenizer to untokenize trajectories for logging.
        example_groups: A dict with values as list of rollouts of one group to log examples of trajectories. Keys are env names.
    """

    wandb_writer = get_wandb_writer()
    tb_writer = get_tensorboard_writer()
    if tb_writer:
        tb_writer.add_scalar('mean_reward', np.mean([np.mean(g) for g in group_stats.rewards]), current_iteration)
    if not wandb_writer:
        return

    # We log these metrics for the aggregated data, no split per env.
    metrics = {
        'min_piold_to_inf_prob': group_stats.min_piold_to_inf_prob,
        'max_piold_to_inf_prob': group_stats.max_piold_to_inf_prob,
        'mean_piold_to_inf_prob': group_stats.mean_piold_to_inf_prob,
        'min_inf_train_prob_abs_diff': group_stats.min_inf_train_prob_abs_diff,
        'max_inf_train_prob_abs_diff': group_stats.max_inf_train_prob_abs_diff,
        'mean_inf_train_prob_abs_diff': group_stats.mean_inf_train_prob_abs_diff,
        'min_inf_prob': group_stats.min_inf_prob,
        'max_inf_prob': group_stats.max_inf_prob,
        'mean_inf_prob': group_stats.mean_inf_prob,
    }

    traj_lens = group_stats.traj_lens
    turn_lens = group_stats.turn_lens
    rewards = group_stats.rewards
    num_turns = group_stats.num_turns
    advantages = group_stats.advantages
    policy_first_epoch = group_stats.policy_first_epoch
    policy_avg_epoch = group_stats.policy_avg_epoch
    policy_last_epoch = group_stats.policy_last_epoch
    kv_first_epoch = group_stats.kv_first_epoch
    kv_avg_epoch = group_stats.kv_avg_epoch
    kv_last_epoch = group_stats.kv_last_epoch
    completed_epochs = group_stats.completed_epochs
    num_evictions = group_stats.num_evictions
    rollout_env_ids = group_stats.rollout_env_ids
    problem_ids = group_stats.problem_ids

    metrics = metrics | prep_wandb_metrics(wandb_writer=wandb_writer,
        traj_lens=traj_lens, turn_lens=turn_lens, rewards=rewards, num_turns=num_turns, advantages=advantages,
        policy_first_epoch=policy_first_epoch, policy_avg_epoch=policy_avg_epoch,
        policy_last_epoch=policy_last_epoch, kv_first_epoch=kv_first_epoch,
        kv_avg_epoch=kv_avg_epoch, kv_last_epoch=kv_last_epoch,
        completed_epochs=completed_epochs,
        num_evictions=num_evictions,
        env_ids=rollout_env_ids, problem_ids=problem_ids,
        current_iteration=current_iteration)
    env_stats = lambda cont, idx: [cont[i] for i in idx]
    group_turn_counts = [sum(nt) for nt in num_turns]

    for env_id in set(group_stats.env_ids):
        env_idx = [i for i, eidx in enumerate(group_stats.env_ids) if eidx == env_id]

        # Advantages are flattened, we need to be more careful with those.
        env_advantages = []
        for i in env_idx:
            st = sum(group_turn_counts[:i])
            end = st + group_turn_counts[i]
            env_advantages.extend(advantages[st:end])

        env_metrics = prep_wandb_metrics(wandb_writer=wandb_writer, traj_lens=env_stats(traj_lens, env_idx),
            turn_lens=env_stats(turn_lens, env_idx),
            rewards=env_stats(rewards, env_idx),
            num_turns=env_stats(num_turns, env_idx),
            advantages=env_advantages,
            policy_first_epoch=env_stats(policy_first_epoch, env_idx),
            policy_avg_epoch=env_stats(policy_avg_epoch, env_idx),
            policy_last_epoch=env_stats(policy_last_epoch, env_idx),
            kv_first_epoch=env_stats(kv_first_epoch, env_idx),
            kv_avg_epoch=env_stats(kv_avg_epoch, env_idx),
            kv_last_epoch=env_stats(kv_last_epoch, env_idx),
            completed_epochs=env_stats(completed_epochs, env_idx),
            num_evictions=env_stats(num_evictions, env_idx),
            env_ids=env_stats(rollout_env_ids, env_idx),
            problem_ids=env_stats(problem_ids, env_idx),
            current_iteration=current_iteration,
            example_group=example_groups[env_id],
            tokenizer=tokenizer,
        )
        for k, v in env_metrics.items():
            metrics[f"{env_id}_{k}"] = v

    wandb_writer.log(metrics, step=current_iteration)


def prepare_trajectories(
    rollouts: Rollouts, tokenizer: MegatronTokenizer, seq_length: int, sequence_packing: bool, skip_bos_token: bool
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

    DEFAULT_PAD_TOKENS = ['<|finetune_right_pad_id|>', '<SPECIAL_999>']

    if tokenizer.library == "huggingface":
        tokenizer : HuggingFaceTokenizer
        if not tokenizer.pad:
            for pad_token in DEFAULT_PAD_TOKENS:
                if pad_token in tokenizer._tokenizer.tokenizer.get_vocab():
                    log_single_rank(
                        logger, logging.INFO, f"Updating tokenizer pad token to {pad_token}"
                    )
                    tokenizer._tokenizer.pad_token = pad_token
                    break
            else:
                raise ValueError("No pad token found in tokenizer vocabulary")
    elif tokenizer.library == "tiktoken":
        assert "<SPECIAL_233>" in tokenizer.vocab, "Pad token is NOT in the tokenizer"
        tokenizer._pad_id = tokenizer.vocab["<SPECIAL_233>"]

    log_single_rank(logger, logging.INFO, f"Tokenizer vocab size: {tokenizer.vocab_size}")
    log_single_rank(
        logger,
        logging.INFO,
        f"Tokenizer PAD: '{tokenizer.detokenize([tokenizer.pad])} ({tokenizer.pad})'",
    )
    log_single_rank(
        logger,
        logging.INFO,
        f"Tokenizer EOD: '{tokenizer.detokenize([tokenizer.eod])} ({tokenizer.eod})'",
    )

    trajs = []
    generation_masks = []
    inference_logprobs = []
    # DAPO-style overlong filtering (gated by --rl-overlong-filtering): drop from the
    # loss any turn truncated at the sequence-length boundary (filled the window without
    # an EOS). See the per-turn check below.
    overlong_filtering = getattr(get_args(), "rl_overlong_filtering", False)
    overlong_filtered = 0
    for rollout in rollouts:
        # traj, gen mask and logprobs are lists now.
        # each list entry is a turn, single-turn environments just have a single-element list.
        # We assume that all lengths of the structs above have the same lengths (number of turns).

        all_turns_trajectories = (
            copy.deepcopy(rollout.trajectory)
            if isinstance(rollout, TokenRollout)
            else tokenizer.tokenize(rollout.trajectory)
        )
        for turn_idx, trajectory in enumerate(all_turns_trajectories):
            inf_logprobs = rollout.logprobs[turn_idx]
            generation_mask = rollout.generation_mask[turn_idx] if isinstance(rollout, TokenRollout) else None
            length = len(trajectory)
            assert length <= seq_length, "Rollout too long, how did this happen?"
            # Multi-turn agents can terminate a turn on a tool-call boundary which is
            # neither tokenizer.eod nor seq_length truncation. Padding logic below
            # handles any short trajectory regardless of terminator. See companion
            # relaxation in compute_group_stats (~line 1013).
            # Old strict check:
            #     if len(trajectory) < seq_length:
            #         assert trajectory[-1] == tokenizer.eod, ...

            if length < seq_length:
                trajectory.extend([tokenizer.pad] * (seq_length - length))
                if generation_mask:
                    generation_mask.extend([False] * (seq_length - length))
            # DAPO-style overlong filtering: a turn that filled the entire window
            # (length == seq_length) without ending in EOS was truncated at the
            # sequence-length boundary. Exclude it from the loss by zeroing its
            # generation mask -- the same zero-loss/zero-grad mechanism the DP-split
            # padding below relies on. Advantages were computed upstream (see the
            # data-parallel split) and are unaffected.
            if (
                overlong_filtering
                and generation_mask is not None
                and length == seq_length
                and trajectory[length - 1] != tokenizer.eod
            ):
                generation_mask = [False] * len(generation_mask)
                overlong_filtered += 1
            trajs.append(trajectory)
            generation_masks.append(generation_mask)

            if inf_logprobs is not None:
                inf_logprobs_tensor = torch.Tensor(inf_logprobs)
                # Don't pad individual logprobs here - padding happens later if needed
                inference_logprobs.append(inf_logprobs_tensor)
            else:
                inference_logprobs.append(None)

        env_id_counts[rollout.env_id] += 1

    if torch.distributed.is_initialized():
        logger.info(f"[{dist.get_rank()}] Rollout counts:")
        for env_id, count in env_id_counts.items():
            logger.info(f"[{dist.get_rank()}] \t{env_id}: {count}")

    if overlong_filtering and overlong_filtered:
        rank_str = str(dist.get_rank()) if torch.distributed.is_initialized() else "0"
        logger.info(
            f"[{rank_str}] overlong filtering: zeroed loss for {overlong_filtered} truncated turn(s) "
            f"(hit seq_length={seq_length} without EOS) out of {len(trajs)} total turns"
        )

    generation_masks = torch.tensor(generation_masks, dtype=torch.bool, device='cpu')
    trajs = torch.tensor(trajs, device='cpu')

    if trajs.ndim == 1:
        # trajs is 1D (shape (0,)) when every rollout had trajectory=[] — i.e. all inference
        # requests returned empty-trajectory placeholders (e.g. 500/TokenOverflowError from KV-cache
        # exhaustion or a chunked-prefill wedge).  The downstream trajs[:, 0] assert would give a
        # confusing IndexError; raise an actionable message here instead.
        rank_str = str(dist.get_rank()) if torch.distributed.is_initialized() else "0"
        raise RuntimeError(
            f"[rank {rank_str}] prepare_trajectories: 0 usable trajectories from {len(rollouts)} rollout(s). "
            f"All rollouts have trajectory=[] (empty-trajectory placeholders). "
            f"Likely cause: inference server returned only 500/TokenOverflowError — check for "
            f"KV-cache exhaustion (too many parallel generations at this SL) or --enable-chunked-prefill wedge."
        )

    # Only process if we have inference_logprobs
    if inference_logprobs and any(lp is not None for lp in inference_logprobs):
        # We need to pad all logprobs to the same size for sequence packing.
        # For non-packing mode, keep as list of tensors (unpadded)
        # This preserves the original behavior where each sequence can have different lengths
        if sequence_packing:
            inference_logprobs = _pad_nonnull_with_zeros(inference_logprobs, seq_length)
    else:
        inference_logprobs = None

    # Some sanity checks regarding the tokenization
    if not skip_bos_token:
        assert (
            tokenizer.bos is None or (trajs[:, 0] == tokenizer.bos).all()
        ), "First token should be bos"
    else:
        assert (
            tokenizer.bos is None or (trajs[:, 0] != tokenizer.bos).all()
        ), "First token should not be bos"  
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


def logprobs_forward_step(data_iterator, model, is_correction, packing_context=None):
    # Avoid self.training checks which will trigger cudagraph capture; this path reuses
    # the forward pass from training after it has been captured on the 1st iteration.
    model.eval()

    if packing_context is not None:
        # When using sequence packing, the data iterator returns a tuple with a single element, the bin index.
        bin_tensor = next(data_iterator)[0]
        #TODO(jalbericiola): change for named tuple
        (b_trajs, _, _, _, b_posids, _, _, _, _, _, b_packed_seq_params) = (
            load_packed_data_by_index(bin_tensor.item(), packing_context, is_correction)
        )
    else:
        b_trajs, b_posids = next(data_iterator)
        b_packed_seq_params = None

    logprobs = (
        get_logprobs(
            model,
            b_trajs.cuda(),
            b_posids.cuda(),
            no_grad=True,
            sequence_packing=packing_context is not None,
            packed_seq_params=b_packed_seq_params,
        ),
        None,
    )
    model.train()
    return logprobs


def compute_logprobs_batch(
    model,
    data_loader,
    forward_backward_func,
    packing_context,
    trajs_batch_size, # n_bins for seq packing, and batch_size for non seq packing
    seq_length,
    logprobs_batch_size,
    decoder_seq_length,
    dtype,
    pp_group,
    is_correction,
    collect_non_loss_data=False,
):
    """Compute logprobs for all batches in the data loader."""
    logprobs_list = []
    data_iterator = iter(data_loader)
    for i in range(len(data_loader)):
        output_tensor = forward_backward_func(
            forward_step_func=partial(logprobs_forward_step, is_correction=is_correction, packing_context=packing_context),
            data_iterator=data_iterator,
            model=model,
            num_microbatches=1,
            seq_length=seq_length,
            micro_batch_size=logprobs_batch_size,
            decoder_seq_length=decoder_seq_length,
            forward_only=True,
            adjust_tensor_shapes_fn=None,
            collect_non_loss_data=collect_non_loss_data,
        )
        if is_pp_last_stage(pp_group):
            logprobs_list.append(output_tensor[0].detach())

    if is_pp_last_stage(pp_group):
        logprobs = torch.concat(logprobs_list, dim=0)
        assert logprobs.dtype == dtype
    else:
        logprobs = torch.empty(
            trajs_batch_size,
            seq_length-1,
            dtype=dtype,
            device=torch.cuda.current_device(),
        )

    # Only PP>1 needs a broadcast from the last stage; for PP=1 the output is already local.
    if get_pg_size(pp_group) > 1:
        dist.broadcast(logprobs, src=get_pp_last_rank(pp_group), group=pp_group)
    return logprobs.cpu()


def prepare_data_for_update(
    model: list[LanguageModule],
    ref_state_dict: Dict[str, Any],
    prox_pi_state_dict: Dict[str, Any],
    rollouts: GroupedRollouts,
    tokenizer: MegatronTokenizer,
    sequence_packing: bool,
    is_correction: bool,
) -> tuple[RerunDataIterator, RolloutStats, dict]:
    """Extract data for the update from raw rollouts.

    Args:
        model: Current policy as the zero-eth element.
        ref_state_dict: Reference policy state dict.
        rollouts: Rollouts to extract the data from.
        tokenizer: Tokenizer to pad/tokenize data.
        sequence_packing: Use sequence packing if True.
        is_correction: Prepare data for IS correction if True.

    Returns:
        Tuple of (cycled iterator over dataset batches, group stats, example groups per env).
    """
    args = get_args()
    nvtx_range = get_nvtx_range()
    runtime_state = get_rl_runtime_state()

    if args.cuda_graph_impl != "none" and not args.rl_training_cuda_graphs:
        lang_module = (
            model[0].module.module if hasattr(model[0].module, "module") else model[0].module
        )
        toggle_cuda_graphs(lang_module, "none")

    model = model[0]
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    with nvtx_range("rl/prepare-data-for-update", time=True):
        with nvtx_range("rl/compute-group-stats", time=True):
            group_stats = compute_group_stats(rollouts, tokenizer, args.seq_length)
            # TODO(vitalyk): why do we need global_advantages here? go inside packing
            advantages = global_advantages = torch.tensor(group_stats.advantages, dtype=dtype).cuda()

        # Now split the rollouts across the data parallel ranks for training
        # This needs to be done at this point because we are about to calculate logprobs
        # Note :- For EP, do not use the expert data parallel group here. Always 
        # use the regular data parallel group. 

        # Get example group per environment to log their rollouts.
        example_groups = {}
        for g in rollouts:
            if g[0].env_id not in example_groups:
                example_groups[g[0].env_id] = g

        # Dump labeled per-rollout staleness/length data (rank 0, opt-in) while the
        # rollouts are still grouped (so batch_id / group index are available).
        dump_staleness_data(rollouts, group_stats, args.curr_iteration)

        # Let's expand rollouts getting rid of the groups.
        # We need this to correctly split the rollouts across dp groups.
        # And we do not actually need them grouped in anything below anyways.
        rollouts = [r for g in rollouts for r in g]
        num_turns = [nt for g in group_stats.num_turns for nt in g]
        total_turns_sampled = len(rollouts)

        # We might sample more than we consume in one step.
        samples_ratio_per_step = args.global_batch_size / (args.grpo_prompts_per_step * args.grpo_group_size)
        assert samples_ratio_per_step <= 1, "You cannot use more data than you sampled."

        if (data_parallel_world_size := mpu.get_data_parallel_world_size()) > 0:
            data_split_size = len(rollouts) // data_parallel_world_size
            data_split_range = (
                mpu.get_data_parallel_rank() * data_split_size,
                (mpu.get_data_parallel_rank() + 1) * data_split_size,
            )
            rollouts = rollouts[data_split_range[0] : data_split_range[1]]
            local_num_turns = sum(num_turns[data_split_range[0] : data_split_range[1]])
            steps_before = sum(num_turns[:data_split_range[0]])
            advantages = advantages[steps_before:steps_before+local_num_turns]
            # First we calculate them on a global level and then we split and recalculate on a local level.
            # Sequence packing and reporting needs it global but non-packing wants it local.

        with nvtx_range("rl/prepare-trajectories", time=True):
            trajs, generation_masks, inference_logprobs = prepare_trajectories(
                rollouts, tokenizer, args.seq_length, sequence_packing, args.rl_skip_bos_token
            )

        # DP-split divergence fix for multi-turn rollouts.
        #
        # The DP split above slices `rollouts` evenly across DP ranks, but multi-turn
        # rollouts produce 1..max_steps trajectories EACH (one per turn). Per-rank
        # trajectory counts therefore vary: one rank may get all 1-turn rollouts
        # (32 trajs), another may get all 6-turn (192 trajs).
        #
        # Downstream `compute_logprobs_batch` iterates `len(data_loader)` times —
        # variable per rank — and each iter calls `forward_backward_no_pipelining`
        # which fires `Timers.start(barrier=True)` on default_pg. Ranks with fewer
        # trajectories race ahead to the next barrier while slower ranks are still
        # in their loop → deadlock at the world-level barrier.
        #
        # Pad each rank's trajectories up to the global max across DP ranks so
        # all ranks loop the same number of times. Padded entries have
        # generation_mask=False everywhere → zero loss / zero grad contribution.
        if (data_parallel_world_size := mpu.get_data_parallel_world_size()) > 1:
            local_count = trajs.shape[0]
            max_count_t = torch.tensor([local_count], device='cuda', dtype=torch.long)
            torch.distributed.all_reduce(
                max_count_t,
                op=torch.distributed.ReduceOp.MAX,
                group=mpu.get_data_parallel_group(),
            )
            target_count = int(max_count_t.item())

            if target_count > local_count:
                pad_n = target_count - local_count
                seq_len = trajs.shape[1]
                pad_trajs = torch.full(
                    (pad_n, seq_len), tokenizer.pad,
                    dtype=trajs.dtype, device=trajs.device,
                )
                pad_gen_masks = torch.zeros(
                    (pad_n, seq_len),
                    dtype=generation_masks.dtype, device=generation_masks.device,
                )
                trajs = torch.cat([trajs, pad_trajs], dim=0)
                generation_masks = torch.cat([generation_masks, pad_gen_masks], dim=0)
                # advantages was sliced to local_num_turns above; pad with zeros.
                pad_adv = torch.zeros(
                    pad_n, dtype=advantages.dtype, device=advantages.device,
                )
                advantages = torch.cat([advantages, pad_adv])
                # inference_logprobs is None, OR a list of per-traj tensors (no-packing path),
                # OR a [N, S] padded tensor (sequence-packing path — `_pad_nonnull_with_zeros`
                # at ~prepare_trajectories line 1376 returns a 2D tensor).
                if inference_logprobs is not None:
                    if isinstance(inference_logprobs, torch.Tensor):
                        # Tensor path: concatenate zero rows of matching shape/dtype/device.
                        pad_lp = torch.zeros(
                            (pad_n, *inference_logprobs.shape[1:]),
                            dtype=inference_logprobs.dtype,
                            device=inference_logprobs.device,
                        )
                        inference_logprobs = torch.cat([inference_logprobs, pad_lp], dim=0)
                    else:
                        # List path: append zero tensors matching existing non-None entry shape.
                        non_none = next(
                            (lp for lp in inference_logprobs if lp is not None), None
                        )
                        if non_none is not None:
                            dummy_lp = torch.zeros_like(non_none)
                        else:
                            dummy_lp = torch.zeros(seq_len - 1, dtype=torch.float)
                        inference_logprobs.extend(
                            [dummy_lp.clone() for _ in range(pad_n)]
                        )

            # Rebuild global_advantages from padded local advantages so that
            # `pack_all_trajectories` (sequence-packing path) — which all-gathers
            # local trajs into a global concat — has a global_advantages tensor
            # that matches the post-gather length. Without this, the dummy
            # trajectories shift indices and `global_advantages[seq_indices]`
            # would go out of range.
            gathered_adv = [
                torch.empty_like(advantages)
                for _ in range(data_parallel_world_size)
            ]
            torch.distributed.all_gather(
                gathered_adv, advantages,
                group=mpu.get_data_parallel_group(),
            )
            global_advantages = torch.cat(gathered_adv, dim=0)

        packing_context = None
        # Build trajectories based on sequence packing or standard processing
        if sequence_packing:
            with nvtx_range("rl/sequence-packing", time=True):
                runtime_state.packing_context = packing_context = pack_all_trajectories(
                    trajs, 
                    generation_masks, 
                    inference_logprobs, 
                    global_advantages, 
                    args.seq_length, 
                    args.rl_sequence_packing_max_sequences_per_bin,
                    args.rl_sequence_packing_algo
                    )
    
                compute_trajs = packing_context.packed_trajs
                compute_position_ids = packing_context.packed_position_ids
                # Use batch_size=1 for packed computation to enable proper attention masking
                # via PackedSeqParams (TE needs cu_seqlens per bin)
                dataset = TensorDataset(torch.arange(len(compute_trajs)))
                data_loader = DataLoader(dataset, batch_size=1)
                logprobs_batch_size = 1

            my_real_tokens = sum(
                packing_context.packing_info.seq_lengths[idx]
                for indices in packing_context.packing_info.bin_seq_indices
                for idx in indices
            )
            real_tokens_tensor = torch.tensor([my_real_tokens], dtype=torch.long, device='cuda')
            torch.distributed.all_reduce(real_tokens_tensor, group=mpu.get_data_parallel_group())
            global_real_tokens = real_tokens_tensor.item()
            try:
                from megatron.training.mfu_tracker import get_mfu_tracker
                get_mfu_tracker().set_iter_real_training_tokens(global_real_tokens)
            except Exception:
                pass
        else:
            # Always compute standard masks for the original data (we'll need them later).
            # We discard the attention_mask, but get_ltor_masks_and_position_ids allocates a
            # [1, seq_length, seq_length] tril tensor unconditionally — that's 68 GB at
            # seq_length=131072 in fp32 and OOMs on iter 2's data prep. Skip the helper in
            # the common case (no EOD-based reset) and compute loss_mask + position_ids
            # inline without the seq^2 allocation. Fall through to the helper if either
            # reset flag is set, since that path needs the attention_mask buffer to mark
            # the inter-document boundaries.
            with nvtx_range("get_ltor_masks_and_position_ids", time=True):
                if args.reset_position_ids or args.reset_attention_mask:
                    _, original_loss_mask, original_position_ids = get_ltor_masks_and_position_ids(
                        trajs,
                        tokenizer.eod,
                        tokenizer.pad,
                        args.reset_position_ids,
                        args.reset_attention_mask,
                        eod_mask_loss=False,
                        pad_mask_loss=True,
                    )
                else:
                    # Mirror the helper's behaviour for eod_mask_loss=False, pad_mask_loss=True.
                    original_loss_mask = torch.ones_like(trajs, dtype=torch.float)
                    original_loss_mask[trajs == tokenizer.pad] = 0.0
                    _, seq_len = trajs.size()
                    original_position_ids = (
                        torch.arange(seq_len, dtype=torch.long, device=trajs.device)
                        .unsqueeze(0)
                        .expand_as(trajs)
                    )
                original_loss_mask[~generation_masks] = 0.0
                compute_trajs = trajs
                compute_position_ids = original_position_ids
                data_loader = DataLoader(
                    TensorDataset(compute_trajs, compute_position_ids),
                    batch_size=args.micro_batch_size,
                )
                logprobs_batch_size = args.micro_batch_size

            # Without sequence packing, training.py defaults to GBS*seq_length which
            # counts padding tokens and inflates TPS metrics.  Report only the real
            # (non-padding) tokens so the metric is comparable to the SP path.
            my_real_tokens = int((trajs != tokenizer.pad).sum().item())
            real_tokens_tensor = torch.tensor([my_real_tokens], dtype=torch.long, device='cuda')
            torch.distributed.all_reduce(real_tokens_tensor, group=mpu.get_data_parallel_group())
            global_real_tokens = real_tokens_tensor.item()
            try:
                from megatron.training.mfu_tracker import get_mfu_tracker
                get_mfu_tracker().set_iter_real_training_tokens(global_real_tokens)
            except Exception:
                pass

        with torch.no_grad(), nvtx_range("rl/compute-logprobs", time=True):
            # Before we can update the model, we need to get the logprobs for the \pi_{old} model.

            forward_backward_func = get_forward_backward_func()
            if args.cuda_graph_impl == "full_iteration":
                forward_backward_func = FullCudaGraphWrapper(
                    forward_backward_func,
                    cuda_graph_warmup_steps=args.cuda_graph_warmup_steps,
                    use_single_mempool=args.cuda_graph_use_single_mempool,
                )

            dtype = (
                torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
            )

            pg_collection = get_attr_wrapped_model(model, "pg_collection")
            pp_group = pg_collection.pp

            cur_st_dict = {
                k: (v.cpu() if v is not None else v) for k, v in model.state_dict().items()
            }
            with torch.no_grad(), nvtx_range("rl/compute-old-logprobs", time=True):
                model.load_state_dict(prox_pi_state_dict)
                old_logprobs = compute_logprobs_batch(
                    model=model,
                    data_loader=data_loader,
                    forward_backward_func=forward_backward_func,
                    packing_context=packing_context,
                    trajs_batch_size=len(compute_trajs),
                    seq_length=args.seq_length,
                    logprobs_batch_size=logprobs_batch_size,
                    decoder_seq_length=args.decoder_seq_length,
                    dtype=dtype,
                    pp_group=pp_group,
                    is_correction=args.rl_inference_logprobs_is_correction,
                )

            with torch.no_grad(), nvtx_range("rl/compute-ref-logprobs", time=True):
                model.load_state_dict(ref_state_dict)
                ref_logprobs = compute_logprobs_batch(
                    model=model,
                    data_loader=data_loader,
                    forward_backward_func=forward_backward_func,
                    packing_context=packing_context,
                    trajs_batch_size=len(compute_trajs),
                    seq_length=args.seq_length,
                    logprobs_batch_size=logprobs_batch_size,
                    decoder_seq_length=args.decoder_seq_length,
                    dtype=dtype,
                    pp_group=pp_group,
                    is_correction=args.rl_inference_logprobs_is_correction,
                )

                # logprobs are [b, seq, h] now.
                model.load_state_dict(cur_st_dict)

                with nvtx_range("rl/synchronize-cuda-and-collect-garbage", time=True):
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()


        if sequence_packing:
            with nvtx_range("rl/pack-logprobs", time=True):
                # Store logprobs on gpu in packing context
                # Since PackingContext is a dataclass, we add these as new attributes
                packing_context.old_logprobs = old_logprobs.cuda()
                packing_context.ref_logprobs = ref_logprobs.cuda()

                if inference_logprobs is not None:
                    # Pack the inference logprobs using the helper function
                    # We do this for logging purposes even if is_correction is disabled
                    packed_inference_logprobs = pack_inference_logprobs(
                        inference_logprobs=packing_context.original_inference_logprobs,
                        packing_info=packing_context.packing_info,
                        generation_masks=packing_context.original_generation_masks,
                        bin_size=args.seq_length,
                    )

                    # Compute statistics for logging using packed data
                    compute_packed_inference_logprobs_stats(
                        old_logprobs=old_logprobs,
                        packed_inference_logprobs=packed_inference_logprobs,
                        packed_loss_mask=packing_context.packed_loss_mask,
                        group_stats=group_stats,
                    )

                    # Store packed inference logprobs in packing context
                    packing_context.packed_inference_logprobs = packed_inference_logprobs.cuda()
                    # Only mark as having inference logprobs for IS correction if enabled
                    packing_context.has_inference_logprobs = args.rl_inference_logprobs_is_correction
            with nvtx_range("rl/create-dataloader", time=True):
                # @vitalyk: This function also reconfigures the data loader to count the
                # global_batch_size in the bins frame of reference.
                # I think it will be a better design if we split the data loader creating and logic
                # that reconfigures the microbatch calculator.

                update_microbatch_calculator(
                    samples_ratio_per_step=samples_ratio_per_step,
                    num_bins_this_rank = len(packing_context.packed_trajs),
                    bin_seq_indices = packing_context.packing_info.bin_seq_indices,
                    global_batch_size=args.global_batch_size,
                    micro_batch_size=args.micro_batch_size,
                    decrease_batch_size_if_needed=args.decrease_batch_size_if_needed,
               )
                loader = get_microbatch_dataloader(len(packing_context.packed_trajs), args.micro_batch_size)
        else:
            with nvtx_range("rl/align-inference-logprobs", time=True):
                if inference_logprobs is not None:
                    inference_logprobs = align_unpacked_inference_logprobs(
                        inference_logprobs=inference_logprobs,
                        old_logprobs_for_data=old_logprobs,
                        generation_masks=generation_masks,
                        group_stats=group_stats,
                    )
                    # We run the above to fill in the inference/train side mismatch stats.
                    # We do the above for logging purposes.
                    # Nullify logprobs if not used in IS correction,
                    if not args.rl_inference_logprobs_is_correction:
                        inference_logprobs = None
            with nvtx_range("rl/create-dataloader", time=True):
                # Because of multiturn, our batch sizes for non-sequence packed trajectories are not fixed anymore.
                # As in sequence packing above, we need to reconfigure it too.
                runtime_state.packing_context = None

                reconfigure_num_microbatches_calculator(
                    rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                    global_batch_size=math.ceil(samples_ratio_per_step*total_turns_sampled),
                    micro_batch_size=args.micro_batch_size,
                    decrease_batch_size_if_needed=args.decrease_batch_size_if_needed,
                    data_parallel_size=mpu.get_data_parallel_world_size(),
                )

                dataset_tensors = [
                    compute_trajs,
                    advantages,
                    old_logprobs,
                    original_loss_mask,
                    original_position_ids,
                    ref_logprobs,
                ]
                if is_correction and inference_logprobs is not None:
                    dataset_tensors.append(inference_logprobs)
                else:
                    dataset_tensors.append(torch.zeros_like(old_logprobs))
                data = TensorDataset(*dataset_tensors)
                loader = DataLoader(data, batch_size=args.micro_batch_size)

    return RerunDataIterator(itertools.cycle(loader)), group_stats, example_groups


def get_grpo_data_iterator(
    model: LanguageModule,
    inference_model: LanguageModule | None,
    optimizer: MegatronOptimizer,
    iteration: int,
    ref_state_dict: Dict[str, torch.Tensor],
    prox_pi_state_dict: Dict[str, torch.Tensor],
    grpo_iterations: int,
    grpo_prompts_per_step: int,
    grpo_group_size: int,
    global_batch_size: int,
    sequence_packing: bool,
    is_correction: bool,
    buffered_rollouts: RerunDataIterator | None = None,
    optimizer_is_on_cpu: bool = False,
) -> RerunDataIterator:
    """
    Get the data iterator for GRPO training.

    Depending on the sampling parameters either performs data collections or returns
    the buffered_rollouts as is.

    Args:
        model: The language model
        optimizer: The Megatron optimizer
        iteration: Current training iteration
        ref_state_dict: Reference model state dict for GRPO
        grpo_iterations: How many steps we reuse the sampled data for.
        grpo_prompts_per_step: How many prompts we sample per data collection.
        grpo_group_size: How many samples we do per prompt.
        global_batch_size: Global batch size.
        sequence_packing: Use sequence packing if True.
        is_correction: Use IS correction if True.
        buffered_rollouts: Previously collected rollouts (if any)
        optimizer_is_on_cpu: If True, the optimizer was offloaded to CPU and must be restored.

    Returns:
        RerunDataIterator for the current training step
    """
    runtime_state = get_rl_runtime_state()
    tokenizer = get_tokenizer()

    # We collect new rollouts when we've gone over the collected data 'grpo_iterations' times.
    global_batches_per_collection = (grpo_prompts_per_step * grpo_group_size) // global_batch_size
    if (
        buffered_rollouts is None or
        iteration == runtime_state.last_collection_iteration +
        (grpo_iterations * global_batches_per_collection)
    ):

        rollouts = get_environment_rollouts(
            model, inference_model, optimizer, grpo_prompts_per_step, grpo_group_size
        )
        buffered_rollouts, group_stats, example_groups = prepare_data_for_update(
            model=model,
            ref_state_dict=ref_state_dict,
            prox_pi_state_dict=prox_pi_state_dict,
            rollouts=rollouts,
            tokenizer=tokenizer,
            sequence_packing=sequence_packing,
            is_correction=is_correction,
        )
        if optimizer_is_on_cpu:
            nvtx_range = get_nvtx_range()
            with nvtx_range("rl/restore-optimizer-after-inference", time=True):
                with nvtx_range("rl/restore/grad-buffers", time=True):
                    model[0].restore_grad_buffers()
                with nvtx_range("rl/restore/optimizer-state", time=True):
                    optimizer.restore_from_cpu()
        runtime_state.group_stats = group_stats
        runtime_state.example_groups = example_groups
        runtime_state.reset_iteration_counters(iteration)

    maybe_log_training_metrics(
        group_stats=runtime_state.group_stats,
        current_iteration=iteration,
        tokenizer=tokenizer,
        example_groups=runtime_state.example_groups,
    )

    return buffered_rollouts


def evaluate_and_print_results_rl(
    data_iterator: Iterator[TensorDataset],
    model: list[LanguageModule],
    optimizer: MegatronOptimizer,
    iteration: int,
    write_to_tensorboard: bool = True,
    training_model: Optional[list[LanguageModule]] = None,
):
    """Helper function to evaluate and dump results on screen.

    Args:
        data_iterator: Iterator over batches of evaluation dataset.
        model: Model to evaluate with (may be separate inference model).
        iteration: Current training iteration.
        write_to_tensorboard: Dumpt stuff to tensorboard or not.
        training_model: Training model (if separate from inference model). Used to offload
            grad buffers and restore to train mode. If None, uses model parameter.
    """
    args = get_args()

    # TODO(vitalyk): I do not track eval loss as in training. We probably should.
    # megatron-lm uses forward_step_func to do the above.

    # Use context manager to temporarily disable sequence parallelism for evaluation

    with torch.no_grad():
        with megatron_rl_inference_mode(
            model,
            optimizer,
            args.cuda_graph_impl,
            args.rl_offload_optimizer_during_inference,
            training_model,
        ) as inference_interface:

            loop = get_asyncio_loop()

            rank = torch.distributed.get_rank()
            if rank == 0:
                logger.info("Collecting evaluation results...")
                agent = get_agent(args)
                request = EvaluationRequest(
                    inference_interface=inference_interface,
                    num_prompts=args.rl_prompts_per_eval,
                    validation=True,
                    rank_info=None,
                    generation_args={
                        'temperature': args.rl_default_temperature,
                        'max_tokens': args.seq_length,
                        'top_p': args.rl_default_top_p,
                        'top_k': args.rl_default_top_k,
                    },
                )
                with get_nvtx_range()("rl/run-evaluation", time=True):
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
            logger.info(
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
            logger.info(
                "Evaluation results:"
                + "".join([f"\n\t{k}: {v:0.4f}" for k, v in eval_metrics.items()])
            )
            if lang_rl_log_dir:
                with open(
                    lang_rl_log_dir
                    + f'/eval_rank{rank}_iteration{args.curr_iteration}_'
                    + f'{Path(args.langrl_env_config).stem}.json',
                    'w',
                ) as f:
                    json.dump([[r.model_dump() for r in group] for group in dp_eval_results], f)


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
        log_single_rank(
            logger,
            logging.WARNING,
            f"WARNING: Shape mismatch - current_logprobs: {current_logprobs.shape}, old_logprobs: {old_logprobs.shape}",
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
    entropy_term = -current_logprobs.exp() * current_logprobs

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
        - entropy_weight * entropy_term
    )

    return loss, kl_term, ratios, entropy_term, truncated_from_above, truncated_from_below


async def _rollout_keepalive(inference_interface, interval_s, requests_per_tick, stop_event):
    """Submit tiny dummy generations while the engine sits in inference mode.

    Agentic environments (SWE containers building repos and running test suites)
    leave the engine with no decode work for tens of minutes, which cluster
    idle-GPU watchdogs read as an abandoned job. A short burst of generation per
    interval keeps mean SM utilization above the kill threshold. Must never
    raise: keepalive is cosmetic and losing it must not affect training.
    """
    import asyncio

    from megatron.rl import GenericGenerationArgs
    from megatron.rl.inference.api import InferenceRequest, LLMChatMessage

    request = InferenceRequest(
        prompt=[LLMChatMessage(role='user', content='keepalive ping')],
        generation_args=GenericGenerationArgs(max_tokens=8, temperature=1.0),
    )
    consecutive_failures = 0
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass
        if stop_event.is_set():
            return
        try:
            # requests_per_tick requests so the coordinator's round-robin
            # reaches every inference DP group, not just one engine.
            await asyncio.gather(
                *[inference_interface.agenerate(request) for _ in range(requests_per_tick)]
            )
            consecutive_failures = 0
        except asyncio.CancelledError:
            raise
        except Exception:
            consecutive_failures += 1
            if consecutive_failures == 1:
                logger.warning("Rollout keepalive generation failed; retrying quietly.", exc_info=True)
            if consecutive_failures >= 10:
                logger.warning("Rollout keepalive disabled after 10 consecutive failures.")
                return


def _maybe_start_rollout_keepalive(args, loop, inference_interface):
    """Start the keepalive task on rank 0 if --rl-rollout-keepalive-interval > 0."""
    import asyncio

    interval_s = getattr(args, 'rl_rollout_keepalive_interval', 0.0) or 0.0
    if interval_s <= 0 or torch.distributed.get_rank() != 0:
        return None, None
    requests_per_tick = max(1, getattr(args, 'data_parallel_size', 1) or 1)
    stop_event = asyncio.Event()
    task = loop.create_task(
        _rollout_keepalive(inference_interface, interval_s, requests_per_tick, stop_event)
    )
    logger.info(
        f"Rollout keepalive enabled: {requests_per_tick} dummy generation(s) "
        f"every {interval_s:.0f}s while in inference mode."
    )
    return stop_event, task


def _stop_rollout_keepalive(loop, stop_event, task):
    """Stop the keepalive task, letting an in-flight tiny generation drain first."""
    import asyncio

    if task is None:
        return
    stop_event.set()
    try:
        loop.run_until_complete(asyncio.wait_for(task, timeout=60))
    except Exception:
        task.cancel()
        try:
            loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
        except Exception:
            pass


@contextmanager
def megatron_rl_inference_mode(
    model: list[LanguageModule],
    optimizer: MegatronOptimizer,
    cuda_graph_impl: str,
    offload_optimizer_during_inference: bool,
    training_model: Optional[list[LanguageModule]] = None,
):
    """Manage the model inference context when collecting rollouts.

    Args:
        model: model to prepare for inference (may be separate inference model).
        optimizer: optimizer used to train the model.
        cuda_graph_impl: which cuda graph implementation to use.
        offload_optimizer_during_inference: move optimizer to cpu during inference or not.
        training_model: training model (if separate from inference model). Used to offload
            grad buffers and restore to train mode. If None, uses model parameter.

    Yields:
        None: this context manager does not return a value.

    """
    args = get_args()
    loop = get_asyncio_loop()
    nvtx_range = get_nvtx_range()

    logger.debug(f"[{dist.get_rank()}] Entering inference mode")

    # Use local CUDA graphs during rollout inference. An empty module list preserves
    # full-layer capture when the configured inference scope is layer.
    model[0].config.cuda_graph_modules = []
    model[0].config.cuda_graph_impl = "local"
    model[0].config.inference_cuda_graph_scope = args.inference_cuda_graph_scope

    # If we get a lower precision wrapper, we go one object deeper.
    lang_module = model[0].module.module if hasattr(model[0].module, "module") else model[0].module

    # Switch MoE layers to full CUDA graph capture for inference
    if args.rl_training_cuda_graphs and args.num_experts is not None:
        transition_moe_cudagraphs(lang_module, 'full')

    lang_module.eval()
    # If this is a separate RL inference model with offloading enabled, ensure weights are on GPU
    # before any CUDA-graph capture/replay or inference. This is a no-op if already on GPU.
    model_core = unwrap_model(model[0])
    with nvtx_range("rl/prefetch-weights-to-gpu", time=True):
        _maybe_prefetch_separate_inference_model_weights(model_core, to_cpu=False)

    rotary_module = getattr(lang_module, "rotary_pos_emb", None)
    # Vanilla RotaryEmbedding module has lru_cache decorator which breaks RL training
    # as it tries to reuse frequences tensors cached in inference mode.
    has_lru_cache = rotary_module is not None and hasattr(rotary_module.forward, "cache_parameters")
    if has_lru_cache:
        rotary_module.forward.cache_clear()

    with torch.no_grad():

        if offload_optimizer_during_inference:
            with nvtx_range("rl/offload-optimizer-before-inference", time=True):
                if not args.rl_training_cuda_graphs:
                    with nvtx_range("rl/offload/grad-buffers", time=True):
                        model_for_grad_offload = training_model if training_model is not None else model
                        model_for_grad_offload[0].offload_grad_buffers()
                else:
                    logger.warning(
                        "Gradient buffers will not be offloaded when training cudagraphs are enabled!")
                with nvtx_range("rl/offload/optimizer-state", time=True):
                    optimizer.offload_to_cpu()

        if cuda_graph_impl != "none" and not args.rl_training_cuda_graphs:
            toggle_cuda_graphs(lang_module, cuda_graph_impl)

        inference_interface = get_inference_interface(args, loop, model)
        inference_interface.set_generation_epoch(get_args().curr_iteration)
        loop.run_until_complete(inference_interface.resume())

        keepalive_stop, keepalive_task = _maybe_start_rollout_keepalive(
            args, loop, inference_interface
        )

        logger.debug(f"[{dist.get_rank()}] Entered inference mode")
        try:
            yield inference_interface
        finally:
            _stop_rollout_keepalive(loop, keepalive_stop, keepalive_task)

        with nvtx_range("rl/suspend-engine", time=True):
            loop.run_until_complete(inference_interface.suspend())

        if cuda_graph_impl != "none" and not args.rl_training_cuda_graphs:
            toggle_cuda_graphs(lang_module, 'none')

        # Reset drop_and_pad leaked from inference decode
        set_decode_expert_padding(unwrap_model(model[0]), set_to=False)

        # Restore cudagraph scope for training.
        # MoE partial capture requires specific scopes that aren't user-facing.
        model[0].config.cuda_graph_impl = args.cuda_graph_impl
        model[0].config.inference_cuda_graph_scope = args.inference_cuda_graph_scope
        if args.num_experts is not None:
            model[0].config.cuda_graph_modules = [
                CudaGraphModule.mamba,
                CudaGraphModule.attn,
                CudaGraphModule.moe_router,
                CudaGraphModule.moe_preprocess,
            ]
        else:
            model[0].config.cuda_graph_modules = copy.copy(args.cuda_graph_modules)

        # Switch MoE layers to partial CUDA graph capture for training
        if args.rl_training_cuda_graphs and args.num_experts is not None:
            transition_moe_cudagraphs(lang_module, 'partial')

        # If this is a separate RL inference model, prefetch weights back to CPU so they
        # don't consume GPU memory during training.
        with nvtx_range("prefetch-inference-model-weights-to-cpu", time=True):
            _maybe_prefetch_separate_inference_model_weights(model_core, to_cpu=True)

        if offload_optimizer_during_inference:
            with nvtx_range("rl/restore-optimizer-after-inference", time=True):
                with nvtx_range("rl/restore/grad-buffers", time=True):
                    model_for_grad_offload = training_model if training_model is not None else model
                    model_for_grad_offload[0].restore_grad_buffers()
                with nvtx_range("rl/restore/optimizer-state", time=True):
                    optimizer.restore_from_cpu()

        # Set training model back to train mode (not inference model if they're separate)
        training_lang_module = unwrap_model(training_model[0]) if training_model is not None else lang_module
        training_lang_module.train()

        if has_lru_cache:
            rotary_module.forward.cache_clear()

        logger.debug(f"[{dist.get_rank()}] Exiting inference mode")


def rl_inference_interface_shutdown():
    global _INFERENCE_INTERFACE
    global _ROLLOUT_GENERATOR

    if _ROLLOUT_GENERATOR is not None:
        loop = get_asyncio_loop()
        loop.run_until_complete(_ROLLOUT_GENERATOR.aclose())
        _ROLLOUT_GENERATOR = None

    if _INFERENCE_INTERFACE is not None:
        loop = get_asyncio_loop()
        loop.run_until_complete(_INFERENCE_INTERFACE.kill())
        _INFERENCE_INTERFACE = None
    else:
        logger.warning("No inference interface to shutdown. This should not happen.")

    # TODO(rkirby): This is a hack to hard exit. There is a bug that is preventing us from using sys.exit(0).
    # It seem the Flask server has non-daemon threads that are preventing the program from exiting.
    # We need to find a way to gracefully complete all in progress requests and shutdown the Flask server.
    import os
    os._exit(0)


def get_iteration_sequence_count(args):
    """Get the total number of sequences processed in this iteration across all ranks."""
    runtime_state = get_rl_runtime_state()
    sequences_tensor = torch.tensor(
        runtime_state.sequences_this_iteration_on_rank, device='cuda', dtype=torch.long
    )
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(sequences_tensor, group=mpu.get_data_parallel_group())
    return int(sequences_tensor.item())
    
def _pad_nonnull_with_zeros(data: list[Optional[torch.Tensor]], max_len: int) -> torch.Tensor:
    """Pad each element of a list of tensors to the length required.
    Args:
        data: List of tensors to pad.
        max_len: Maximum length to pad to. Must be higher or equal than the max len of the data tensors.
    Returns:
        A padded tensor which is a stacked list of padded input tensors.

    """
    if all([el is None for el in data]):
        raise ValueError("At least one element of the data list should be not None.")
    padded_data = []
    for chunk in data:
        if chunk is not None:
            padding_size = max_len - len(chunk)
            if padding_size > 0:
                # Pad with zeros (these positions will be masked anyway)
                padded = torch.nn.functional.pad(chunk, (0, padding_size), value=0.0)
                padded_data.append(padded)
            elif padding_size == 0:
                padded_data.append(chunk)
            else:
                raise ValueError("One of the input tensors has larger length than padding max len.")
        else:
            # Create zero tensor for None logprobs
            padded_data.append(torch.zeros(max_len))
    return torch.stack(padded_data)
