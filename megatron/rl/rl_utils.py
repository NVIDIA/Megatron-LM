# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc

# Keep this to make the env registered.
import itertools
import json
import logging
import math
import pickle
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from megatron.core import mpu
from megatron.core.datasets.megatron_tokenizer import MegatronLegacyTokenizer
from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.optimizer import MegatronOptimizer
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.pipeline_parallel.utils import is_pp_last_stage, get_pp_last_rank
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.core.transformer.cuda_graphs import _CudagraphGlobalRecord
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.utils import toggle_cuda_graphs
from megatron.core.resharding.refit import swap_model_weights
from megatron.core.inference.unified_memory import (
    advise_managed_module_parameters_preferred_location,
    prefetch_managed_module_parameters,
)
from megatron.core.utils import get_asyncio_loop, log_single_rank
from megatron.rl.sequence_packing_utils import (
    get_microbatch_dataloader,
    pack_inference_logprobs,
    compute_packed_inference_logprobs_stats,
    pack_all_trajectories,
    load_packed_data_by_index,
    update_sequence_packing_metrics,
    get_sequence_packing_tensorboard_metrics,
    get_sequence_packing_log_info,
    get_default_packed_seq_params,
)
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
from megatron.training.utils import (
    get_ltor_masks_and_position_ids,
    get_nvtx_range,
    print_rank_0,
    unwrap_model,
)
from megatron.core.utils import get_pg_rank, get_pg_size, get_attr_wrapped_model
from megatron.core.process_groups_config import ProcessGroupCollection
from wandb import wandb_run
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    is_batch_invariant_mode_enabled,
)


logger = logging.getLogger(__name__)

# Global variable to store packing context for forward_step
_GLOBAL_PACKING_CONTEXT = None


def _maybe_prefetch_separate_inference_model_weights(model_core, *, to_cpu: bool) -> None:
    """Prefetch RL *separate inference model* weights to CPU/GPU (UVM-only path).

    Gated only by user args; this assumes the separate inference model was allocated with UVM when enabled.
    """
    args = get_args()
    if not args.rl_offload_inference_model_weights_when_idle:
        return
    if args.rl_inference_model_unified_memory_level != 1:
        return

    device = -1 if to_cpu else int(torch.cuda.current_device())
    advise_managed_module_parameters_preferred_location(model_core, device=device, include_buffers=True)
    nbytes = prefetch_managed_module_parameters(model_core, device=device, include_buffers=True)
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
    """Container for runtime state that is not checkpointed, tracking state between rollout collections"""

    def __init__(self):
        self.packing_context = None
        self.last_collection_iteration = 0
        self.global_batches_per_collection = 0
        self.sequences_this_iteration_on_rank = 0
        self.latest_batch_num_sequences = 0

    def reset_iteration_counters(self, iteration):
        """Reset per-iteration counters."""
        self.sequences_this_iteration_on_rank = 0
        self.last_collection_iteration = iteration

    def increment_sequences(self, count):
        """Increment the sequence counter."""
        self.sequences_this_iteration_on_rank += count
        self.latest_batch_num_sequences = count


# Global runtime state instance
_rl_runtime_state = RLRuntimeState()


def get_rl_runtime_state():
    """Get the global RL runtime state."""
    return _rl_runtime_state


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
        agent = get_agent(args, parallel_generation_tasks=args.rl_parallel_generation_tasks)
        # Collect Rollouts
        request = GroupedRolloutRequest(
            num_groups=-1 if args.rl_partial_rollouts else n_prompts,
            rollouts_per_group=samples_per_group,
            inference_interface=inference_interface,
            generation_args={
                'temperature': args.rl_default_temperature,
                'max_tokens': args.inference_max_seq_length,
                'top_p': args.rl_default_top_p,
                'top_k': args.rl_default_top_k,
            },
            filter_groups_with_same_reward=args.grpo_filter_groups_with_same_reward,
        )
        _ROLLOUT_GENERATOR = agent.get_grouped_rollouts(request)
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

    # If we have seperate training and inference models we to refit weights from the training model to the inference model.
    if inference_model is not None:
        # If the separate inference model weights were prefetched to CPU while idle, bring them
        # back to GPU before refit/copy and before any CUDA-graph'd inference.
        with nvtx_range("prefetch-inference-model-weights-to-gpu"):
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
    assert (
        n_prompts % get_pg_size(inference_pg_collection.ep) == 0
    ), "n_prompts must be divisible by data_parallel_world_size"

    with nvtx_range("rollout-collection"):
        loop = get_asyncio_loop()
        with megatron_rl_inference_mode(
            inference_model,
            optimizer,
            args.cuda_graph_impl,
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
                    log_single_rank(
                        logger,
                        logging.INFO,
                        f"Collecting rollouts, Iteration {args.curr_iteration}...",
                    )
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
        logger.debug(f"Got rollouts on rank {rank}")

    if lang_rl_log_dir and rank == get_pg_rank(inference_pg_collection.tp):
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
        Logprobs of input sequences.

    """

    # Ensure packed_seq_params is always provided for CUDA graph signature consistency
    if packed_seq_params is None and sequence_packing:
        packed_seq_params = get_default_packed_seq_params(
            seq_length=tokens.shape[1],
            device=tokens.device,
        )

    nvtx_range = get_nvtx_range()

    with nvtx_range("get-logprobs", time=False):

        with nvtx_range("forward-pass", time=False):
            # TODO(vitalyk): use fp16/bf16 as a function argument. Do not use args.
            args = get_args()

            attention_mask_for_forward = None

            # This is a hack to fix megatron's behaviour when flash-decode affects the training code flow.
            flash_decode = model.config.flash_decode
            model.config.flash_decode = False
            fp32_output = not (args.fp16 or args.bf16)
            with torch.no_grad() if no_grad else nullcontext():
                logits_or_hidden_states = model(
                    tokens,
                    position_ids,
                    attention_mask_for_forward,
                    packed_seq_params=packed_seq_params,
                    runtime_gather_output=True,
                    fp32_output=fp32_output,
                )
            model.config.flash_decode = flash_decode

        pg_collection = get_attr_wrapped_model(model, "pg_collection")
        pp_group = pg_collection.pp

        if not is_pp_last_stage(pp_group):
            return logits_or_hidden_states
        else:
            logits = logits_or_hidden_states
            with nvtx_range("log-softmax", time=False):
                # We do not need logprobs for the n+1 token.
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
                    'nonzero_groups_ratio': np.count_nonzero(group_stats.advantages)
                    / len(group_stats.advantages),
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
                    log_single_rank(
                        logger, logging.INFO, f"Updating tokenizer pad token to {pad_token}"
                    )
                    tokenizer._tokenizer.pad_token_id = tokenizer.vocab[pad_token]
                    break
            else:
                raise ValueError("No pad token found in tokenizer vocabulary")
    elif isinstance(tokenizer, CustomTikTokenizer):
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

    logger.info(f"[{dist.get_rank()}] Rollout counts:")
    for env_id, count in env_id_counts.items():
        logger.info(f"[{dist.get_rank()}] \t{env_id}: {count}")

    generation_masks = torch.tensor(generation_masks, dtype=torch.bool, device='cpu')
    trajs = torch.tensor(trajs, device='cpu')

    args = get_args()
    # Only process if we have inference_logprobs
    if inference_logprobs and any(lp is not None for lp in inference_logprobs):
        if args.rl_use_sequence_packing:
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
    if not args.rl_skip_bos_token:
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
    wandb_writer = get_wandb_writer()
    tb_writer = get_tensorboard_writer()
    nvtx_range = get_nvtx_range()
    runtime_state = get_rl_runtime_state()

    if args.cuda_graph_impl != "none" and not args.rl_training_cuda_graphs:
        lang_module = (
            model[0].module.module if hasattr(model[0].module, "module") else model[0].module
        )
        toggle_cuda_graphs(lang_module, "none", reset_cuda_graphs=False)

    model = model[0]
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

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
            global_rollout_count = len(group_stats.rewards)

        with nvtx_range("prepare_advantages", time=True):        
            # [g, group_size]
            # Making an assumption that all groups are of the same size!
            rewards = torch.tensor(rewards, device='cpu')
            advantages = (rewards - rewards.mean(axis=1, keepdim=True)) / (
                1e-4 + rewards.std(axis=1, keepdim=True)
            )

            # Flatten advantages for training and move to GPU
            advantages = global_advantages = advantages.view(-1).cuda()

        # Now split the rollouts across the data parallel ranks for training
        # This needs to be done at this point because we are about to calculate logprobs
        # Note :- For EP, do not use the expert data parallel group here. Always 
        # use the regular data parallel group. 
        if (data_parallel_world_size := mpu.get_data_parallel_world_size()) > 0:
            data_split_size = len(rollouts) // data_parallel_world_size
            data_split_range = (
                mpu.get_data_parallel_rank() * data_split_size,
                (mpu.get_data_parallel_rank() + 1) * data_split_size,
            )
            rollouts = rollouts[data_split_range[0] : data_split_range[1]]
            # First we calculate them on a global level and then we split and recalculate on a local level.
            # Sequence packing and reporting needs it global but non-packing wants it local.
            rewards = torch.tensor([[r.reward for r in group] for group in rollouts], device='cpu')
            advantages = (rewards - rewards.mean(axis=1, keepdim=True)) / (
                1e-4 + rewards.std(axis=1, keepdim=True)
            )

            # Flatten advantages for training and move to GPU
            advantages = advantages.view(-1).cuda()

        with nvtx_range("prepare_trajectories"):
            trajs, generation_masks, inference_logprobs = prepare_trajectories(
                rollouts, tokenizer, args.seq_length
            )

        # Build trajectories based on sequence packing or standard processing
        if args.rl_use_sequence_packing:
            with nvtx_range("sequence_packing", time=True):
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
        else:
            # Always compute standard masks for the original data (we'll need them later)
            with nvtx_range("get_ltor_masks_and_position_ids"):
                _, original_loss_mask, original_position_ids = get_ltor_masks_and_position_ids(
                    trajs,
                    tokenizer.eod,
                    tokenizer.pad,
                    args.reset_position_ids,
                    args.reset_attention_mask,
                    eod_mask_loss=False,
                    pad_mask_loss=True,
                )
                original_loss_mask[~generation_masks] = 0.0
                compute_trajs = trajs
                compute_position_ids = original_position_ids
                data_loader = DataLoader(
                    TensorDataset(compute_trajs, compute_position_ids),
                    batch_size=args.micro_batch_size,
                )
                logprobs_batch_size = args.micro_batch_size


        with torch.no_grad(), nvtx_range("compute_logprobs", time=True):
            # Before we can update the model, we need to get the logprobs for the \pi_{old} model.

            # Wrap forward_backward_func for Full iteration CUDA graph
            forward_backward_func = get_forward_backward_func()
            if args.cuda_graph_impl == "local" and CudaGraphScope.full_iteration in args.cuda_graph_scope:
                forward_backward_func = FullCudaGraphWrapper(
                    forward_backward_func, cuda_graph_warmup_steps=args.cuda_graph_warmup_steps
                )

            def logprobs_forward_step(data_iterator, model):

                # Avoid self.training checks which will trigger cudagraph capture; this path reuses
                # the forward pass from training after it has been captured on the 1st iteration.
                model.eval()

                if args.rl_use_sequence_packing:
                    # When using sequence packing, the data iterator returns a tuple with a single element, the bin index.
                    bin_tensor = next(data_iterator)[0]
                    #TODO(jalbericiola): change for named tuple
                    (b_trajs, _, _, _, b_posids, _, _, _, _, _, b_packed_seq_params) = (
                        load_packed_data_by_index(bin_tensor.item(), packing_context, args.rl_inference_logprobs_is_correction)
                    )
                else:
                    batch_data = next(data_iterator)
                    b_trajs, b_posids = batch_data
                    b_packed_seq_params = None

                b_trajs = b_trajs.cuda()
                b_posids = b_posids.cuda()

                logprobs = (
                    get_logprobs(
                        model,
                        b_trajs,
                        b_posids,
                        no_grad=True,
                        sequence_packing=args.rl_use_sequence_packing,                       
                        packed_seq_params=b_packed_seq_params,
                    ),
                    None,
                )

                model.train()
                return logprobs

            dtype = (
                torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
            )

            pg_collection = get_attr_wrapped_model(model, "pg_collection")
            pp_group = pg_collection.pp

            def _compute_logprobs_batch():
                """Compute logprobs for all batches in the data loader."""
                logprobs_list = []
                data_iterator = iter(data_loader)
                for i in range(len(data_loader)):
                    output_tensor = forward_backward_func(
                        forward_step_func=logprobs_forward_step,
                        data_iterator=data_iterator,
                        model=model,
                        num_microbatches=1,
                        seq_length=args.seq_length,
                        micro_batch_size=logprobs_batch_size,
                        decoder_seq_length=args.decoder_seq_length,
                        forward_only=True,
                        adjust_tensor_shapes_fn=None,
                    )
                    if is_pp_last_stage(pp_group):
                        logprobs_list.append(output_tensor[0].detach())

                if is_pp_last_stage(pp_group):
                    logprobs = torch.concat(logprobs_list, dim=0)
                    assert logprobs.dtype == dtype
                else:
                    logprobs = torch.empty(
                        len(compute_trajs),
                        args.seq_length - 1,
                        dtype=dtype,
                        device=torch.cuda.current_device(),
                    )

                # Only PP>1 needs a broadcast from the last stage; for PP=1 the output is already local.
                if get_pg_size(pp_group) > 1:
                    dist.broadcast(logprobs, src=get_pp_last_rank(pp_group), group=pp_group)
                return logprobs.cpu()

            with torch.no_grad(), nvtx_range("compute_old_logprobs", time=True):
                old_logprobs = _compute_logprobs_batch()

            with torch.no_grad(), nvtx_range("compute_ref_logprobs", time=True):
                # We need to load the ref model state dict and compute the logprobs for the ref model
                cur_st_dict = {
                    k: (v.cpu() if v is not None else v) for k, v in model.state_dict().items()
                }
                model.load_state_dict(ref_state_dict)

                ref_logprobs = _compute_logprobs_batch()

                # logprobs are [b, seq, h] now.
                model.load_state_dict(cur_st_dict)

            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()


        if args.rl_use_sequence_packing:
            with nvtx_range("pack_logprobs", time=True):
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
        else:
            with nvtx_range("align_inference_logprobs", time=True):
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

        with nvtx_range("create_dataloader"):
            if args.rl_use_sequence_packing:
               loader, optimizer_steps = get_microbatch_dataloader(packing_context)
               runtime_state.global_batches_per_collection = optimizer_steps
            else:
                runtime_state.packing_context = None
                runtime_state.global_batches_per_collection = global_rollout_count / args.global_batch_size
                dataset_tensors = [
                    compute_trajs,
                    advantages,
                    old_logprobs,
                    original_loss_mask,
                    original_position_ids,
                    ref_logprobs,
                ]
                if args.rl_inference_logprobs_is_correction and inference_logprobs is not None:
                    dataset_tensors.append(inference_logprobs)
                else:
                    dataset_tensors.append(torch.zeros_like(old_logprobs))

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
    inference_model: LanguageModule | None,
    optimizer: MegatronOptimizer,
    iteration: int,
    ref_state_dict: Dict[str, torch.Tensor],
) -> RerunDataIterator:

    args = get_args()
    tokenizer = get_tokenizer()

    buffered_rollouts = get_environment_rollouts(
        model, inference_model, optimizer, args.grpo_prompts_per_step, args.grpo_group_size
    )
    buffered_rollouts = prepare_data_for_update(model, ref_state_dict, buffered_rollouts, tokenizer)

    return buffered_rollouts


def setup_grpo_data_iterator(
    model: LanguageModule,
    inference_model: LanguageModule | None,
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
    runtime_state = get_rl_runtime_state()

    if inference_model is not None:
        inference_pg_collection = unwrap_model(inference_model[0]).pg_collection
    else:
        inference_pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    # We collect new rollouts when we've gone over the collected data 'grpo_iterations' times.
    if (
        buffered_rollouts is None or
        iteration == runtime_state.last_collection_iteration + 
        (args.grpo_iterations * runtime_state.global_batches_per_collection)
    ):
        train_data_iterator = get_rollout_data_iterator(model,inference_model, optimizer, iteration, ref_state_dict)
        runtime_state.reset_iteration_counters(iteration)
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
            args.cuda_graph_impl,
            args.rl_reset_cuda_graphs,
            args.rl_offload_optimizer_during_inference,
            args.rl_offload_kv_cache_during_training,
            args.rl_remove_kv_cache_during_training,
        ) as inference_interface:

            loop = get_asyncio_loop()

            rank = torch.distributed.get_rank()
            if rank == 0:
                logger.info(f"Collecting evaluation results...")
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


@contextmanager
def megatron_rl_inference_mode(
    model: list[LanguageModule],
    optimizer: MegatronOptimizer,
    cuda_graph_impl: str,
    reset_cuda_graphs: bool,
    offload_optimizer_during_inference: bool,
    offload_kv_cache_during_training: bool,
    remove_kv_cache_during_training: bool,
):
    """Manage the model inference context when collecting rollouts.

    Args:
        model: model to prepare.
        optimizer: optimizer used to train the model.
        cuda_graph_impl: which cuda graph implementation to use.
        reset_cuda_graphs: rebuild cuda graphs for each inference stage or not.
        offload_optimizer_during_inference: move optimizer to cpu during inference or not.
        offload_kv_cache_during_training: manually offload kv cache to host before training or not.
        remove_kv_cache_during_training: manually remove kv cache before training or not.

    Yields:
        None: this context manager does not return a value.

    """
    args = get_args()
    loop = get_asyncio_loop()
    nvtx_range = get_nvtx_range()

    logger.debug(f"[{dist.get_rank()}] Entering inference mode")

    # If we get a lower precision wrapper, we go one object deeper.
    lang_module = model[0].module.module if hasattr(model[0].module, "module") else model[0].module

    lang_module.eval()
    # If this is a separate RL inference model allocated with UVM, ensure weights are resident on GPU
    # before any CUDA-graph capture/replay or inference.
    with nvtx_range("prefetch-inference-model-weights-to-gpu"):
        model_core = unwrap_model(model[0])
        _maybe_prefetch_separate_inference_model_weights(model_core, to_cpu=False)

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

        # TODO: Remove this if statement once a change to `toggle_cuda_graphs` makes it safe to.
        if cuda_graph_impl != "none" and not args.rl_training_cuda_graphs:
            toggle_cuda_graphs(lang_module, cuda_graph_impl, reset_cuda_graphs=reset_cuda_graphs)

        inference_interface = get_inference_interface(args, loop, model)

        with nvtx_range("onload-kv-cache-before-inference"):
            if offload_kv_cache_during_training:
                assert (
                    reset_cuda_graphs
                ), "reset_cuda_graphs must be True when offloading kv cache during training"
                logger.debug(
                    f"[{dist.get_rank()}] Restoring kv cache ({inference_interface._inference_engine.context.memory_buffer.numel() / 1024**3:.2f} GB) to GPU"
                )
                kv_cache = inference_interface._inference_engine.context.memory_buffer
                inference_interface._inference_engine.context.memory_buffer = kv_cache.cuda()
            elif remove_kv_cache_during_training:
                if inference_interface._inference_engine.context.memory_buffer is None:
                    inference_interface._inference_engine.context.build_memory_buffer()

        # TODO: Improve this if statement once a change is made to CUDA graph handling.
        cuda_graph_exists = len(_CudagraphGlobalRecord.cudagraph_inference_record) != 0
        if cuda_graph_impl != "none" and not cuda_graph_exists:
            with nvtx_range("wait-for-decode-only"):
                while not inference_interface._inference_engine.context.is_decode_only():
                    active_requests, finished_requests, step_time = loop.run_until_complete(
                        inference_interface._inference_engine.async_step()
                    )
            with nvtx_range("build-cuda-graphs"):
                inference_interface._inference_engine.create_cuda_graphs(reset_context=True)

        loop.run_until_complete(inference_interface.resume())

        logger.debug(f"[{dist.get_rank()}] Entered inference mode")
        yield inference_interface

        with nvtx_range("suspend-engine"):
            loop.run_until_complete(inference_interface.suspend())

        with nvtx_range("offload-kv-cache-after-inference"):
            if offload_kv_cache_during_training:
                kv_cache = inference_interface._inference_engine.context.memory_buffer
                logger.debug(
                    f"[{dist.get_rank()}] Offloading kv cache ({kv_cache.numel() * kv_cache.element_size() / 1024**3:.2f} GB) to CPU"
                )
                inference_interface._inference_engine.context.memory_buffer = kv_cache.cpu()
            elif remove_kv_cache_during_training:
                inference_interface._inference_engine.context.memory_buffer = None

        # TODO: Remove this if statement once a change to `toggle_cuda_graphs` makes it safe to.
        if cuda_graph_impl != "none" and not args.rl_training_cuda_graphs:
            toggle_cuda_graphs(lang_module, 'none', reset_cuda_graphs=reset_cuda_graphs)

        # If this is a separate RL inference model, prefetch weights back to CPU so they don't consume
        # GPU memory during training.
        with nvtx_range("prefetch-inference-model-weights-to-cpu"):
            _maybe_prefetch_separate_inference_model_weights(model_core, to_cpu=True)

        if offload_optimizer_during_inference:
            with nvtx_range("onload-optimizer-after-inference"):
                optimizer.restore_from_cpu()

        lang_module.train()

        if has_lru_cache:
            rotary_module.forward.cache_clear()

        logger.debug(f"[{dist.get_rank()}] Exiting inference mode")


def rl_inference_interface_shutdown():
    global _INFERENCE_INTERFACE
    if _INFERENCE_INTERFACE is not None:
        loop = get_asyncio_loop()
        loop.run_until_complete(_INFERENCE_INTERFACE.kill())
        _INFERENCE_INTERFACE = None
    else:
        logger.warning("No inference interface to shutdown. This should not happen.")


def get_iteration_sequence_count(args):
    """Get the total number of sequences processed in this iteration across all ranks."""
    runtime_state = get_rl_runtime_state()
    sequences_tensor = torch.tensor(
        runtime_state.sequences_this_iteration_on_rank, device='cuda', dtype=torch.long
    )
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(sequences_tensor, group=mpu.get_data_parallel_group())
    return int(sequences_tensor.item())
