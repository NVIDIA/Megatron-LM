# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""RL utilities."""

import gc

# Keep this to make the env registered.
import itertools
import logging
import pickle
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterator

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
from megatron.core.optimizer import MegatronOptimizer
from megatron.core.parallel_state import get_tensor_model_parallel_src_rank
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.core.transformer.cuda_graphs import _CudagraphGlobalRecord
from megatron.core.transformer.utils import toggle_cuda_graphs
from megatron.rl.agent.api import (
    EvaluationRequest,
    EvaluationResponse,
    GroupedRolloutRequest,
    Rollout,
    TokenRollout,
)
from megatron.rl.agent.reward_only_agent import RewardOnlyEvaluationResult
from megatron.rl.agent.weighted_multi_task import WeightedMultiTask
from megatron.rl.inference.megatron import MegatronChatLocal, MegatronLocal
from megatron.rl.logging import LOG_DIR as lang_rl_log_dir
from megatron.rl.logging import log as lang_rl_log
from megatron.training.global_vars import (
    get_args,
    get_tensorboard_writer,
    get_tokenizer,
    get_wandb_writer,
)
from megatron.training.tokenizer.tokenizer import CustomTikTokenizer, _HuggingFaceTokenizer
from megatron.training.utils import get_ltor_masks_and_position_ids, get_nvtx_range, print_rank_0

logger = logging.getLogger(__name__)


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
            raise ValueError(f"Unknown inference_server_type {args.langrl_inference_server_type}")
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

    with nvtx_range("rollout-collection", time=True):
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

            with nvtx_range("inference-setup", time=True):
                # Asyncronously run inference and rollout collection
                rollout_generator = get_rollout_generator(
                    args, inference_interface, n_prompts, samples_per_group
                )

            # NOTE(jbarker): we need to double check this when using PP>1
            rank = torch.distributed.get_rank()
            with nvtx_range("collect-rollouts", time=True):
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

        with nvtx_range("sync-rollouts", time=True):
            # Wait for Rollouts to be collected
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


def compute_group_stats(rollouts: GroupedRollouts, tokenizer: MegatronLegacyTokenizer) -> RolloutStats:
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
    rollouts: GroupedRollouts,
    tokenizer: MegatronLegacyTokenizer,
    seq_length: int,
    add_extra_token_to_sequence: bool,
):
    """Pad trajectories and extract the generation masks.

    Args:
        rollouts: Rollouts to extract trajectories from.
        tokenizer: Tokenizer to get the padding token and potentially tokenize.
        seq_length:  Maximum sequence length to pad to.
        add_extra_token_to_sequence: Add one extra token to sequence when padding or not.

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

            if length < (seq_length + add_extra_token_to_sequence):
                trajectory.extend(
                    [tokenizer.pad] * (seq_length + add_extra_token_to_sequence - length)
                )
                if generation_mask:
                    generation_mask.extend(
                        [False] * (seq_length + add_extra_token_to_sequence - length)
                    )
            trajs.append(trajectory)
            generation_masks.append(generation_mask)
            inference_logprobs.append(torch.Tensor(inf_logprobs))

            env_id = rollout.env_id
            env_id_counts[env_id] += 1

    print(
        "Rollout counts:"
        + "".join([f"\n\t{env_id}: {count}" for env_id, count in env_id_counts.items()])
    )

    generation_masks = torch.tensor(generation_masks, dtype=torch.bool, device='cpu')
    trajs = torch.tensor(trajs, device='cpu')

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


def prepare_data_for_update(
    model: list[LanguageModule],
    ref_state_dict: Dict[str, Any],
    rollouts: GroupedRollouts,
    tokenizer: MegatronLegacyTokenizer,
    add_extra_token_to_sequence: bool = True,
) -> RerunDataIterator:
    """Extract data for the update from raw rollouts.

    Args:
        model: Current policy as the zero-eth element.
        ref_state_dict: Reference policy state dict.
        rollouts: Rollouts to extract the data from.
        tokenizer: Tokenizer to pad/tokenize data.
        add_extra_token_to_sequence: Add extra token to sequence or not when padding.

    Returns:
        Cycled iterator over dataset batches. In GRPO we might want to go over the same data multiple times.
    """
    args = get_args()
    wandb_writer = get_wandb_writer()
    tb_writer = get_tensorboard_writer()
    nvtx_range = get_nvtx_range()
    model = model[0]

    with nvtx_range("prepare-data-for-update", time=True):
        with nvtx_range("compute-group-stats", time=True):
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
        rewards = torch.tensor(
            [[rollout.reward for rollout in group] for group in rollouts], device='cpu'
        )
        # We flatten them for logging.
        with nvtx_range("prepare-trajectories", time=True):
            trajs, generation_masks, inference_logprobs = prepare_trajectories(
                rollouts, tokenizer, args.seq_length, add_extra_token_to_sequence
            )

        with nvtx_range("get-ltor-masks-and-position-ids", time=True):
            _, loss_mask, position_ids = get_ltor_masks_and_position_ids(
                trajs,
                tokenizer.eod,
                tokenizer.pad,
                args.reset_position_ids,
                args.reset_attention_mask,
                eod_mask_loss=False,
                pad_mask_loss=True,
            )
            loss_mask[~generation_masks] = 0.0

        with nvtx_range("create-logprobs-dataloader", time=True):
            data_iter = DataLoader(
                TensorDataset(trajs, position_ids), batch_size=args.micro_batch_size
            )
            old_logprobs = []

            # Temporarily disable sequence parallelism for logprob computation
            for b_trajs, b_posids in data_iter:
                logprobs = get_logprobs(model, b_trajs.cuda(), b_posids.cuda(), None, no_grad=True)
                old_logprobs.append(logprobs.detach().cpu())

            old_logprobs = torch.concat(old_logprobs, dim=0)

        with torch.no_grad(), nvtx_range("compute_ref_logprobs", time=True):
            # We need to load the ref model state dict and compute the logprobs for the ref model
            cur_st_dict = {
                k: (v.cpu() if v is not None else v) for k, v in model.state_dict().items()
            }
            model.load_state_dict(ref_state_dict)
            ref_logprobs = []

            for b_trajs, b_posids in data_iter:
                logprobs = get_logprobs(model, b_trajs.cuda(), b_posids.cuda(), None, no_grad=True)
                ref_logprobs.append(logprobs.detach().cpu())

            ref_logprobs = torch.concat(ref_logprobs, dim=0)
            # logprobs are [b, seq, h] now.
            model.load_state_dict(cur_st_dict)

        with torch.no_grad(), nvtx_range("compute-logprobs", time=True):
            # Before we can update the model, we need to get the logprobs for the \pi_{old} model.
            old_logprobs = []

            for b_trajs, b_posids in data_iter:
                logprobs = get_logprobs(model, b_trajs.cuda(), b_posids.cuda(), None, no_grad=True)
                old_logprobs.append(logprobs.detach().cpu())

            old_logprobs = torch.concat(old_logprobs, dim=0)

        with nvtx_range("compute-prob-stats", time=True):
            # Inference logprobs 2 tokens shorter than old_logprobs.
            # One token difference is because we remove the last one in get_logprobs(), the other one is eod padding, if I got it correct. The difference should be one token if we are cut by the sequence length.

            # Get first occurrence of a generation token.
            # @vitalyk:
            # In get_logprobs() we chop off the first token -> the generation mask is shifted by one.
            # We have to correct for it.
            first_gen_tok = generation_masks.int().argmax(dim=1) - 1

            padded_inference_logprobs = old_logprobs.clone()
            # We need to align old_logprobs and inference logprobs as the latter are only for generations.
            for i, inf_logprobs in enumerate(inference_logprobs):
                first_gen_idx = first_gen_tok[i]
                # We subtract -1 here because we append eod token on the train side, and we do not
                # get it from the inference. For the eod token, we reuse old_logprobs value.
                padded_inference_logprobs[i, first_gen_idx : first_gen_idx + len(inf_logprobs)] = (
                    inf_logprobs
                )

            truncated_mask = generation_masks[:, 1:].bool()
            inference_logprobs = padded_inference_logprobs
            n_elems = truncated_mask.sum()

            ratios = (old_logprobs - inference_logprobs).exp()[truncated_mask]
            abs_diffs = (old_logprobs.exp() - inference_logprobs.exp()).abs()[truncated_mask]
            # Two probability values cannot be more than 1.0 apart.
            assert all(abs_diffs <= 1.0)

            group_stats.min_piold_to_inf_prob = ratios.min().item()
            group_stats.max_piold_to_inf_prob = ratios.max().item()
            group_stats.mean_piold_to_inf_prob = (ratios.sum() / n_elems).item()
            group_stats.min_inf_train_prob_abs_diff = abs_diffs.min().item()
            group_stats.max_inf_train_prob_abs_diff = abs_diffs.max().item()
            group_stats.mean_inf_train_prob_abs_diff = (abs_diffs.sum() / n_elems).item()

            inf_probs = inference_logprobs.exp()[truncated_mask]
            group_stats.min_inf_prob = inf_probs.min().item()
            group_stats.max_inf_prob = inf_probs.max().item()
            group_stats.mean_inf_prob = inf_probs.mean().item()

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        with nvtx_range("prepare-advantages", time=True):
            advantages = (rewards - rewards.mean(axis=1, keepdim=True)) / (
                1e-4 + rewards.std(axis=1, keepdim=True)
            )
        with nvtx_range("create-dataloader", time=True):
            # Mask out the prompts, we do not want to add them to the loss and backprop.
            data = TensorDataset(
                trajs,
                advantages.view(-1),
                old_logprobs,
                loss_mask,
                position_ids,
                ref_logprobs,
                inference_logprobs,
            )
            loader = DataLoader(data, batch_size=args.micro_batch_size)

        with nvtx_range("log-wandb-tb", time=True):
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
                        if isinstance(result, RewardOnlyEvaluationResult):
                            try:
                                lang_rl_log(
                                    f"Evaluation: [{result.env_id}] [{result.reward}] {result.prompt} {result.response}"
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get GRPO loss, the kl term of the loss and the pi/pi_{old} ratios.

    Args:
        current_logprobs: pi logprobs, [batch, seq, 1].
        old_logprobs: pi_{old} logprobs, [batch, seq, 1].
        ref_logprobs: pi_{ref} logprobs, [batch, seq, 1].
        advantages: advantages tensor, [batch,].
        clamp_eps_lower: eps to clamp ratios from below.
        clamp_eps_upper: eps to clamp ratios from above, if vanilla GRPO, this should be equal to clamp_eps_lower.
        kl_beta: weight for the KL penalty term measuring the distance between pi and pi_{ref}.
        entropy_weight: weight for the entropy term.
        inference_logprobs: pi_{old} logprobs calculated by the inference engine.
            If not None, importance sampling correction will be applied.
        is_truncation_coef: importance sampling truncation coefficient. Will be applied if it is not None and inference_logprobs are present.

    Returns:
        total per-token GRPO loss [batch, seq],
        kl_term of the loss [batch, seq],
        pi/pi_{old} ratios [batch, seq],
        entropy_term of the loss [batch, sec],
        truncated_from_above [batch, seq] (whether we clamped the ratios or not),
        truncated_from_below [batch, seq] (whether we clamped the ratios or not).
    """
    ratios = (current_logprobs - old_logprobs).exp()
    clamped_ratios = ratios.clamp(1 - clamp_eps_lower, 1 + clamp_eps_upper)
    truncated_from_above = torch.gt(ratios, 1 + clamp_eps_upper)
    truncated_from_below = torch.lt(ratios, 1 - clamp_eps_lower)
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

    with torch.no_grad():

        if offload_optimizer_during_inference:
            with nvtx_range("offload-optimizer-before-inference", time=True):
                optimizer.offload_to_cpu()

        if enable_cuda_graph:
            toggle_cuda_graphs(lang_module, True, reset_cuda_graphs=reset_cuda_graphs)

        inference_interface = get_inference_interface(args, loop, model)

        with nvtx_range("onload-kv-cache-before-inference", time=True):
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

        if not _CudagraphGlobalRecord.cudagraph_created:
            with nvtx_range("wait-for-decode-only", time=True):
                while not inference_interface._coordinator.engine.context.is_decode_only():
                    active_requests, finished_requests, step_time = loop.run_until_complete(
                        inference_interface._coordinator.engine.async_step()
                    )
            with nvtx_range("build-cuda-graphs", time=True):
                inference_interface._coordinator.engine.build_cuda_graphs(reset_context=False)

        inference_interface._coordinator.resume_engine()
        yield inference_interface

        with nvtx_range("suspend-engine", time=True):
            loop.run_until_complete(inference_interface.suspend())

        with nvtx_range("offload-kv-cache-after-inference", time=True):
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
            with nvtx_range("onload-optimizer-after-inference", time=True):
                optimizer.restore_from_cpu()

        lang_module.train()

        print(f"[{dist.get_rank()}:DP] Exiting inference mode")
