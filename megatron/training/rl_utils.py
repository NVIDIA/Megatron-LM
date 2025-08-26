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

from lang_rl.agent.api import (
    EvaluationRequest,
    EvaluationResponse,
    GroupedRolloutRequest,
    Rollout,
    TokenRollout,
)
from lang_rl.agent.reward_only_agent import RewardOnlyEvaluationResult
from lang_rl.agent.weighted_multi_task import WeightedMultiTask
from lang_rl.inference.megatron import MegatronChatLocal, MegatronLocal
from lang_rl.logging import LOG_DIR as lang_rl_log_dir
from lang_rl.logging import log as lang_rl_log
from megatron.core import mpu
from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
from megatron.core.inference.utils import get_event_loop
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.optimizer import MegatronOptimizer
from megatron.core.parallel_state import get_tensor_model_parallel_src_rank
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.core.transformer.utils import toggle_cuda_graphs
from megatron.training.tokenizer.tokenizer import CustomTikTokenizer, _HuggingFaceTokenizer
from megatron.training.utils import get_ltor_masks_and_position_ids
from wandb import wandb_run

from .global_vars import get_args, get_tensorboard_writer, get_timers, get_wandb_writer
from .utils import get_nvtx_range, print_rank_0

logger = logging.getLogger(__name__)


GroupedRollouts = list[list[TokenRollout | Rollout]]


@dataclass(slots=True)
class RolloutStats:
    mean_reward: float
    mean_sim: float
    mean_length: float
    mean_length_std: float
    max_length: float
    min_length: float
    reward_means: list[float]
    reward_stds: list[float]
    rewards: list[float]
    advantages: None | list[float]


def get_agent(args):
    """Get an agent based on environment configuration.

    If args.env_config is provided, uses weighted environment selection.
    Otherwise falls back to legacy single environment selection.
    """
    with open(args.env_config, 'r') as f:
        config = yaml.safe_load(f)

    return WeightedMultiTask.from_config(config)


def get_inference_interface(args, loop, model):
    if args.inference_server_type == 'inplace_megatron':
        return loop.run_until_complete(MegatronLocal.launch(model[0]))
    elif args.inference_server_type == 'inplace_megatron_chat':
        return loop.run_until_complete(
            MegatronChatLocal.launch(
                model[0], conversation_template=args.inference_server_conversation_template
            )
        )
    else:
        raise ValueError(f"Unknown inference_server_type {args.inference_server_type}")


def get_environment_rollouts(model: LanguageModule, n_prompts: int, samples_per_group: int):
    """Sample environment rollouts from an LLM.

    Args:
        model: Model to sample from.
        n_prompts: Number of prompts to sample for across *all* data parallel workers.
        samples_per_group: Amount of trajectories per prompt.

    Returns:
        GroupedRollouts object which is a nested list with each element being a list of rollouts of a group.
    """
    args = get_args()
    timers = get_timers()
    nvtx_range = get_nvtx_range()

    assert (
        n_prompts % mpu.get_expert_data_parallel_world_size() == 0
    ), "n_prompts must be divisible by data_parallel_world_size"

    with nvtx_range("rollout_collection"):
        timers('rollout-collection', log_level=0).start(barrier=True)
        model[0].eval()

        with torch.no_grad():
            with nvtx_range("inference_setup"):
                # Asyncronously run inference and rollout collection
                loop = get_event_loop()

                inference_interface = get_inference_interface(args, loop, model)

            # NOTE(jbarker): we need to double check this when using PP>1
            rank = torch.distributed.get_rank()
            with nvtx_range("collect_rollouts"):
                if rank == 0:
                    print(f"Collecting rollouts on rank {rank}...")
                    agent = get_agent(args)
                    # Collect Rollouts
                    request = GroupedRolloutRequest(
                        num_groups=n_prompts,
                        rollouts_per_group=samples_per_group,
                        inference_interface=inference_interface,
                        generation_args={
                            'temperature': args.grpo_default_temperature,
                            'max_tokens': args.seq_length,
                            'top_p': args.grpo_default_top_p,
                        },
                        filter_groups_with_same_reward=args.grpo_filter_groups_with_same_reward,
                    )
                    rollouts = loop.run_until_complete(agent.get_grouped_rollouts(request))

                else:
                    # Just set up space to collect the rollouts
                    rollouts = [[None for _ in range(samples_per_group)] for _ in range(n_prompts)]

            with nvtx_range("cleanup"):
                loop.run_until_complete(inference_interface.kill())

            with nvtx_range("sync_rollouts"):
                # Wait for Rollouts to be collected
                # TODO(jbarker): double check why this isn't causing rank 0 memory allocations
                torch.distributed.broadcast_object_list(rollouts, src=0)

        model[0].train()

        # For some reason we need to force garbage collection to prevent memory leak
        gc.collect()
        timers('rollout-collection').stop()

    if lang_rl_log_dir and rank == get_tensor_model_parallel_src_rank():
        with open(
            lang_rl_log_dir
            + f'/rollouts_rank{rank}_iteration{args.curr_iteration}_'
            + f'{Path(args.env_config).stem}.pkl',
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
        use_sequence_parallel: Override sequence_parallel setting for this call.
                              If None, use model's current setting.

    Returns:
        Logprobs of input sequences.

    """
    nvtx_range = get_nvtx_range()

    with nvtx_range("get_logprobs"):

        with nvtx_range("forward_pass"):
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
        with nvtx_range("log_softmax"):
            logprobs = selective_log_softmax(logits[:, :-1, :], tokens[:, 1:])

    return logprobs


def compute_group_stats(rollouts: GroupedRollouts, tokenizer: MegatronTokenizer) -> RolloutStats:
    """Add group-based rollout stats for logging.

    Args:
        rollouts: Rollouts to generate the stats for. Each inner list is a group (as in GRPO group), i.e. all rollouts are for the same prompt.
        tokenizer: Tokenizer to tokenize the rollouts in case they are raw strings.

    Returns:
       RolloutStats object containing all the stats.
    """
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
                # assert (len(rollout.trajectory) == args.seq_length) or (rollout.trajectory[-1] == tokenizer.eod), f"Rollout is not the correct length: {len(rollout.trajectory)} {rollout.trajectory[-1]}\n{tokenizer.detokenize(rollout.trajectory)}"
            else:
                lang_rl_log(
                    f"Rollout: [{rollout.env_id}] [{rollout.reward} : {len(rollout.trajectory)} chars] {rollout.trajectory}"
                )
            group_rewards.append(rollout.reward)
            group_lengths.append(len(rollout.trajectory))
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
        mean_sim=np.mean(group_rollout_similarities),
        mean_length=np.mean(group_length_means),
        mean_length_std=np.mean(group_length_stds),
        max_length=np.max(group_length_maxs),
        min_length=np.min(group_length_mins),
        reward_means=group_reward_means,
        reward_stds=group_reward_stds,
        rewards=None,  # We will fill those in later in prepare_data_for_update.
        advantages=None,  # We will fill those in later in prepare_data_for_update.
    )
    return stats


def maybe_log_training_metrics(
    group_stats: RolloutStats,
    current_iteration: int,
    tokenizer: MegatronTokenizer,
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
                'group_means_hist': wandb_writer.plot.histogram(
                    group_table, 'group_means', 'Group Means'
                ),
                'group_stds_hist': wandb_writer.plot.histogram(
                    group_table, 'group_stds', 'Group STDs'
                ),
                'rewards_hist': wandb_writer.plot.histogram(rollout_table, 'reward', 'All Rewards'),
                'mean_length': group_stats.mean_length,
                'mean_intra_group_similarity': group_stats.mean_sim,
                'mean_length_std': group_stats.mean_length_std,
                'max_length': group_stats.max_length,
                'min_length': group_stats.min_length,
                'mean_reward': group_stats.mean_reward,
                'mean_advantage': np.mean(group_stats.advantages),
                'advantages_hist': wandb_writer.plot.histogram(
                    advantages, 'advantages', 'Advantages'
                ),
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
            step=current_iteration,
        )
    if tb_writer:
        tb_writer.add_scalar('mean_reward', group_stats.mean_reward, current_iteration)


def prepare_trajectories(
    rollouts: GroupedRollouts,
    tokenizer: MegatronTokenizer,
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
    for gidx, group in enumerate(rollouts):
        for ridx, rollout in enumerate(group):
            # TODO(vitalyk): make this model-agnostic. Take lens from the inference server.
            generation_mask = rollout.generation_mask if isinstance(rollout, TokenRollout) else None

            trajectory = (
                rollout.trajectory.copy()
                if isinstance(rollout, TokenRollout)
                else tokenizer.tokenize(rollout.trajectory)
            )
            if len(trajectory) < seq_length:
                trajectory = trajectory + [tokenizer.eod]
                generation_mask = generation_mask + [True]
            length = len(trajectory)

            # TODO(rkirby): Why do we have add_extra_token_to_sequence? I'm now adding a single eod to the end of the trajectory, so the added token might end up being the eod token.
            # See https://gitlab-master.nvidia.com/ADLR/megatron-rl/-/issues/49
            assert length <= seq_length, "Rollout too long, how did this happen?"
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
    # assert ((trajs == tokenizer.eod).sum(axis=0) == 1).all(), "Only one eod per trajectory"
    # TODO(rkirby):
    # We should avoid the tokenizer pad token being the same as the eod token for proper loss masking,
    # But now the deepseek tokenizer has the pad token set to eod, we need to handle this.
    # assert (tokenizer.pad != tokenizer.eod), "Pad and eod should be different"
    return trajs, generation_masks


def prepare_data_for_update(
    model: list[LanguageModule],
    ref_state_dict: Dict[str, Any],
    rollouts: GroupedRollouts,
    tokenizer: MegatronTokenizer,
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
    timers = get_timers()
    wandb_writer = get_wandb_writer()
    tb_writer = get_tensorboard_writer()
    nvtx_range = get_nvtx_range()
    model = model[0]

    with nvtx_range("prepare_data_for_update"):
        with nvtx_range("compute_group_stats"):
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
        data_split_size = len(rollouts) // mpu.get_expert_data_parallel_world_size()
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
        with nvtx_range("prepare_trajectories"):
            trajs, generation_masks = prepare_trajectories(
                rollouts, tokenizer, args.seq_length, add_extra_token_to_sequence
            )

        with nvtx_range("get_ltor_masks_and_position_ids"):
            _, loss_mask, position_ids = get_ltor_masks_and_position_ids(
                trajs,
                tokenizer.eod,
                tokenizer.pad,
                args.reset_position_ids,
                args.reset_attention_mask,
                eod_mask_loss=False,
                pad_mask_loss=True,
            )

        with torch.no_grad(), nvtx_range("compute_logprobs"):
            timers('compute-logprobs', log_level=0).start()
            # Before we can update the model, we need to get the logprobs for the \pi_{old} model.
            data_iter = DataLoader(
                TensorDataset(trajs, position_ids), batch_size=args.micro_batch_size
            )
            model.eval()

            old_logprobs = []

            # Temporarily disable sequence parallelism for logprob computation
            for b_trajs, b_posids in data_iter:
                logprobs = get_logprobs(model, b_trajs.cuda(), b_posids.cuda(), None, no_grad=True)
                old_logprobs.append(logprobs.detach().cpu())

            old_logprobs = torch.concat(old_logprobs, dim=0)
            timers('compute-logprobs').stop()

        with torch.no_grad(), nvtx_range("compute_ref_logprobs"):
            timers('compute-ref-logprobs', log_level=0).start()
            cur_st_dict = {
                k: (v.cpu() if v is not None else v) for k, v in model.state_dict().items()
            }
            model.load_state_dict(ref_state_dict)
            # TODO(vitalyk): should this be eval?
            ref_logprobs = []

            # Temporarily disable sequence parallelism for reference logprob computation
            for b_trajs, b_posids in data_iter:
                logprobs = get_logprobs(model, b_trajs.cuda(), b_posids.cuda(), None, no_grad=True)
                ref_logprobs.append(logprobs.detach().cpu())

            ref_logprobs = torch.concat(ref_logprobs, dim=0)
            # logprobs are [b, seq, h] now.
            model.load_state_dict(cur_st_dict)
            timers('compute-ref-logprobs').stop()

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        model.train()

        with nvtx_range("prepare_advantages"):
            timers('prepare-advantages', log_level=0).start()
            advantages = (rewards - rewards.mean(axis=1, keepdim=True)) / (
                1e-4 + rewards.std(axis=1, keepdim=True)
            )
            timers('prepare-advantages').stop()
        with nvtx_range("create_dataloader"):
            # Mask out the prompts, we do not want to add them to the loss and backprop.
            loss_mask[~generation_masks] = 0.0
            data = TensorDataset(
                trajs, advantages.view(-1), old_logprobs, loss_mask, position_ids, ref_logprobs
            )
            loader = DataLoader(data, batch_size=args.micro_batch_size)

        with nvtx_range("log_wandb_tb"):
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

    if args.rl_offload_optimizer_during_inference:
        optimizer.offload_to_cpu()

    with megatron_rl_inference_mode(
        model,
        optimizer,
        args.enable_cuda_graph,
        args.rl_offload_optimizer_during_inference
    ):
        buffered_rollouts = get_environment_rollouts(
            model,
            args.grpo_prompts_per_step,
            args.grpo_group_size
        )
    buffered_rollouts = prepare_data_for_update(
        model,
        ref_state_dict,
        buffered_rollouts,
        tokenizer,
    )

    if args.rl_offload_optimizer_during_inference:
        optimizer.restore_from_cpu()

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
        model[0].eval()
        with megatron_rl_inference_mode(
            model,
            optimizer,
            args.sequence_parallel,
            args.enable_cuda_graph,
            args.rl_offload_optimizer_during_inference,
        ):

            loop = get_event_loop()
            inference_interface = get_inference_interface(args, loop, model)
            data_parallel_rank = mpu.get_data_parallel_rank()
            data_parallel_size = mpu.get_expert_data_parallel_world_size()

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

            loop.run_until_complete(inference_interface.kill())

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
                    + f'{Path(args.env_config).stem}.pkl',
                    'wb',
                ) as f:
                    pickle.dump(dp_eval_results, f)

    model[0].train()


def calculate_grpo_loss(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clamp_eps_lower: float,
    clamp_eps_upper: float,
    kl_beta: float,
    entropy_weight: float,
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

    Returns:
        total per-token GRPO loss [batch, seq], kl_term of the loss [batch, seq], and pi/pi_{old} ratios [batch, seq].
    """
    ratios = (current_logprobs - old_logprobs).exp()
    clamped_ratios = ratios.clamp(1 - clamp_eps_lower, 1 + clamp_eps_upper)
    advantages = advantages.view(-1, 1)

    ref_diff = ref_logprobs - current_logprobs
    kl_term = ref_diff.exp() - ref_diff - 1

    entropy_term = current_logprobs.exp() * current_logprobs

    loss = (
        -torch.min(ratios * advantages, clamped_ratios * advantages)
        + kl_beta * kl_term
        + entropy_weight * entropy_term
    )
    return loss, kl_term, ratios, entropy_term


@contextmanager
def megatron_rl_inference_mode(
    model: list[LanguageModule],
    optimizer: MegatronOptimizer,
    enable_cuda_graph: bool,
    rl_offload_optimizer_during_inference: bool,
):
    """Manage the model inference context when collecting rollouts.

    Args:
        model: model to prepare.
        optimizer: optimizer used to train the model.
        sequence_parallel: use sequence parallel or not.
        enable_cuda_graph: use cuda graphs or not.
        rl_offload_optimizer_during_inference: move optimizer to cpu during inference or not.

    Yields:
        None: this context manager does not return a value.

    """
    # TODO(vitalyk): add model.eval() and no_grad() here.
    if rl_offload_optimizer_during_inference:
        optimizer.offload_to_cpu()

    if enable_cuda_graph:
        toggle_cuda_graphs(model[0].module.module, True)

    yield

    if enable_cuda_graph:
        toggle_cuda_graphs(model[0].module.module, False)

    if rl_offload_optimizer_during_inference:
        optimizer.restore_from_cpu()
