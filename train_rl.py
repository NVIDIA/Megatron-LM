# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Train GPT with rl."""

import inspect
import os
from contextlib import nullcontext
from functools import partial

import torch

from gpt_builders import gpt_builder
from mamba_builders import mamba_builder
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import StragglerDetector
from megatron.rl.rl_utils import calculate_grpo_loss, get_logprobs
from megatron.training import get_args, get_timers, pretrain, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from model_provider import model_provider

stimer = StragglerDetector()


def _gpt_builder(args, pre_process, post_process, vp_stage=None, config=None):
    # TODO(Peter): This is a hack to get around the fact that we are activation recomputation for training but not
    # for inference with cuda graphs. Without out this the post checks in the transformer config will assert error.
    if config is None:
        recompute_granularity_from_args = None
        if args.recompute_granularity is not None:
            recompute_granularity_from_args = args.recompute_granularity
            args.recompute_granularity = None

        config = core_transformer_config_from_args(args)

        if recompute_granularity_from_args is not None:
            config.recompute_granularity = recompute_granularity_from_args

    build_model_context = nullcontext
    build_model_context_args = {}
    if args.fp8_param_gather:
        try:
            from transformer_engine.pytorch import fp8_model_init

            build_model_context = fp8_model_init
            build_model_context_args["enabled"] = True

            # Check if fp8_model_init supports preserve_high_precision_init_val
            if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                build_model_context_args["preserve_high_precision_init_val"] = True
        except:  # noqa E722
            raise RuntimeError(
                "--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found."
            )

    with build_model_context(**build_model_context_args):
        return gpt_builder(args, pre_process, post_process, vp_stage=vp_stage, config=config)


# define spiky loss as a variation of 20% or more
SPIKY_LOSS_PERC = 0.2


def loss_func(
    loss_mask: torch.Tensor,
    kl_term: torch.Tensor,
    ratios: torch.Tensor,
    entropy_term: torch.Tensor,
    truncated_from_above: torch.Tensor,
    truncated_from_below: torch.Tensor,
    output_tensor: torch.Tensor,
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        kl_term (torch.Tensor): KL term of the loss. Used for logging.
        ratios (torch.Tensor): pi/pi_{old} ratios. Used for logging.
        entropy (torch.Tensor): Current policy entropy on the trajectories. Used for logging.
        truncated_from_above(torch.Tensor): A boolean mask that tells whether the ratios were truncated from above. Used for logging.
        truncated_from_below(torch.Tensor): A boolean mask that tells whether the ratios were truncated from below. Used for logging.
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    losses = output_tensor.float().cuda()
    total_tokens = loss_mask.sum().cuda()

    loss_mask = loss_mask.view(-1).float().cuda()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

    masked_kl = torch.sum(loss_mask * kl_term.view(-1).cuda())
    masked_ratios = torch.sum(loss_mask * ratios.view(-1).cuda())
    masked_entropy = torch.sum(loss_mask * entropy_term.view(-1).cuda())
    masked_truncated_from_above = torch.sum(
        loss_mask * truncated_from_above.float().view(-1).cuda()
    )
    masked_truncated_from_below = torch.sum(
        loss_mask * truncated_from_below.float().view(-1).cuda()
    )

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=partial(rerun_state_machine.is_spiky_loss, threshold=SPIKY_LOSS_PERC),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    reporting_kl = torch.cat([masked_kl.clone().detach().view(1), total_tokens.view(1)])
    reporting_ratios = torch.cat([masked_ratios.clone().detach().view(1), total_tokens.view(1)])
    reporting_entropy = torch.cat([masked_entropy.clone().detach().view(1), total_tokens.view(1)])
    reporting_truncated_from_above = torch.cat(
        [masked_truncated_from_above.clone().detach().view(1), total_tokens.view(1)]
    )
    reporting_truncated_from_below = torch.cat(
        [masked_truncated_from_below.clone().detach().view(1), total_tokens.view(1)]
    )

    return (
        loss[0] * args.context_parallel_size,
        total_tokens.int(),
        {
            'lm loss': loss.clone().detach(),
            'rl/kl_term': reporting_kl,
            'rl/pi_over_pi_old': reporting_ratios,
            'rl/entropy_term': reporting_entropy,
            'rl/truncated_from_above': reporting_truncated_from_above,
            'rl/truncated_from_below': reporting_truncated_from_below,
        },
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        (
            tokens,
            advantages,
            old_logprobs,
            loss_mask,
            position_ids,
            ref_logprobs,
            inference_logprobs,
        ) = next(data_iterator)
    timers('batch-generator').stop()

    tokens = tokens.cuda()
    position_ids = position_ids.cuda()
    old_logprobs = old_logprobs.cuda()
    ref_logprobs = ref_logprobs.cuda()
    advantages = advantages.cuda()
    inference_logprobs = (
        inference_logprobs.cuda() if args.rl_inference_logprobs_is_correction else None
    )

    with stimer:
        current_logprobs = get_logprobs(model, tokens, position_ids, None, no_grad=False)
        loss, kl_term, ratios, entropy_term, truncated_from_above, truncated_from_below = (
            calculate_grpo_loss(
                current_logprobs=current_logprobs,
                old_logprobs=old_logprobs,
                ref_logprobs=ref_logprobs,
                advantages=advantages,
                clamp_eps_lower=args.grpo_clamp_eps_lower,
                clamp_eps_upper=args.grpo_clamp_eps_upper,
                kl_beta=args.grpo_kl_beta,
                entropy_weight=args.grpo_entropy_term_weight,
                inference_logprobs=inference_logprobs,
                is_truncation_coef=args.rl_importance_sampling_truncation_coef,
            )
        )

    # loss_mask will not be applied to 0th token as we do not have a logprob for it.
    return loss, partial(
        loss_func,
        loss_mask[:, 1:].contiguous(),
        kl_term,
        ratios,
        entropy_term,
        truncated_from_above,
        truncated_from_below,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """For GRPO, use lightweight minimal datasets instead of heavyweight mocks."""
    del train_val_test_num_samples
    args = get_args()

    class MinimalDataset:
        def __init__(self, size=1):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # Return empty tensors with expected shapes but minimal memory footprint
            return {
                'text': torch.ones(1, dtype=torch.long),  # Just a single token
                'tokens': torch.ones(1, dtype=torch.long),
                'labels': torch.ones(1, dtype=torch.long),
                'position_ids': torch.zeros(1, dtype=torch.long),
                'attention_mask': torch.ones(1, dtype=torch.bool),
                'loss_mask': torch.ones(1, dtype=torch.float),
            }

    # Create minimal datasets instead of None
    train_ds = MinimalDataset(
        size=(
            (args.global_batch_size * args.train_iters) if args.train_iters else args.train_samples
        )
    )
    valid_ds = MinimalDataset(
        size=(args.eval_iters * args.global_batch_size)
        * (
            (
                args.train_iters
                if args.train_iters
                else (args.train_samples // args.global_batch_size)
            )
            // args.eval_interval
        )
    )
    test_ds = MinimalDataset()

    print_rank_0("> finished creating minimal datasets for RL")
    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    def _model_builder(args, pre_process, post_process, vp_stage=None):
        if getattr(args, "is_hybrid_model", False):
            return mamba_builder(args, pre_process, post_process, vp_stage)
        else:
            return _gpt_builder(args, pre_process, post_process, vp_stage)

    pretrain(
        train_valid_test_datasets_provider,  # This is currently a mock dataset that will be overriden inside of the train_step call for rl.
        partial(model_provider, _model_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
