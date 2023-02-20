# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT"""

import torch
from functools import partial

from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import tensor_parallel, parallel_state
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel, ModelType, megablocks_utils
from megatron.model.megablocks_utils import moe
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    model = GPTModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids

def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}

def moe_loss_func(loss_mask, output_tensor=None):
    # NOTE: For pipeline parallelism this function will be run on the
    # non-final stages to calculate load balancing loss contribution
    # for the MoE layers within the stage. For these cases, output_tensor
    # will be None.
    loss, loss_dict = (None, {})
    if parallel_state.is_pipeline_last_stage():
        assert output_tensor is not None
        loss, loss_dict = loss_func(loss_mask, output_tensor)
        assert loss.numel() == 1

    # NOTE: If recompute is enabled we will collect duplicate load
    # balancing loss contributions. Prune these before calculating
    # the load balancing loss.
    args = get_args()
    if args.recompute_granularity is not None:
        # Ignore load balancing loss contributions compute during
        # the forward pass if recompute is turned on.
        load_balancing_loss_data = moe.get_load_balancing_loss()
        if args.num_layers * 2 == len(load_balancing_loss_data):
            load_balancing_loss_data = (
                load_balancing_loss_data[args.num_layers:])
            moe.clear_load_balancing_loss()
            moe.save_load_balancing_loss(load_balancing_loss_data)

    # Compute the load balancing loss for all MoE layers.
    megablocks_args = megablocks_utils.arguments.from_megatron(args)
    lbl = moe.batched_load_balancing_loss(megablocks_args)
    moe.clear_load_balancing_loss()

    # Average the load balancing loss across data parallel
    # replicas and save for logging.
    averaged_lbl = average_losses_across_data_parallel_group([lbl])
    loss_dict['load balancing loss'] = averaged_lbl[0]

    # Compute the total loss, if necessary.
    total_loss = loss + lbl if loss is not None else lbl
    return total_loss, loss_dict

def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    loss_fn = (
        moe_loss_func if args.moe_num_experts is not None else loss_func)
    return output_tensor, partial(loss_fn, loss_mask)

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path,)
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder, forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
