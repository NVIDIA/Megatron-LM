# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.


"""Fine-tune GPT"""

import torch
from functools import partial
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))
from megatron import get_args
from megatron import get_timers
from megatron import get_tokenizer
from megatron import print_rank_0
from megatron.core import mpu
from megatron.data.blendable_dataset import BlendableDataset
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel
from megatron.arguments import core_transformer_config_from_args
from megatron.core.enums import ModelType
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    config = core_transformer_config_from_args(args)

    print_rank_0('building GPT model ...')
    model = GPTModel(
        config=config,
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
    data_b = mpu.broadcast_data(keys, data, datatype)

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


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds1, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating finetuning GPT datasets ...")

    _, valid_ds, _ = build_train_valid_test_datasets(
        data_prefix=args.data_path2,
        data_impl="mmap",
        splits_string="98,2,0",
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=2048,
        seed=1234,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating pretrained GPT datasets ...")

    return train_ds, valid_ds, test_ds


def add_validation_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='validation set')
    group.add_argument('--data-path2', nargs='*', default=None,
                       help='Path to the validation dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ...')
    group.add_argument('--eval-ppl', action='store_true', default=False)
    group.add_argument('--stored_params', type=dict, default=dict())
    return parser


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step, args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
             extra_args_provider=add_validation_args,)
