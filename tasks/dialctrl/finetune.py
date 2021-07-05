
"""Controllable Dialogue Finetuning"""

import torch
from functools import partial
from megatron import get_args
from megatron import get_timers
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron import mpu
from megatron.model import GPTModel
from megatron.training import evaluate_and_print_results
from megatron.utils import average_losses_across_data_parallel_group
from tasks.finetune_utils import finetune
from tasks.dialctrl.data import build_train_valid_test_datasets
from tasks.dialctrl.utils import get_ltor_attention_masks_and_position_ids


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


def train_valid_datasets_provider():
    """Build train, valid, and test datasets for dialog/control module"""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets for %s module ...' % args.train_module)
    
    train_ds, valid_ds, _ = build_train_valid_test_datasets(
        data_folder=args.data_folder,
        dataset_name=args.dataset_name,
        train_module=args.train_module,
        max_seq_len=args.max_seq_len,
        seed=args.seed)
    print_rank_0("> finished creating datasets for %s module ..." % args.train_module)

    args.eval_interval = len(train_ds) // args.global_batch_size
    print_rank_0(' > evaluation interval: %d' % args.eval_interval)

    return train_ds, valid_ds


def process_batch(batch):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text', 'loss_mask']
    datatype = torch.int64

    data_b = mpu.broadcast_data(keys, batch, datatype)

    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    loss_mask = data_b['loss_mask'].float()

    # Get the attention_mask and postition ids.
    attention_mask, position_ids = \
        get_ltor_attention_masks_and_position_ids(tokens, tokenizer.eod_id)

    return tokens, labels, loss_mask, attention_mask, position_ids


def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(batch, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()
    
    try:
        batch_ = next(batch)
    except BaseException:
        batch_ = batch

    tokens, labels, loss_mask, attention_mask, position_ids = process_batch(batch_)

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def main():
    
    finetune(train_valid_datasets_provider, model_provider, \
             forward_step=forward_step)

