# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain Retro."""

from functools import partial
import torch

from megatron import get_args, get_retro_args
from megatron import get_timers
from megatron import get_tokenizer
from megatron import print_rank_0
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from tools.retro.query.retro_dataset import get_retro_datasets

from pretrain_gpt import (
    loss_func,
    model_provider,
    core_gpt_dataset_config_from_args
)


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    retro_args = get_retro_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    if args.retro_add_retriever:
        keys += 'neighbor_tokens',

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

    if args.retro_add_retriever:
        # note: [bs * l * k, r]
        # note: 2x == neighbor, continuation
        neighbor_tokens = data_b['neighbor_tokens'] \
            .view(-1, retro_args.retro_gpt_retrieved_length).long()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    if args.retro_add_retriever:
        _, _, neighbor_position_ids = get_ltor_masks_and_position_ids(
            neighbor_tokens,
            tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)
        neighbor_attention_mask = None
        return tokens, labels, loss_mask, attention_mask, position_ids, \
               neighbor_tokens, neighbor_attention_mask, neighbor_position_ids
    else:
        return tokens, labels, loss_mask, attention_mask, position_ids


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    if args.retro_add_retriever:
        tokens, labels, loss_mask, attention_mask, position_ids, \
            neighbor_tokens, neighbor_attention_mask, neighbor_position_ids = \
                get_batch(data_iterator)
    else:
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator)
        neighbor_tokens, neighbor_attention_mask, neighbor_position_ids = \
            None, None, None
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          retriever_input_ids=neighbor_tokens,
                          retriever_position_ids=neighbor_position_ids,
                          retriever_attn_mask=neighbor_attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    if args.retro_add_retriever:
        return get_retro_datasets()
    else:
        print_rank_0("> building train, validation, and test datasets for GPT ...")

        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            GPTDataset,
            train_val_test_num_samples,
            core_gpt_dataset_config_from_args(args)
        ).build()

        print_rank_0("> finished creating GPT datasets ...")

        return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    # Temporary for transitiont to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.retro_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                            'retro_add_retriever': True})
