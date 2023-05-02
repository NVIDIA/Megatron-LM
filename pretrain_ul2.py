# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain UL2"""

import argparse
from functools import partial

import torch

from megatron import (
    get_args,
    get_timers,
    print_rank_0
)
from megatron.arguments import core_transformer_config_from_args
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.dataset_utils import build_train_valid_test_datasets
from megatron.data.ul2_dataset import (
    is_decoder_only as _is_decoder_only,
    is_prefix_lm as _is_prefix_lm,
)
from megatron.model import GPTModel, T5Model
from megatron.model.enums import UL2ModelType
from megatron.model.t5_model import t5_position_ids
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group


"""
Pipeline parallelism for UL2
============================

Since UL2 re-uses the T5 model architecture for encoder-decoder models
and the GPT model architecture for decoder-only models, please see their
documentation for more information.
"""


def is_decoder_only():
    """Return whether we use a decoder-only model."""
    args = get_args()
    return _is_decoder_only(args.ul2_model_type)


def is_prefix_lm():
    """Return whether we use a non-causal decoder-only model."""
    args = get_args()
    return _is_prefix_lm(args.ul2_model_type)


def model_provider(pre_process=True, post_process=True,
                   add_encoder=True, add_decoder=True):
    """Build the model."""

    print_rank_0('building UL2 model ...')
    config = core_transformer_config_from_args(get_args())
    if is_decoder_only():
        print_rank_0('Using decoder-only UL2 model.')
        model = GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
            prefix_lm=is_prefix_lm(),
        )
    else:
        print_rank_0('Using encoder-decoder UL2 model.')
        model = T5Model(config,
                        num_tokentypes=0,
                        parallel_output=True,
                        pre_process=pre_process,
                        post_process=post_process,
                        add_encoder=add_encoder,
                        add_decoder=add_decoder)
    return model


def get_batch(data_iterator):
    """Build the batch."""

    if is_decoder_only():
        keys = ['text', 'labels', 'loss_mask', 'dec_mask']
    else:
        keys = ['text_enc', 'text_dec', 'labels', 'loss_mask',
                'enc_mask', 'dec_mask', 'enc_dec_mask']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    if is_decoder_only():
        tokens = data_b['text'].long()
        labels = data_b['labels'].long()
        loss_mask = data_b['loss_mask'].float()

        dec_mask = (data_b['dec_mask'] < 0.5)
        dec_mask = dec_mask.unsqueeze(1)
        return tokens, loss_mask, labels, dec_mask
    else:
        tokens_enc = data_b['text_enc'].long()
        tokens_dec = data_b['text_dec'].long()
        labels = data_b['labels'].long()
        loss_mask = data_b['loss_mask'].float()

        enc_mask = (data_b['enc_mask'] < 0.5)
        dec_mask = (data_b['dec_mask'] < 0.5)
        enc_dec_mask = (data_b['enc_dec_mask'] < 0.5)

        return tokens_enc, tokens_dec, loss_mask, labels, \
               enc_mask, dec_mask, enc_dec_mask


def loss_func(loss_mask, output_tensor):
    lm_loss_ = output_tensor.float()
    lm_loss = torch.sum(
        lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    loss = lm_loss
    averaged_losses = average_losses_across_data_parallel_group([lm_loss])

    return loss, {'lm loss': averaged_losses[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch generator', log_level=2).start()
    if is_decoder_only():
        (tokens, loss_mask, lm_labels, dec_mask) = get_batch(data_iterator)
    else:
        (
            tokens_enc, tokens_dec, loss_mask, lm_labels,
            enc_mask, dec_mask, enc_dec_mask,
        ) = get_batch(data_iterator)
    timers('batch generator').stop()

    # Forward model lm_labels
    if is_decoder_only():
        position_ids = t5_position_ids(tokens)
        output_tensor = model(tokens, position_ids, dec_mask,
                              labels=lm_labels)
    else:
        output_tensor = model(tokens_enc,
                              tokens_dec,
                              enc_mask,
                              dec_mask,
                              enc_dec_mask,
                              tokentype_ids=None,
                              lm_labels=lm_labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for UL2 ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        max_seq_length=args.encoder_seq_length,
        max_seq_length_dec=args.decoder_seq_length,
        masked_lm_prob=args.mask_prob,
        short_seq_prob=args.short_seq_prob,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        dataset_type='ul2')
    print_rank_0("> finished creating UL2 datasets ...")

    return train_ds, valid_ds, test_ds


def extra_args_provider(parser):
    parser.add_argument('--_is_ul2', default=True, help=argparse.SUPPRESS)
    return parser


def model_type_fn():
    args = get_args()
    if args.ul2_model_type is UL2ModelType.encoder_decoder:
        return ModelType.encoder_and_decoder
    else:
        return ModelType.encoder_or_decoder


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider, model_type_fn,
             forward_step, extra_args_provider=extra_args_provider,
             args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
