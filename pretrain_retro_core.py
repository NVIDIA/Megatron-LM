# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain Retro"""

# import torch
# from functools import partial

from megatron import get_args
# from megatron import get_timers
# from megatron import get_tokenizer
from megatron import print_rank_0
from megatron.arguments import core_transformer_config_from_args
# from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
# from megatron.core.models.gpt import GPTModel
from megatron.core.models.retro import (
    get_decoder_model_spec,
    get_encoder_model_spec,
    RetroDecoderModel,
    RetroEncoderModel,
)
# from megatron.core.transformer.spec_utils import import_module
# from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.training import pretrain
# from megatron.utils import average_losses_across_data_parallel_group
# from megatron.utils import get_ltor_masks_and_position_ids

from pretrain_retro import (
    forward_step,
    train_valid_test_datasets_provider,
)

# >>>
from lutil import pax
# <<<


# def get_spec(encoder=None):
#     # NOTE: Experimental customization feature
#     args = get_args()
#     if args.model_spec is not None:
#         return import_module(args.model_spec)()
#     else:
#         return get_model_spec(encoder=encoder)


def get_encoder(config):
    args = get_args()
    return RetroEncoderModel(
        config=config,
        # spec=get_spec(None),
        spec=get_encoder_model_spec(),
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=True,
        post_process=False,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent
    )


def get_decoder(config, pre_process, post_process, encoder):
    args = get_args()
    return RetroDecoderModel(
        config=config,
        # spec=get_spec(encoder),
        spec=get_decoder_model_spec(encoder),
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        # retriever=retriever,
    )


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    args = get_args()
    config = core_transformer_config_from_args(args)

    print_rank_0('building Retro model ...')
    encoder = get_encoder(config)
    decoder = get_decoder(config, pre_process, post_process, encoder)

    pax("encoder", "decoder")

    return decoder


# def get_batch(data_iterator):
#     raise Exception("hi.")
#     """Generate a batch"""
#     args = get_args()
#     tokenizer = get_tokenizer()

#     # Items and their type.
#     keys = ['text']
#     datatype = torch.int64

#     # Broadcast data.
#     if data_iterator is not None:
#         data = next(data_iterator)
#     else:
#         data = None
#     data_b = tensor_parallel.broadcast_data(keys, data, datatype)

#     # Unpack.
#     tokens_ = data_b['text'].long()
#     labels = tokens_[:, 1:].contiguous()
#     tokens = tokens_[:, :-1].contiguous()

#     # Get the masks and postition ids.
#     attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
#         tokens,
#         tokenizer.eod,
#         args.reset_position_ids,
#         args.reset_attention_mask,
#         args.eod_mask_loss)

#     return tokens, labels, loss_mask, attention_mask, position_ids

# def loss_func(loss_mask, output_tensor):
#     raise Exception("hi.")
#     losses = output_tensor.float()
#     loss_mask = loss_mask.view(-1).float()
#     loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

#     # Reduce loss for logging.
#     averaged_loss = average_losses_across_data_parallel_group([loss])

#     return loss, {'lm loss': averaged_loss[0]}


# def forward_step(data_iterator, model):
#     raise Exception("hi.")
#     """Forward step."""
#     args = get_args()
#     timers = get_timers()

#     # Get the batch.
#     timers('batch-generator', log_level=2).start()
#     tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
#         data_iterator)
#     timers('batch-generator').stop()

#     output_tensor = model(tokens, position_ids, attention_mask,
#                           labels=labels)

#     return output_tensor, partial(loss_func, loss_mask)


# def train_valid_test_datasets_provider(train_val_test_num_samples):
#     raise Exception("hi.")
#     """Build train, valid, and test datasets."""
#     args = get_args()

#     print_rank_0('> building train, validation, and test datasets '
#                  'for Retro ...')
#     train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
#         data_prefix=args.data_path,
#         data_impl=args.data_impl,
#         splits_string=args.split,
#         train_valid_test_num_samples=train_val_test_num_samples,
#         seq_length=args.seq_length,
#         seed=args.seed,
#         skip_warmup=(not args.mmap_warmup),
#         train_data_prefix=args.train_data_path,
#         valid_data_prefix=args.valid_data_path,
#         test_data_prefix=args.test_data_path)
#     print_rank_0("> finished creating Retro datasets ...")

#     return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}
    )
