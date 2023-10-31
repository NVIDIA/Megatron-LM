# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain Retro."""

from functools import partial
import torch

from megatron import get_args, get_retro_args
from megatron import get_timers
from megatron import get_tokenizer
from megatron import print_rank_0
from megatron.arguments import core_transformer_config_from_args
from megatron.core import tensor_parallel
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset
from megatron.core.enums import ModelType
from megatron.core.models.retro import get_retro_decoder_block_spec, RetroModel
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from tools.retro.query.retro_dataset import get_retro_datasets

from pretrain_gpt import loss_func, model_provider as default_model_provider


def core_model_provider(pre_process=True, post_process=True):
    """Build the model using Megatron-Core."""

    args = get_args()
    config = core_transformer_config_from_args(args)

    # NOTE: Experimental customization featuress
    if args.spec is not None:
        block_spec = import_module(args.spec)()
    else:
        block_spec = get_retro_decoder_block_spec(config, use_transformer_engine=True)

    print_rank_0('building GPT model ...')
    model = RetroModel(
        config=config,
        transformer_layer_spec=block_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent
    )
    return model


def model_provider(pre_process=True, post_process=True):
    """Build the model.

    Select between two different model classes:
      1. Default model (uses megatron/models/gpt_model.py).
      2. Core model (uses megatron/core/models/retro/model.py).
    """

    args = get_args()
    provider = core_model_provider if args.use_mcore_models else default_model_provider
    return provider(pre_process=pre_process, post_process=post_process)


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    retro_args = get_retro_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text', 'neighbor_tokens']
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
    _, _, neighbor_position_ids = get_ltor_masks_and_position_ids(
        neighbor_tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)
    neighbor_attention_mask = None

    return tokens, labels, loss_mask, attention_mask, position_ids, \
           neighbor_tokens, neighbor_attention_mask, neighbor_position_ids


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids, \
        neighbor_tokens, neighbor_attention_mask, neighbor_position_ids = \
            get_batch(data_iterator)
    timers('batch-generator').stop()

    # Model call.
    if args.use_mcore_models:
        forward_kwargs = {
            "context_input_ids" : neighbor_tokens,
            "context_position_ids" : neighbor_position_ids,
            "context_mask" : neighbor_attention_mask,
        }
    else:
        forward_kwargs = {
            "retriever_input_ids" : neighbor_tokens,
            "retriever_position_ids" : neighbor_position_ids,
            "retriever_attn_mask" : neighbor_attention_mask,
        }

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels, **forward_kwargs)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    return get_retro_datasets()


if __name__ == "__main__":

    # Temporary for transitiont to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.retro_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                            'retro_add_retriever': True})
