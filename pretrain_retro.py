# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain Retro."""

from functools import partial
import torch

from megatron import get_args
from megatron import get_timers
from megatron import get_tokenizer
from megatron import print_rank_0
from megatron.arguments import core_transformer_config_from_args
from megatron.core import tensor_parallel
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.retro.query.retro_dataset import get_retro_datasets
from megatron.core.datasets.retro.query.multi_split_gpt_dataset import MultiSplitGPTDataset, MultiSplitGPTDatasetConfig
from megatron.core.enums import ModelType
from megatron.core.models.retro import get_retro_decoder_block_spec, RetroConfig, RetroModel
from megatron.core.models.retro.utils import get_all_true_mask
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from pretrain_gpt import (
    is_dataset_built_on_rank,
    loss_func,
    model_provider as default_model_provider,
    train_valid_test_datasets_provider as gpt_train_valid_test_datasets_provider,
)


def get_retro_config():
    return core_transformer_config_from_args(get_args(), RetroConfig)


def core_model_provider(pre_process=True, post_process=True):
    """Build the model using Megatron-Core."""

    args = get_args()
    config = get_retro_config()

    # NOTE: Experimental customization feature
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
    provider = core_model_provider if (args.use_mcore_models and args.retro_add_retriever) else default_model_provider
    model = provider(pre_process=pre_process, post_process=post_process)
    return model


def get_batch(data_iterator):
    """Generate a batch"""

    args = get_args()
    tokenizer = get_tokenizer()
    config = get_retro_config()

    # Items and their type.
    keys = ['text']
    if args.retro_add_retriever:
        keys.append('neighbor_tokens')
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

    if args.retro_add_retriever:
        # note: [bs * l * k, r]
        # note: 2x == neighbor, continuation
        neighbor_tokens = data_b['neighbor_tokens'] \
            .view(-1, config.retro_retrieved_length).long()
        _, _, neighbor_position_ids = get_ltor_masks_and_position_ids(
            neighbor_tokens,
            tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)
        neighbor_attention_mask = get_all_true_mask(
            (1, 1, config.retro_retrieved_length, config.retro_retrieved_length),
            neighbor_tokens.device)
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

    # Model call.
    if args.use_mcore_models:
        if args.retro_add_retriever:
            forward_kwargs = {
                "context_input_ids" : neighbor_tokens,
                "context_position_ids" : neighbor_position_ids,
                "context_mask" : neighbor_attention_mask,
            }
        else:
            forward_kwargs = {}
    else:
        forward_kwargs = {
            "retriever_input_ids" : neighbor_tokens,
            "retriever_position_ids" : neighbor_position_ids,
            "retriever_attn_mask" : neighbor_attention_mask,
        }

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels, **forward_kwargs)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_valid_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    # Dataset config.
    retro_config = get_retro_config()
    data_config = MultiSplitGPTDatasetConfig(
        is_built_on_rank=is_dataset_built_on_rank,
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=args.data_path,
        blend_per_split=[args.train_data_path, args.valid_data_path, args.test_data_path],
        split=args.split,
        split_preprocessing=retro_config.retro_split_preprocessing,
        path_to_cache=args.data_cache_path,
        return_document_ids=False,
        tokenizer=get_tokenizer(),
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        vocab_size=get_tokenizer().vocab_size,
        mock=args.mock_data,
    )

    # GPT datasets.
    print_rank_0(" > multi-split gpt datasets.")
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        MultiSplitGPTDataset,
        train_valid_test_num_samples,
        data_config,
    ).build()

    gpt_datasets = {
        "train" : (train_ds, train_valid_test_num_samples[0]),
        "valid" : (valid_ds, train_valid_test_num_samples[1]),
        "test"  : (test_ds, train_valid_test_num_samples[2]),
    }

    # Retro datasets.
    if args.retro_add_retriever:
        return get_retro_datasets(
            config=retro_config,
            gpt_datasets=gpt_datasets,
            sample_length=args.seq_length,
            eod_token_id=get_tokenizer().eod,
        )

    # Multi-split GPT datasets.
    else:
        return (
            gpt_datasets["train"][0],
            gpt_datasets["valid"][0],
            gpt_datasets["test"][0],
        )


if __name__ == "__main__":

    # Temporary for transition to core datasets.
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.retro_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
