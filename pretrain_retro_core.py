# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain Retro with Megatron Core"""

from functools import partial

from megatron import get_args, print_rank_0
from megatron.arguments import core_transformer_config_from_args
from megatron.core.enums import ModelType
from megatron.core.models.retro import get_retro_decoder_block_spec, RetroModel
from megatron.training import pretrain

# from pretrain_gpt_core import model_provider as gpt_model_provider
from pretrain_retro import (
    forward_step,
    train_valid_test_datasets_provider,
)


# def model_provider(pre_process=True, post_process=True):
#     args = get_args()
#     config = core_transformer_config_from_args(args)
#     model = gpt_model_provider(pre_process, post_process,
#                                block_spec=get_retro_decoder_block_spec(config))
#     return model
def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    args = get_args()
    config = core_transformer_config_from_args(args)

    # NOTE: Experimental customization feature
    if args.block_spec is not None:
        block_spec_func = import_module(args.block_spec)
        block_spec = block_spec_func()
    else:
        block_spec = get_retro_decoder_block_spec(config)

    print_rank_0('building GPT model ...')
    model = RetroModel(
        config=config,
        spec=block_spec,
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


def get_forward_kwargs(input_ids, position_ids, attn_mask):
    return {
        "context_input_ids" : input_ids,
        "context_position_ids" : position_ids,
        "context_mask" : attn_mask,
    }


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             partial(forward_step, get_forward_kwargs=get_forward_kwargs),
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}
    )
