# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain Retro with Megatron Core"""

from megatron import get_args
from megatron.arguments import core_transformer_config_from_args
from megatron.core.enums import ModelType
from megatron.core.models.retro import get_retro_decoder_block_spec
from megatron.training import pretrain

from pretrain_gpt_core import model_provider as gpt_model_provider
from pretrain_retro import (
    forward_step,
    train_valid_test_datasets_provider,
)


def model_provider(pre_process=True, post_process=True):
    args = get_args()
    config = core_transformer_config_from_args(args)
    return gpt_model_provider(pre_process, post_process,
                              block_spec=get_retro_decoder_block_spec(config))


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}
    )
