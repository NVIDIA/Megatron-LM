# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain T5"""

from functools import partial

import torch

from megatron.training import (
    get_args,
    get_timers,
    get_tokenizer,
    print_rank_0
)
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.models.T5 import T5Model
from megatron.training import pretrain
from megatron.training.utils import average_losses_across_data_parallel_group
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.t5_dataset import T5MaskedWordPieceDataset, T5MaskedWordPieceDatasetConfig
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.models.T5.t5_spec import (get_t5_encoder_with_transformer_engine_block_spec,
                                            get_t5_decoder_with_transformer_engine_block_spec,
                                            get_t5_encoder_with_local_block_spec,
                                            get_t5_decoder_with_local_block_spec)
from megatron.legacy.model import T5Model as NonCoreT5Model

"""
Pipeline parallelism for T5
(Caveat: currently, mcore T5 model has not supported pipeline-parallelism)
===========================

T5 is a model architecture with both encoder and decoder blocks.
Consequently, pipeline parallelism is implemented slightly differently
compared to architectures like GPT and BERT.

In particular, when pipeline_model_parallel_world_size > 1, each stage
either executes an encoder block or a decoder block. The
--pipeline-model-parallel-split-rank argument controls the rank at which
the split happens: all ranks lower than this argument execute the
encoder block, and all ranks equal to or higher than this argument value
execute the decoder block.

In the encoder section of the model, only one tensor is sent downstream:
the intermediate encoder_hidden_state. In the decoder section of the
model, two tensors are sent downstream in the forward pass: the fully
computed encoder_hidden_state, and the intermediate decoder_hidden_state.

In particular, these are the shapes of the tensors sent between
different workers:
    If rank is in decoder section:
        intermediate decoder_hidden_state (pre-transpose),
        complete encoder_hidden_state (post-transpose).
    If rank is at boundary between encoder and decoder sections:
        complete encoder_hidden_state (post-transpose).
    If rank is in encoder section:
        intermediate encoder_hidden_state (pre-transpose).

Additionally, we have code in the backward_step function in schedules.py
to accumulate the encoder_hidden_state gradient across skip connections
(encoder_hidden_state fed in as input to each layer in the decoder).
"""

def model_provider(pre_process=True, post_process=True, add_encoder=True, add_decoder=True) -> T5Model:
    """Builds the model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.
        add_encoder (bool, optional): Defaults to True
        add_decoder (bool, optional): Defaults to True
    Returns:
        T5Model: The returned T5 model
    """


    args = get_args()
    config = core_transformer_config_from_args(args)
    if args.use_mcore_models:
        if args.transformer_impl=="local":
            en_block_spec = get_t5_encoder_with_local_block_spec(args.encoder_num_layers)
            de_block_spec = get_t5_decoder_with_local_block_spec(args.decoder_num_layers)
        elif args.transformer_impl=="transformer_engine":
            en_block_spec = get_t5_encoder_with_transformer_engine_block_spec(args.encoder_num_layers)
            de_block_spec = get_t5_decoder_with_transformer_engine_block_spec(args.decoder_num_layers)
        print_rank_0('building T5 model ...')
        model = T5Model(
            config=config,
            transformer_encoder_layer_spec=en_block_spec,
            transformer_decoder_layer_spec=de_block_spec,
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
    else:
        model = NonCoreT5Model(config=config,
                        num_tokentypes=0,
                        parallel_output=True,
                        pre_process=pre_process,
                        post_process=post_process,
                        add_encoder=add_encoder,
                        add_decoder=add_decoder)
    return model


def get_batch(data_iterator):
    """Build the batch."""

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
    tokens_enc = data_b['text_enc'].long()
    tokens_dec = data_b['text_dec'].long()
    labels = data_b['labels'].long()
    loss_mask = data_b['loss_mask'].float()

    enc_mask = (data_b['enc_mask'] < 0.5)
    dec_mask = (data_b['dec_mask'] < 0.5)
    enc_dec_mask = (data_b['enc_dec_mask'] < 0.5)

    return tokens_enc, tokens_dec, loss_mask, labels, \
           enc_mask, dec_mask, enc_dec_mask


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
    """   
    lm_loss_ = output_tensor.float()
    lm_loss = torch.sum(
        lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    loss = lm_loss
    averaged_losses = average_losses_across_data_parallel_group([lm_loss])

    return loss, {'lm loss': averaged_losses[0]}


def forward_step(data_iterator, model: T5Model):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (T5Model): The T5 Model
    """

    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch generator', log_level=2).start()
    tokens_enc, tokens_dec, loss_mask, lm_labels, enc_mask, dec_mask, enc_dec_mask \
        = get_batch(data_iterator)
    timers('batch generator').stop()

    # Forward model lm_labels
    output_tensor = model(tokens_enc,
                        tokens_dec,
                        enc_mask,
                        dec_mask,
                        enc_dec_mask,
                        lm_labels=lm_labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples: int):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    tokenizer = get_tokenizer()

    config = T5MaskedWordPieceDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.encoder_seq_length,
        sequence_length_decoder=args.decoder_seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path)
        ],
        split=args.split,
        path_to_cache=args.data_cache_path,
        mock=False,
        tokenizer=tokenizer,
        masking_probability=args.mask_prob,
        short_sequence_probability=args.short_seq_prob,
        masking_max_ngram=10,
        masking_do_full_word=True,
        masking_do_permutation=False,
        masking_use_longer_ngrams=False,
        masking_use_geometric_distribution=True,
    )

    print_rank_0('> building train, validation, and test datasets '
                 'for T5 ...')

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        T5MaskedWordPieceDataset,
        train_val_test_num_samples,
        lambda: mpu.get_tensor_model_parallel_rank() == 0,
        config,
    ).build()

    print_rank_0("> finished creating T5 datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(train_valid_test_datasets_provider, model_provider, ModelType.encoder_and_decoder,
             forward_step, args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})