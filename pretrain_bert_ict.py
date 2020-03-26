# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain BERT for Inverse Cloze Task"""

import torch
import torch.nn.functional as F

from configure_data import configure_data
from megatron import mpu
from megatron.model import ICTBertModel
from megatron.utils import print_rank_0
from megatron.utils import reduce_losses
from megatron.utils import vocab_size_with_padding
from megatron.training import run


def model_provider(args):
    """Build the model."""

    print_rank_0('building BERT models ...')

    model = ICTBertModel(
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        embedding_dropout_prob=args.hidden_dropout,
        attention_dropout_prob=args.attention_dropout,
        output_dropout_prob=args.hidden_dropout,
        max_sequence_length=args.max_position_embeddings,
        checkpoint_activations=args.checkpoint_activations,
        ict_head_size=128,
        checkpoint_num_layers=args.checkpoint_num_layers,
        layernorm_epsilon=args.layernorm_epsilon,
        num_tokentypes=args.tokentype_size,
        parallel_output=True,
        apply_query_key_layer_scaling=args.apply_query_key_layer_scaling,
        attention_softmax_in_fp32=args.attention_softmax_in_fp32)

    return model


def get_batch(data_iterator, timers):

    # Items and their type.
    keys = ['input_text', 'input_types', 'input_pad_mask',
            'context_text', 'context_types', 'context_pad_mask']
    datatype = torch.int64

    # Broadcast data.
    timers('data loader').start()
    if data_iterator is None:
        data = None
    else:
        data = next(data_iterator)

    timers('data loader').stop()
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    input_tokens = data_b['input_text'].long()
    input_types = data_b['input_types'].long()
    input_pad_mask = data_b['input_pad_mask'].long()
    context_tokens = data_b['context_text'].long()
    context_types = data_b['context_types'].long()
    context_pad_mask = data_b['context_pad_mask'].long()

    return input_tokens, input_types, input_pad_mask,\
           context_tokens, context_types, context_pad_mask


def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    input_tokens, input_types, input_pad_mask,\
    context_tokens, context_types, context_pad_mask = get_batch(data_iterator, timers)
    timers('batch generator').stop()

    # Forward model.
    retrieval_scores = model(input_tokens, 1 - input_pad_mask, input_types,
                             context_tokens, 1 - context_pad_mask, context_types)

    softmaxed = F.softmax(retrieval_scores, dim=0).float()
    retrieval_loss = F.cross_entropy(softmaxed, torch.arange(softmaxed.size()[0]))

    reduced_losses = reduce_losses([retrieval_loss])

    return retrieval_loss, {'retrieval loss': reduced_losses[0]}


def get_train_val_test_data(args):
    """Load the data on rank zero and boradcast number of tokens to all GPUS."""

    (train_data, val_data, test_data) = (None, None, None)

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0:
        if (args.data_loader == 'raw'
                or args.data_loader == 'lazy'
                or args.data_loader == 'tfrecords'):
            data_config = configure_data()
            ds_type = 'BERT'
            data_config.set_defaults(data_set_type=ds_type, transpose=False)
            (train_data, val_data, test_data), tokenizer = data_config.apply(args)
            num_tokens = vocab_size_with_padding(tokenizer.num_tokens, args)
            # Need to broadcast num_tokens and num_type_tokens.
            token_counts = torch.cuda.LongTensor([num_tokens,
                                                  tokenizer.num_type_tokens,
                                                  int(args.do_train),
                                                  int(args.do_valid),
                                                  int(args.do_test)])
        else:
            print("Unsupported data loader for BERT.")
            exit(1)
    else:
        token_counts = torch.cuda.LongTensor([0, 0, 0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(token_counts,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    num_tokens = token_counts[0].item()
    num_type_tokens = token_counts[1].item()
    args.do_train = token_counts[2].item()
    args.do_valid = token_counts[3].item()
    args.do_test = token_counts[4].item()

    args.vocab_size = num_tokens
    args.tokentype_size = num_type_tokens

    return train_data, val_data, test_data


if __name__ == "__main__":

    run('Pretrain ICT BERT model', get_train_val_test_data,
        model_provider, forward_step)
