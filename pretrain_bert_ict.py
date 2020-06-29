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
import torch.distributed as dist
import torch.nn.functional as F

from megatron import get_args, print_rank_0
from megatron import get_timers
from megatron import mpu
from megatron.data.dataset_utils import build_train_valid_test_datasets
from megatron.model import ICTBertModel
from megatron.training import pretrain
from megatron.utils import reduce_losses

num_batches = 0


def general_model_provider(only_query_model=False, only_block_model=False):
    """Build the model."""
    args = get_args()
    assert args.ict_head_size is not None, \
        "Need to specify --ict-head-size to provide an ICTBertModel"

    assert args.model_parallel_size == 1, \
        "Model parallel size > 1 not supported for ICT"

    print_rank_0('building ICTBertModel...')

    # simpler to just keep using 2 tokentypes since the LM we initialize with has 2 tokentypes
    model = ICTBertModel(
        ict_head_size=args.ict_head_size,
        num_tokentypes=2,
        parallel_output=True,
        only_query_model=only_query_model,
        only_block_model=only_block_model)

    return model


def model_provider():
    return general_model_provider(False, False)


def get_batch(data_iterator):
    # Items and their type.
    keys = ['query_tokens', 'query_pad_mask',
            'block_tokens', 'block_pad_mask', 'block_data']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is None:
        data = None
    else:
        data = next(data_iterator)
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    query_tokens = data_b['query_tokens'].long()
    query_pad_mask = data_b['query_pad_mask'].long()
    block_tokens = data_b['block_tokens'].long()
    block_pad_mask = data_b['block_pad_mask'].long()
    block_indices = data_b['block_data'].long()

    return query_tokens, query_pad_mask,\
           block_tokens, block_pad_mask, block_indices


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch generator').start()
    query_tokens, query_pad_mask, \
    block_tokens, block_pad_mask, block_indices = get_batch(data_iterator)
    timers('batch generator').stop()

    # Forward model.
    query_logits, block_logits = model(query_tokens, query_pad_mask, block_tokens, block_pad_mask)

    data_parallel_size = dist.get_world_size() / args.model_parallel_size
    batch_size = query_logits.shape[0]
    global_batch_size = int(batch_size * data_parallel_size)

    all_logits_shape = (int(global_batch_size), int(query_logits.shape[1]))
    all_query_logits = torch.cuda.FloatTensor(*all_logits_shape).type(query_logits.dtype).fill_(0.0)
    all_block_logits = all_query_logits.clone()

    # record this processes' data
    all_query_logits[args.rank * batch_size:(args.rank + 1) * batch_size] = query_logits
    all_block_logits[args.rank * batch_size:(args.rank + 1) * batch_size] = block_logits

    # merge data from all processes
    dist.all_reduce(all_query_logits)
    dist.all_reduce(all_block_logits)

    # scores are inner products between query and block embeddings
    retrieval_scores = all_query_logits.float().matmul(torch.transpose(all_block_logits, 0, 1).float())
    softmaxed = F.softmax(retrieval_scores, dim=1)
    sorted_vals, sorted_indices = torch.topk(softmaxed, k=softmaxed.shape[1], sorted=True)

    def topk_acc(k):
        return torch.cuda.FloatTensor([sum([int(i in sorted_indices[i, :k]) for i in range(global_batch_size)]) / global_batch_size])
    top_accs = [topk_acc(k) for k in [1, 8, 20, 100]]

    retrieval_loss = torch.nn.CrossEntropyLoss()(retrieval_scores, torch.arange(global_batch_size).long().cuda())
    reduced_losses = reduce_losses([retrieval_loss, *top_accs])
    stats_dict = {
        'retrieval loss': reduced_losses[0],
        'top1_acc': reduced_losses[1],
        'top8_acc': reduced_losses[2],
        'top20_acc': reduced_losses[3],
        'top100_acc': reduced_losses[4],
    }

    return retrieval_loss, stats_dict


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid and test datasets."""
    args = get_args()
    print_rank_0('> building train, validation, and test datasets '
                 'for BERT ICT...')

    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        max_seq_length=args.seq_length,
        masked_lm_prob=args.mask_prob,
        short_seq_prob=args.short_seq_prob,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        dataset_type='ict')
    print_rank_0("> finished creating BERT ICT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
             args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
