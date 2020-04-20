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

from megatron import get_args
from megatron import get_timers
from megatron import mpu
from megatron import print_rank_0
from megatron.data.bert_dataset import build_train_valid_test_datasets
from megatron.model import ICTBertModel, REALMBertModel
from megatron.training import pretrain
from megatron.utils import reduce_losses

num_batches = 0

def model_provider():
    """Build the model."""
    args = get_args()
    print_rank_0('building BERT models ...')

    realm_model = REALMBertModel(args.ict_model_path,
                                 args.block_hash_data_path)

    return ict_model


def get_batch(data_iterator):

    # Items and their type.
    keys = ['query_tokens', 'query_types', 'query_pad_mask']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is None:
        data = None
    else:
        data = next(data_iterator)
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    query_tokens = data_b['query_tokens'].long()
    query_types = data_b['query_types'].long()
    query_pad_mask = data_b['query_pad_mask'].long()

    return query_tokens, query_types, query_pad_mask


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers('batch generator').start()
    query_tokens, query_types, query_pad_mask = get_batch(data_iterator)
    timers('batch generator').stop()

    # Forward model.
    query_logits, block_logits = model(query_tokens, query_pad_mask, query_types,
                                       block_tokens, block_pad_mask, block_types).float()

    # [batch x h] * [h x batch]
    retrieval_scores = query_logits.matmul(torch.transpose(block_logits, 0, 1))
    softmaxed = F.softmax(retrieval_scores, dim=1)

    top5_vals, top5_indices = torch.topk(softmaxed, k=5, sorted=True)
    batch_size = softmaxed.shape[0]

    top1_acc = torch.cuda.FloatTensor([sum([int(top5_indices[i, 0] == i) for i in range(batch_size)]) / batch_size])
    top5_acc = torch.cuda.FloatTensor([sum([int(i in top5_indices[i]) for i in range(batch_size)]) / batch_size])

    retrieval_loss = F.cross_entropy(softmaxed, torch.arange(batch_size).cuda())
    reduced_losses = reduce_losses([retrieval_loss, top1_acc, top5_acc])

    return retrieval_loss, {'retrieval loss': reduced_losses[0],
                            'top1_acc': reduced_losses[1],
                            'top5_acc': reduced_losses[2]}


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid and test datasets."""
    args = get_args()
    print_rank_0('> building train, validation, and test datasets '
                 'for BERT ...')

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
        ict_dataset=True)
    print_rank_0("> finished creating BERT ICT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
             args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
