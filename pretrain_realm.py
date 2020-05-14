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

from indexer import load_ict_checkpoint, get_ict_dataset
from megatron.data.realm_index import BlockData, RandProjectionLSHIndex, FaissMIPSIndex
from megatron import get_args
from megatron import get_timers
from megatron import mpu
from megatron import print_rank_0
from megatron.data.dataset_utils import build_train_valid_test_datasets
from megatron.model import REALMBertModel, REALMRetriever
from megatron.training import pretrain
from megatron.utils import reduce_losses

num_batches = 0


def model_provider():
    """Build the model."""
    args = get_args()
    print_rank_0('building REALM models ...')

    ict_model = load_ict_checkpoint()
    ict_dataset = get_ict_dataset(use_titles=False)
    all_block_data = BlockData.load_from_file(args.block_data_path)
    # hashed_index = RandProjectionLSHIndex.load_from_file(args.block_index_path)
    hashed_index = FaissMIPSIndex(index_type='flat_ip', embed_size=128)
    hashed_index.add_block_embed_data(all_block_data)

    # top_k + 1 because we may need to exclude trivial candidate
    retriever = REALMRetriever(ict_model, ict_dataset, all_block_data, hashed_index, args.block_top_k + 1)
    model = REALMBertModel(retriever)

    return model


def get_batch(data_iterator):
    # Items and their type.
    keys = ['tokens', 'labels', 'loss_mask', 'pad_mask', 'query_block_indices']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is None:
        data = None
    else:
        data = next(data_iterator)
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens = data_b['tokens'].long()
    labels = data_b['labels'].long()
    loss_mask = data_b['loss_mask'].long()
    pad_mask = data_b['pad_mask'].long()
    query_block_indices = data_b['query_block_indices'].long()

    return tokens, labels, loss_mask, pad_mask, query_block_indices


def get_qa_batch(data_iterator):
    question_tokens, question_attention_mask, answer_tokens, answer_token_lengths = next(data_iterator)
    return question_tokens, question_attention_mask, answer_tokens, answer_token_lengths


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, loss_mask, pad_mask, query_block_indices = get_batch(data_iterator)
    timers('batch generator').stop()

    # Forward model.
    lm_logits, block_probs = model(tokens, pad_mask, query_block_indices)
    with torch.no_grad():
        retrieval_utility = get_retrieval_utility(lm_logits, labels, loss_mask)

    # P(y|x) = sum_z(P(y|z, x) * P(z|x))
    block_probs = block_probs.unsqueeze(2).unsqueeze(3).expand_as(lm_logits)
    lm_logits = torch.sum(lm_logits * block_probs, dim=1)[:, :labels.shape[1]]

    lm_loss_ = mpu.vocab_parallel_cross_entropy(lm_logits.contiguous().float(),
                                                labels.contiguous())
    lm_loss = torch.sum(
        lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    reduced_loss = reduce_losses([lm_loss, retrieval_utility])
    torch.cuda.synchronize()
    return lm_loss, {'lm_loss': reduced_loss[0], 'retrieval_utility': reduced_loss[1]}


def get_retrieval_utility(lm_logits, labels, loss_mask):
    """log P(y | z, x) - log P(y | null, x)"""
    # [batch x seq_len x vocab_size]
    null_block_lm_logits = lm_logits[:, -1, :, :]
    null_block_loss_ = mpu.vocab_parallel_cross_entropy(null_block_lm_logits.contiguous().float(),
                                                       labels.contiguous())
    null_block_loss = torch.sum(
        null_block_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    retrieved_block_losses = []
    for block_num in range(lm_logits.shape[1] - 1):
        retrieved_block_lm_logits = lm_logits[:, block_num, :, :]
        retrieved_block_loss_ = mpu.vocab_parallel_cross_entropy(retrieved_block_lm_logits.contiguous().float(),
                                                                 labels.contiguous())
        retrieved_block_loss = torch.sum(
            retrieved_block_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()
        retrieved_block_losses.append(retrieved_block_loss)
    avg_retrieved_block_loss = torch.sum(retrieved_block_losses) / (lm_logits.shape[1] - 1)

    retrieval_utility = null_block_loss - avg_retrieved_block_loss
    return retrieval_utility


def qa_forward_step(data_iterator, model):
    timers = get_timers()

    # this dataset interface needs to be implemented
    timers('batch generator').start()
    question_tokens, question_attention_mask, answer_tokens, answer_token_lengths = get_qa_batch(data_iterator)
    timers('batch generator').stop()

    batch_span_logits, batch_loss_masks, block_probs = model(question_tokens, question_attention_mask,
                                                             answer_tokens, answer_token_lengths)
    # [batch_size x k x num_spans]
    block_probs = block_probs.unsqueeze(2).expand_as(batch_span_logits)
    batch_span_probs = F.softmax(batch_span_logits, dim=2)
    reduced_block_span_probs = torch.sum(batch_span_probs * block_probs, dim=1)
    qa_span_loss_ = -torch.log(reduced_block_span_probs)
    qa_span_loss = torch.sum(
        qa_span_loss_.view(-1) * batch_loss_masks
    )


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
        dataset_type='realm')
    print_rank_0("> finished creating BERT ICT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
             args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
