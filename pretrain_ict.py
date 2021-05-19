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

from functools import partial
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import mpu
from megatron.data.biencoder_dataset_utils import get_ict_batch
from megatron.data.dataset_utils import build_train_valid_test_datasets
from megatron.model.biencoder_model import biencoder_model_provider
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group


def pretrain_ict_model_provider(pre_process=True, post_process=True):
    args = get_args()

    model = biencoder_model_provider(
                only_context_model=False,
                only_query_model=False,
                biencoder_shared_query_context_model=\
                args.biencoder_shared_query_context_model,
                pre_process=pre_process, post_process=post_process)

    return model

def get_group_world_size_rank():

    group = mpu.get_data_parallel_group()
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)

    return group, rank, world_size


class AllgatherFromDataParallelRegion(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_):
        assert input_.dim() == 2
        group, rank, world_size = get_group_world_size_rank()

        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        tensor_list[rank] = input_
        torch.distributed.all_gather(tensor_list, input_, group=group)

        output = torch.cat(tensor_list, dim=0).contiguous()

        return output


    @staticmethod
    def backward(ctx, grad_output):
        group, rank, world_size = get_group_world_size_rank()

        assert grad_output.shape[0] % world_size == 0
        dim_size = grad_output.shape[0] // world_size
        output_list = torch.split(grad_output, dim_size, dim=0)

        # get chunk from this rank
        output = output_list[rank].contiguous()
        return output

def loss_func(output_tensor):
    args = get_args()
    query_logits, context_logits = output_tensor

    micro_batch_size = query_logits.shape[0]
    # recall we assert that tensor_model_parallel_size == 1
    assert mpu.get_tensor_model_parallel_world_size() == 1, \
        "Model parallel size > 1 not supported for ICT"

    global_batch_size = dist.get_world_size() * micro_batch_size
    all_query_logits = AllgatherFromDataParallelRegion.apply(query_logits)
    all_context_logits = AllgatherFromDataParallelRegion.apply(context_logits)

    # scores are inner products between query and context embeddings
    retrieval_scores = torch.matmul(all_query_logits,
                        torch.transpose(all_context_logits, 0, 1))
    # scaling the retriever scores
    if args.retriever_score_scaling:
        retrieval_scores = retrieval_scores / math.sqrt(args.hidden_size)

    softmax_scores = F.log_softmax(retrieval_scores, dim=1)
    sorted_vals, sorted_indices = torch.topk(softmax_scores,
                                    k=softmax_scores.shape[1], sorted=True)

    def topk_accuracy(k):
        return torch.cuda.FloatTensor([sum([int(i in sorted_indices[i, :k]) \
            for i in range(global_batch_size)]) / global_batch_size])

    topk_accs = [topk_accuracy(int(k)) for k in args.retriever_report_topk_accuracies]

    labels = torch.arange(global_batch_size).long().cuda()
    loss = F.nll_loss(softmax_scores, labels, reduction='mean')
    reduced_losses = average_losses_across_data_parallel_group([loss, *topk_accs])

    # Scale the retrieval loss
    loss = loss * mpu.get_data_parallel_world_size()

    # create stats_dict with retrieval loss and all specified top-k accuracies
    topk_acc_dict = {'top{}_acc'.format(k): v * 100 for k, v in \
                        zip(args.retriever_report_topk_accuracies, reduced_losses[1:])}
    stats_dict = dict(loss=reduced_losses[0], **topk_acc_dict)
    return loss, stats_dict



def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    query_tokens, query_mask, \
    context_tokens, context_mask, context_indices = get_ict_batch(data_iterator)
    timers('batch-generator').stop()

    # Query and Context Types
    query_types = torch.cuda.LongTensor(*query_tokens.shape).fill_(0)
    context_types = torch.cuda.LongTensor(*context_tokens.shape).fill_(0)

    # Forward model.
    output_tensor = model(query_tokens, query_mask, query_types, context_tokens,
                        context_mask, context_types)

    return output_tensor, partial(loss_func)

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
        binary_head=False,
        dataset_type='ict')
    print_rank_0("> finished creating BERT ICT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    pretrain(train_valid_test_datasets_provider,
             pretrain_ict_model_provider,
             forward_step,
             args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
