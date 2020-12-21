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

from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import mpu
from megatron.data.dataset_utils import build_train_valid_test_datasets
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group
from megatron.model.realm_model import general_ict_model_provider
from megatron.data.realm_dataset_utils import get_ict_batch


def pretrain_ict_model_provider():
    args = get_args()
    return general_ict_model_provider(False, False)


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


def forward_step(data_iterator, model, input_tensor):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    query_tokens, query_pad_mask, \
    block_tokens, block_pad_mask, block_indices = get_ict_batch(data_iterator)
    timers('batch-generator').stop()


    # Forward model.
    query_logits, block_logits = model(query_tokens, query_pad_mask, block_tokens, block_pad_mask)
    micro_batch_size = query_logits.shape[0]
    global_batch_size = dist.get_world_size() * micro_batch_size  # recall we assert that tensor_model_parallel_size == 1

    all_query_logits = AllgatherFromDataParallelRegion.apply(query_logits)
    all_block_logits = AllgatherFromDataParallelRegion.apply(block_logits)

    # scores are inner products between query and block embeddings
    retrieval_scores = all_query_logits.float().matmul(torch.transpose(all_block_logits, 0, 1).float())
    softmaxed = F.softmax(retrieval_scores, dim=1)
    sorted_vals, sorted_indices = torch.topk(softmaxed, k=softmaxed.shape[1], sorted=True)

    def topk_accuracy(k):
        return torch.cuda.FloatTensor([sum([int(i in sorted_indices[i, :k]) for i in range(global_batch_size)]) / global_batch_size])

    topk_accs = [topk_accuracy(int(k)) for k in args.report_topk_accuracies]
    retrieval_loss = torch.nn.CrossEntropyLoss()(retrieval_scores, torch.arange(global_batch_size).long().cuda())
    retrieval_loss = retrieval_loss.float()
    averaged_losses = average_losses_across_data_parallel_group([retrieval_loss, *topk_accs])

    # create stats_dict with retrieval loss and all specified top-k accuracies
    topk_acc_dict = {'top{}_acc'.format(k): v for k, v in zip(args.report_topk_accuracies, averaged_losses[1:])}
    stats_dict = dict(retrieval_loss=averaged_losses[0], **topk_acc_dict)

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
    pretrain(train_valid_test_datasets_provider, pretrain_ict_model_provider, forward_step,
             args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
