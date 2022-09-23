# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Evaluation utilities."""
from collections import OrderedDict
import math
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from megatron import get_args, print_rank_0
from megatron.core import mpu
from megatron.utils import average_losses_across_data_parallel_group
from tasks.finetune_utils import build_data_loader

def task_collate_fn(batch_data):
    # generate batch
    batch_size = len(batch_data)
    tensorized = OrderedDict()
    for d in batch_data:
        for k, v in d.items():
            tensorized.setdefault(k, []).append(v)

    tensorized['query'] = torch.LongTensor(tensorized['query'])
    tensorized['query_mask'] = torch.LongTensor(tensorized['query_mask'])
    tensorized['query_types'] = torch.LongTensor(tensorized['query_types'])
    tensorized['query_pad_mask'] = \
        torch.LongTensor(tensorized['query_pad_mask'])

    tensorized['context'] = torch.LongTensor(tensorized['context'])
    tensorized['context_mask'] = \
        torch.LongTensor(tensorized['context_mask'])
    tensorized['context_types'] = \
        torch.LongTensor(tensorized['context_types'])
    tensorized['context_pad_mask'] = \
        torch.LongTensor(tensorized['context_pad_mask'])

    if 'neg_context' in tensorized:
        tensorized['neg_context'] = \
            torch.LongTensor(np.concatenate(tensorized['neg_context']))
        tensorized['neg_context_mask'] = \
            torch.LongTensor(np.concatenate(tensorized['neg_context_mask']))
        tensorized['neg_context_types'] = \
            torch.LongTensor(np.concatenate(tensorized['neg_context_types']))

    return tensorized



def process_batch(batch):
    """Process batch and produce inputs for the model."""
    query_tokens = batch['query'].long().cuda()
    query_mask = (batch['query_mask'] < 0.5).cuda()
    query_types = batch['query_types'].long().cuda()
    query_pad_mask = batch['query_pad_mask'].long().cuda()

    context_tokens = batch['context'].long().cuda()
    context_mask = (batch['context_mask'] < 0.5).cuda()
    context_types = batch['context_types'].long().cuda()
    context_pad_mask = batch['context_pad_mask'].long().cuda()

    if 'neg_context' in batch:
        neg_context_tokens = batch['neg_context'].long().cuda()
        neg_context_mask = (batch['neg_context_mask'] < 0.5).cuda()
        neg_context_types = batch['neg_context_types'].long().cuda()
    else:
        neg_context_tokens = None
        neg_context_mask = None
        neg_context_types = None

    reference = batch['reference']

    return query_tokens, query_mask, query_types, query_pad_mask, \
           context_tokens, context_mask, context_types, context_pad_mask, \
           neg_context_tokens, neg_context_mask, neg_context_types, reference

def accuracy_func_provider(single_dataset_provider, rank0sampler=False):
    """Provide function that calculates accuracies."""
    args = get_args()

    print_rank_0("accuracy_func_provider is CALLED")

    # Build dataloaders
    datapath = args.valid_data
    dataset = single_dataset_provider(datapath)

    drop_last = False
    if mpu.get_data_parallel_world_size() > 1 and not rank0sampler:
        drop_last = True

    print_rank_0(datapath)
    print_rank_0(rank0sampler)

    dataloader = build_data_loader(dataset,
                                   args.eval_micro_batch_size,
                                   num_workers=args.num_workers,
                                   drop_last=drop_last,
                                   task_collate_fn=task_collate_fn)
    dataloaders = (dataset.dataset_name, dataloader)

    def metrics_func(model, epoch, output_predictions=False):
        print_rank_0('calculating metrics by accuracy func in ORQA...')

        if output_predictions:
            assert rank0sampler
            names = 'predictions'
        name, dataloader = dataloaders
        if args.task == "RET-FINETUNE-NQ":
            start_time = time.time()
            output = retrieval_loss(model, dataloader)
            stats_dict, total = output
            format_string = ""
            for k, v in stats_dict.items():
                format_string += "|{} = {:.2f}".format(k, v / total)
            print_rank_0("epoch:{}{}".format(epoch, format_string))
            print_rank_0("taken time to calcuate metrics {:.3f}".format(\
                time.time() - start_time))
        else:
            raise AssertionError("{} Task not supported".format(args.task))

    return metrics_func


def retrieval_loss(model, dataloader):
    args = get_args()
    total = 0
    topk_stats_dict = {'top{}_acc'.format(k): 0 for k in \
        args.retriever_report_topk_accuracies}
    stats_dict = dict(rank=0, **topk_stats_dict)

    assert len(model) == 1
    unwrapped_model = model[0]
    unwrapped_model.eval()

    with torch.no_grad():
        # For all the batches in the dataset.
        for batch in dataloader:
            # Run the model forward.
            query_tokens, query_mask, query_types, _, \
            context_tokens, context_mask, context_types, _, \
            neg_context_tokens, neg_context_mask, neg_context_types, \
            reference = process_batch(batch)

            query_logits, context_logits = unwrapped_model(query_tokens,
                query_mask, query_types,
                torch.cat([context_tokens, neg_context_tokens]),
                torch.cat([context_mask, neg_context_mask]),
                torch.cat([context_types, neg_context_types]))

            retrieval_scores = torch.matmul(query_logits,
                                    torch.transpose(context_logits, 0, 1))

            if args.retriever_score_scaling:
                retrieval_scores = retrieval_scores / \
                    math.sqrt(args.hidden_size)

            local_batch_size = query_logits.shape[0]
            labels = torch.arange(local_batch_size).long().cuda()

            softmax_scores = F.softmax(retrieval_scores, dim=1)
            sorted_vals, sorted_indices = torch.topk(softmax_scores,
                                                     k=softmax_scores.shape[1],
                                                     sorted=True)

            def topk_accuracy(k):
                return torch.cuda.FloatTensor(
                    [sum([int(labels[i] in sorted_indices[i, :k]) for i in \
                        range(local_batch_size)])])

            def get_rank():
                return torch.cuda.FloatTensor(
                    [sum([torch.nonzero(labels[i] == sorted_indices[i])[0][0] \
                        for i in range(local_batch_size)])])

            topk_accs = [topk_accuracy(k) for k in \
                args.retriever_report_topk_accuracies]
            rank = get_rank()
            losses = average_losses_across_data_parallel_group([rank, \
                *topk_accs])

            # create stats_dict with retrieval loss and all specified
            # top-k accuracies
            topk_acc_dict = {'top{}_acc'.format(k): v * 100 for k, v in \
                zip(args.retriever_report_topk_accuracies, losses[1:])}
            temp_stats_dict = dict(rank=losses[0], **topk_acc_dict)
            for k in stats_dict.keys():
                stats_dict[k] += temp_stats_dict[k]
            total += local_batch_size

    unwrapped_model.train()

    return stats_dict, total
