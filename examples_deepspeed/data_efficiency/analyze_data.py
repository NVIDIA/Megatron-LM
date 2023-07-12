# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

'''
Copyright 2022 The Microsoft DeepSpeed Team
'''

import os
import time
import sys
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir,os.path.pardir)))
from datetime import datetime
import numpy as np
import torch

from deepspeed.runtime.data_pipeline.data_sampling.data_analyzer \
    import DataAnalyzer
from deepspeed.runtime.data_pipeline.data_sampling.indexed_dataset \
    import MMapIndexedDataset

from megatron import get_args
from megatron import print_rank_0
from megatron.initialize import initialize_megatron

def get_tasks_args(parser):
    """Provide extra arguments required for data analyzing."""
    group = parser.add_argument_group(title='data_analyzing')

    group.add_argument('--analyzing-task', type=str, required=True,
                       default=None,
                       choices=['map',
                                'reduce'],
                       help='What type of analyzing task to perform.')
    group.add_argument('--analyzing-data-type', type=str, required=True,
                       default=None,
                       choices=['BERT',
                                'GPT'],
                       help='What type of data.')
    group.add_argument('--analyzing-metric', type=str, nargs='+', default=[],
                       help='What kinds of metrics to analyze.')
    group.add_argument('--analyzing-num-workers', type=int, default=1,
                       help='Number of workers. Each worker could be a single CPU node.')
    group.add_argument('--analyzing-worker-id', type=int, default=0,
                       help='Worker id of current node.')
    group.add_argument('--analyzing-num-threads', type=int, default=1,
                       help='Number of threads for each worker.')
    group.add_argument('--analyzing-num-threads-reduce', type=int, default=1,
                       help='Number of threads for each worker.')
    group.add_argument('--analyzing-specific-threads', type=int, nargs='+', default=[],
                       help='Which specific threads to run. Helpful when there are specific thread failed in previous run.')
    return parser

def train_valid_test_datasets_provider_gpt():
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    from megatron.data.gpt_dataset import build_train_valid_test_datasets
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=[1,1,1], # Just dummy numbers since we assume args.train_data_exact_num_epochs will override them
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds

def train_valid_test_datasets_provider_bert():
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for BERT ...')
    from megatron.data.dataset_utils import build_train_valid_test_datasets
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=[1,1,1], # Just dummy numbers since we assume args.train_data_exact_num_epochs will override them
        max_seq_length=args.seq_length,
        masked_lm_prob=args.mask_prob,
        short_seq_prob=args.short_seq_prob,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        binary_head=args.bert_binary_head)
    print_rank_0("> finished creating BERT datasets ...")

    return train_ds, valid_ds, test_ds

def metric_seqlen(data):
    metric = torch.count_nonzero(data['padding_mask'], dim=1)
    return metric

def metric_total_vocab_freq(data):
    args = get_args()
    if args.analyzing_data_type == 'BERT':
        frequency = torch.bincount(data['text'].view(-1),
            minlength=args.padded_vocab_size+1,
            weights=data['padding_mask'].view(-1))
    elif args.analyzing_data_type == 'GPT':
        frequency = torch.bincount(data['text'].view(-1),
            minlength=args.padded_vocab_size+1)
    return frequency

def metric_vocab_rarity(data):
    args = get_args()
    if args.analyzing_data_type == 'BERT':
        rarity = torch.sum(data['padding_mask'] * \
            args.total_vocab_freq[data['text']], dim=1).to(torch.long)
    elif args.analyzing_data_type == 'GPT':
        rarity = []
        # Do one by one to avoid too high memory consumption
        for row in range(data['text'].size()[0]):
            rarity.append(int(torch.sum(args.total_vocab_freq[data['text'][row]]).item()))
        rarity = torch.tensor(rarity, dtype=torch.long)
    print(f"rarity min {min(rarity)}, max {max(rarity)}, len {len(rarity)}, avg {sum(rarity)/len(rarity)}")
    return rarity

def metric_seqlen_vocab_rarity(data):
    args = get_args()
    metric = torch.count_nonzero(data['padding_mask'], dim=1).to(torch.long) * args.seqlen_coeff
    metric += torch.sum(data['padding_mask'] * \
        args.total_vocab_freq[data['text']], dim=1).to(torch.long)
    print(f"metric min {min(metric)}, max {max(metric)}, len {len(metric)}, avg {sum(metric)/len(metric)}")
    return metric

def get_metric_function(metric_name):
    if metric_name == 'seqlen':
        return metric_seqlen
    if metric_name == 'total_vocab_freq':
        return metric_total_vocab_freq
    if metric_name == 'vocab_rarity':
        return metric_vocab_rarity
    if metric_name == 'seqlen_vocab_rarity':
        return metric_seqlen_vocab_rarity

def get_metric_type(metric_name):
    if metric_name == 'seqlen':
        return 'single_value_per_sample'
    if metric_name == 'total_vocab_freq':
        return 'accumulate_value_over_samples'
    if metric_name == 'vocab_rarity':
        return 'single_value_per_sample'
    if metric_name == 'seqlen_vocab_rarity':
        return 'single_value_per_sample'

def run_map():
    args = get_args()
    if args.analyzing_data_type == 'BERT':
        args.mask_prob = 0 # When analyzing data, we don't want any mask.
        train_ds, _, _ = train_valid_test_datasets_provider_bert()
    elif args.analyzing_data_type == 'GPT':
        train_ds, _, _ = train_valid_test_datasets_provider_gpt()
        assert 'seqlen' not in args.analyzing_metric, 'GPT data has fixed seqlen, thus unnecessary to analyze seqlen metric.'
        assert 'seqlen_vocab_rarity' not in args.analyzing_metric, 'GPT data has fixed seqlen, thus unnecessary to analyze seqlen metric.'
    if 'vocab_rarity' in args.analyzing_metric or 'seqlen_vocab_rarity' in args.analyzing_metric:
        total_vocab_freq_fname = f"{args.save}/total_vocab_freq/total_vocab_freq_metric_value"
        assert os.path.isfile(f"{total_vocab_freq_fname}.bin") and os.path.isfile(f"{total_vocab_freq_fname}.idx"), "To analyze vocab rarity, first need to analyze the total vocab freq."
        total_vocab_freq = MMapIndexedDataset(total_vocab_freq_fname, skip_warmup=True)
        total_vocab_freq = np.copy(total_vocab_freq[0])
        total_vocab_freq[total_vocab_freq == 0] = 1 # Avoid log(0) error
        total_vocab_freq = np.log(total_vocab_freq/sum(total_vocab_freq)) * -1
        args.total_vocab_freq = torch.tensor(total_vocab_freq, dtype=torch.double)
        if 'seqlen_vocab_rarity' in args.analyzing_metric:
            # Use large coeff to make seqlen dominates vocab_rarity
            max_possible_rarity = args.seq_length * torch.max(args.total_vocab_freq).item()
            args.seqlen_coeff = 10 ** (math.ceil(math.log(max_possible_rarity, 10)) + 1)
            print(f"Metric seqlen_vocab_rarity: using {args.seqlen_coeff} as coefficient for seqlen.")
    metric_functions = [get_metric_function(x) for x in args.analyzing_metric]
    metric_types = [get_metric_type(x) for x in args.analyzing_metric]
    # For metric_dtypes we int64 by default since it could be hard to estimate
    # the appropriate dtype before the mapping analysis. During reduce where
    # we merge the analysis results, the DataAnalyzer will automatically choose
    # the dtype of merged result file as the smallest one that meet the range
    # requirement.
    metric_dtypes = [np.int64 for x in args.analyzing_metric]
    start = time.time()
    data_analyzer = DataAnalyzer(train_ds,
        num_workers=args.analyzing_num_workers,
        worker_id=args.analyzing_worker_id,
        num_threads=args.analyzing_num_threads,
        specific_threads=args.analyzing_specific_threads,
        batch_size=args.global_batch_size, metric_names=args.analyzing_metric,
        metric_functions=metric_functions, metric_types=metric_types,
        metric_dtypes=metric_dtypes, save_path=args.save)
    data_analyzer.run_map()
    duration = (time.time() - start) / 3600.0
    print(f"map job finished in {duration} hr.")

def run_reduce():
    args = get_args()
    if args.analyzing_data_type == 'BERT':
        args.mask_prob = 0 # When analyzing data, we don't want any mask.
        train_ds, _, _ = train_valid_test_datasets_provider_bert()
    elif args.analyzing_data_type == 'GPT':
        train_ds, _, _ = train_valid_test_datasets_provider_gpt()
    metric_functions = [get_metric_function(x) for x in args.analyzing_metric]
    metric_types = [get_metric_type(x) for x in args.analyzing_metric]
    metric_dtypes = [np.int64 for x in args.analyzing_metric]
    start = time.time()
    data_analyzer = DataAnalyzer(train_ds,
        num_workers=args.analyzing_num_workers,
        num_threads=args.analyzing_num_threads,
        num_threads_reduce=args.analyzing_num_threads_reduce,
        batch_size=args.global_batch_size, metric_names=args.analyzing_metric,
        metric_functions=metric_functions, metric_types=metric_types,
        metric_dtypes=metric_dtypes, save_path=args.save)
    data_analyzer.run_reduce()
    duration = (time.time() - start) / 3600.0
    print(f"reduce job finished in {duration} hr.")

if __name__ == "__main__":
    initialize_megatron(extra_args_provider=get_tasks_args, allow_no_cuda=True)
    args = get_args()
    if args.analyzing_task == 'map':
        run_map()
    elif args.analyzing_task == 'reduce':
        run_reduce()
    else:
        raise NotImplementedError('Task {} is not implemented.'.format(
            args.analyzing_task))
