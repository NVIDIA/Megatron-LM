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

"""
 Data Loader for Google NQ dataset
"""

from abc import ABC
import csv
from collections import OrderedDict
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, BatchSampler

from megatron import print_rank_0, get_args, get_tokenizer, mpu
from megatron.data.biencoder_dataset_utils import make_attention_mask

def get_nq_dataset(qa_data, split):
    args = get_args()
    tokenizer = get_tokenizer()

    dataset = NQDataset('Google NQ {} Split'.format(split),
                        'Google Natural Questions',
                        qa_data,
                        tokenizer,
                        args.retriever_seq_length)
    return dataset


def process_nq_batch(batch):
    query_tokens = batch['token_ids'].long().cuda()
    query_mask = (batch['token_mask'] < 0.5).cuda()
    query_types = batch['token_types'].long().cuda()
    query_len = batch['seq_len'].long().cuda()
    reference = batch['reference']

    return query_tokens, query_mask, query_types, query_len, reference


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, eval=False, **kwargs):
        if kwargs.get('collate_fn', None) is None:
            kwargs['collate_fn'] = self._collate_fn
        self.eval = eval
        super().__init__(dataset, **kwargs)

    def _collate_fn(self, batch_data):
        # generate batch
        batch_size = len(batch_data)
        tensorized = OrderedDict()
        for d in batch_data:
            for k, v in d.items():
                tensorized.setdefault(k, []).append(v)
        assert len(tensorized) == 5

        tensorized['token_ids'] = torch.LongTensor(tensorized['token_ids'])
        tensorized['token_mask'] = torch.LongTensor(tensorized['token_mask'])
        tensorized['token_types'] = torch.LongTensor(tensorized['token_types'])
        tensorized['seq_len'] = torch.LongTensor(tensorized['seq_len'])
        return tensorized


def get_one_epoch_nq_dataloader(dataset, micro_batch_size=None):
    """Data loader. Note that batch-size is the local (per GPU) batch-size.
       NOTE: This dataloader is not distributed !!!
    """

    args = get_args()
    if micro_batch_size is None:
        micro_batch_size = args.micro_batch_size
    num_workers = args.num_workers

    sampler = torch.utils.data.SequentialSampler(dataset)
    # importantly, drop_last must be False to get all the data.
    batch_sampler = BatchSampler(sampler,
                                 batch_size=micro_batch_size,
                                 drop_last=False)

    # Data loader. Note that batch size is the per GPU batch size.
    data_loader = CustomDataLoader(dataset,
                                   batch_sampler=batch_sampler,
                                   num_workers=num_workers,
                                   pin_memory=True)
    return data_loader


def build_tokens_types_paddings_from_text(src_text, tokenizer, max_seq_length):
    """Build token types and paddings, trim if needed, and pad if needed."""

    src_text_ids = tokenizer.tokenize(src_text)

    return build_tokens_types_paddings_from_ids(src_text_ids,
                                                max_seq_length,
                                                tokenizer.cls,
                                                tokenizer.sep,
                                                tokenizer.pad)


def build_tokens_types_paddings_from_ids(src_ids, max_seq_length, cls_id, \
    sep_id, pad_id):
    """
    Build token types and paddings, trim if needed, and pad if needed.

    TODO: Design modular interface to reuse this function. This is getting
    repeated multiple times in different tasks
    """

    enc_ids = []
    tokentypes_enc = []

    # [CLS].
    enc_ids.append(cls_id)
    tokentypes_enc.append(0)

    # A.
    len_src = len(src_ids)
    enc_ids.extend(src_ids)
    tokentypes_enc.extend([0] * len_src)

    # Cap the size.
    if len(enc_ids) > max_seq_length - 1:
        enc_ids = enc_ids[0: max_seq_length - 1]
        tokentypes_enc = tokentypes_enc[0: max_seq_length - 1]

    # [SEP].
    enc_ids.append(sep_id)
    tokentypes_enc.append(0)

    num_tokens_enc = len(enc_ids)
    # Padding.
    padding_length = max_seq_length - len(enc_ids)
    if padding_length > 0:
        enc_ids.extend([pad_id] * padding_length)
        tokentypes_enc.extend([pad_id] * padding_length)

    return enc_ids, tokentypes_enc, num_tokens_enc


def build_sample(token_ids, token_types, num_tokens, reference):
    """
    Convert to numpy and return a sample consumed by the
    batch producer.
    """

    token_ids = np.array(token_ids, dtype=np.int64)
    token_types = np.array(token_types, dtype=np.int64)
    token_mask = make_attention_mask(token_ids, token_ids)

    sample = ({
        'token_ids': token_ids,
        'token_mask': token_mask,
        'token_types': token_types,
        'seq_len': num_tokens,
        'reference': reference
    })
    return sample


class NQDataset(ABC, Dataset):
    """
    Open Retrieval Question Answering evaluation using Google NQ dataset.
    """

    def __init__(self, task_name, dataset_name, datapath,
                 tokenizer, max_seq_length):
        # Store inputs.
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        print_rank_0(' > building {} dataset for {}:'.format(self.task_name,
                                                             self.dataset_name))
        print_rank_0(datapath)
        self.samples = self.process_samples_from_single_path(datapath)
        print_rank_0('  >> total number of samples: {}'.format(\
                                                        len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]

        ques_tokens, tokentypes_enc, num_tokens_ques = \
            build_tokens_types_paddings_from_text(raw_sample['question'],
                self.tokenizer, self.max_seq_length)

        sample = build_sample(ques_tokens,
                              tokentypes_enc,
                              num_tokens_ques,
                              raw_sample['answers'])
        return sample

    @staticmethod
    def process_samples_from_single_path(filename):
        print_rank_0(' > Processing {} ...'.format(filename))
        samples = []
        total = 0

        with open(filename, 'r') as ifile:
            reader = csv.reader(ifile, delimiter='\t')
            for row in reader:
                question = row[0]
                answers = eval(row[1])

                sample = {'question': question, 'answers': answers}
                total += 1
                samples.append(sample)

                if total % 1000 == 0:
                    print_rank_0('  > processed {} so far ...'.format(total))

        print_rank_0(' >> processed {} samples.'.format(len(samples)))
        return samples
