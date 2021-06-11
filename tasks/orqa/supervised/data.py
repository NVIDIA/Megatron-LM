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

"""ORQA dataset."""

import json
import random
from abc import ABC
from abc import abstractmethod

import numpy as np
from torch.utils.data import Dataset

from megatron import print_rank_0, get_args
from megatron.data.biencoder_dataset_utils import make_attention_mask

def build_token_types_from_context_list(ctx_list, tokenizer, max_seq_length):
    ctx_id_list, ctx_types_list = [], []
    for context in ctx_list:
        title_ids = tokenizer.tokenize(context['title'])
        ctx_ids = tokenizer.tokenize(context['text'])
        ctx_ids = title_ids + [tokenizer.sep_id] + ctx_ids

        ctx_ids, ctx_types, _ = build_tokens_types_paddings_from_ids(ctx_ids,
                                    max_seq_length, tokenizer.cls,
                                    tokenizer.sep, tokenizer.pad)
        ctx_id_list.append(ctx_ids)
        ctx_types_list.append(ctx_types)

    return ctx_id_list, ctx_types_list


def build_tokens_types_paddings_from_text(query, context,
                                          tokenizer, max_seq_length):
    """Build token types and paddings, trim if needed, and pad if needed."""

    query_ids = tokenizer.tokenize(query)
    query_ids, query_types, query_pad_mask = \
        build_tokens_types_paddings_from_ids(query_ids, max_seq_length, \
            tokenizer.cls, tokenizer.sep, tokenizer.pad)

    # Appending the title of the context at front
    extended_ctx_ids = None
    if context is not None:
        title_ids = tokenizer.tokenize(context['title'])
        ctx_ids = tokenizer.tokenize(context['text'])
        extended_ctx_ids = title_ids + [tokenizer.sep] + ctx_ids

    ctx_ids, ctx_types, ctx_pad_mask = \
        build_tokens_types_paddings_from_ids(extended_ctx_ids,
            max_seq_length, tokenizer.cls, tokenizer.sep, tokenizer.pad)

    return query_ids, query_types, query_pad_mask, \
           ctx_ids, ctx_types, ctx_pad_mask


# Similar code tasks/data_utils with some changes
def build_tokens_types_paddings_from_ids(text_ids, max_seq_length,
                                         cls_id, sep_id, pad_id):
    """Build token types and paddings, trim if needed, and pad if needed."""
    enc_ids = []
    tokentypes_enc = []

    # [CLS].
    enc_ids.append(cls_id)
    tokentypes_enc.append(0)

    # A.
    len_src = len(text_ids)
    enc_ids.extend(text_ids)
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

    pad_mask = ([1] * num_tokens_enc) + ([0] * padding_length)
    pad_mask = np.array(pad_mask, dtype=np.int64)

    return enc_ids, tokentypes_enc, pad_mask


def build_sample(query_ids, query_types, query_pad_mask,
                ctx_ids, ctx_types, ctx_pad_mask, answers,
                neg_ctx_id_list=None, neg_ctx_types_list=None,
                include_neg=False):
    """Convert to numpy and return a sample consumed by the batch producer."""

    query_ids = np.array(query_ids, dtype=np.int64)
    query_types = np.array(query_types, dtype=np.int64)
    query_mask = make_attention_mask(query_ids, query_ids)

    ctx_ids = np.array(ctx_ids, dtype=np.int64)
    ctx_types = np.array(ctx_types, dtype=np.int64)
    ctx_mask = make_attention_mask(ctx_ids, ctx_ids)

    sample = ({
        'query': query_ids,
        'query_mask': query_mask,
        'query_types': query_types,
        'query_pad_mask': query_pad_mask,
        'context': ctx_ids,
        'context_mask': ctx_mask,
        'context_types': ctx_types,
        'context_pad_mask': ctx_pad_mask,
        'reference': answers
    })

    if include_neg:
        neg_ctx_ids = np.array(neg_ctx_id_list, dtype=np.int64)
        neg_ctx_id_types = np.array(neg_ctx_types_list, dtype=np.int64)
        neg_ctx_mask = np.array([make_attention_mask(ids, ids) \
            for ids in neg_ctx_ids], dtype=np.int64)

        sample['neg_context'] = neg_ctx_ids
        sample['neg_context_types'] = neg_ctx_id_types
        sample['neg_context_mask'] = neg_ctx_mask

    return sample


class OpenRetrievalAbstractDataset(ABC, Dataset):
    """Open Retrieval base dataset class."""

    def __init__(self, task_name, dataset_name, datapaths, tokenizer, \
                max_seq_length, evaluate=False):
        # Store inputs.
        args = get_args()
        self.evaluate = evaluate
        self.val_av_rank_hard_neg = args.val_av_rank_hard_neg
        self.val_av_rank_other_neg = args.val_av_rank_other_neg
        self.train_with_neg = args.train_with_neg
        self.train_hard_neg = args.train_hard_neg

        self.task_name = task_name
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        print_rank_0(' > building {} dataset for {}:'.format(self.task_name,
                                                             self.dataset_name))
        # Process the files.
        string = '  > paths:'
        for path in datapaths:
            string += ' ' + path
        print_rank_0(string)
        self.samples = []
        for datapath in datapaths:
            self.samples.extend(self.process_samples_from_single_path(datapath))

        args = get_args()
        if args.sample_rate < 1:  # subsample
            k = int(len(self.samples) * args.sample_rate)
            self.samples = random.sample(self.samples, k)

        print_rank_0('  >> total number of samples: {}'.format(
            len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]

        query_ids, query_types, query_pad_mask, ctx_ids, ctx_types, \
            ctx_pad_mask = build_tokens_types_paddings_from_text( \
                raw_sample['question'], raw_sample['pos_context'], \
                self.tokenizer, self.max_seq_length)

        if self.evaluate:
            neg_ctx_list = \
                raw_sample['negative_context'][:self.val_av_rank_other_neg] + \
                raw_sample['hard_negative_context'][:self.val_av_rank_hard_neg]
            neg_ctx_id_list, neg_ctx_types_list = \
                build_token_types_from_context_list(neg_ctx_list, \
                    self.tokenizer, self.max_seq_length)

        elif self.train_with_neg:
            hard_negative_ctx = raw_sample['hard_negative_context']
            negative_ctx = raw_sample['negative_context']
            if True:  # TODO: fix this or remove this condition
                random.shuffle(hard_negative_ctx)
                random.shuffle(negative_ctx)

            neg_ctx_list = hard_negative_ctx[:self.train_hard_neg]
            # In the Google NQ dataset by DPR paper, there are around more than
            # 50 missing hard negatives in training data.
            # In those cases, substitute hard negatives by simple negatives.
            if len(neg_ctx_list) < self.train_hard_neg:
                neg_ctx_list += negative_ctx[:self.train_hard_neg - \
                    len(neg_ctx_list)]

            neg_ctx_id_list, neg_ctx_types_list = \
                build_token_types_from_context_list(neg_ctx_list,
                    self.tokenizer, self.max_seq_length)
        else:
            neg_ctx_id_list = None
            neg_ctx_types_list = None

        sample = build_sample(query_ids, query_types, query_pad_mask,
                              ctx_ids, ctx_types, ctx_pad_mask,
                              raw_sample['answers'],
                              neg_ctx_id_list, neg_ctx_types_list,
                              include_neg=self.evaluate or self.train_with_neg)

        return sample

    @staticmethod
    @abstractmethod
    def process_samples_from_single_path(filename):
        """Abstract method that takes a filename and
        returns a list of dataset samples, each sample being a dict of
            {'text': string, 'text': string}
        """
        pass



def normalize_question(question):
    if question[-1] == '?':
        question = question[:-1]
    return question

# The following class reads the datasets for training retriever as
# prepared by the DPR codebase (https://github.com/facebookresearch/DPR)

class NQSupervisedDataset(OpenRetrievalAbstractDataset):

    def __init__(self, name, datapaths, tokenizer, max_seq_length, \
                evaluate=False):
        super().__init__('natural_questions_ret',
                         name,
                         datapaths,
                         tokenizer,
                         max_seq_length,
                         evaluate=evaluate)

    @staticmethod
    def process_samples_from_single_path(filename):
        """"Implement abstract method."""
        print_rank_0(' > Processing {} ...'.format(filename))
        samples = []
        total = 0

        with open(filename, 'r', encoding="utf-8") as f:
            data = json.load(f)
            for row in data:
                question = normalize_question(row['question'])
                pos_context = row['positive_ctxs'][0]

                # Hard Negative Contexts
                if len(row['hard_negative_ctxs']) > 0:
                    hard_neg_context = row['hard_negative_ctxs']
                else:
                    hard_neg_context = []

                # Negative Contexts
                if len(row['negative_ctxs']) > 0:
                    neg_context = row['negative_ctxs']
                else:
                    neg_context = []

                answers = row['answers']
                sample = {'question': question,
                          'pos_context': pos_context,
                          'hard_negative_context': hard_neg_context,
                          'negative_context': neg_context,
                          'answers': answers}
                total += 1
                samples.append(sample)

                if total % 5000 == 0:
                    print_rank_0('  > processed {} so far ...'.format(total))

        print_rank_0(' >> processed {} samples.'.format(len(samples)))
        return samples

