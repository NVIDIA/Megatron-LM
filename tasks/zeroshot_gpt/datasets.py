# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Zero-shot datasets."""

import json
import math

import numpy as np
import torch

from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from .detokenizer import get_detokenizer


def build_dataset(task):
    """Helper function to select and build dataset."""

    if task == 'LAMBADA':
        return _build_lambada_dataset()
    if task == 'WIKITEXT103':
        return _build_wikitext103_dataset()

    raise NotImplementedError('dataset for {} task is not '
                              'implemented.'.format(task))


class _LMDataset(torch.utils.data.Dataset):

    def __init__(self, tokens, seq_len, pad_idx, num_original_tokens,
                 num_tokenized_tokens, overalapping_eval=None):
        self.tokens = tokens
        self.seq_len = seq_len
        self.pad_idx = pad_idx
        self.overalapping_eval = overalapping_eval
        if self.overalapping_eval is None:
            self.overalapping_eval = self.seq_len
        self.overalapping_eval = max(1, self.overalapping_eval)
        self.num_original_tokens = num_original_tokens
        self.num_tokenized_tokens = num_tokenized_tokens
        self.total_targets = len(self.tokens) - 1
        # remove first sequence tokens
        targets = max(self.total_targets - self.overalapping_eval, 0)
        self.total_sequences = max(
            math.ceil(targets / self.overalapping_eval) + 1, 1)

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.overalapping_eval
        end_idx = start_idx + self.seq_len
        tokens = self.tokens[start_idx:end_idx + 1]
        num_tokens = len(tokens)
        pad_mask = [1] * num_tokens
        if num_tokens < self.seq_len + 1:
            num_pad = (self.seq_len + 1 - num_tokens)
            pad_mask += [0] * (num_pad)
            tokens += [self.pad_idx] * num_pad
        pad_mask = np.array(pad_mask[1:])
        if self.overalapping_eval != self.seq_len and idx != 0:
            pad_mask[:-self.overalapping_eval] *= 0

        return {'text': np.array(tokens), 'pad_mask': pad_mask}


class _LambadaDataset(torch.utils.data.Dataset):

    def __init__(self, path, pad_idx, tokenizer, seq_len, strict=False):
        print_rank_0('> building lambada dataset from {} ...'.format(path))
        self.seq_len = seq_len
        self.pad_idx = pad_idx
        self.tokenizer = tokenizer
        self.strict = strict

        self.tokens = []
        self.labels = []
        with open(path, 'r') as f:
            for line in f.readlines():
                text = json.loads(line)['text']
                tokens, labels = self.get_tokens(text)
                self.tokens.append(tokens)
                self.labels.append(labels)

    def get_tokens(self, text):
        if not self.strict:
            tokens = self.tokenizer.tokenize(text)
            return tokens[:-1], [tokens[-1]]
        last_token = text.split()[-1]
        start_idx = text.rfind(last_token)
        beginning_tokens = self.tokenizer.tokenize(text[:start_idx].strip())
        last_token = self.tokenizer.tokenize(' ' + last_token)
        return beginning_tokens, last_token

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        num_tokens = len(tokens)
        pad_mask = [0] * num_tokens
        labels = self.labels[idx]
        pad_mask += [1] * len(labels)
        tokens = tokens + labels
        num_tokens = len(tokens)
        if num_tokens < self.seq_len + 1:
            num_pad = (self.seq_len + 1 - num_tokens)
            pad_mask += [0] * (num_pad)
            tokens += [self.pad_idx] * num_pad
        pad_mask = np.array(pad_mask[1:])

        return {'text': np.array(tokens), 'pad_mask': pad_mask}


def _build_lambada_dataset():
    """Build lambada dataset."""
    args = get_args()
    tokenizer = get_tokenizer()

    assert len(args.valid_data) == 1
    val_dataset = _LambadaDataset(args.valid_data[0], tokenizer.eod, tokenizer,
                                  args.seq_length, args.strict_lambada)
    print_rank_0(' > found {} samples.'.format(len(val_dataset)))

    return val_dataset


def _build_wikitext103_dataset():
    """"""
    args = get_args()
    tokenizer = get_tokenizer()

    assert len(args.valid_data) == 1
    with open(args.valid_data[0], "rb") as reader:
        entire_data = reader.read().decode('utf-8')
    num_original_tokens = len(entire_data.strip().split(" "))
    entire_data = get_detokenizer(args.valid_data[0])(entire_data)
    tokenized_data = tokenizer.tokenize(entire_data)
    num_tokenized_tokens = len(tokenized_data)

    val_dataset = _LMDataset(tokenized_data, args.seq_length, tokenizer.eod,
                             num_original_tokens, num_tokenized_tokens,
                             args.overlapping_eval)
    print_rank_0(' > number of original tokens: {}, number of detokenized '
                 'tokens: {}'.format(num_original_tokens, num_tokenized_tokens))

    return val_dataset
