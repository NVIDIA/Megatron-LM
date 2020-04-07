import random
import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from megatron import get_tokenizer
from megatron import print_rank_0
from megatron import mpu
from megatron.data import helpers

class InverseClozeDataset(Dataset):
    """Dataset containing sentences and various 'blocks' for an inverse cloze task."""
    def __init__(self, name, indexed_dataset, data_prefix,
                 num_epochs, max_num_samples, max_seq_length,
                 short_seq_prob, seed):
        self.name = name
        self.seed = seed
        self.max_seq_length = max_seq_length
        self.indexed_dataset = indexed_dataset
        self.short_seq_prob = short_seq_prob

        tokenizer = get_tokenizer()
        self.vocab_id_list = list(tokenizer.inv_vocab.keys())
        self.vocab_id_to_token_list = tokenizer.inv_vocab
        self.cls_id = tokenizer.cls
        self.sep_id = tokenizer.sep
        self.mask_id = tokenizer.mask
        self.pad_id = tokenizer.pad

    def __len__(self):
        return self.indexed_dataset.doc_idx.shape[0]

    def __getitem__(self, idx):
        # get rng state corresponding to index (allows deterministic random pair)
        rng = random.Random(idx + 20000 + self.seed)
        np_rng = np.random.RandomState(seed=[rng.randint(0, 2**32-1) for _ in range(16)])

        # get seq length. Save 2 tokens for beginning and end
        target_seq_length = self.max_seq_length - 2
        if rng.random() < self.short_seq_prob:
            target_seq_length = rng.randint(5, target_seq_length)

        input_data, context_data = self.get_input_and_context(target_seq_length, rng, np_rng)
        input_tokens, input_token_types, input_pad_mask = input_data
        context_tokens, context_token_types, context_pad_mask = context_data

        sample = {
            'input_text': np.array(input_tokens),
            'input_types': np.array(input_token_types),
            'input_pad_mask': np.array(input_pad_mask),
            'context_text': np.array(context_tokens),
            'context_types': np.array(context_token_types),
            'context_pad_mask': np.array(context_pad_mask)
        }

        return sample

    def get_sentence_split_doc(self, idx):
        """fetch document at index idx and split into sentences"""
        doc_start = self.indexed_dataset.doc_idx[idx]
        doc_end = self.indexed_dataset.doc_idx[idx + 1]

        doc_sentences_array = self.indexed_dataset[doc_start:doc_end]
        doc_sentences = [list(arr) for arr in doc_sentences_array]

        return doc_sentences

    def concat_and_pad_tokens(self, tokens):
        """concat with special tokens and pad sequence to self.max_seq_length"""
        tokens = [self.cls_id] + tokens + [self.sep_id]
        assert len(tokens) <= self.max_seq_length

        num_pad = self.max_seq_length - len(tokens)
        pad_mask = [0] * len(tokens) + [1] * num_pad
        tokens += [self.pad_id] * num_pad
        token_types = [0] * self.max_seq_length
        return tokens, token_types, pad_mask

    def get_input_and_context(self, target_seq_length, rng, np_rng):
        """fetches a sentence and its surrounding context"""
        num_tries = 0
        while num_tries < 20:
            num_tries += 1
            doc = None
            while doc is None:
                doc_idx = np_rng.randint(len(self) - 1)
                # doc is a list of sentences
                doc = self.get_sentence_split_doc(doc_idx)
                if not doc:
                    doc = None

            num_sentences = len(doc)
            padless_max_len = self.max_seq_length - 2

            # select a random sentence from the document as input
            # TODO: consider adding multiple input sentences.
            input_sentence_idx = rng.randint(0, num_sentences - 1)
            input_tokens = doc[input_sentence_idx][:target_seq_length]
            if not len(input_tokens) > 0:
                continue

            context_tokens = []
            # 10% of the time, the input sentence is left in the context.
            # The other 90% of the time, keep it out.
            if rng.random() < 0.1:
                context_tokens = input_tokens.copy()

            view_preceding = True
            view_radius = 1
            while len(context_tokens) < padless_max_len:
                # keep adding sentences while the context can accommodate more.
                if view_preceding:
                    examine_idx = input_sentence_idx - view_radius
                    if examine_idx >= 0:
                        new_tokens = doc[examine_idx]
                        context_tokens = new_tokens + context_tokens
                else:
                    examine_idx = input_sentence_idx + view_radius
                    if examine_idx < num_sentences:
                        new_tokens = doc[examine_idx]
                        context_tokens += new_tokens
                    view_radius += 1
                view_preceding = not view_preceding
                if view_radius > num_sentences:
                    break

            # assemble the tokens and token types of the context
            context_tokens = context_tokens[:padless_max_len]
            if not len(context_tokens) > 0:
                continue

            # concatenate 'CLS' and 'SEP' tokens and add extra token types
            input_tokens, input_token_types, input_pad_mask = self.concat_and_pad_tokens(input_tokens)
            context_tokens, context_token_types, context_pad_mask = self.concat_and_pad_tokens(context_tokens)

            return (input_tokens, input_token_types, input_pad_mask), \
                   (context_tokens, context_token_types, context_pad_mask)
        else:
            raise RuntimeError("Could not get a valid data point from InverseClozeDataset")


