# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

import numpy as np
import torch

from megatron import get_args, get_tokenizer


class BertEmbeddingDataset(torch.utils.data.Dataset):
    '''Dataset to convert a text dataset to Bert tokens.'''

    def __init__(self, text_dataset, max_seq_length):

        super().__init__()

        args = get_args()

        # Dataset, tokenizer.
        self.text_dataset = text_dataset
        self.max_seq_length = max_seq_length
        self.bert_tokenizer = get_tokenizer()

    def __len__(self):
        return len(self.text_dataset)

    @classmethod
    def build_sample(cls, tokenizer, token_ids):
        get_constant_array = lambda c : np.full((len(token_ids) + 2,), c, "int64")
        return {
            "text" : np.array([ tokenizer.cls, *token_ids, tokenizer.sep ], dtype="int64"),
            "types" : get_constant_array(0),
            "labels" : get_constant_array(-1),
            "is_random" : 0,
            "loss_mask" : get_constant_array(0),
            "padding_mask" : get_constant_array(1),
            "truncated" : 0,
        }

    def __getitem__(self, idx):

        # Text.
        text_sample = self.text_dataset[idx]
        text = text_sample["text"]
        text = text.replace("<|endoftext|>", "")

        # Bert/Wordpiece tokens (+truncate).
        bert_token_ids = self.bert_tokenizer.tokenize(text)
        bert_token_ids = bert_token_ids[:self.max_seq_length - 2] # cls+sep.
        if not bert_token_ids:
            bert_token_ids = [ self.bert_tokenizer.pad_id ] # hack when empty seq

        # Bert sample.
        sample = self.build_sample(self.bert_tokenizer, bert_token_ids)

        return sample
