# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import json
import numpy as np
import torch
from tqdm import tqdm

from megatron import get_args, print_rank_0
from tools.retro.external_libs import h5py
from tools.retro.utils import get_gpt_tokenizer


class DBDataset(torch.utils.data.Dataset):
    '''Dataset for iterating chunks.

    Requires:
    - List of indexed datasets
    - Chunk index array, with format:
        [dataset_idx, doc_id, start_idx, end_idx, bert_length])
    '''

    def __init__(self, db_path, indexed_datasets, chunks, max_chunk_length):

        assert chunks.shape[1] == 5, "expected 5 columns (dataset_idx, " \
        "doc_idx, token_start_idx, token_end_idx, bert_chunk_length); " \
        "found %d columns." % chunks.shape[1]

        self.db_path = db_path
        self.indexed_datasets = indexed_datasets
        self.chunks = chunks
        self.doc_chunk_map = None

        self.max_chunk_length = max_chunk_length
        self.eod_token_id = get_gpt_tokenizer().eod

    def __len__(self):
        return self.chunks.shape[0]

    def __getitem__(self, chunk_id):

        # Chunk start/end indexes.
        indexed_dataset_id, doc_id, token_start_idx, token_end_idx, _ = \
            [ value.item() for value in self.chunks[chunk_id] ]
        chunk_length = token_end_idx - token_start_idx
        indexed_dataset = self.indexed_datasets[indexed_dataset_id]

        # Chunk token ids.
        token_ids = indexed_dataset.get(doc_id,
                                        offset=token_start_idx,
                                        length=chunk_length)

        # Extend chunks to max_chunk_length by padding with EOD tokens.
        if chunk_length != self.max_chunk_length:
            assert chunk_length < self.max_chunk_length, "invalid chunk len."
            token_ids = token_ids.tolist()
            token_ids += [self.eod_token_id] * \
                (self.max_chunk_length - chunk_length)

        return {
            "doc_id" : doc_id,
            "text" : np.array(token_ids, dtype=np.int64),
        }

    def load_doc_tuples(self):
        '''Load the dataset & document ids.

        Load the dataset id & document id of each chunk in the database, to
        be used for causality filtering during querying.
        '''
        self.doc_tuples = np.zeros(shape=(len(self), 2), dtype="uint32")
        block_size = int(1e6)
        for start_idx in tqdm(range(0, len(self), block_size)):
            end_idx = min(len(self), start_idx + block_size)
            self.doc_tuples[start_idx:end_idx]=self.chunks[start_idx:end_idx,:2]
