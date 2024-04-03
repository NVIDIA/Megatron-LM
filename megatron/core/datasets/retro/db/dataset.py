# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""A DBDataset is for iterating the chunks of the chunk database.

This dataset is used for both training a vector index, and adding vectors to a
trained index.
"""

from typing import List

import numpy as np
import torch
from tqdm import tqdm

from megatron.core.datasets.indexed_dataset import IndexedDataset


class DBDataset(torch.utils.data.Dataset):
    """Dataset for iterating chunks.
    
    Args:
        db_path (str): Path of HDF5-format chunk database.
        indexed_datasets (List[IndexedDataset]): Indexed datasets used to build database.
        chunks (np.ndarray): Array of chunk indexes, for indexing into indexed datasets. Format [dataset_idx, doc_id, start_idx, end_idx, bert_length].
        chunk_length (int): Max GPT chunk length (e.g., 64).
        eod_token_id (int): EOD token ID.
    """

    def __init__(
        self,
        db_path: str,
        indexed_datasets: List[IndexedDataset],
        chunks: np.ndarray,
        chunk_length: int,
        eod_token_id: int,
    ):

        assert chunks.shape[1] == 5, (
            "expected 5 columns (dataset_idx, "
            "doc_idx, token_start_idx, token_end_idx, bert_chunk_length); "
            "found %d columns." % chunks.shape[1]
        )

        self.db_path = db_path
        self.indexed_datasets = indexed_datasets
        self.chunks = chunks
        self.doc_chunk_map = None

        self.max_chunk_length = chunk_length
        self.eod_token_id = eod_token_id

    def __len__(self) -> int:
        """Length of DB dataset.

        Returns:
            Number of chunks contained in the dataset.
        """
        return self.chunks.shape[0]

    def __getitem__(self, chunk_id: int) -> dict:
        """DB dataset sample.

        Args:
            chunk_id (int): Index of chunk within dataset.

        Returns:
            A dict containing:
            - 'doc_id': Document index within indexed dataset.
            - 'text': GPT token IDs.
        """

        # Chunk start/end indexes.
        indexed_dataset_id, doc_id, token_start_idx, token_end_idx, _ = [
            value.item() for value in self.chunks[chunk_id]
        ]
        chunk_length = token_end_idx - token_start_idx
        indexed_dataset = self.indexed_datasets[indexed_dataset_id]

        # Chunk token ids.
        token_ids = indexed_dataset.get(doc_id, offset=token_start_idx, length=chunk_length)

        # Extend chunks to max_chunk_length by padding with EOD tokens.
        if chunk_length != self.max_chunk_length:
            assert chunk_length < self.max_chunk_length, "invalid chunk len."
            token_ids = token_ids.tolist()
            token_ids += [self.eod_token_id] * (self.max_chunk_length - chunk_length)

        return {
            "doc_id": doc_id,
            "text": np.array(token_ids, dtype=np.int64),
        }

    def load_doc_tuples(self) -> None:
        """Load the dataset & document ids.

        Load the dataset id & document id of each chunk in the database, to
        be used for causality filtering during querying.
        """
        self.doc_tuples = np.zeros(shape=(len(self), 2), dtype="uint32")
        block_size = int(1e6)
        for start_idx in tqdm(
            range(0, len(self), block_size),
            "load doc tuples",
            miniters=(len(self) // block_size) // 10,
            disable=torch.distributed.get_rank() != 0,
        ):
            end_idx = min(len(self), start_idx + block_size)
            self.doc_tuples[start_idx:end_idx] = self.chunks[start_idx:end_idx, :2]
