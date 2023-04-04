# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import numpy as np
import os
import torch

from megatron import get_args, get_retro_args
from tools.bert_embedding.utils import get_index_path_map
from tools.retro.db.utils import get_merged_train_dataset as get_db_dataset
from tools.retro.external_libs import h5py

from .chunk_dataset import get_chunk_dataset_map


class RetroDataset(torch.utils.data.Dataset):
    '''Dataset of retro samples.

    Each sample contains the original GPT sample, along with the token IDs
    of each neighbor of each chunk within the sequence. Neighbor array has
    shape (num_chunks_per_sample, num_neighbors, num_retrieved_tokens).
    '''

    def __init__(self,
                 num_neighbors,
                 num_retrieved_chunks,
                 block_size,
                 db_dataset,
                 chunk_dataset,
                 neighbor_path_map):
        '''Note: chunk dataset wraps original GPT dataset (see
        chunk_dataset.py).'''

        super().__init__()

        self.num_neighbors = num_neighbors
        self.num_retrieved_chunks = num_retrieved_chunks
        self.block_size = block_size
        self.db_dataset = db_dataset
        self.chunk_dataset = chunk_dataset
        self.neighbor_path_map = neighbor_path_map

    def __len__(self):
        return len(self.chunk_dataset.sample_dataset)

    def __getitem__(self, sample_idx):

        n_chunks_per_sample = self.chunk_dataset.n_chunks_per_sample

        # Get standard sample.
        sample = self.chunk_dataset.sample_dataset[sample_idx]

        # Sample idx to chunk idxs.
        chunk_idxs = list(range(
            sample_idx * n_chunks_per_sample,
            (sample_idx + 1) * n_chunks_per_sample,
        ))

        # Collect retrieved tokens.
        all_retrieved_chunk_ids = []
        all_retrieved_token_ids = []
        for chunk_idx in chunk_idxs:

            # Neighbor chunk ids.
            neighbor_path = self.neighbor_path_map[chunk_idx]
            with h5py.File(neighbor_path, "r") as f:
                neighbor_chunk_ids = f["neighbors"] \
                    [chunk_idx % self.block_size, :self.num_neighbors].tolist()

            # Retrieved (neighbor + continuation) token ids.
            retrieved_chunk_ids = []
            retrieved_token_ids = []
            for neighbor_chunk_id in neighbor_chunk_ids:
                current_chunk_ids = [
                    i % len(self.db_dataset)
                    for i in range(
                            neighbor_chunk_id,
                            neighbor_chunk_id + self.num_retrieved_chunks)]
                current_token_ids = [self.db_dataset[ci]["text"]
                                     for ci in current_chunk_ids]
                retrieved_chunk_ids.append(current_chunk_ids)
                retrieved_token_ids.append(current_token_ids)

            # Collect retrieved tokens.
            all_retrieved_chunk_ids.append(retrieved_chunk_ids)
            all_retrieved_token_ids.append(retrieved_token_ids)

        # Reshape retrieved tokens.
        all_retrieved_chunk_ids = np.array(all_retrieved_chunk_ids) \
            .reshape((n_chunks_per_sample, self.num_neighbors, -1))
        all_retrieved_token_ids = np.array(all_retrieved_token_ids) \
            .reshape((n_chunks_per_sample, self.num_neighbors, -1))

        # Sample.
        sample = {
            **sample,
            "neighbor_chunks" : all_retrieved_chunk_ids,
            "neighbor_tokens" : all_retrieved_token_ids,
        }

        return sample


def get_retro_datasets():
    '''Get train, valid, test retro datasets.'''

    args = get_args()
    retro_args = get_retro_args()

    # DB dataset.
    db_dataset = get_db_dataset()

    # Retro datasets.
    chunk_ds_info_map = get_chunk_dataset_map()
    retro_dataset_map = {}
    for data_key, chunk_ds_info in chunk_ds_info_map.items():

        chunk_dataset = chunk_ds_info["data"]
        neighbor_dir = chunk_ds_info["neighbor_dir"]
        neighbor_path_map = get_index_path_map(neighbor_dir)

        # Verify dataset prefixes.
        sample_prefix = chunk_dataset.sample_dataset.datasets[0].index_prefix
        neighbor_prefix = os.path.basename(neighbor_dir)
        assert sample_prefix == neighbor_prefix, \
            "inconsistent dataset source; '%s' vs. '%s'." % \
            (sample_prefix, neighbor_prefix)

        # Verify num chunks.
        n_sample_chunks = len(chunk_dataset)
        n_neighbor_chunks = len(neighbor_path_map.id_index_map)

        if n_sample_chunks != n_neighbor_chunks:
            print("neighbor_dir : %s" % neighbor_dir)
            print("neighbor_path_map : %s" % neighbor_path_map)
            raise Exception("num sampled chunks (%d) != num neighbor chunks (%d)"
                            % (n_sample_chunks, n_neighbor_chunks))

        # Retro dataset.
        retro_dataset_map[data_key] = RetroDataset(
            num_neighbors=args.retro_num_neighbors,
            num_retrieved_chunks=args.retro_num_retrieved_chunks,
            block_size=retro_args.retro_block_size,
            db_dataset=db_dataset,
            chunk_dataset=chunk_dataset,
            neighbor_path_map=neighbor_path_map,
        )

    # Extract datasets.
    train_ds = retro_dataset_map.get("train", None)
    valid_ds = retro_dataset_map.get("valid", None)
    test_ds = retro_dataset_map.get("test", None)

    return train_ds, valid_ds, test_ds
