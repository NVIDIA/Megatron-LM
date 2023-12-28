# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import os
import torch

from megatron import get_args, get_retro_args, print_rank_0
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.training import (
    build_train_valid_test_datasets as build_pretraining_train_valid_test_datasets,
    update_train_iters,
)
from pretrain_gpt import is_dataset_built_on_rank
from tools.retro.db.utils import get_indexed_dataset_infos
from tools.retro.utils import get_num_chunks_per_sample

from .multi_split_gpt_dataset import MultiSplitGPTDataset, MultiSplitGPTDatasetConfig
from .utils import get_neighbor_dirname, get_query_workdir


class ChunkDataset(torch.utils.data.Dataset):
    '''Pretraining chunk dataset wraps a standard GPT dataset.

    This dataset conceptually divides each sample (e.g., length 2048)
    into chunks (e.g., length 64) and restructures them into a list of
    chunks (e.g., length num_samples * num_chunks_per_sample).
    '''

    def __init__(self, sample_dataset, chunk_length):

        super().__init__()

        self.sample_dataset = sample_dataset

        self.chunk_length = chunk_length
        self.n_chunks_per_sample = get_num_chunks_per_sample()
        self.n_samples = len(sample_dataset)
        self.n_chunks = self.n_samples * self.n_chunks_per_sample

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):

        # Convert global chunk index to global sample index & local chunk index.
        sample_idx = idx // self.n_chunks_per_sample
        chunk_idx = idx % self.n_chunks_per_sample

        # Extract sample data.
        sample = self.sample_dataset[sample_idx]
        sample_token_ids = sample["text"]
        sample_doc_ids = sample["document_ids"]

        # Chunk start/end token idxs.
        token_start_idx = chunk_idx * self.chunk_length
        token_end_idx = token_start_idx + self.chunk_length
        chunk_token_ids = sample_token_ids[token_start_idx:token_end_idx]

        # Sample.
        return {
            "doc_ids" : sample_doc_ids,
            "text" : chunk_token_ids,
        }


def core_retro_dataset_config_from_args(args, retro_args):
    return MultiSplitGPTDatasetConfig(
        is_built_on_rank=is_dataset_built_on_rank,
        random_seed=retro_args.retro_gpt_seed,
        sequence_length=retro_args.retro_gpt_seq_length,
        blend=args.data_path if args.data_path is not None else retro_args.retro_gpt_data_path,
        split=args.split,
        path_to_cache=args.data_cache_path,
        return_document_ids=retro_args.retro_return_doc_ids,
        split_preprocessing=retro_args.retro_gpt_split,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""

    args = get_args()
    retro_args = get_retro_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        MultiSplitGPTDataset,
        train_val_test_num_samples,
        core_retro_dataset_config_from_args(args, retro_args)
    ).build()
    print_rank_0("> finished creating pretrained GPT datasets ...")

    return train_ds, valid_ds, test_ds


def get_chunk_dataset_map():
    '''Get train, valid, test chunk datasets.'''

    args = get_retro_args()

    # Update train iters.
    update_train_iters(args)

    args.iteration = 0
    args.consumed_train_samples = 0

    # Datasets.
    print_rank_0(" > datasets.")
    train_ds, valid_ds, test_ds = build_pretraining_train_valid_test_datasets(
        train_valid_test_datasets_provider)

    sample_dataset_map = {
        "train" : train_ds,
        "valid" : valid_ds,
        "test" : test_ds,
    }

    # Info dict.
    chunk_dataset_map = {
        key : {
            "neighbor_dir" : get_neighbor_dirname(key, sample_ds),
            "data" : ChunkDataset(sample_ds, args.retro_gpt_chunk_length),
        }
        for key, sample_ds in sample_dataset_map.items() if sample_ds
    }

    return chunk_dataset_map
