# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import os
import torch

from megatron import get_retro_args, print_rank_0
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.training import (
    build_train_valid_test_data_loaders,
    update_train_iters,
)
from tools.retro.db.utils import get_indexed_dataset_infos
from tools.retro.utils import get_num_chunks_per_sample

from .utils import get_pretraining_workdir


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
        sample_doc_ids = sample["doc_ids"]

        # Chunk start/end token idxs.
        token_start_idx = chunk_idx * self.chunk_length
        token_end_idx = token_start_idx + self.chunk_length
        chunk_token_ids = sample_token_ids[token_start_idx:token_end_idx]

        # Sample.
        return {
            "doc_ids" : sample_doc_ids,
            "text" : chunk_token_ids,
        }


def verify_indexed_dataset_order():
    '''Verify pretraining order same as DB order.'''

    args = get_retro_args()

    # DB dataset prefixes.
    db_indexed_dataset_infos = get_indexed_dataset_infos()
    db_prefixes = [ info["prefix"] for info in db_indexed_dataset_infos ]

    # Verify order & prefixes.
    assert len(args.data_path) >= 2, "blendable dataset supported only."
    pretraining_prefixes = args.data_path[1:None:2]

    if len(db_prefixes) != len(pretraining_prefixes):
        raise Exception("inconsistent dataset count between db & pretraining.")
    if db_prefixes != pretraining_prefixes:
        raise Exception("inconsistent dataset order between db & pretraining.")


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""

    args = get_retro_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.retro_gpt_seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        return_doc_ids=args.retro_return_doc_ids)
    print_rank_0("> finished creating pretrained GPT datasets ...")

    return train_ds, valid_ds, test_ds


def get_chunk_dataset_map():
    '''Get train, valid, test chunk datasets.'''

    args = get_retro_args()

    # Update train iters.
    update_train_iters(args)

    args.iteration = 0
    args.consumed_train_samples = 0

    # Verify indexed dataset order.
    verify_indexed_dataset_order()

    # Datasets.
    print_rank_0(" > data loader.")
    train_data_loader, valid_data_loader, test_data_loader \
        = build_train_valid_test_data_loaders(
            train_valid_test_datasets_provider)

    data_loader_map = {
        "train" : train_data_loader,
        "valid" : valid_data_loader,
        "test" : test_data_loader,
    }

    # Info dict.
    workdir = get_pretraining_workdir()
    dataset_map = {
        key : {
            "neighbor_dir" : os.path.join(
                workdir,
                os.path.basename(loader.dataset.datasets[0].index_prefix),
            ),
            "data" : ChunkDataset(loader.dataset, args.retro_gpt_chunk_length),
        }
        for key, loader in data_loader_map.items() if loader
    }

    return dataset_map
