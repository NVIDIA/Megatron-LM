# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import numpy as np
import os
import shutil
import torch
from tqdm import tqdm

from megatron import get_retro_args, print_rank_0
from tools.bert_embedding import DiskDataParallelBertEmbedder
from tools.retro.db.utils import (
    get_indexed_dataset_infos,
    get_merged_sampled_dataset,
    get_merged_train_dataset,
)
from tools.retro.external_libs import h5py
from tools.retro.index.factory import IndexFactory
from tools.retro.utils import GPTToTextDataset

from .utils import (
    get_training_data_block_dir,
    get_training_data_block_paths,
    get_training_data_merged_path,
    get_training_data_root_dir,
)


##################################################
# Train index.
##################################################


def get_empty_index_path():
    '''Path of empty index.'''
    args = get_retro_args()
    index = IndexFactory.get_index(args.retro_index_type)
    empty_index_path = index.get_empty_index_path()
    return empty_index_path


def get_block_nload(block_path, load_fraction):
    with h5py.File(block_path) as fi:
        return int(load_fraction * fi["data"].shape[0])


def merge_embedding_blocks():

    if torch.distributed.get_rank() != 0:
        return

    args = get_retro_args()

    # Get block, merged paths.
    load_fraction = args.retro_index_train_load_fraction
    block_paths = get_training_data_block_paths()
    bin_path = get_training_data_merged_path()

    # Skip, if already built.
    if os.path.exists(bin_path):
        return

    # Merge blocks.
    with open(bin_path, "wb") as fo:
        byte_offset = 0
        for block_idx, block_path in \
            enumerate(tqdm(block_paths, "merge train embeddings")):
            with h5py.File(block_path) as fi:

                nload = get_block_nload(block_path, load_fraction)
                block = np.array(fi["data"][:nload], copy = False)

                fo.write(block.tobytes())

                byte_offset += block.size * block.itemsize
                fo.seek(byte_offset)


def embed_db():
    '''Embed DB chunks.

    Store chunks in blocks on disk. These blocks will later be merged into
    a single dataset for training the index.
    '''

    args = get_retro_args()

    merged_train_data_path = get_training_data_merged_path()
    if os.path.exists(merged_train_data_path):
        return

    # Get db dataset.
    gpt_dataset = get_merged_sampled_dataset()
    text_dataset = GPTToTextDataset(gpt_dataset)

    # Embed dataset.
    embedder = DiskDataParallelBertEmbedder(args.retro_bert_batch_size,
                                            args.retro_bert_max_chunk_length,
                                            args.retro_block_size,
                                            args.bert_embedder_type)
    embedder.embed_text_dataset("index",
                                get_training_data_block_dir(),
                                text_dataset)

    # Merge embeddings.
    merge_embedding_blocks()


def train_on_embeddings():
    '''Train index on embedded DB chunks.'''
    args = get_retro_args()
    index = IndexFactory.get_index(args.retro_index_type)
    index.train()


def remove_embeddings():
    '''Remove embeddings after training.'''
    torch.distributed.barrier()
    if torch.distributed.get_rank() != 0:
        return
    empty_index_path = get_empty_index_path()
    assert os.path.isfile(empty_index_path)
    shutil.rmtree(get_training_data_root_dir(), ignore_errors=True)


def train_index():
    '''Train index on DB chunks.'''

    args = get_retro_args()

    # Check if trained index already exists.
    if not os.path.isfile(get_empty_index_path()):

        # Embed training chunks.
        embed_db()

        # Train index on embeddings.
        train_on_embeddings()

    # Wait for (single-process) training to complete.
    torch.distributed.barrier()

    # Remove embeddings.
    if args.retro_index_delete_training_embeddings:
        remove_embeddings()


##################################################
# Add to index.
##################################################


def add_to_index():
    '''Add DB chunks to index.'''

    args = get_retro_args()

    # Get index.
    index = IndexFactory.get_index(args.retro_index_type)

    # Get text dataset.
    gpt_dataset = get_merged_train_dataset()
    text_dataset = GPTToTextDataset(gpt_dataset)

    # Add to index.
    output_index_path = index.add(text_dataset)

    return output_index_path


##################################################
# Build index (train + add).
##################################################


def build_index():
    '''Build index.

    Building index involves sequentially running stages above:
    - Train index (on sampled training chunks).
    - Add to index (on all training chunks).
    '''

    # Train index.
    train_index()

    # Add to index.
    add_to_index()
