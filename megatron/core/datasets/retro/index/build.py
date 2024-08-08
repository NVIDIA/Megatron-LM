# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Construct an index.

Constructing an index generally happens in two phases:

  - index.train(): Train an index on a representative set of vectors.
  - index.add(): Add vectors to an index, to be available for retrieval.
"""

import os
import shutil

import numpy as np
import torch
from tqdm import tqdm

from megatron.core.datasets.retro.config import RetroPreprocessingConfig
from megatron.core.datasets.retro.db.utils import (
    get_merged_sampled_dataset,
    get_merged_train_dataset,
)
from megatron.core.datasets.retro.external_libs import h5py
from megatron.core.datasets.retro.utils import GPTToTextDataset

from .factory import IndexFactory
from .utils import (
    get_training_data_block_dir,
    get_training_data_block_paths,
    get_training_data_merged_path,
    get_training_data_root_dir,
)

##################################################
# Train index.
##################################################


def get_empty_index_path(config: RetroPreprocessingConfig) -> str:
    """Path of empty index.

    Args:
        config (RetroPreprocessingConfig): Retro preprocessing config.

    Returns:
        Path to the empty (trained, but without added samples) vector index.
    """
    index = IndexFactory.get_index(config.retro_index_type)
    empty_index_path = index.get_empty_index_path(config)
    return empty_index_path


def get_block_nload(block_path: str, load_fraction: float) -> int:
    """Compute number of blocks to load.

    This is computed by multiplying the total number of available blocks with the
    fraction of blocks to load.

    Args:
        block_path (str): Path to HDF5 file containing block of data. File must contain key 'data'.
        load_fraction (float): Fraction (0 < load_fraction <= 1) of block samples to load.

    Returns:
        Number of block samples to load.
    """
    with h5py.File(block_path) as fi:
        return int(load_fraction * fi["data"].shape[0])


def merge_embedding_blocks(config: RetroPreprocessingConfig) -> None:
    """Merge individual embedding blocks into a single binary mmap file.

    The embeddings are initially stored in block-sized (e.g., ~100k embeddings per
    block) HDF5 files. These individual block files must be merged into a single
    file before training, to be based as a numpy mmap array to the index.

    Args:
        config (RetroPreprocessingConfig): Retro preprocessing config.
    """

    if torch.distributed.get_rank() != 0:
        return

    # Get block, merged paths.
    load_fraction = config.retro_index_train_load_fraction
    block_paths = get_training_data_block_paths(config)
    bin_path = get_training_data_merged_path(config)

    # Skip, if already built.
    if os.path.exists(bin_path):
        return

    # Merge blocks.
    with open(bin_path, "wb") as fo:
        byte_offset = 0
        for block_idx, block_path in enumerate(
            tqdm(
                block_paths,
                "merge train embeddings",
                miniters=len(block_paths) // 10,
                disable=torch.distributed.get_rank() != 0,
            )
        ):
            with h5py.File(block_path) as fi:

                nload = get_block_nload(block_path, load_fraction)
                block = np.array(fi["data"][:nload], copy=False)

                fo.write(block.tobytes())

                byte_offset += block.size * block.itemsize
                fo.seek(byte_offset)


def get_text_dataset_for_training(config: RetroPreprocessingConfig) -> GPTToTextDataset:
    """Convert GPT token chunk dataset to a text dataset for passing to the
    embedder.

    Args:
        config (RetroPreprocessingConfig): Retro preprocessing config.

    Returns:
        The text dataset consisting of tokens converted from sampled chunk database.
    """
    gpt_dataset = get_merged_sampled_dataset(
        project_dir=config.retro_project_dir,
        chunk_length=config.retro_gpt_chunk_length,
        eod_token_id=config.retro_tokenizers.gpt.eod,
    )
    text_dataset = GPTToTextDataset(gpt_dataset, config.retro_tokenizers.gpt)
    return text_dataset


def embed_training_chunks(config: RetroPreprocessingConfig) -> None:
    """Embed DB chunks.

    Store chunks in blocks on disk. These blocks will later be merged into
    a single dataset for training the index.

    Args:
        config (RetroPreprocessingConfig): Retro preprocessing config.
    """

    merged_train_data_path = get_training_data_merged_path(config)
    if os.path.exists(merged_train_data_path):
        return

    # Get training text dataset.
    text_dataset = get_text_dataset_for_training(config)

    # Embed dataset.
    embedder = config.retro_bert_embedders.disk
    embedder.embed_text_dataset("index", get_training_data_block_dir(config), text_dataset)

    # Merge embeddings.
    merge_embedding_blocks(config)


def train_on_embeddings(config: RetroPreprocessingConfig) -> None:
    """Train index on embedded DB chunks.

    Args:
        config (RetroPreprocessingConfig): Retro preprocessing config.
    """
    index = IndexFactory.get_index(config.retro_index_type)
    index.train(config)


def remove_embeddings(config: RetroPreprocessingConfig) -> None:
    """Remove embeddings after training.

    Args:
        config (RetroPreprocessingConfig): Retro preprocessing config.
    """
    torch.distributed.barrier()
    if torch.distributed.get_rank() != 0:
        return
    empty_index_path = get_empty_index_path(config)
    assert os.path.isfile(empty_index_path)
    shutil.rmtree(get_training_data_root_dir(config), ignore_errors=True)


def _train_index(config: RetroPreprocessingConfig) -> None:
    """Train index on DB chunks.

    Args:
        config (RetroPreprocessingConfig): Retro preprocessing config.
    """

    # Check if trained index already exists.
    if not os.path.isfile(get_empty_index_path(config)):

        # Embed training chunks.
        embed_training_chunks(config)

        # Train index on embeddings.
        train_on_embeddings(config)

    # Wait for (single-process) training to complete.
    torch.distributed.barrier()

    # Remove embeddings.
    if config.retro_index_delete_training_embeddings:
        remove_embeddings(config)


def train_index(config: RetroPreprocessingConfig) -> None:
    """Entry point for training the index.

    We select whether to train a new index, or validate an existing index.

    Args:
        config (RetroPreprocessingConfig): Retro preprocessing config.
    """

    # Train new index.
    if config.retro_task_validate is None:
        _train_index(config)

    # Validate existing trained index.
    else:
        from .validate import validate_training_embeddings

        validate_training_embeddings(config)


##################################################
# Add to index.
##################################################


def get_text_dataset_for_adding(config: RetroPreprocessingConfig) -> GPTToTextDataset:
    """Convert GPT token chunk dataset to a text dataset for passing to the
    embedder.

    Args:
        config (RetroPreprocessingConfig): Retro preprocessing config.

    Returns:
        The text dataset that consists of tokens converted from the 'train' chunk database. These are the chunks used for retrieval by the pretraining 'train' dataset.
    """
    gpt_dataset = get_merged_train_dataset(
        project_dir=config.retro_project_dir,
        chunk_length=config.retro_gpt_chunk_length,
        eod_token_id=config.retro_tokenizers.gpt.eod,
    )
    text_dataset = GPTToTextDataset(gpt_dataset, config.retro_tokenizers.gpt)
    return text_dataset


def _add_to_index(config: RetroPreprocessingConfig) -> str:
    """Add DB chunks to index.

    Args:
        config (RetroPreprocessingConfig): Retro preprocessing config.

    Returns:
        Path to the populated index.
    """

    # Get index.
    index = IndexFactory.get_index(config.retro_index_type)

    # Get text dataset.
    text_dataset = get_text_dataset_for_adding(config)

    # Add to index.
    output_index_path = index.add(config, text_dataset)

    return output_index_path


def add_to_index(config: RetroPreprocessingConfig) -> None:
    """Entry point for adding to the index.

    We select whether to add to a new index, or validate an existing index.

    Args:
        config (RetroPreprocessingConfig): Retro preprocessing config.
    """

    # Add to new index.
    if config.retro_task_validate is None:
        _add_to_index(config)

    # Validate existing encodings.
    else:
        from .validate import validate_added_encodings

        validate_added_encodings(config)


##################################################
# Build index (train + add).
##################################################


def build_index(config: RetroPreprocessingConfig) -> None:
    """Build index.

    Building index involves sequentially running stages above:
    - Train index (on sampled training chunks).
    - Add to index (on all training chunks).

    Args:
        config (RetroPreprocessingConfig): Retro preprocessing config.
    """

    # Train index.
    train_index(config)

    # Add to index.
    add_to_index(config)
