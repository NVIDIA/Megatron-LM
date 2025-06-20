# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Utilities for building an index."""

import glob
import os
from typing import List, Tuple

from megatron.core.datasets.retro.config import RetroPreprocessingConfig
from megatron.core.datasets.retro.utils import retro_makedir


def get_index_dir(config: RetroPreprocessingConfig) -> str:
    """Create sub-directory for this index.

    Args:
        config (RetroPreprocessingConfig): Retro preprocessing config.

    Returns:
        Path to index sub-directory within Retro project.
    """

    # Directory path.
    index_dir_path = os.path.join(
        config.retro_project_dir, "index", config.retro_index_type, config.retro_index_str
    )

    # Make directory.
    retro_makedir(config, index_dir_path)

    return index_dir_path


def num_samples_to_block_ranges(
    config: RetroPreprocessingConfig, num_samples: int
) -> List[Tuple[int, int]]:
    """Split a range (length num_samples) into sequence of block ranges
    of size block_size.

    Args:
        config (RetroPreprocessingConfig): Retro preprocessing config.
        num_samples (int): Split `num_samples` into consecutive block ranges, where each block is size `config.retro_block_size`.

    Returns:
        A list of tuples where each item is the (start, end) index for a given block.
    """
    block_size = config.retro_block_size
    start_idxs = list(range(0, num_samples, block_size))
    end_idxs = [min(num_samples, s + block_size) for s in start_idxs]
    ranges = list(zip(start_idxs, end_idxs))
    return ranges


def get_training_data_root_dir(config: RetroPreprocessingConfig) -> str:
    """Get root directory for embeddings (blocks and merged data).

    Args:
        config (RetroPreprocessingConfig): Retro preprocessing config.

    Returns:
        Path to the training data directory, which contains both training embedding blocks and the final merged training embeddings.
    """
    return os.path.join(config.retro_project_dir, "index", "train_emb")


def get_training_data_block_dir(config: RetroPreprocessingConfig) -> str:
    """Get directory for of saved embedding blocks.

    Args:
        config (RetroPreprocessingConfig): Retro preprocessing config.

    Returns:
        Path to the directory containing the training embedding blocks, which will be later merged into a single embedding array.
    """
    return os.path.join(get_training_data_root_dir(config), "blocks")


def get_training_data_block_paths(config: RetroPreprocessingConfig) -> List[str]:
    """Get paths to saved embedding blocks.

    Args:
        config (RetroPreprocessingConfig): Retro preprocessing config.

    Returns:
        Paths of all training embedding blocks.
    """
    return sorted(glob.glob(get_training_data_block_dir(config) + "/*.hdf5"))


def get_training_data_merged_path(config: RetroPreprocessingConfig) -> str:
    """Get path to merged training embeddings.

    Args:
        config (RetroPreprocessingConfig): Retro preprocessing config.

    Returns:
        Path to the merged training embedding binary file.
    """
    return os.path.join(
        get_training_data_root_dir(config),
        "train_%.3f.bin" % config.retro_index_train_load_fraction,
    )


def get_added_codes_dir(config: RetroPreprocessingConfig) -> str:
    """Get directory of saved encodings.

    Args:
        config (RetroPreprocessingConfig): Retro preprocessing config.

    Returns:
        Path to the directory containing the vector encodings for adding to the index.
    """
    return os.path.join(get_index_dir(config), "add_codes")


def get_added_code_paths(config: RetroPreprocessingConfig) -> List[str]:
    """Get paths to all saved encodings.

    Args:
        config (RetroPreprocessingConfig): Retro preprocessing config.

    Returns:
        Paths of all vector encoding blocks, for adding to the index.
    """
    return sorted(glob.glob(get_added_codes_dir(config) + "/*.hdf5"))
