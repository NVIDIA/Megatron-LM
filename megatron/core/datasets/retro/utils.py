# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Utilities for Retro preprocessing."""

import glob
import logging
import os
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from megatron.core import parallel_state
from megatron.core.datasets.retro.config import RetroPreprocessingConfig
from megatron.core.datasets.retro.query.multi_split_gpt_dataset import (
    MultiSplitGPTDataset,
    MultiSplitGPTDatasetConfig,
)
from megatron.core.utils import log_single_rank

from .external_libs import h5py

logger = logging.getLogger(__name__)


def log_retro_rank_0(message: str) -> None:
    """Log on rank 0.

    Args:
        message (str): Message to log.
    """
    log_single_rank(logger, logging.INFO, "[RETRO] " + message)


def retro_makedir(config: RetroPreprocessingConfig, path: str) -> None:
    """Make a directory, conditional on not being in validation mode.

    Args:
        config (RetroPreprocessingConfig): Retro preprocessing config.
        path (str): Path to directory.
    """
    if config.retro_task_validate is None:
        os.makedirs(path, exist_ok=True)


def extract_data_config(config: RetroPreprocessingConfig) -> MultiSplitGPTDatasetConfig:
    """Extract data config from dataset.

    Args:
        config (RetroPreprocessingConfig): Retro preprocessing config.

    Returns:
        The config object used to build the dataset.
    """
    return config.retro_gpt_chunk_datasets.train["dataset"].sample_dataset.config


def get_num_chunks_per_sample(sample_length: int, chunk_length: int) -> int:
    """Compute seq_length // chunk_length.

    Args:
        sample_length (int): Alias of `sequence_length`.
        chunk_length (int): Retro chunk length (e.g., 64).

    Returns:
        Number of chunks per sample (i.e., `sequence_length` / `chunk_length`).
    """
    assert sample_length % chunk_length == 0
    return sample_length // chunk_length


class GPTToTextDataset(torch.utils.data.Dataset):
    """Dataset to convert GPT tokens to text.

    Args:
        gpt_dataset (MultiSplitGPTDataset): GPT dataset, which outputs GPT token samples.
        gpt_tokenizer (Any): GPT tokenizer.
    """

    def __init__(self, gpt_dataset: MultiSplitGPTDataset, gpt_tokenizer: Any):

        super().__init__()

        self.gpt_dataset = gpt_dataset
        self.gpt_tokenizer = gpt_tokenizer

    def __len__(self) -> int:
        """Dataset length.

        Returns:
            Number of samples in the dataset.
        """
        return len(self.gpt_dataset)

    def __getitem__(self, idx: int) -> dict:
        """Get dataset sample.

        Args:
            idx (int): Index of sample.

        Returns:
            A dict containing attribute 'text' of type string.
        """
        gpt_token_ids = self.gpt_dataset[idx]["text"].tolist()
        text = self.gpt_tokenizer.detokenize(gpt_token_ids)
        return {"text": text}


def get_blocks(
    dirname: str,
    n_samples: int,
    block_size: int,
    validate: Callable = None,
) -> SimpleNamespace:
    """Divide range [0, num_samples) to sequence of block ranges.

    This is a core method within the concept of block processing. The idea
    is to divide a range (size n_samples) into a sequence of blocks. Each
    block corresponds to a file within 'dirname' with name
    '{start_idx}-{end_idx}.hdf5'. This method checks for the existence of
    these files, and returns two lists, one for existing blocks and one for
    missing blocks.

    Args:
        dirname (str): Path to directory containing block files.
        n_samples (int): Ideal number of samples. The total number of saved block data is <=n_samples.
        block_size (int): Max number of samples per block file (e.g., 100000).
        validate (Callable): Method for validating each block file during load.

    Returns:
        A namespace consisting of 2 lists: existing blocks, and missing blocks. The total number of samples between the existing and missing blocks should equal n_samples above.
    """

    assert os.path.isdir(dirname), "missing directory '%s.'" % dirname

    # Block ranges.
    block_start_idxs = list(range(0, n_samples, block_size))
    block_end_idxs = [min(n_samples, i + block_size) for i in block_start_idxs]
    block_ranges = list(zip(block_start_idxs, block_end_idxs))

    # All block files (existing + missing).
    n_digits = int(np.ceil(np.log(n_samples) / np.log(10)) + 1)
    all_blocks = [
        {
            "range": r,
            "path": os.path.join(
                dirname,
                "%s-%s.hdf5" % tuple([str(i).zfill(n_digits) for i in r]),
            ),
        }
        for r in block_ranges
    ]
    all_block_path_set = set(block["path"] for block in all_blocks)

    # Validate function.
    validate = (lambda f: None) if validate is None else validate

    # Delete corrupt files.
    if torch.distributed.get_rank() == 0:
        existing_block_paths = [
            block["path"] for block in all_blocks if os.path.exists(block["path"])
        ]
        for index, path in enumerate(tqdm(existing_block_paths, "validating block.")):

            assert path in all_block_path_set, "unexpected filename, '%s'." % path

            try:
                f = h5py.File(path, "r")
            except Exception:
                os.remove(path)
                continue

            try:
                validate(f)
            except Exception:
                os.remove(path)
            finally:
                f.close()

    # Wait for files to be deleted.
    torch.distributed.barrier()

    # Collect blocks.
    blocks = SimpleNamespace(
        existing=[b for b in all_blocks if os.path.exists(b["path"])],
        missing=[b for b in all_blocks if not os.path.exists(b["path"])],
    )

    return blocks


def get_blocks_by_rank(
    dirname: str,
    n_samples: int,
    block_size: int,
    validate: Callable = None,
    sample: Optional[float] = None,
) -> SimpleNamespace:
    """Divide existing and missing blocks evenly across all ranks.

    See 'get_blocks()' above for description. The returned lists of existing and
    missing blocks are split evenly across ranks via interleaving. This way,
    each rank has a roughly equal number of blocks to process for a
    downstream operation.

    Args:
        dirname (str): Path to directory containing block files.
        n_samples (int): Ideal number of samples. The total number of saved block data is <=n_samples.
        block_size (int): Max number of samples per block file (e.g., 100000).
        validate (Callable): Method for validating each block file during load.
        sample (Optional[float]): If provided, sample a random subset of the blocks. Used for validating preprocessing correctness.

    Returns:
        A namespace consisting of 2 lists: existing blocks, and missing blocks. Each of these two lists is potentially a sub-sample of the total set of existing and missing blocks, depending on whether sampling is used. Additionally, the attributes n_existing_world and n_missing_world are the total number of existing and missing blocks, independent of samples. Therefore, (n_existing_world + n_missing_world) * block_size == n_samples.
    """

    # Get world blocks.
    blocks = get_blocks(dirname, n_samples, block_size, validate)

    # This rank's existing and missing files.
    data_parallel_rank = parallel_state.get_data_parallel_rank()
    data_parallel_world_size = parallel_state.get_data_parallel_world_size()
    rank_existing_blocks = blocks.existing[
        data_parallel_rank : len(blocks.existing) : data_parallel_world_size
    ]
    rank_missing_blocks = blocks.missing[
        data_parallel_rank : len(blocks.missing) : data_parallel_world_size
    ]

    # Extend rank's existing and missing blocks (with None) such that all ranks
    # have equal length lists. This allows for easier tracking of global progress.
    def get_world_max(n: int) -> int:
        """Get max value across ranks.

        Args:
            n (int): Value on this rank.

        Returns:
            Max value across all ranks.
        """
        n_tensor = torch.cuda.LongTensor([n])
        torch.distributed.all_reduce(n_tensor, op=torch.distributed.ReduceOp.MAX)
        return n_tensor.item()

    max_n_existing = get_world_max(len(rank_existing_blocks))
    max_n_missing = get_world_max(len(rank_missing_blocks))

    rank_existing_blocks += [None] * (max_n_existing - len(rank_existing_blocks))
    rank_missing_blocks += [None] * (max_n_missing - len(rank_missing_blocks))

    # Collect blocks.
    blocks = SimpleNamespace(
        n_existing_world=len(blocks.existing),
        n_missing_world=len(blocks.missing),
        existing=rank_existing_blocks,
        missing=rank_missing_blocks,
    )

    if sample is not None:
        # Sample existing and missing blocks evenly across all ranks. The
        # returned lists of blocks are randomly sampled (without replacement)
        # to yield `sample * len(blocks)` number of blocks.

        # Randomly sample blocks.
        def sample_blocks(_blocks: List[Optional[Dict]]) -> List[Optional[Dict]]:
            """Sample a random subset of all blocks.

            Args:
                _blocks (List[Optional[Dict]]): List of all blocks.

            Returns:
                A random subset of the blocks.
            """
            n_blocks_sample = int(np.ceil(sample * len(_blocks)))
            sampled_blocks: List[Optional[Dict]] = [b for b in _blocks if b is not None]

            np.random.seed(None)
            np.random.shuffle(sampled_blocks)

            sampled_blocks = sampled_blocks[:n_blocks_sample]
            sampled_blocks += [None] * (n_blocks_sample - len(sampled_blocks))

            return sampled_blocks

        blocks.existing = sample_blocks(blocks.existing)
        blocks.missing = sample_blocks(blocks.missing)

    return blocks


class BlockPathMap:
    """Map an index to its containing block path.

    The common use for this class is to have a directory of files containing
    blocks of processed data, of uniform block size (e.g., 100k samples per
    file). Each file must follow a naming convention of 'startIdx-endIdx.[ext]',
    where 'endIdx' minus 'startIdx' must equal the block size, with the possible
    exception of the final block. Given an input index, this class maps the
    index to the containing block file.

    Args:
        block_paths (List[str]): List of paths to saved block files.
        block_size (int): Max number of samples per block file (e.g., 100000).
    """

    @classmethod
    def from_dir(cls, dir: str, block_size: int, ext: str = "hdf5") -> Any:
        """Get list of block files, and create map.

        Args:
            dir (str): Path to directory containing saved block files.
            block_size (int): Max number of samples per block file (e.g., 100000).
            ext (str): Block file extension (e.g., 'hdf5').

        Returns:
            A mapping of sample index to block file path.
        """
        assert os.path.isdir(dir), f"directory not found, '{dir}'."
        return cls(sorted(glob.glob(dir + f"/*.{ext}")), block_size)

    def __init__(self, block_paths: List[str], block_size: int):
        self.max_idx = 0
        self.block_path_map = {}
        for block_path in block_paths:
            name = os.path.splitext(os.path.basename(block_path))[0]
            start_idx, end_idx = [int(i) for i in name.split("-")]
            self.block_path_map[start_idx] = block_path
            self.max_idx = max(self.max_idx, end_idx)
        self.block_size = block_size

    def __str__(self) -> str:
        """Stringify the mapping.

        Returns:
            A string representation of this block path map.
        """
        return "%d paths" % len(self.block_path_map)

    def __getitem__(self, idx: int) -> str:
        """Get block path from index.

        Args:
            idx (int): Index of sample.

        Returns:
            The path to the block file containing the sample index.
        """
        block_start_idx = self.block_size * (idx // self.block_size)
        block_path = self.block_path_map[block_start_idx]
        return block_path
