# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import logging
import os
import time
from dataclasses import dataclass, field
from math import ceil
from typing import Dict, Optional, Tuple, List

import numpy
import torch

from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, _build_document_index, _build_shuffle_index
from megatron.core.datasets.megatron_dataset import MegatronDataset
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.utils import Split
from megatron.core.datasets.object_storage_utils import ObjectStorageConfig, is_object_storage_path
from megatron.core.utils import log_single_rank

from bisect import bisect

logger = logging.getLogger(__name__)


@dataclass
class SFTDatasetConfig(GPTDatasetConfig):
    train_on_completitions_only: bool = False
    """If True, only train on completitions, otherwise train on full conversations"""

    train_on_thinking_traces: bool = True
    """If False, mask thinking traces from the completitions"""
    
    #completitions_start_tokens: List[int]
    """The tokens that indicate the start of a completition"""

    #completitions_end_tokens: List[int]
    """The tokens that indicate the end of a completition"""

    truncate_conversations: bool = True # TODO(asolergi-nv): 

class SFTDataset(MegatronDataset):
    def __init__(
        self,
        indexed_dataset: IndexedDataset,
        dataset_path: Optional[str],
        indexed_indices: numpy.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: SFTDatasetConfig,
    ) -> None:
        super().__init__(
            indexed_dataset, dataset_path, indexed_indices, num_samples, index_split, config
        )

        (self.document_index, self.sample_index, self.shuffle_index) = (
            self._build_document_sample_shuffle_indices()
        )
        
    @staticmethod
    def numel_low_level_dataset(low_level_dataset: IndexedDataset) -> int:
        """Abstract method implementation

        For GPT, the underlying IndexedDataset should be split by sequence, as opposed to, say,
        BERT, which should be split by document

        Args:
            low_level_dataset (IndexedDataset): The underlying IndexedDataset

        Returns:
            int: The number of unique elements in the underlying IndexedDataset
        """
        return low_level_dataset.sequence_lengths.shape[0]

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: SFTDatasetConfig) -> IndexedDataset:
        """Abstract method implementation

        Args:
            dataset_path (str): The real path prefix to the IndexedDataset .bin and .idx files

            config (SFTDatasetConfig): The config

        Returns:
            IndexedDataset: The underlying IndexedDataset
        """
        if is_object_storage_path(dataset_path):
            assert config.object_storage_cache_path is not None
            return IndexedDataset(
                dataset_path,
                multimodal=False,
                mmap=config.mmap_bin_files,
                object_storage_config=ObjectStorageConfig(
                    path_to_idx_cache=config.object_storage_cache_path
                ),
            )
        return IndexedDataset(
            dataset_path,
            multimodal=False,
            mmap=config.mmap_bin_files,
        )
    
    def __len__(self) -> int:
        """Abstract method implementation

        Returns:
            int: The length of the dataset
        """

        return self.shuffle_index.shape[0]
    
    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        """Abstract method implementation

        Args:
            idx (Optional[int]): The index into the dataset

        Returns:
            Dict[str, torch.Tensor]: The sample information wrapped in a dictionary
        """
        tokens, lengths = self._query_document_sample_shuffle_indices(idx)

        tokens = torch.from_numpy(tokens).long()
        cu_seqlens = torch.cat((torch.zeros(1, dtype=torch.int64), torch.cumsum(torch.from_numpy(lengths), dim = 0)))
        position_ids = torch.arange(tokens.shape[0], dtype=torch.long) # TODO(asolergi-nv): Create position_ids from lengths
        
        # TODO(asolergi-nv): Create labels, remember self.config.train_on_completitions_only

        return {
            'tokens': tokens[:-1].contiguous(),
            'labels': tokens[1:].contiguous(),
            'position_ids': position_ids,
            'cu_seqlens': cu_seqlens,
        }
    def _query_document_sample_shuffle_indices(
        self, idx: int
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Get the text (token ids) and document ids for a given index

        Args:
            idx (int): The index into the dataset

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The text ids and document ids
        """
        # Do the shuffle mapping
        idx = self.shuffle_index[idx]

        # Get the beginning and end documents and offsets
        doc_index_beg = self.sample_index[idx]
        doc_index_end = self.sample_index[idx + 1]

        document_ids = []
        sample_parts = []

        # TODO(asolergi-nv): For PP add flag to JUST read cu_seqlens
        # TODO(asolergi-nv): sequence_pointer, sequence_length, sequence_mode = self.dataset.index[document_index[i or doc_index_beg]]
        
        for i in range(doc_index_beg, doc_index_end):
            # Add the document id # TODO(asolergi-nv): Remove
            document_ids.append(self.document_index[i])
            
            # Add the sample part
            sample_parts.append(self.dataset.get(self.document_index[i]))
        assert len(document_ids) == len(
            sample_parts
        ), f"len(document_ids) ({len(document_ids)}) != len(sample_parts) ({len(sample_parts)})"

        return (
            numpy.concatenate(sample_parts, dtype=numpy.int64),
            numpy.array([len(sample_part) for sample_part in sample_parts], dtype=numpy.int64), # NOTE(asolergi-nv): cu_seqlens
        )

    def _build_document_sample_shuffle_indices(
        self,
    ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Build the document index, the sample index, and the shuffle index

        The document index:
            -- 1-D
            -- An ordered array of document ids

        The sample index:
            -- 2-D
            -- The document indices and offsets which mark the start of every sample

        The shuffle index:
            -- 1-D
            -- A random permutation of index range of the sample index

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: The document index, the sample
            index, and the shuffle index
        """
        path_to_cache = self.config.path_to_cache
        if path_to_cache is None and not self.config.mock:
            path_to_cache = os.path.join(
                self.dataset.path_prefix, "cache", f"{type(self).__name__}_indices"
            )

        if path_to_cache:
            base = f"{self.unique_description_hash}-{type(self).__name__}-{self.index_split.name}"
            get_path_to = lambda affix: os.path.join(path_to_cache, f"{base}-{affix}")
            path_to_description = get_path_to("description.txt")
            path_to_document_index = get_path_to("document_index.npy")
            path_to_sample_index = get_path_to("sample_index.npy")
            path_to_shuffle_index = get_path_to("shuffle_index.npy")
            cache_hit = (
                True
                if self.config.fast_cache_load
                else all(
                    map(
                        os.path.isfile,
                        [
                            path_to_description,
                            path_to_document_index,
                            path_to_sample_index,
                            path_to_shuffle_index,
                        ],
                    )
                )
            )
        else:
            cache_hit = False

        if not path_to_cache or (
            not cache_hit
            and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)
        ):
            log_single_rank(
                logger,
                logging.DEBUG,
                f"Build and save the {type(self).__name__} {self.index_split.name} indices",
            )
            t_beg = time.time()

            sequence_length = self.config.sequence_length
            sample_lengths = self.dataset.sequence_lengths[self.indices]
            num_samples = 1 if self.num_samples is None else self.num_samples

            packed_samples = pack_samples(sample_lengths, sequence_length)
            document_index = [j for pack in packed_samples for j in pack]
            sample_index = torch.cumsum(torch.tensor([0] + [len(sample) for sample in packed_samples]), dim=0)

            num_packed = len(packed_samples)
            num_epochs = (num_samples // num_packed) + 1 if num_samples % num_packed else num_samples // num_packed
            shuffle_index = []
            for epoch in range(num_epochs):
                shuffle_index.extend(torch.randperm(num_packed).tolist())

            # TODO(asolergi-nv): Refactor
            # Convert to numpy arrays
            document_index = numpy.array(document_index, dtype=numpy.int32)
            sample_index = numpy.array(sample_index, dtype=numpy.int32)
            shuffle_index = numpy.array(shuffle_index, dtype=numpy.int32)

            if path_to_cache:
                os.makedirs(path_to_cache, exist_ok=True)
                # Write the description
                with open(path_to_description, "wt") as writer:
                    writer.write(self.unique_description)
                numpy.save(path_to_document_index, document_index, allow_pickle=True)
                numpy.save(path_to_sample_index, sample_index, allow_pickle=True)
                numpy.save(path_to_shuffle_index, shuffle_index, allow_pickle=True)
            else:
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f"Unable to save {type(self).__name__} indexes because path_to_cache is None",
                )

            t_end = time.time()
            log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

            log_single_rank(
                logger, logging.DEBUG, f"> total number of samples: {sample_index.shape[0] - 1}"
            )
            log_single_rank(logger, logging.DEBUG, f"> total number of epochs: {num_epochs}")

            return document_index, sample_index, shuffle_index

        log_single_rank(
            logger, logging.DEBUG, f"Load the {type(self).__name__} {self.index_split.name} indices"
        )

        log_single_rank(
            logger,
            logging.DEBUG,
            f"\tLoad the document index from {os.path.basename(path_to_document_index)}",
        )
        t_beg = time.time()
        document_index = numpy.load(path_to_document_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(
            logger,
            logging.DEBUG,
            f"\tLoad the sample index from {os.path.basename(path_to_sample_index)}",
        )
        t_beg = time.time()
        sample_index = numpy.load(path_to_sample_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(
            logger,
            logging.DEBUG,
            f"\tLoad the shuffle index from {os.path.basename(path_to_shuffle_index)}",
        )
        t_beg = time.time()
        shuffle_index = numpy.load(path_to_shuffle_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(
            logger, logging.DEBUG, f"> total number of samples: {sample_index.shape[0] - 1}"
        )

        return document_index, sample_index, shuffle_index

def _classify_items(
    items: List[Tuple[int, int]], bin_capacity: int
) -> Tuple[
    List[Tuple[int, int]],
    List[Tuple[int, int]],
    List[Tuple[int, int]],
    List[Tuple[int, int]],
]:
    """Split items into large / medium / small / tiny classes.

    Follows the classification used by Johnson & Garey:
        large   : (C/2, C]
        medium  : (C/3, C/2]
        small   : (C/6, C/3]
        tiny    : (0  , C/6]

    Args:
        items: List of (index, size) tuples

    Returns:
        Tuple of four lists (large, medium, small, tiny) without additional sorting.
    """
    large, medium, small, tiny = [], [], [], []
    for idx, size in items:
        if size > bin_capacity / 2:
            large.append((idx, size))
        elif size > bin_capacity / 3:
            medium.append((idx, size))
        elif size > bin_capacity / 6:
            small.append((idx, size))
        else:
            tiny.append((idx, size))
    return large, medium, small, tiny

def pack_samples(sequence_lengths: List[int], bin_capacity: int) -> List[List[int]]:
    """Pack sequences using the Modified First-Fit Decreasing algorithm.

    Args:
        sequence_lengths: A list of sequence lengths to pack.

    Returns:
        A list of bins, where each bin is a list of indices into the original
        sequence_lengths list.
    """
    # Validate inputs
    if bin_capacity <= 0:
        raise ValueError("bin_capacity must be positive")
    if any(l <= 0 for l in sequence_lengths):
        raise ValueError("sequence lengths must be positive")

    # Drop documents that exceed capacity and warn
    long_mask = [l > bin_capacity for l in sequence_lengths]
    if any(long_mask):
        n_dropped = sum(long_mask)
        log_single_rank(
            logger,
            logging.WARNING,
            f"Dropping {n_dropped} document(s) with sequence length > bin_capacity "
            f"(bin_capacity={bin_capacity}).",
        )
    items: List[Tuple[int, int]] = [
        (i, l) for i, l in enumerate(sequence_lengths) if l <= bin_capacity
    ]

    # Phase-0: classify
    large, medium, small, tiny = _classify_items(items, bin_capacity)

    # Sort according to the rules of MFFD
    large.sort(key=lambda x: x[1], reverse=True)  # descending size
    medium.sort(key=lambda x: x[1], reverse=True)
    small.sort(key=lambda x: x[1])  # ascending size
    tiny.sort(key=lambda x: x[1])

    # Phase-1: start one bin per large item
    bins: List[List[Tuple[int, int]]] = [[item] for item in large]

    # Phase-2: try to add one medium item to each large bin (forward pass)
    for b in bins:
        remaining = bin_capacity - sum(size for _, size in b)
        for i, (idx, size) in enumerate(medium):
            if size <= remaining:
                b.append(medium.pop(i))
                break

    # Phase-3: backward pass – fill with two small items where possible
    for b in reversed(bins):
        has_medium = any(
            bin_capacity / 3 < size <= bin_capacity / 2 for _, size in b
        )
        if has_medium or len(small) < 2:
            continue
        remaining = bin_capacity - sum(size for _, size in b)
        if small[0][1] + small[1][1] > remaining:
            continue
        first_small = small.pop(0)
        # pick the *largest* small that fits with first_small (so iterate from end)
        second_idx = None
        for j in range(len(small) - 1, -1, -1):
            if small[j][1] <= remaining - first_small[1]:
                second_idx = j
                break
        if second_idx is not None:
            second_small = small.pop(second_idx)
            b.extend([first_small, second_small])

    # Phase-4: forward greedy fit of remaining items
    remaining_items = sorted(
        medium + small + tiny, key=lambda x: x[1], reverse=True
    )
    for b in bins:
        while remaining_items:
            rem = bin_capacity - sum(size for _, size in b)
            # if even the smallest remaining doesn't fit we break
            if rem < remaining_items[-1][1]:
                break

            # pick the first (largest) that fits
            chosen_idx = None
            for i, (_, size) in enumerate(remaining_items):
                if size <= rem:
                    chosen_idx = i
                    break
            if chosen_idx is None:
                break
            b.append(remaining_items.pop(chosen_idx))

    # Phase-5: FFD on leftovers
    leftovers = remaining_items  # renamed for clarity

    # New O(n * logn) implementation
    ffd_bins: List[List[Tuple[int, int]]] = [[]]
    ffd_bin_sizes: List[int] = [0]
    for idx, size in sorted(leftovers, key=lambda x: x[1], reverse=True):
        # We only need to check the first bin since we guarantee the order of ffd_bin_sizes to be sorted from smallest to largest.
        if size <= (bin_capacity - ffd_bin_sizes[0]):
            new_bin = ffd_bins.pop(0)
            new_bin_size = ffd_bin_sizes.pop(0)
        else:
            new_bin = []
            new_bin_size = 0

        new_bin.append((idx, size))
        new_bin_size += size

        new_idx = bisect(ffd_bin_sizes, new_bin_size)
        ffd_bins.insert(new_idx, new_bin)
        ffd_bin_sizes.insert(new_idx, new_bin_size)

    bins.extend(ffd_bins)

    # Convert to list of index lists (discard sizes)
    return [[idx for idx, _ in b] for b in bins if b]