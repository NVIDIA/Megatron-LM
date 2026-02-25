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

logger = logging.getLogger(__name__)


@dataclass
class SFTDatasetConfig(GPTDatasetConfig):
    train_on_completitions_only: bool = False
    """If True, only train on completitions, otherwise train on full conversations"""

    train_on_thinking_traces: bool = True
    """If False, mask thinking traces from the completitions"""

    completitions_start_tokens: List[int]
    """The tokens that indicate the start of a completition"""

    completitions_end_tokens: List[int]
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

        return self.sample_index.shape[0] - 1
    
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
        doc_index_beg, doc_index_beg_offset = self.sample_index[idx]
        doc_index_end, doc_index_end_offset = self.sample_index[idx + 1]

        assert doc_index_beg_offset == 0 # TODO(asolergi-nv): Remove
        assert doc_index_end_offset == 0 # TODO(asolergi-nv): Remove

        document_ids = []
        sample_parts = []

        # TODO(asolergi-nv): For PP add flag to JUST read cu_seqlens
        # TODO(asolergi-nv): sequence_pointer, sequence_length, sequence_mode = self.dataset.index[document_index[i or doc_index_beg]]
        
        # Sample spans a single document
        if doc_index_beg == doc_index_end:
            # Add the document id
            document_ids.append(self.document_index[doc_index_beg])

            # Add the entire sample
            sample_parts.append(
                self.dataset.get(
                    self.document_index[doc_index_beg],
                    offset=int(doc_index_beg_offset),
                    length=doc_index_end_offset
                    - doc_index_beg_offset
                    + self.config.add_extra_token_to_sequence,
                )
            )

        # Sample spans multiple documents
        else:
            for i in range(doc_index_beg, doc_index_end + 1):
                # Add the sample part
                offset = 0 if i > doc_index_beg else doc_index_beg_offset
                length = (
                    None
                    if i < doc_index_end
                    else doc_index_end_offset + self.config.add_extra_token_to_sequence
                )
                # TODO(asolergi-nv): Improve this logic, we will always be in this else statement & always get a full sample
                if length is not None and length < 2:
                    break
                
                # Add the document id
                document_ids.append(self.document_index[i])
                
                sample_parts.append(
                    self.dataset.get(self.document_index[i], offset=int(offset), length=length)
                )
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
            num_tokens_per_epoch = self._get_num_tokens_per_epoch()
            num_epochs = self._get_num_epochs(num_tokens_per_epoch)

            if num_epochs == 1:
                separate_final_epoch = False
            else:
                # Get the number of samples for the last epoch
                num_samples_sans_final_epoch = (
                    (num_epochs - 1) * num_tokens_per_epoch
                    - self.config.add_extra_token_to_sequence
                ) // sequence_length
                num_samples_from_final_epoch = self.num_samples - num_samples_sans_final_epoch
                num_samples_per_epoch = (
                    num_tokens_per_epoch - self.config.add_extra_token_to_sequence
                ) // sequence_length

                # num_samples_from_final_epoch should be non-negative
                assert num_samples_from_final_epoch >= 0

                # num_samples_from_final_epoch should not exceed max value
                assert num_samples_from_final_epoch <= num_samples_per_epoch + 1

                # Separate the final epoch if it falls below the threshold
                threshold = 0.80
                separate_final_epoch = num_samples_from_final_epoch < int(
                    threshold * num_samples_per_epoch
                )

                log_single_rank(
                    logger,
                    logging.DEBUG,
                    f"> num_samples_from_final_epoch: {num_samples_from_final_epoch}",
                )
                log_single_rank(logger, logging.DEBUG, f"> threshold: {threshold}")
                log_single_rank(
                    logger, logging.DEBUG, f"> num_samples_per_epoch: {num_samples_per_epoch}"
                )

            log_single_rank(
                logger, logging.DEBUG, f"> separate_final_epoch: {separate_final_epoch}"
            )

            numpy_random_state = numpy.random.RandomState(self.config.random_seed)

            # Build the document index
            document_index = _build_document_index(
                self.indices, num_epochs, numpy_random_state, separate_final_epoch
            )

            # Build the sample index
            from megatron.core.datasets import helpers

            if self.index_split == Split.valid:
                drop_last_partial_sequence = self.config.drop_last_partial_validation_sequence
            else:
                drop_last_partial_sequence = True

            assert document_index.dtype == numpy.int32
            assert self.dataset.sequence_lengths.dtype == numpy.int32
            if len(document_index) * 2 > len(self.dataset.sequence_lengths):
                # If "access density" of sequence_lengths is high, force load the mmap-ed array
                # into memory by making a copy.
                #
                # System performance benefits come from two aspects:
                #   1. We sequentially pre-load the whole file, most of which we expect to read
                #   2. The GIL is held when entering the c++ program, improving the speed of which
                #      improves parallelism
                sequence_lengths_for_cpp = self.dataset.sequence_lengths.copy()
            else:
                sequence_lengths_for_cpp = self.dataset.sequence_lengths
            sample_index = helpers.build_sft_sample_idx(
                sequence_lengths_for_cpp,
                document_index,
                sequence_length,
                num_epochs,
                num_tokens_per_epoch,
                drop_last_partial_sequence,
                self.config.add_extra_token_to_sequence,
            )

            # Build the shuffle index
            if separate_final_epoch:
                shuffle_index = _build_shuffle_index(
                    num_samples_sans_final_epoch, sample_index.shape[0] - 1, numpy_random_state
                )
            else:
                shuffle_index = _build_shuffle_index(
                    sample_index.shape[0] - 1, sample_index.shape[0] - 1, numpy_random_state
                )

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

    def _get_num_tokens_per_epoch(self) -> int:
        """Calculate the number of tokens in a single epoch

        Returns:
            int: The number of tokens in a single epoch
        """
        # TODO(asolergi-nv): For the truncated conversations it'll be less than this 
        return int(numpy.sum(self.dataset.sequence_lengths[self.indices]))

    def _get_num_epochs(self, num_tokens_per_epoch: int) -> int:
        """Calculate the number of epochs

        Args:
            num_tokens_per_epoch (int): The number of tokens in a single epoch

        Returns:
            int: The number of epochs
        """
        num_epochs = 1
        num_tokens = num_tokens_per_epoch
        if self.num_samples is not None:
            num_tokens_requested = (
                self.num_samples * self.config.sequence_length
            ) + self.config.add_extra_token_to_sequence
            while num_tokens < num_tokens_requested:
                num_epochs += 1
                num_tokens += num_tokens_per_epoch
        return num_epochs
