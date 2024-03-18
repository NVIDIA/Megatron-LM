# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy

from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.masked_dataset import (
    MaskedWordPieceDataset,
    MaskedWordPieceDatasetConfig,
)
from megatron.core.datasets.utils import Split


@dataclass
class BERTMaskedWordPieceDatasetConfig(MaskedWordPieceDatasetConfig):
    """Configuration object for Megatron Core BERT WordPiece datasets

    Args:
        classification_head (bool): Option to perform the next sequence prediction during sampling
    """

    classification_head: bool = None

    def __post_init__(self) -> None:
        """Do asserts and set fields post init
        """
        super().__post_init__()

        assert self.classification_head is not None


class BERTMaskedWordPieceDataset(MaskedWordPieceDataset):
    """The BERT dataset that assumes WordPiece tokenization

    Args:
        indexed_dataset (IndexedDataset): The IndexedDataset around which to build the MegatronDataset

        dataset_path (str): The real path on disk to the dataset, for bookkeeping

        indexed_indices (numpy.ndarray): The set of the documents indices to expose

        num_samples (int): The number of samples to draw from the indexed dataset

        index_split (Split): The indexed_indices Split

        config (BERTMaskedWordPieceDatasetConfig): The config
    """

    def __init__(
        self,
        indexed_dataset: IndexedDataset,
        dataset_path: str,
        indexed_indices: numpy.ndarray,
        num_samples: int,
        index_split: Split,
        config: BERTMaskedWordPieceDatasetConfig,
    ) -> None:
        super().__init__(
            indexed_dataset, dataset_path, indexed_indices, num_samples, index_split, config
        )

    def _finalize(self) -> None:
        """Abstract method implementation
        """
        self.token_lookup = list(self.config.tokenizer.inv_vocab.keys())
        # Account for the single <cls> and two <sep> token ids
        self.sample_index = self._build_sample_index(
            self.config.sequence_length - 3, 2 if self.config.classification_head else 1
        )

    @staticmethod
    def _key_config_attributes() -> List[str]:
        """Inherited method implementation

        Returns:
            List[str]: The key config attributes
        """
        return super(
            BERTMaskedWordPieceDataset, BERTMaskedWordPieceDataset
        )._key_config_attributes() + ["classification_head",]

    def __getitem__(self, idx: int) -> Dict[str, Union[int, numpy.ndarray]]:
        """Abstract method implementation
 
        Args:
            idx (int): The index into the dataset

        Returns:
            Dict[str, Union[int, numpy.ndarray]]: The 
        """
        idx_beg, idx_end, target_sequence_length = self.sample_index[idx]
        sample = [self.dataset[i] for i in range(idx_beg, idx_end)]
        numpy_random_state = numpy.random.RandomState(
            seed=(self.config.random_seed + idx) % 2 ** 32
        )

        assert target_sequence_length <= self.config.sequence_length

        # Split the sample into contiguous subsegments A and B
        pivot = len(sample)
        is_next_random = False
        if self.config.classification_head:
            assert len(sample) > 1, "the sample must contain at least two sentences"
            pivot = 1
            if len(sample) >= 3:
                pivot = numpy_random_state.randint(low=1, high=len(sample))
            is_next_random = numpy_random_state.random() < 0.5
        split_A = []
        for sample_a in sample[:pivot]:
            split_A.extend(sample_a)
        split_B = []
        for sample_b in sample[pivot:]:
            split_B.extend(sample_b)
        if is_next_random:
            split_A, split_B = split_B, split_A

        # Trim the subsegments from either end to a desired joint length
        length_A = len(split_A)
        length_B = len(split_B)
        if length_A + length_B <= target_sequence_length:
            truncated = False
        else:
            while length_A + length_B > target_sequence_length:
                split = split_A if length_A > length_B else split_B
                if numpy_random_state.random() < 0.5:
                    del split[0]
                else:
                    del split[-1]
                length_A = len(split_A)
                length_B = len(split_B)
            truncated = True

        # Merge the subsegments and create the token assignment labels
        tokens = [
            self.config.tokenizer.cls,
            *split_A,
            self.config.tokenizer.sep,
        ]
        assignments = [0 for _ in range(1 + len(split_A) + 1)]
        if split_B:
            tokens += [*split_B, self.config.tokenizer.sep]
            assignments += [1 for _ in range(len(split_B) + 1)]

        # Masking
        tokens, masked_positions, masked_labels, _, _ = self._create_masked_lm_predictions(
            tokens, target_sequence_length, numpy_random_state
        )

        # Pad the sequences and convert to NumPy
        length_toks = len(tokens)
        length_pads = self.config.sequence_length - length_toks
        assert length_pads >= 0

        tokens = numpy.array(tokens, dtype=numpy.int64)
        tokens = numpy.pad(tokens, (0, length_pads), constant_values=self.config.tokenizer.pad)

        assignments = numpy.array(assignments, dtype=numpy.int64)
        assignments = numpy.pad(
            assignments, (0, length_pads), constant_values=self.config.tokenizer.pad
        )

        # Get the padding mask
        mask_pads = numpy.ones(length_toks, dtype=numpy.int64)
        mask_pads = numpy.pad(
            mask_pads, (0, length_pads), constant_values=self.config.tokenizer.pad
        )

        # Mask the labels
        labels = numpy.zeros(self.config.sequence_length, dtype=numpy.int64) - 1
        labels[masked_positions] = masked_labels

        # Get the loss mask
        mask_loss = numpy.zeros(self.config.sequence_length, dtype=numpy.int64)
        mask_loss[masked_positions] = 1

        return {
            "text": tokens,
            "types": assignments,
            "labels": labels,
            "is_random": int(is_next_random),
            "padding_mask": mask_pads,
            "loss_mask": mask_loss,
            "truncated": int(truncated),
        }

    def _get_token_mask(self, numpy_random_state: numpy.random.RandomState) -> Optional[int]:
        """Abstract method implementation

        80% of the time, replace the token id with mask token id. 10% of the time, replace token id
        with a random token id from the vocabulary. 10% of the time, do nothing.

        Args:
            numpy_random_state (RandomState): The NumPy random state

        Returns:
            Optional[int]: The replacement token id or None
        """
        if numpy_random_state.random() < 0.8:
            return self.config.tokenizer.mask
        else:
            if numpy_random_state.random() >= 0.5:
                return self.token_lookup[numpy_random_state.randint(0, len(self.token_lookup))]
        return None
