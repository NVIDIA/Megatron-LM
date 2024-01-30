# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy
import torch

from megatron.core.datasets.blended_megatron_dataset_config import (
    convert_split_vector_to_split_matrix,
    parse_and_normalize_split,
)
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.indexed_dataset import MMapIndexedDataset
from megatron.core.datasets.utils import Split, log_single_rank

logger = logging.getLogger(__name__)


@dataclass
class MultiSplitGPTDatasetConfig(GPTDatasetConfig):
    """Configuration object for Megatron Core blended and megatron Retro datasets

    Attributes:
        return_document_ids (bool): Whether to return the document ids when querying the dataset.
        Turn this option on during preprocessing.

        split_preprocessing (str): The Retro preprocessing split string. It follows the same
        pattern convention as 'split'. Not to be used with 'blend_per_split'.
    """

    return_document_ids: bool = None

    split_preprocessing: str = None

    def __post_init__(self):
        super().__post_init__()
        assert self.split is not None, "the Retro data pipeline does not support 'blend_per_split'"
        assert self.return_document_ids is not None, "this attribute must be user defined"
        assert self.split_preprocessing is not None, "this attribute must be user defined"
        split_vector = parse_and_normalize_split(self.split)
        split_preprocessing_vector = parse_and_normalize_split(self.split_preprocessing)
        if not numpy.allclose(split_vector, split_preprocessing_vector):
            self.split_matrix = convert_split_vector_to_split_matrix(
                split_vector, split_preprocessing_vector
            )
            log_single_rank(
                logger,
                logging.WARNING,
                f"split =/= split_preprocessing. Let split_matrix = {self.split_matrix}",
            )


class MultiSplitGPTDataset(GPTDataset):
    """Retro's customized GPT dataset.

    Args:
        indexed_dataset (MMapIndexedDataset): The MMapIndexedDataset around which to build the
        MegatronDataset

        dataset_path (str): The real path on disk to the dataset, for bookkeeping

        indexed_indices (numpy.ndarray): The set of the documents indices to expose

        num_samples (int): The number of samples to draw from the indexed dataset

        index_split (Split): The indexed_indices Split

        config (MultiSplitGPTDatasetConfig): The Retro-specific container for all config sourced parameters
    """

    def __init__(
        self,
        indexed_dataset: MMapIndexedDataset,
        dataset_path: str,
        indexed_indices: numpy.ndarray,
        num_samples: int,
        index_split: Split,
        config: MultiSplitGPTDatasetConfig,
    ) -> None:
        super().__init__(indexed_dataset, dataset_path, indexed_indices, num_samples, index_split, config)

    def __getitem__(self, idx: int) -> Dict[str, numpy.ndarray]:
        """Abstract method implementation

        Args:
            idx (int): The index into the dataset

        Returns:
            Dict[str, numpy.ndarray]: The text ids and (optionally) the document ids wrapped in a
            dictionary
        """
        text, document_ids = self._query_document_sample_shuffle_indices(idx)
        if self.config.return_document_ids:
            return {"text": text, "document_ids": document_ids}
        else:
            return {"text": text}

    @staticmethod
    def _key_config_attributes() -> List[str]:
        """Inherited method implementation

        The preprocessing split used for preprocessing will constrain the samples available for 
        pretraining.

        Returns:
            List[str]: The key config attributes
        """
        return super(MultiSplitGPTDataset, MultiSplitGPTDataset)._key_config_attributes() + [
            "split_preprocessing"
        ]
