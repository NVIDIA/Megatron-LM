# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import hashlib
import json
from abc import ABC, abstractmethod, abstractstaticmethod
from collections import OrderedDict
from typing import Dict, List, Union

import numpy
import torch

from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.indexed_dataset import MMapIndexedDataset
from megatron.core.datasets.utils import Split


class MegatronDataset(ABC, torch.utils.data.Dataset):
    """The wrapper class from which dataset classes should inherit e.g. GPTDataset

    Args:
        indexed_dataset (MMapIndexedDataset): The MMapIndexedDataset around which to build the
        MegatronDataset

        indexed_indices (numpy.ndarray): The set of the documents indices to expose

        num_samples (int): The number of samples to draw from the indexed dataset

        index_split (Split): The indexed_indices Split

        config (BlendedMegatronDatasetConfig): The container for all config sourced parameters
    """

    def __init__(
        self,
        indexed_dataset: MMapIndexedDataset,
        indexed_indices: numpy.ndarray,
        num_samples: int,
        index_split: Split,
        config: BlendedMegatronDatasetConfig,
    ) -> None:
        assert indexed_indices.size > 0
        assert num_samples > 0
        assert self.is_multimodal() == indexed_dataset.multimodal
        assert self.is_split_by_sequence() != self.is_split_by_document()

        self.indexed_dataset = indexed_dataset
        self.indexed_indices = indexed_indices
        self.num_samples = num_samples
        self.index_split = index_split
        self.config = config

        self.unique_identifiers = OrderedDict()
        self.unique_identifiers["class"] = type(self).__name__
        self.unique_identifiers["path_prefix"] = self.indexed_dataset.path_prefix
        self.unique_identifiers["num_samples"] = self.num_samples
        self.unique_identifiers["index_split"] = self.index_split.name
        for attr in self._key_config_attributes():
            self.unique_identifiers[attr] = getattr(self.config, attr)

        self.unique_description = json.dumps(self.unique_identifiers, indent=4)
        self.unique_description_hash = hashlib.md5(
            self.unique_description.encode("utf-8")
        ).hexdigest()

        self._finalize()

    @abstractmethod
    def _finalize(self) -> None:
        """Build the dataset and assert any subclass-specific conditions
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the length of the dataset

        Returns:
            int: See abstract implementation
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, numpy.ndarray]]:
        """Return from the dataset

        Args:
            idx (int): The index into the dataset

        Returns:
            Dict[str, Union[torch.Tensor, numpy.ndarray]]: See abstract implementation
        """
        pass

    @abstractstaticmethod
    def is_multimodal() -> bool:
        """Return True if the inheritor class and its internal MMapIndexedDataset are multimodal

        Returns:
            bool: See abstract implementation
        """
        pass

    @abstractstaticmethod
    def is_split_by_sequence() -> bool:
        """Return whether the dataset is split by sequence

        For example, the GPT train/valid/test split is document agnostic

        Returns:
            bool: See abstract implementation
        """
        pass

    @classmethod
    def is_split_by_document(cls) -> bool:
        """Return whether the dataset is split by document

        For example, the BERT train/valid/test split is document aware

        Returns:
            bool: The negation of cls.is_split_by_sequence
        """
        return not cls.is_split_by_sequence()

    @staticmethod
    def _key_config_attributes() -> List[str]:
        """Return all config attributes which contribute to uniquely identifying the dataset.

        These attributes will be used to build a uniquely identifying string and MD5 hash which
        will be used to cache/load the dataset from run to run.

        Returns:
            List[str]: The key config attributes
        """
        return ["random_seed", "sequence_length", "split", "split_matrix"]
