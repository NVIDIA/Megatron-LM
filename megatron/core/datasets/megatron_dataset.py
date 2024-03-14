# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import hashlib
import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy
import torch

from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.utils import Split

LowLevelDataset = Union[IndexedDataset, Iterable]


class MegatronDataset(ABC, torch.utils.data.Dataset):
    """The highest level wrapper class from which all dataset classes should inherit

    Args:
        dataset (LowLevelDataset): The dataset around which to build the MegatronDataset

        dataset_path (str): The real path on disk to the dataset, for bookkeeping. TODO: subsume
        this argument by enforcing auto-bookkeeping in the dataset class type.

        indices (numpy.ndarray): The set of the documents indices to expose

        num_samples (int): The number of samples to draw from the indexed dataset

        index_split (Split): The indices Split

        config (BlendedMegatronDatasetConfig): The config
    """

    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: str,
        indices: numpy.ndarray,
        num_samples: int,
        index_split: Split,
        config: BlendedMegatronDatasetConfig,
    ) -> None:
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.indices = indices
        self.num_samples = num_samples
        self.index_split = index_split
        self.config = config

        if not self.config.mock:
            self.unique_identifiers = OrderedDict()
            self.unique_identifiers["class"] = type(self).__name__
            self.unique_identifiers["dataset_path"] = self.dataset_path
            self.unique_identifiers["num_samples"] = self.num_samples
            self.unique_identifiers["index_split"] = self.index_split.name
            for attr in self._key_config_attributes():
                self.unique_identifiers[attr] = getattr(self.config, attr)

            self.unique_description = json.dumps(
                self.unique_identifiers, indent=4, default=lambda obj: obj.unique_identifiers
            )
            self.unique_description_hash = hashlib.md5(
                self.unique_description.encode("utf-8")
            ).hexdigest()

        self._finalize()

    def _finalize(self) -> None:
        """Build the dataset and assert any subclass-specific conditions
        """
        pass

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        """Return the number of elements in the underlying low level dataset for the purpose of
        segregating the train/valid/test split indices

        It may be that the low level dataset can be split any number of ways, depending on the mid
        level dataset it supports, which is why we define the "number of elements" function
        separately from the __len__ function here in the mid level dataset class

        Args:
            low_level_dataset (LowLevelDataset): The underlying low level dataset

        Returns:
            int: The number of elements in the underlying low level dataset
        """
        raise NotImplementedError

    @staticmethod
    def build_low_level_dataset(
        dataset_path: str, config: BlendedMegatronDatasetConfig
    ) -> LowLevelDataset:
        """Build the low level dataset via a function to be called from within
        BlendedMegatronDatasetBuilder.build_generic_dataset

        It may be that the low level dataset spans any subset of train/valid/test splits, which is
        why we define a static "build" function separately from the constructor in the mid level
        dataset class

        Args:
            dataset_path (str): The real path on disk to the dataset

            config (BlendedMegatronDatasetConfig): The dataset config

        Returns:
            LowLevelDataset: The low level dataset
        """
        raise NotImplementedError

    @staticmethod
    def _key_config_attributes() -> List[str]:
        """Return all config attributes which contribute to uniquely identifying the dataset.

        These attributes will be used to build a uniquely identifying string and MD5 hash which
        will be used to cache/load dataset resources from run to run.

        Returns:
            List[str]: The key config attributes
        """
        return ["random_seed", "sequence_length", "split", "split_matrix", "tokenizer"]

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


class MockDataset(MegatronDataset):
    """The highest level wrapper class from which all mock dataset classes should inherit

    The MockDataset is a special, one-off class that should not serve as a precedent for developers
    seeking to extend the MegatronDataset. This class is incompatible with BlendedDataset

    This class cannibalizes the constructor of the parent class. As such, we do not need to
    pass in some constructor parameters. They may be populated, but most are superfluous and can
    be None. Only num_samples, index_split, and config are required.


    Args:
        dataset (Optional[LowLevelDataset]): The dataset around which to build the MegatronDataset

        dataset_path (Optional[str]): The real path on disk to the dataset, for bookkeeping. TODO: subsume
        this argument by enforcing auto-bookkeeping in the dataset class type.

        indices (Optional[numpy.ndarray]): The set of the documents indices to expose

        num_samples (int): The number of samples to draw from the indexed dataset

        index_split (Split): The indices Split

        config (BlendedMegatronDatasetConfig): The config
    """

    def __init__(
        self,
        dataset: Optional[LowLevelDataset],
        dataset_path: Optional[str],
        indices: Optional[numpy.ndarray],
        num_samples: int,
        index_split: Split,
        config: BlendedMegatronDatasetConfig,
    ) -> None:
        self.config = config
        assert self.config.mock

        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

    def __len__(self) -> int:
        """Return an arbitrary length

        Returns:
            int: The total number of samples that are present in the dataset
        """
        return self.num_samples
