# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import hashlib
import json
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Union

import numpy
import torch

from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.utils import Split

LowLevelDataset = Union[IndexedDataset, Iterable]


_PAD_TOKEN_ID = -1


class MegatronDataset(ABC, torch.utils.data.Dataset):
    """The highest level wrapper class from which all dataset classes should inherit

    Args:
        dataset (LowLevelDataset): The dataset around which to build the MegatronDataset

        dataset_path (Optional[str]): The real path on disk to the dataset, for bookkeeping

        indices (numpy.ndarray): The set of the documents indices to expose

        num_samples (Optional[int]): The minimum number of samples to build from the indexed
            dataset. When None, build as many samples as correspond to one epoch.

        index_split (Split): The indices Split

        config (BlendedMegatronDatasetConfig): The config
    """

    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: Optional[str],
        indices: numpy.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: BlendedMegatronDatasetConfig,
    ) -> None:
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.indices = indices
        self.num_samples = num_samples
        self.index_split = index_split
        self.config = config

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
            self.unique_description.encode("utf-8"), usedforsecurity=False
        ).hexdigest()

        # Handle pad token id provided by the tokenizer
        try:
            self._pad_token_id = self.config.tokenizer.pad
        except Exception:
            self._pad_token_id = _PAD_TOKEN_ID

        # Check if pad token id collides with any other special tokens
        try:
            _special_tokens_list = [
                v for k, v in self.config.tokenizer.special_tokens_dict.items() if k != "pad_token"
            ]
        except (AttributeError, IndexError, ValueError):
            _special_tokens_list = []
        # If the tokenizer does not have a special_tokens_dict attribute, at least check eos and eod
        if not _special_tokens_list:
            try:
                _special_tokens_list.append(self.config.tokenizer.eos)
            except (AttributeError, NotImplementedError):
                pass
            try:
                _special_tokens_list.append(self.config.tokenizer.eod)
            except (AttributeError, NotImplementedError):
                pass

        if self._pad_token_id in _special_tokens_list:
            if self.config.allow_ambiguous_pad_tokens:
                # This will break training, but users must explicitly opt-in to this behavior.
                warnings.warn(
                    "The pad token id in the tokenizer collides with another special token id. "
                    "This may cause instability and lack of covergence during training. "
                    "Do not ignore this warning if you do not understand the implications. "
                )
            else:
                # Reset the pad token id to a value which is guaranteed not to be in the dataset.
                self._pad_token_id = _PAD_TOKEN_ID
                warnings.warn(
                    "The pad token id in the tokenizer collides with another special token id. "
                    "This may cause instability and lack of covergence during training. "
                    "As such, the training flow will avoid masking out any pad tokens already "
                    "present in the dataset. If you would like to disable this behavior, "
                    "please provide a tokenizer with a uniquely-defined pad token id."
                )

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
