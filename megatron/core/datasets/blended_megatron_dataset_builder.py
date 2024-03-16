# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import logging
import math
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union

import numpy
import torch

from megatron.core.datasets.blended_dataset import BlendedDataset
from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset, MockDataset
from megatron.core.datasets.utils import Split, normalize

logger = logging.getLogger(__name__)

MidLevelDataset = Union[MegatronDataset, MockDataset]

TopLevelDataset = Union[BlendedDataset, MidLevelDataset]

DistributedDataset = Union[
    TopLevelDataset, MidLevelDataset, LowLevelDataset, torch.utils.data.Dataset
]


class BlendedMegatronDatasetBuilder(object):
    """Builder class for the BlendedDataset and MegatronDataset classes

    Args:
        cls (Type[MegatronDataset]): The class to instantiate, must inherit from MegatronDataset

        sizes (List[int]): The minimum number of total samples to draw from each split, varies with blend

        config (BlendedMegatronDatasetConfig): The config object which informs dataset creation
    """

    def __init__(
        self, cls: Type[MidLevelDataset], sizes: List[int], config: BlendedMegatronDatasetConfig,
    ):
        self.cls = cls
        self.sizes = sizes
        self.config = config

        assert not self.config.mock or issubclass(self.cls, MockDataset)

    def build(self) -> List[Optional[TopLevelDataset]]:
        """Build all dataset splits according to the provided blend(s)
        
        This method is distributed-aware and must be called on all ranks.
        
        The dataset splits returned can vary according to the config. Supply config.blend and
        config.split to build BlendedDataset and/or MegatronDataset splits from the same
        distribution. Supply config.blend_per_split to build BlendedDataset and/or MegatronDataset
        splits from separate distributions.

        Returns:
            List[Optional[TopLevelDataset]]: A list containing a dataset instance (or None) per split
        """
        return self._build_blended_dataset_splits()

    def _build_blended_dataset_splits(self,) -> List[Optional[TopLevelDataset]]:
        """Build all dataset splits according to the provided blend(s)
        
        See the BlendedMegatronDatasetBuilder.build alias for more information.

        Returns:
            List[Optional[TopLevelDataset]]: A list containing a dataset instance (or None) per split
        """

        # Return fake "mock" datasets
        if self.config.mock:

            return self._build_megatron_dataset_splits(None, None, self.sizes)

        # All splits come from the same distribution
        elif self.config.blend:
            blend = self.config.blend
            split = self.config.split_matrix

            # Blend consists of a single prefix
            if len(blend) == 1:
                return self._build_megatron_dataset_splits(blend[0], split, self.sizes)

            # Blend consists of multiple weights and prefixes
            (
                prefix_per_dataset,
                weight_per_dataset,
                sizes_per_dataset,
            ) = _get_prefixes_weights_and_sizes_for_blend(blend, self.sizes)

            megatron_datasets = [[] for _ in range(len(Split))]

            for i in range(len(prefix_per_dataset)):
                megatron_datasets_split = self._build_megatron_dataset_splits(
                    prefix_per_dataset[i], split, sizes_per_dataset[i]
                )
                for j in range(len(megatron_datasets_split)):
                    megatron_datasets[j].append(megatron_datasets_split[j])

            # Sum over all contributing datasets, per split
            size_per_split = list(map(sum, zip(*sizes_per_dataset)))

            blended_datasets = []

            for i in range(len(megatron_datasets)):
                is_none = map(lambda _: _ is None, megatron_datasets[i])

                if split[i] is None:
                    assert all(is_none)
                    blended_datasets.append(None)
                else:
                    assert all(is_none) or not any(is_none)
                    blended_datasets.append(
                        self.build_generic_dataset(
                            BlendedDataset,
                            self.config.is_built_on_rank,
                            megatron_datasets[i],
                            weight_per_dataset,
                            size_per_split[i],
                            self.config,
                        )
                    )

            return blended_datasets

        # Each split comes from a separate distribution
        else:
            blended_datasets = []
            for i in range(len(Split)):
                blend = self.config.blend_per_split[i]

                # Blend is not provided
                if not blend:
                    blended_datasets.append(None)
                    continue

                split_spoof = [None] * len(Split)
                split_spoof[i] = (0.0, 1.0)
                sizes_spoof = [0] * len(Split)
                sizes_spoof[i] = self.sizes[i]

                # Blend consists of a sigle prefix
                if len(blend) == 1:
                    blended_datasets.append(
                        self._build_megatron_dataset_splits(blend[0], split_spoof, sizes_spoof)[i]
                    )

                # Blend consists of multiple weights and prefixes
                else:
                    (
                        prefix_per_dataset,
                        weight_per_dataset,
                        sizes_per_dataset,
                    ) = _get_prefixes_weights_and_sizes_for_blend(blend, sizes_spoof)

                    megatron_datasets = []
                    for j in range(len(prefix_per_dataset)):
                        megatron_datasets.append(
                            self._build_megatron_dataset_splits(
                                prefix_per_dataset[j], split_spoof, sizes_per_dataset[j],
                            )[i]
                        )

                    size_per_split = list(map(sum, zip(*sizes_per_dataset)))

                    blended_datasets.append(
                        self.build_generic_dataset(
                            BlendedDataset,
                            self.config.is_built_on_rank,
                            megatron_datasets,
                            weight_per_dataset,
                            size_per_split[i],
                            self.config,
                        )
                    )

            return blended_datasets

    def _build_megatron_dataset_splits(
        self, dataset_path: Optional[str], split: List[float], sizes: List[int],
    ) -> List[Optional[MidLevelDataset]]:
        """Build each MidLevelDataset split from a single LowLevelDataset

        Args:
            dataset_path (Optional[str]): The path on disk which defines the underlying LowLevelDataset, e.g. the .bin and .idx file prefix when self.cls is of type IndexedMegatronDataset or None when self.cls is of type MockDataset

            split (List[Tuple[float, float]]): The dataset split matrix

            sizes (List[int]): The number of total samples to draw from each split

        Returns:
            List[Optional[MidLevelDataset]]: The MidLevelDataset (or None) per split
        """
        # Build the low level dataset
        if issubclass(self.cls, MockDataset):
            low_level_dataset = None
        elif issubclass(self.cls, MegatronDataset):
            low_level_dataset = self.cls.build_low_level_dataset(dataset_path, self.config)
        else:
            raise NotImplementedError

        # Build the split indices for the low level dataset
        if low_level_dataset is not None:
            num_elements = self.cls.numel_low_level_dataset(low_level_dataset)
            split_indices = []
            for i, _ in enumerate(Split):
                if split[i] is not None:
                    beg = int(round(split[i][0] * float(num_elements)))
                    end = int(round(split[i][1] * float(num_elements)))
                    split_indices.append(
                        numpy.arange(start=beg, stop=end, step=1, dtype=numpy.int32)
                    )
                else:
                    split_indices.append(None)
        else:
            split_indices = [None for _ in Split]

        # Build the mid level dataset
        mid_level_datasets = []
        for i, _split in enumerate(Split):
            if not self.config.mock and split[i] is None:
                mid_level_datasets.append(None)
            else:
                mid_level_datasets.append(
                    self.build_generic_dataset(
                        self.cls,
                        self.config.is_built_on_rank,
                        low_level_dataset,
                        dataset_path,
                        split_indices[i],
                        sizes[i],
                        _split,
                        self.config,
                    )
                )

        return mid_level_datasets

    @staticmethod
    def build_generic_dataset(
        cls: Union[Type[DistributedDataset], Callable], is_built_on_rank: Callable, *args: Any
    ) -> Optional[Union[DistributedDataset, Iterable]]:
        """Build the DistributedDataset

        Return None if and only if the underlying dataset class is not built on the current rank
        and torch.distributed is initialized.

        Args:
            cls (Union[Type[DistributedDataset], Callable]): The DistributedDataset class to be built. In special cases, e.g. when we are building the low level dataset for a RawMegatronDataset instance, we can accept a Callable which returns an Iterable.

            args (Tuple[Any]): The positional arguments used to build the provided DistributedDataset class

        Raises:
            Exception: When the dataset constructor raises an OSError

        Returns:
            Optional[Union[DistributedDataset, Iterable]]: The DistributedDataset instantion, the Iterable instantiation, or None
        """
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()

            dataset = None

            # First, build on rank 0
            if rank == 0 and is_built_on_rank():
                try:
                    dataset = cls(*args)
                except OSError as err:
                    log = (
                        f"Failed to write dataset materials to the data cache directory. "
                        + f"Please supply a directory to which you have write access via "
                        + f"the path_to_cache attribute in BlendedMegatronDatasetConfig and "
                        + f"retry. Refer to the preserved traceback above for more information."
                    )
                    raise Exception(log) from err

            torch.distributed.barrier()

            # After, build on other ranks
            if rank != 0 and is_built_on_rank():
                dataset = cls(*args)

            return dataset

        return cls(*args)


def _get_prefixes_weights_and_sizes_for_blend(
    blend: List[str], target_num_samples_per_split: List[int]
) -> Tuple[List[str], List[float], List[List[int]]]:
    """Determine the contribution of the MegatronDataset splits to the BlendedDataset splits
    
    Args:
        blend (List[str]): e.g. ["30", "path/to/dataset_1_prefix", "70", "path/to/dataset_2_prefix"]

        target_num_samples_per_split (List[int]): The number of samples to target for each BlendedDataset split

    Returns:
        Tuple[List[str], List[float], List[List[int]]]: The prefix strings e.g. ["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], the normalized weights e.g. [0.3, 0.7], and the number of samples to request per MegatronDataset per split
    """
    weights, prefixes = zip(
        *[(float(blend[i]), blend[i + 1].strip()) for i in range(0, len(blend), 2)]
    )

    weights = normalize(weights)

    # Use 0.5% target margin to ensure we satiate the network
    sizes_per_dataset = [
        [
            int(math.ceil(target_num_samples * weight * 1.005))
            for target_num_samples in target_num_samples_per_split
        ]
        for weight in weights
    ]

    return prefixes, weights, sizes_per_dataset
