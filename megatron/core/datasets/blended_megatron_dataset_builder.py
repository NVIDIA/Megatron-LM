# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import logging
import math
from typing import Any, Callable, Iterable, List, Optional, Type, Union

import numpy
import torch

from megatron.core.datasets.blended_dataset import BlendedDataset
from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset, MockDataset
from megatron.core.datasets.utils import Split, log_single_rank, normalize
from megatron.core.parallel_state import get_virtual_pipeline_model_parallel_rank

logger = logging.getLogger(__name__)

MidLevelDataset = MegatronDataset

TopLevelDataset = Union[BlendedDataset, MidLevelDataset]

DistributedDataset = Union[
    TopLevelDataset, MidLevelDataset, LowLevelDataset, torch.utils.data.Dataset
]


class BlendedMegatronDatasetBuilder(object):
    """Builder class for the BlendedDataset and MegatronDataset classes

    Args:
        cls (Type[MegatronDataset]): The class to instantiate, must inherit from MegatronDataset

        sizes (List[Optional[int]]): The minimum total number of samples to draw, or None, per split

        is_built_on_rank (Callable): A callable which returns True if the dataset should be built on the current rank and False otherwise. It should be Megatron Core parallelism aware i.e. global rank, local group rank, and virtual rank may inform its return value.

        config (BlendedMegatronDatasetConfig): The config object which informs dataset creation
    """

    def __init__(
        self,
        cls: Type[MidLevelDataset],
        sizes: List[int],
        is_built_on_rank: Callable,
        config: BlendedMegatronDatasetConfig,
    ):
        self.cls = cls
        self.sizes = sizes
        self.is_built_on_rank = is_built_on_rank
        self.config = config

        log_single_rank(
            logger,
            logging.WARNING,
            f"Building dataset splits with cls={cls.__name__}, sizes={self.sizes}, and config={self.config}",
        )

        if self.config.mock:
            assert issubclass(self.cls, MockDataset)
        else:
            for split in Split:
                size_is_none = self.sizes[split.value] is None
                if self.config.blend_per_split is None:
                    weights_are_none = self.config.blend[1] is None
                else:
                    if self.config.blend_per_split[split.value] is None:
                        continue
                    weights_are_none = self.config.blend_per_split[split.value][1] is None
                if size_is_none:
                    assert (
                        weights_are_none
                    ), f"size_is_none => weights_are_none fails for {split.name} split"

        if torch.distributed.is_initialized():
            gb_rank = torch.distributed.get_rank()
            vp_rank = get_virtual_pipeline_model_parallel_rank()
            if gb_rank == 0 and (vp_rank == 0 or vp_rank is None):
                assert (
                    self.is_built_on_rank()
                ), "is_built_on_rank must return True when global rank = 0 and vp rank = 0"

    def build(self) -> List[Optional[TopLevelDataset]]:
        """Build all dataset splits according to the provided blend(s)
        
        This method is distributed-aware and must be called on all ranks.
        
        The dataset splits returned can vary according to the config. Supply config.blend and
        config.split to build BlendedDataset and/or MegatronDataset splits from the same
        distribution. Supply config.blend_per_split to build BlendedDataset and/or MegatronDataset
        splits from separate distributions. In either case, for each split, handle the following
        cases:

        (1) The split is None
            - do nothing

        (2) The split has one contributing dataset, and...

            (a) 'size' is not None
                - Build a mid-level dataset with low-level dataset sampling in proportion to the size            

            (b) 'size' is None
                - Build mid-level datasets with no excess low-level dataset sampling

        (3) The split has multiple contributing datasets, and...

            (a) 'weights' is not None and 'size' is not None
                - Build mid-level datasets with low-level dataset sampling in proportion to their weights and the size
                - Build a top-level dataset of length marginally greater than 'size' with mid-level dataset sampling in proportion to their weights and the size

            (b) 'weights' is not None and 'size' is None
                - Error

            (c) 'weights' is None and 'size' is not None
                - Build mid-level datasets with no excess low-level dataset sampling
                - Build a top-level dataset of length 'size' with mid-level dataset sampling in proportion to their lengths and the size
                    - The 'size' of the top-level dataset is capped at the sum of the mid-level dataset lengths

            (d) 'weights' is None and 'size' is None
                - Build mid-level datasets with no excess low-level dataset sampling
                - Build a top-level dataset with no excess mid-level dataset sampling

        Returns:
            List[Optional[TopLevelDataset]]: A list containing a dataset instance (or None) per split
        """
        datasets = self._build_blended_dataset_splits()

        for dataset in datasets:
            if dataset is not None and len(dataset) > 0:
                if isinstance(dataset, BlendedDataset):
                    # Check blend size
                    assert dataset.size is None or dataset.size == dataset.dataset_index.shape[0]
                    # Check blend access of mid-level datasets
                    _, sizes = numpy.unique(dataset.dataset_index, return_counts=True)
                    for i, dataset_and_size in enumerate(zip(dataset.datasets, sizes)):
                        if len(dataset_and_size[0]) < dataset_and_size[1]:
                            raise IndexError(
                                f"{type(dataset).__name__} blend goes out of bounds for {type([dataset_and_size[0]]).__name__} {i} for {dataset.split.name} split"
                            )

        return datasets

    def _build_blended_dataset_splits(self,) -> List[Optional[TopLevelDataset]]:
        """Build all dataset splits according to the provided blend(s)
        
        See the BlendedMegatronDatasetBuilder.build alias for more information.

        Returns:
            List[Optional[TopLevelDataset]]: A list containing a dataset instance (or None) per split
        """
        ##
        # Return fake "mock" datasets
        ##
        if self.config.mock:
            return self._build_megatron_dataset_splits(None, None, self.sizes)

        ##
        # All splits come from the same distribution
        ##
        elif self.config.blend:
            prefixes, weights = self.config.blend
            if weights is not None:
                weights = normalize(weights)

            split = self.config.split_matrix

            # Blend consists of a single prefix
            if len(prefixes) == 1:
                return self._build_megatron_dataset_splits(prefixes[0], split, self.sizes)

            # Build the mid-level datasets
            if weights is None:
                sizes_per_dataset = [[None for split in Split] for prefix in prefixes]
            else:
                sizes_per_dataset = _get_size_per_split_per_dataset(weights, self.sizes)
            megatron_datasets = [[] for _ in range(len(Split))]
            for i in range(len(prefixes)):
                megatron_datasets_split = self._build_megatron_dataset_splits(
                    prefixes[i], split, sizes_per_dataset[i]
                )
                for j in range(len(megatron_datasets_split)):
                    megatron_datasets[j].append(megatron_datasets_split[j])

            # Build the top-level datasets
            blended_datasets = [None] * len(Split)
            for i in range(len(Split)):
                if split[i] is not None:
                    weights_i = weights
                    if weights_i is not None and self.sizes[i] is not None:
                        size_i = sum(list(zip(*sizes_per_dataset))[i])
                    elif weights_i is None:
                        try:
                            weights_i = [
                                len(megatron_dataset) for megatron_dataset in megatron_datasets[i]
                            ]
                        except TypeError:
                            weights_i = [0 for _ in prefixes]
                        if self.sizes[i] is not None:
                            size_i = min(self.sizes[i], sum(weights_i))
                        else:
                            size_i = None  # => the size will be sum(weights_i)
                    else:
                        raise RuntimeError
                    blended_datasets[i] = self.build_generic_dataset(
                        BlendedDataset,
                        self.is_built_on_rank,
                        megatron_datasets[i],
                        weights_i,
                        size_i,
                        self.config,
                    )

            return blended_datasets

        ##
        # Each split comes from a separate distribution
        ##
        else:
            blended_datasets = [None] * len(Split)
            for i in range(len(Split)):
                split_spoof = [None] * len(Split)
                split_spoof[i] = (0.0, 1.0)
                sizes_spoof = [0] * len(Split)
                sizes_spoof[i] = self.sizes[i]

                # Blend is provided for the split
                blend = self.config.blend_per_split[i]
                if blend is not None:
                    prefixes, weights = blend
                    if weights is not None:
                        weights = normalize(weights)

                    # Blend consists of a sigle prefix
                    if len(prefixes) == 1:
                        blended_datasets[i] = self._build_megatron_dataset_splits(
                            prefixes[0], split_spoof, sizes_spoof
                        )[i]
                        continue

                    # Build mid-level datasets
                    if weights is None:
                        sizes_per_dataset = [[None for split in Split] for prefix in prefixes]
                    else:
                        sizes_per_dataset = _get_size_per_split_per_dataset(weights, sizes_spoof)
                    megatron_datasets = []
                    for j in range(len(prefixes)):
                        megatron_datasets.append(
                            self._build_megatron_dataset_splits(
                                prefixes[j], split_spoof, sizes_per_dataset[j],
                            )[i]
                        )

                    # Build top-level dataset
                    if weights is not None and self.sizes[i] is not None:
                        size = list(map(sum, zip(*sizes_per_dataset)))[i]
                    elif weights is None:
                        try:
                            weights = [
                                len(megatron_dataset) for megatron_dataset in megatron_datasets
                            ]
                        except TypeError:
                            weights = [0 for _ in prefixes]
                        if self.sizes[i] is not None:
                            size = min(self.sizes[i], sum(weights))
                        else:
                            size = None  # => the size will be sum(weights)
                    else:
                        raise RuntimeError
                    blended_datasets[i] = self.build_generic_dataset(
                        BlendedDataset,
                        self.is_built_on_rank,
                        megatron_datasets,
                        weights,
                        size,
                        self.config,
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
                        self.is_built_on_rank,
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


def _get_size_per_split_per_dataset(
    normalized_weights: List[float], target_size_per_split: List[int]
) -> List[List[int]]:
    """Determine the contribution of the MegatronDataset splits to the BlendedDataset splits
    
    Args:
        normalized_weights (List[float]): e.g. [0.3, 0.7]

        target_size_per_split (List[int]): The number of samples to target for each BlendedDataset split

    Returns:
        List[List[int]]: The number of samples to request per MegatronDataset per split
    """
    assert numpy.isclose(sum(normalized_weights), 1.0)

    # Use 0.5% target margin to ensure we satiate the request
    sizes_per_dataset = [
        [int(math.ceil(target_size * weight * 1.005)) for target_size in target_size_per_split]
        for weight in normalized_weights
    ]

    return sizes_per_dataset
