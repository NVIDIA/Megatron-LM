# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import logging
import math
from typing import Any, List, Optional, Tuple, Type, Union

import numpy
import torch

from megatron.core.datasets.blended_dataset import BlendedDataset
from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.indexed_dataset import MMapIndexedDataset
from megatron.core.datasets.megatron_dataset import MegatronDataset
from megatron.core.datasets.utils import Split, normalize

logger = logging.getLogger(__name__)

DistributedDataset = Union[BlendedDataset, MegatronDataset, MMapIndexedDataset]


class BlendedMegatronDatasetBuilder(object):
    """Builder class for the BlendedDataset and MegatronDataset classes

    Args:
        cls (Type[MegatronDataset]): The class to instantiate, must inherit from MegatronDataset

        sizes (List[int]): The minimum number of total samples to draw from each split, varies
        with blend

        config (BlendedMegatronDatasetConfig): The config object which informs dataset creation
    """

    def __init__(
        self, cls: Type[MegatronDataset], sizes: List[int], config: BlendedMegatronDatasetConfig,
    ):
        self.cls = cls
        self.sizes = sizes
        self.config = config

    def build(self) -> List[Optional[Union[BlendedDataset, MegatronDataset]]]:
        """Build all dataset splits according to the provided blend(s)
        
        This method is distributed-aware and must be called on all ranks.
        
        The dataset splits returned can vary according to the config. Supply config.blend and
        config.split to build BlendedDataset and/or MegatronDataset splits from the same
        distribution. Supply config.blend_per_split to build BlendedDataset and/or MegatronDataset
        splits from separate distributions.

        Returns:
            List[Optional[Union[BlendedDataset, MegatronDataset]]]: A list of either
            MegatronDataset or BlendedDataset (or None) per split
        """
        return self._build_blended_dataset_splits()

    def _build_blended_dataset_splits(
        self,
    ) -> List[Optional[Union[BlendedDataset, MegatronDataset]]]:
        """Build all dataset splits according to the provided blend(s)
        
        See the BlendedMegatronDatasetBuilder.build alias for more information.

        Returns:
            List[Optional[Union[BlendedDataset, MegatronDataset]]]: A list of either
            MegatronDataset or BlendedDataset (or None) per split
        """

        if getattr(self.config, "blend"):
            blend = getattr(self.config, "blend")
            split = getattr(self.config, "split_vector")

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

            # >>>
            # import json
            # from lutil import pax
            # def print_ds(ds):
            #     desc = json.loads(ds.unique_description)
            #     pax("desc")
            #     return "%s / %s" % (desc["index_split"], desc["path_prefix"])
            # pax(
            #     {f"megatron_datasets / {i}":"%s ... %s" % (len(d) if d else "--", d) for i,d in enumerate(megatron_datasets)},
            #     {"ds / 0": megatron_datasets[0]},
            #     {"ds / 1": megatron_datasets[1]},
            #     {"ds / 0 / 0": print_ds(megatron_datasets[0][0])},
            #     {"ds / 0 / 1": print_ds(megatron_datasets[0][1])},
            #     {"ds / 1 / 0": print_ds(megatron_datasets[1][0])},
            #     {"ds / 1 / 1": print_ds(megatron_datasets[1][1])},
            # )
            # <<<
            for i in range(len(megatron_datasets)):
                is_none = map(lambda _: _ is None, megatron_datasets[i])

                if split[i] == 0.0:
                    assert all(is_none)
                    blended_datasets.append(None)
                else:
                    assert all(is_none) or not any(is_none)
                    # >>>
                    # from lutil import pax
                    # pax({"dss": megatron_datasets[i]})
                    # <<<
                    blended_datasets.append(
                        self._build_generic_dataset(
                            BlendedDataset,
                            megatron_datasets[i],
                            weight_per_dataset,
                            size_per_split[i],
                            self.config,
                        )
                    )

            return blended_datasets

        else:
            blended_datasets = []
            for i in range(len(Split)):
                blend = getattr(self.config, "blend_per_split")[i]

                # Blend is not provided
                if not blend:
                    blended_datasets.append(None)
                    continue

                split_spoof = [0.0] * len(Split)
                split_spoof[i] = 1.0
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
                        self._build_generic_dataset(
                            BlendedDataset,
                            megatron_datasets,
                            weight_per_dataset,
                            size_per_split[i],
                            self.config,
                        )
                    )

            return blended_datasets

    def _build_megatron_dataset_splits(
        self, path_prefix: str, split: List[float], sizes: List[int],
    ) -> List[Optional[MegatronDataset]]:
        """Build each MegatronDataset split from a single MMapIndexedDataset

        Args:
            path_prefix (str): The MMapIndexedDataset .bin and .idx file prefix

            split (List[float]): The dataset split ratios (must sum to 1.00)

            sizes (List[int]): The number of total samples to draw from each split

        Returns:
            List[Optional[MegatronDataset]]: The MegatronDatset (or None) per split
        """
        indexed_dataset = self._build_generic_dataset(
            MMapIndexedDataset, path_prefix, self.cls.is_multimodal()
        )

        if indexed_dataset is not None:
            if self.cls.is_split_by_sequence():
                split_idx_bounds = _get_split_indices(
                    split, indexed_dataset.sequence_lengths.shape[0]
                )
            else:
                split_idx_bounds = _get_split_indices(
                    split, indexed_dataset.document_indices.shape[0] - 1
                )
            split_indices = [
                numpy.arange(
                    start=split_idx_bounds[i],
                    stop=split_idx_bounds[i + 1],
                    step=1,
                    dtype=numpy.int32,
                )
                for i, _ in enumerate(Split)
            ]
        else:
            split_indices = [None for _ in Split]

        megatron_datasets = []
        for i, _split in enumerate(Split):
            if split[i] == 0.0:
                megatron_datasets.append(None)
            else:
                megatron_datasets.append(
                    self._build_generic_dataset(
                        self.cls, indexed_dataset, split_indices[i], sizes[i], _split, self.config
                    )
                )

        return megatron_datasets

    def _build_generic_dataset(
        self, cls: Type[DistributedDataset], *args: Any,
    ) -> Optional[DistributedDataset]:
        """Build the DistributedDataset

        Return None if and only if the underlying MegatronDataset class is not built on the current
        rank and torch.distributed is initialized.

        Args:
            cls (Type[DistributedDataset]): The DistributedDataset class to be built

            args (Tuple[Any]): The positional arguments used to build the provided
            DistributedDataset class

        Raises:
            Exception: When the dataset constructor raises an OSError

        Returns:
            Optional[DistributedDataset]: The DistributedDataset instantion or None
        """
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()

            dataset = None

            # First, build on rank 0
            if rank == 0 and getattr(self.config, "is_built_on_rank")():
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
            if rank != 0 and getattr(self.config, "is_built_on_rank")():
                dataset = cls(*args)

            return dataset

        return cls(*args)


def _get_split_indices(split: List[float], num_elements: int) -> List[int]:
    """Determine the document index bounds per split

    Args:
        split (List[float]): The dataset split ratios (must sum to 1.00)

        num_elements (int): The number of elements, e.g. sequences or documents, available for
        the split

    Returns:
        List[int]: The indices for all three splits e.g. [0, 900, 990, 1000] for a 1000-document
        set and a [90.0, 9.0, 1.0] split
    """
    split_indices = [0]
    for split_pct in split:
        split_indices.append(split_indices[-1] + int(round(split_pct * float(num_elements))))
    split_indices[1:] = list(
        map(lambda _: _ - (split_indices[-1] - num_elements), split_indices[1:])
    )

    assert len(split_indices) == len(split) + 1
    assert split_indices[-1] == num_elements

    return split_indices


def _get_prefixes_weights_and_sizes_for_blend(
    blend: List[str], target_num_samples_per_split: List[int]
) -> Tuple[List[str], List[float], List[List[int]]]:
    """Determine the contribution of the MegatronDataset splits to the BlendedDataset splits
    
    Args:
        blend (List[str]): e.g. ["30", "path/to/dataset_1_prefix", "70", 
        "path/to/dataset_2_prefix"]

        target_num_samples_per_split (List[int]): The number of samples to target for each
        BlendedDataset split

    Returns:
        Tuple[List[str], List[float], List[List[int]]]: The prefix strings e.g.
        ["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], the normalized weights e.g.
        [0.3, 0.7], and the number of samples to request per MegatronDataset per split
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
