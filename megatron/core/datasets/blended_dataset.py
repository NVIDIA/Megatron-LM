# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import numpy
import torch

from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.megatron_dataset import MegatronDataset
from megatron.core.datasets.utils import normalize
from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)

_VERBOSE = False


class BlendedDataset(torch.utils.data.Dataset):
    """Conjugating class for a set of MegatronDataset instances

    Args:
        datasets (List[MegatronDataset]): The MegatronDataset instances to blend

        weights (List[Union[int, float]]): The weights that determine the dataset blend ratios

        size (Optional[int]): The number of samples to draw from the blend. If None, for each
            dataset index idx draw exactly weights[idx] samples from datasets[idx].

        config (BlendedMegatronDatasetConfig): The config

    Raises:
        RuntimeError: When the dataset has fewer or more samples than 'size' post-initialization
    """

    def __init__(
        self,
        datasets: List[MegatronDataset],
        weights: List[Union[int, float]],
        size: Optional[int],
        config: BlendedMegatronDatasetConfig,
    ) -> None:
        assert len(datasets) == len(weights)
        assert len(datasets) < 32767
        assert all(map(lambda _: type(_) == type(datasets[0]), datasets))
        assert all(map(lambda _: _.index_split == datasets[0].index_split, datasets))
        assert all(map(lambda _: _ > 0, weights))
        assert all(map(lambda _: type(_) == type(weights[0]), weights))
        if size is None and isinstance(weights[0], float):
            assert all(map(lambda _: _ == int(_), weights))

        # Alert user to unnecessary blending
        if len(datasets) == 1:
            log_single_rank(
                logger, logging.WARNING, f"Building a BlendedDataset for a single MegatronDataset"
            )

        if size is not None:
            weights = normalize(weights)

        self.datasets = datasets
        self.split = self.datasets[0].index_split
        self.weights = weights
        self.size = size
        self.config = config

        unique_identifiers = OrderedDict()
        unique_identifiers["class"] = type(self).__name__
        unique_identifiers["datasets"] = [dataset.unique_identifiers for dataset in self.datasets]
        unique_identifiers["split"] = self.split.name
        unique_identifiers["weights"] = self.weights
        unique_identifiers["size"] = self.size

        self.unique_description = json.dumps(
            unique_identifiers, indent=4, default=lambda obj: obj.unique_identifiers
        )
        self.unique_description_hash = hashlib.md5(
            self.unique_description.encode("utf-8")
        ).hexdigest()

        self.dataset_index, self.dataset_sample_index = self._build_indices()

    def __len__(self) -> int:
        return self.dataset_index.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Union[int, numpy.ndarray]]:
        dataset_id = self.dataset_index[idx]
        dataset_sample_id = self.dataset_sample_index[idx]
        return {"dataset_id": dataset_id, **self.datasets[dataset_id][dataset_sample_id]}

    def _build_indices(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Build and optionally cache the dataset index and the dataset sample index

        The dataset index is a 1-D mapping which determines the dataset to query. The dataset
        sample index is a 1-D mapping which determines the sample to request from the queried
        dataset.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The dataset index and the dataset sample index
        """
        path_to_cache = self.config.path_to_cache

        if path_to_cache:
            get_path_to = lambda suffix: os.path.join(
                path_to_cache,
                f"{self.unique_description_hash}-{type(self).__name__}-{self.split.name}-{suffix}",
            )
            path_to_description = get_path_to("description.txt")
            path_to_dataset_index = get_path_to("dataset_index.npy")
            path_to_dataset_sample_index = get_path_to("dataset_sample_index.npy")
            cache_hit = all(
                map(
                    os.path.isfile,
                    [path_to_description, path_to_dataset_index, path_to_dataset_sample_index],
                )
            )
        else:
            cache_hit = False

        if not path_to_cache or (not cache_hit and torch.distributed.get_rank() == 0):
            log_single_rank(
                logger, logging.INFO, f"Build and save the {type(self).__name__} indices"
            )

            # Build the dataset and dataset sample indexes
            log_single_rank(
                logger, logging.INFO, f"\tBuild and save the dataset and dataset sample indexes"
            )
            t_beg = time.time()
            from megatron.core.datasets import helpers

            if self.size is not None:
                dataset_index = numpy.zeros(self.size, dtype=numpy.int16)
                dataset_sample_index = numpy.zeros(self.size, dtype=numpy.int64)
                helpers.build_blending_indices(
                    dataset_index,
                    dataset_sample_index,
                    self.weights,
                    len(self.datasets),
                    self.size,
                    _VERBOSE,
                )
            else:
                size = sum(self.weights)
                dataset_index = numpy.zeros(size, dtype=numpy.int16)
                dataset_sample_index = numpy.zeros(size, dtype=numpy.int64)
                helpers.build_exhaustive_blending_indices(
                    dataset_index, dataset_sample_index, self.weights, len(self.datasets)
                )

            dataset_indices, dataset_sizes = numpy.unique(dataset_index, return_counts=True)
            for i, (_index, _size) in enumerate(zip(dataset_indices, dataset_sizes)):
                if len(self.datasets[_index]) < _size:
                    raise IndexError(
                        f"The {self.split.name} blend oversamples the contributing datasets and, "
                        f"for example, requests {_size} samples from "
                        f"{type(self.datasets[_index]).__name__} number {i} in excess of its size "
                        f"{len(self.datasets[_index])}. The current value of the config attribute "
                        f"mid_level_dataset_surplus may be increased, e.g. two- or ten-fold, from "
                        f"its current value ({self.config.mid_level_dataset_surplus}) to ensure a "
                        f"sufficient mid-level dataset sample margin from which to draw."
                    )

            if path_to_cache:
                os.makedirs(path_to_cache, exist_ok=True)
                # Write the description
                with open(path_to_description, "wt") as writer:
                    writer.write(self.unique_description)
                # Save the indexes
                numpy.save(path_to_dataset_index, dataset_index, allow_pickle=True)
                numpy.save(path_to_dataset_sample_index, dataset_sample_index, allow_pickle=True)
            else:
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f"Cannot save the {type(self).__name__} indexes because path_to_cache is None",
                )

            t_end = time.time()
            log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

            return dataset_index, dataset_sample_index

        log_single_rank(logger, logging.INFO, f"Load the {type(self).__name__} indices")

        log_single_rank(
            logger, logging.INFO, f"\tLoad the dataset index from {path_to_dataset_index}"
        )
        t_beg = time.time()
        dataset_index = numpy.load(path_to_dataset_index, allow_pickle=True, mmap_mode='r')
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(
            logger,
            logging.INFO,
            f"\tLoad the dataset sample index from {path_to_dataset_sample_index}",
        )
        t_beg = time.time()
        dataset_sample_index = numpy.load(
            path_to_dataset_sample_index, allow_pickle=True, mmap_mode='r'
        )
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        return dataset_index, dataset_sample_index
