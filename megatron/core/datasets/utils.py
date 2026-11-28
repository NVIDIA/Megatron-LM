# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import logging
import os
import tempfile
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy

from ..utils import log_single_rank

logger = logging.getLogger(__name__)


class Split(Enum):
    """Dataset partition used for training, validation, and testing."""

    train = 0
    valid = 1
    test = 2


def compile_helpers():
    """Compile C++ helper functions at runtime. Make sure this is invoked on a single process."""
    import subprocess

    command = ["make", "-C", os.path.abspath(os.path.dirname(__file__))]
    if subprocess.run(command).returncode != 0:
        import sys

        log_single_rank(logger, logging.ERROR, "Failed to compile the C++ dataset helper functions")
        sys.exit(1)


def normalize(weights: List[float]) -> List[float]:
    """Do non-exponentiated normalization

    Args:
        weights (List[float]): The weights

    Returns:
        List[float]: The normalized weights
    """

    w = numpy.array(weights, dtype=numpy.float64)
    w_sum = numpy.sum(w)
    w = (w / w_sum).tolist()
    return w


def get_blend_from_list(
    blend: Optional[List[str]],
) -> Optional[Tuple[List[str], Optional[List[float]]]]:
    # pylint: disable=line-too-long
    """Get the blended_megatron_dataset_config.BlendedMegatronDatasetConfig blend
    from the blend list

    Args:
        blend (Optional[List[str]]): The blend list, which can be either
            (1) a list of prefixes, e.g. ["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], or
            (2) a flattened, zipped list of weights and prefixes, e.g. ["30", "path/to/dataset_1_prefix", "70", "path/to/dataset_2_prefix"]

    Returns:
        Optional[Tuple[List[str], Optional[List[float]]]]: The blend, consisting of a list of dataset prefixes and optionally a list of dataset weights, e.g. [["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], [30.0, 70.0]].
    """
    # pylint: enable=line-too-long
    if blend is None:
        return None

    if len(blend) % 2 == 1:
        weight_per_dataset = None
        raw_prefix_per_dataset = blend
    else:
        raw_weight_per_dataset, raw_prefix_per_dataset = zip(
            *[(blend[i], blend[i + 1]) for i in range(0, len(blend), 2)]
        )

        weight_per_dataset = []
        for rwpd in raw_weight_per_dataset:
            try:
                weight = float(rwpd)
            except ValueError:
                weight = None
            weight_per_dataset.append(weight)

        is_none = map(lambda _: _ is None, weight_per_dataset)
        if any(is_none):
            assert all(is_none)
            weight_per_dataset = None
            raw_prefix_per_dataset = blend

    prefix_per_dataset = [rppd.strip() for rppd in raw_prefix_per_dataset]

    return prefix_per_dataset, weight_per_dataset


def save_index_cache(
    path_to_description: str, description: str, indices: Dict[str, numpy.ndarray]
) -> None:
    """Publish complete index files before their cache description.

    Args:
        path_to_description (str): Path to the cache description.
        description (str): The dataset's unique description.
        indices (Dict[str, numpy.ndarray]): Index arrays keyed by their cache paths.
    """
    # Stage on the same filesystem so interrupted writes never replace a complete index.
    with tempfile.TemporaryDirectory(dir=os.path.dirname(path_to_description)) as temporary_dir:
        for path, index in indices.items():
            numpy.save(
                os.path.join(temporary_dir, os.path.basename(path)), index, allow_pickle=True
            )
        with open(
            os.path.join(temporary_dir, os.path.basename(path_to_description)), "wt"
        ) as writer:
            writer.write(description)
        for path in (*indices, path_to_description):
            os.replace(os.path.join(temporary_dir, os.path.basename(path)), path)
