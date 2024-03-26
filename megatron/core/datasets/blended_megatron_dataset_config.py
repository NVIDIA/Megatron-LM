# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import functools
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
from megatron.core.datasets.utils import Split, log_single_rank, normalize

logger = logging.getLogger(__name__)


@dataclass
class BlendedMegatronDatasetConfig:
    """Configuration object for Megatron Core datasets

    Args:
        random_seed (int): The seed for all RNG during dataset creation.

        sequence_length (int): The sequence length.

        blend (Optional[List[str]]): The blend string, consisting of either a single dataset or a flattened sequential sequence of weight-dataset pairs. For exampe, ["dataset-path1"] and ["50", "dataset-path1", "50", "dataset-path2"] are both valid. Not to be used with 'blend_per_split'. Defaults to None.

        blend_per_split (blend_per_split: Optional[List[Optional[List[str]]]]): A set of blend strings, as defined above, one for each split distribution. Not to be used with 'blend'. Defauls to None.

        split (Optional[str]): The split string, a comma separated weighting for the dataset splits when drawing samples from a single distribution. Not to be used with 'blend_per_split'. Defaults to None.

        split_matrix (Optional[List[Tuple[float, float]]]): The split matrix consisting of non-overlapping book-ends of each split in order. For more information, refer to 'convert_split_vector_to_split_matrix'. Created automatically from 'split'. Not to be passed in to the constructor.

        path_to_cache (str): Where all re-useable dataset indices are to be cached.

        mmap_bin_files (bool): Whether to mmap the .bin files or use file pointer.

        mock (bool): Whether to bypass real data loading and validation in favor of mock data generation.

        tokenizer (Optional[MegatronTokenizer]): The MegatronTokenizer instance or None. Required for datasets which do online tokenization.
    """

    random_seed: int

    sequence_length: int

    blend: Optional[List[str]] = None

    blend_per_split: Optional[List[Optional[List[str]]]] = None

    split: Optional[str] = None

    split_matrix: Optional[List[Tuple[float, float]]] = field(init=False, default=None)

    path_to_cache: Optional[str] = None

    mmap_bin_files: bool = False

    mock: bool = False

    tokenizer: Optional[MegatronTokenizer] = None

    def __post_init__(self) -> None:
        """Do asserts and set fields post init
        """
        log_single_rank(logger, logging.INFO, f"mock = {self.mock}")

        if not self.mock:
            if self.blend_per_split is not None and any(self.blend_per_split):
                assert self.blend is None, "blend and blend_per_split are incompatible"
                assert self.split is None, "split and blend_per_split are incompatible"
                assert len(self.blend_per_split) == len(
                    Split
                ), f"blend_per_split must contain {len(Split)} blends"
            else:
                assert (
                    self.blend is not None
                ), "one of either blend or blend_per_split must be provided"
                assert self.split is not None, "both blend and split must be provided"
                split_vector = parse_and_normalize_split(self.split)
                self.split_matrix = convert_split_vector_to_split_matrix(split_vector)
                log_single_rank(logger, logging.INFO, f"Let split_matrix = {self.split_matrix}")


def parse_and_normalize_split(split: str) -> List[float]:
    """Parse the dataset split ratios from a string

    Args:
        split (str): The train valid test split string e.g. "99,1,0"

    Returns:
        List[float]: The trian valid test split ratios e.g. [0.99, 0.01, 0.0]
    """
    split = list(map(float, re.findall(r"[.0-9]+", split)))
    split = split + [0.0 for _ in range(len(Split) - len(split))]

    assert len(split) == len(Split)
    assert all(map(lambda _: _ >= 0.0, split))

    split = normalize(split)

    return split


def convert_split_vector_to_split_matrix(
    vector_a: List[float], vector_b: Optional[List[float]] = None
) -> List[Optional[Tuple[float, float]]]:
    """Build the split matrix from one or optionally two contributing split vectors.

    Ex. a standard conversion:

    [0.99, 0.01, 0.0] -> [(0, 0.99), (0.99, 1.0), None]

    Ex. a conversion for Retro when Retro pretraining uses a [0.99, 0.01, 0.0] split and Retro
    preprocessing used a [0.98, 0.02, 0.0] split:

    [0.99, 0.01, 0.0], [0.98, 0.02, 0.0] -> [(0, 0.98), (0.99, 1.0), None]

    Args:
        vector_a (List[float]): The primary split vector

        vector_b (Optional[List[float]]): An optional secondary split vector which constrains the primary split vector. Defaults to None.

    Returns:
        List[Tuple[float, float]]: The split matrix consisting of book-ends of each split in order
    """
    if vector_b is None:
        vector_b = vector_a

    # [.900, .090, .010] -> [0.00, .900, .990, 100]
    expansion_a = functools.reduce(lambda a, b: a + [a[len(a) - 1] + b], [[0], *vector_a])
    expansion_b = functools.reduce(lambda a, b: a + [a[len(a) - 1] + b], [[0], *vector_b])

    # [0.00, .900, .990, 100.0] -> [(0.00, .900), (.900, .990), (.990, 100)]
    bookends_a = list(zip(expansion_a[:-1], expansion_a[1:]))
    bookends_b = list(zip(expansion_b[:-1], expansion_b[1:]))

    # gather per-split overlap or None
    matrix = []
    for bookend_a, bookend_b in zip(bookends_a, bookends_b):
        if min(bookend_a[1], bookend_b[1]) <= max(bookend_a[0], bookend_b[0]):
            overlap = None
        else:
            overlap = (max(bookend_a[0], bookend_b[0]), min(bookend_a[1], bookend_b[1]))
        matrix.append(overlap)

    return matrix
