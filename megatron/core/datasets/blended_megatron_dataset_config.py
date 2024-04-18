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
    """Configuration object for Megatron Core datasets"""

    random_seed: int
    """The seed for all RNG during dataset creation."""

    sequence_length: int
    """The sequence length."""

    blend: Optional[Tuple[List[str], Optional[List[float]]]] = None
    """The blend, consisting of a list of dataset prefixes and optionally a list of dataset
       weights. For example, [["dataset-path1", "dataset-path2"], [0.3, 0.7]]. When the weights are
       None, they are inferred from the lengths of the contributing datasets. Not to be used with
       'blend_per_split'. Defaults to None.
    """

    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]] = None
    """A set of blends, as defined above, one for each split distribution. Not to be used with
       'blend'. Defauls to None.
    """

    split: Optional[str] = None
    """The split string, a comma separated weighting for the dataset splits when drawing samples
       from a single distribution. Not to be used with 'blend_per_split'.  Defaults to None.
    """

    split_matrix: Optional[List[Tuple[float, float]]] = field(init=False, default=None)
    """The split matrix consisting of non-overlapping book-ends of each split in order. For more
       information, refer to 'convert_split_vector_to_split_matrix'. Created automatically from
       'split'. Not to be passed in to the constructor.
    """

    path_to_cache: Optional[str] = None
    """Where all re-useable dataset indices are to be cached."""

    mmap_bin_files: bool = True
    """Whether to mmap the .bin files or use file pointers."""

    mock: bool = False
    """Whether to bypass real data loading and validation in favor of mock data generation."""

    tokenizer: Optional[MegatronTokenizer] = None
    """The MegatronTokenizer instance or None. Required for datasets which do online tokenization."""

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
                for split in Split:
                    if self.blend_per_split[split.value] is None:
                        log_single_rank(
                            logger, logging.INFO, f"blend not provided for {split.name} split"
                        )
                    else:
                        assert self.blend_per_split[split.value][1] is None or len(
                            self.blend_per_split[split.value][0]
                        ) == len(
                            self.blend_per_split[split.value][1]
                        ), "blend per split prefixes and weights must be equal in number"
            else:
                assert (
                    self.blend is not None
                ), "one of either blend or blend_per_split must be provided"
                assert self.split is not None, "both blend and split must be provided"
                assert self.blend[1] is None or len(self.blend[0]) == len(
                    self.blend[1]
                ), "blend prefixes and weights must be equal in number"
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
