# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import functools
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch

from megatron.core.datasets.utils import Split, log_single_rank, normalize
from megatron.core.parallel_state import get_virtual_pipeline_model_parallel_rank

logger = logging.getLogger(__name__)


@dataclass
class BlendedMegatronDatasetConfig:
    """Configuration object for megatron-core blended and megatron datasets
    
    Attributes:
        is_built_on_rank (Callable): A callable which returns True if the dataset should be built
        on the current rank. It should be Megatron Core parallelism aware i.e. global rank, group
        rank, and virtual rank may inform its return value.

        random_seed (int): The seed for all RNG during dataset creation.

        sequence_length (int): The sequence length.

        blend (Optional[List[str]]): The blend string, consisting of either a single dataset or a
        flattened sequential sequence of weight-dataset pairs. For exampe, ["dataset-path1"] and
        ["50", "dataset-path1", "50", "dataset-path2"] are both valid. Not to be used with
        'blend_per_split'. Defaults to None.

        blend_per_split (blend_per_split: Optional[List[Optional[List[str]]]]): A set of blend
        strings, as defined above, one for each split distribution. Not to be used with 'blend'.
        Defauls to None.

        split (Optional[str]): The split string, a comma separated weighting for the dataset splits
        when drawing samples from a single distribution. Not to be used with 'blend_per_split'.
        Defaults to None.

        split_matrix (Optional[List[Tuple[float, float]]]): The split matrix consisting of
        non-overlapping book-ends of each split in order. For more information, refer to
        'convert_split_vector_to_split_matrix'. Created automatically from 'split'. Not to be
        passed in to the constructor.

        path_to_cache (str): Where all re-useable dataset indices are to be cached.
    """

    is_built_on_rank: Callable

    random_seed: int

    sequence_length: int

    blend: Optional[List[str]] = None

    blend_per_split: Optional[List[Optional[List[str]]]] = None

    split: Optional[str] = None

    split_matrix: Optional[List[Tuple[float, float]]] = field(init=False, default=None)

    path_to_cache: str = None

    def __post_init__(self):
        if torch.distributed.is_initialized():
            gb_rank = torch.distributed.get_rank()
            vp_rank = get_virtual_pipeline_model_parallel_rank()
            if gb_rank == 0 and (vp_rank == 0 or vp_rank is None):
                assert (
                    self.is_built_on_rank()
                ), "is_built_on_rank must return True when global rank = 0 and vp rank = 0"

        if self.blend_per_split is not None and any(self.blend_per_split):
            assert self.blend is None, "blend and blend_per_split are incompatible"
            assert len(self.blend_per_split) == len(
                Split
            ), f"blend_per_split must contain {len(Split)} blends"
            if self.split is not None:
                self.split = None
                log_single_rank(logger, logging.WARNING, f"Let split = {self.split}")
        else:
            assert self.blend is not None, "one of either blend or blend_per_split must be provided"
            assert self.split is not None, "both blend and split must be provided"
            split_vector = parse_and_normalize_split(self.split)
            self.split_matrix = convert_split_vector_to_split_matrix(split_vector)
            log_single_rank(logger, logging.INFO, f"Let split_matrix = {self.split_matrix}")


@dataclass
class GPTDatasetConfig(BlendedMegatronDatasetConfig):
    """Configuration object for megatron-core blended and megatron GPT datasets

    Attributes:
        return_document_ids (bool): Whether to return the document ids when querying the dataset.
    """

    return_document_ids: bool = False
    reset_position_ids: bool = False
    reset_attention_mask: bool = False
    eod_mask_loss: bool = False
    eod_id: int = 0


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

        vector_b (Optional[List[float]]): An optional secondary split vector which constrains the
        primary split vector. Defaults to None.

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
