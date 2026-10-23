# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import functools
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from megatron.core.datasets.utils import Split, normalize
from megatron.core.tokenizers import MegatronTokenizerBase


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
       'blend'. Defaults to None.
    """

    multiple_validation_sets: Optional[bool] = None
    """Whether the validation split should be treated as multiple separated datasets."""

    full_validation: Optional[bool] = None
    """Whether to run a full epoch of validation each time validation occurs."""

    split: Optional[str] = None
    """The split string, a comma separated weighting for the dataset splits when drawing samples
       from a single distribution. Not to be used with 'blend_per_split'.  Defaults to None.
    """

    num_dataset_builder_threads: int = 1
    """The number of threads to use for dataset building."""

    path_to_cache: Optional[str] = None
    """Where all re-useable dataset indices are to be cached."""

    mmap_bin_files: bool = True
    """Whether to mmap the .bin files or use file pointers."""

    tokenizer: Optional[MegatronTokenizerBase] = None
    """The MegatronTokenizerBase instance. Required for datasets that do online tokenization."""

    mid_level_dataset_surplus: float = 0.005
    """The sample surplus to build for the mid-level datasets(s). Defaults arbitrarily to 0.005.
       This value is irrelevant for single source data blends. This value may need to be increased
       if the top level dataset oversamples the mid level dataset(s). This value may be set to 0.0
       in future if the top level dataset is constrained to not oversample the mid level
       datasets(s).
    """

    allow_ambiguous_pad_tokens: Optional[bool] = False
    """Whether to prevent pad tokens already present in the dataset from being masked out
       when the pad token incorrectly shares the same id with other special tokens.
       Treating such tokens as pad tokens results in training instability and divergence.
       Such a scenario is best resolved by fixing the tokenizer, but leaving this option as False
       provides a workaround.
       This argument will have no effect if the tokenizer is correct. However, should the user
       desire to train on a dataset that intentionally contains pad tokens - while also using an
       incorrect tokenizer - this option may be set to True. This is typically not recommended.
    """

    fast_cache_load: bool = False
    """Option to use the fast cache loading path. Requires all the dataset caches to be built."""

    defer_npy_index_mmap: bool = False
    """Option to defer the mmap of the dataset indexes until the first access.
       Requires all the dataset caches to be built.
    """

    def validate(self) -> None:
        """Check the blend and split: assertions only, idempotent.

        Nothing runs at construction. A config can be constructed before its tokenizer exists and
        completed later: run this once it is complete. ``BlendedMegatronDatasetBuilder`` runs it
        before building, so a config handed complete to the builder needs no explicit call.
        Subclasses add the checks that need the tokenizer.
        """
        if self.fast_cache_load:
            assert (
                self.path_to_cache is not None
            ), "--data-cache-path must be provided when using --dataloader-fast-cache-load."
            assert (
                self.blend is None
            ), f"--dataloader-fast-cache-load and --data-path cannot be used together. \
            Use --per-split-data-args-path or --train-data-path, --valid-data-path and \
            --test-data-path instead."
        if self.defer_npy_index_mmap:
            assert (
                self.path_to_cache is not None
            ), "--data-cache-path must be provided when using --dataloader-defer-npy-index-mmap."
        if self.blend_per_split is not None and any(self.blend_per_split):
            assert self.blend is None, "blend and blend_per_split are incompatible"
            assert self.split is None, "split and blend_per_split are incompatible"
            assert len(self.blend_per_split) == len(
                Split
            ), f"blend_per_split must contain {len(Split)} blends"
            for blend in self.blend_per_split:
                assert (
                    blend is None or blend[1] is None or len(blend[0]) == len(blend[1])
                ), "blend per split prefixes and weights must be equal in number"
        else:
            if self.blend is not None:
                assert self.blend[1] is None or len(self.blend[0]) == len(
                    self.blend[1]
                ), "blend prefixes and weights must be equal in number"
                assert self.split is not None, "split must be provided when blend is not None"
            if self.split is not None:
                parse_and_normalize_split(self.split)  # a malformed split string is rejected here

    @property
    def mock(self) -> bool:
        """Whether to bypass real data loading and validation in favor of mock data generation:
        True when neither ``blend`` nor ``blend_per_split`` provides a data source."""
        return self.blend is None and not (
            self.blend_per_split is not None and any(self.blend_per_split)
        )

    @property
    def split_matrix(self) -> Optional[List[Tuple[float, float]]]:
        """The split matrix consisting of non-overlapping book-ends of each split in order, from
        ``split`` (an arbitrarily even split when the config is mock and no split was given). For
        more information, refer to 'convert_split_vector_to_split_matrix'. None with
        ``blend_per_split``, which carries one blend per split."""
        if self.blend_per_split is not None and any(self.blend_per_split):
            return None
        split = self.split
        if split is None:
            if not self.mock:
                return None  # validate() rejects a blend without a split
            split = "1,1,1"
        return convert_split_vector_to_split_matrix(parse_and_normalize_split(split))


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
