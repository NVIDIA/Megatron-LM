# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import functools
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from megatron.core.datasets.utils import Split, log_single_rank, normalize
from megatron.core.tokenizers import MegatronTokenizerBase

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

    split_matrix: Optional[List[Tuple[float, float]]] = field(init=False, default=None)
    """The split matrix consisting of non-overlapping book-ends of each split in order. For more
       information, refer to 'convert_split_vector_to_split_matrix'. Created automatically from
       'split'. Not to be passed in to the constructor.
    """

    num_dataset_builder_threads: int = 1
    """The number of threads to use for dataset building."""

    path_to_cache: Optional[str] = None
    """Where all re-useable dataset indices are to be cached."""

    mmap_bin_files: bool = True
    """Whether to mmap the .bin files or use file pointers."""

    mock: bool = field(init=False, default=False)
    """Whether to bypass real data loading and validation in favor of mock data generation.
       Created automatically from 'blend' and 'blend_per_split'. Not to be passed in to the
       constructor.
    """

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

    dataloader_type: Optional[str] = None
    """The dataloader / sampler strategy used to build the torch DataLoader on top of the dataset:
       'single' (sequential), 'cyclic' (randomized, resumable), 'batch' (full-global-batch, for
       variable-length fine-tuning), or 'external' (the dataset is passed through untouched). When
       None, the consuming training loop selects a default. Carried here so a config container can
       drive data-loader construction without reading global args.
    """

    num_workers: int = 2
    """Number of subprocesses used by the torch DataLoader for data loading."""

    data_sharding: bool = True
    """Whether the randomized ('cyclic') sampler shards the dataset into per-rank buckets before
       shuffling. Only consulted by MegatronPretrainingRandomSampler.
    """

    pin_memory: bool = True
    """Whether the torch DataLoader pins host memory for faster host-to-device transfer."""

    persistent_workers: bool = True
    """Whether the torch DataLoader keeps worker subprocesses alive between epochs.
       Forced to False when num_workers == 0.
    """

    def __post_init__(self) -> None:
        """Do asserts and set fields post init"""
        # Persistent workers require at least one worker subprocess.
        if self.num_workers == 0 and self.persistent_workers:
            self.persistent_workers = False
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
            if self.blend is not None:
                assert self.blend[1] is None or len(self.blend[0]) == len(
                    self.blend[1]
                ), "blend prefixes and weights must be equal in number"
                assert self.split is not None, "split must be provided when blend is not None"
            else:
                self.mock = True
                log_single_rank(
                    logger,
                    logging.INFO,
                    f"Let mock = True, as both blend and blend_per_split are None",
                )
                self.split = "1,1,1"
                log_single_rank(
                    logger,
                    logging.INFO,
                    f"Let split = {self.split}, an arbitrarily even split, as mock is True",
                )
            split_vector = parse_and_normalize_split(self.split)
            self.split_matrix = convert_split_vector_to_split_matrix(split_vector)
            log_single_rank(logger, logging.INFO, f"Let split_matrix = {self.split_matrix}")

        # Tokenizer-dependent state is materialized eagerly when a tokenizer was
        # supplied at construction (backwards compatible). When the tokenizer is
        # deferred (None) -- e.g. for an early, serializable config container that
        # is built before the tokenizer exists -- call ``finalize(tokenizer)``
        # later to complete it.
        if self.tokenizer is not None:
            self._finalize_with_tokenizer()

    def _finalize_with_tokenizer(self) -> None:
        """Run tokenizer-dependent validation / derivation.

        Subclasses override this to compute fields that require a built tokenizer.
        The base config has no tokenizer-derived state, so this is a no-op here.
        Split into its own method (rather than living inline in ``__post_init__``)
        so it can be deferred and re-run by ``finalize`` once the tokenizer is
        available.
        """
        pass

    def finalize(self, tokenizer=None) -> "BlendedMegatronDatasetConfig":
        """Materialize tokenizer-dependent state once the tokenizer is built.

        Config containers are constructed early as a declarative job description,
        before heavy / environment-specific objects like the tokenizer exist. This
        injects an already-built ``tokenizer`` and runs the deferred
        tokenizer-dependent setup. Build the tokenizer via the usual mechanism and
        pass it here so tokenizer construction stays in its current place / ranks.

        Args:
            tokenizer: The built tokenizer to inject. If None, the config must
                already carry one (e.g. supplied at construction).

        Returns:
            self, for chaining.
        """
        if tokenizer is not None:
            self.tokenizer = tokenizer
        assert self.tokenizer is not None, (
            "Dataset config has no tokenizer; call finalize(tokenizer) with a built "
            "tokenizer before using the config to build datasets."
        )
        self._finalize_with_tokenizer()
        return self


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
