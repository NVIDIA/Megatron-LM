# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Base class for all vector indexes.

A vector index is a type of retrieval database that is queried using vectors,
and returns vectors that are 'similar' (e.g., by cosine distance) to the query
vector. The construction and usage of an index generally has the following
pattern:

  - Train the index on representative vectors.
  - Add vectors to the index (i.e., vectors available for retrieval)
  - Query index with new vector, to retrieve similar vector indexes.
"""

import abc
import os
from typing import List, Tuple

import numpy as np
import torch

from megatron.core.datasets.retro.config import Embedder, RetroPreprocessingConfig
from megatron.core.datasets.retro.external_libs import faiss
from megatron.core.datasets.retro.utils import GPTToTextDataset

from .utils import get_index_dir


class Index(abc.ABC):

    """Abstract base class for indexes.

    *Note* : While currently only Faiss-based classes are implemented, in the
    future, this class will be extended with other types of indexes that have
    different performance-accuracy trade-offs.

    The primary methods to override are:
    - train() : Train index on the sampled training chunks.
    - add() : Add all training chunks to index.
    """

    @classmethod
    def make_object_verbose(cls, index: faiss.Index, verbose: bool) -> None:
        """Make index object verbose.

        Args:
            index (faiss.Index): Faiss object to set verbose.
            verbose (bool): Sets whether index should log status updates during training and adding.
        """
        assert isinstance(verbose, bool)
        faiss.ParameterSpace().set_index_parameter(index, "verbose", verbose)

    def get_empty_index_path(self, config: RetroPreprocessingConfig) -> str:
        """Get file path to empty index (i.e., trained, but unpopulated).

        Args:
            config (RetroPreprocessingConfig): Retro preprocessing config.

        Returns:
            File path to empty index (i.e., this index has had index.train() called, but not yet index.add()).
        """
        return os.path.join(
            get_index_dir(config), "empty_%.3f.faissindex" % config.retro_index_train_load_fraction,
        )

    def get_empty_index(self, config: RetroPreprocessingConfig) -> faiss.Index:
        """Get empty index (i.e., trained, but unpopulated).

        Args:
            config (RetroPreprocessingConfig): Retro preprocessing config.

        Returns:
            Empty Faiss index, loaded from storage.
        """
        return faiss.read_index(self.get_empty_index_path(config))

    def get_added_index_path(self, config: RetroPreprocessingConfig) -> str:
        """Get file path to index that has been populated with vectors.

        Args:
            config (RetroPreprocessingConfig): Retro preprocessing config.

        Returns:
            File path to added index (i.e., this index has had both index.train() and index.add() called).
        """
        return os.path.join(
            get_index_dir(config),
            "added_%.3f_%.3f.faissindex"
            % (config.retro_index_train_load_fraction, config.retro_index_add_load_fraction,),
        )

    def get_added_index(self, config: RetroPreprocessingConfig) -> faiss.Index:
        """Get index that has been populated with vectors.

        Args:
            config (RetroPreprocessingConfig): Retro preprocessing config.

        Returns:
            'Added' (i.e., populated) Faiss index, loaded from storage.
        """
        return faiss.read_index(self.get_added_index_path(config))

    @abc.abstractmethod
    def train(self, config: RetroPreprocessingConfig) -> None:
        """Train index on a representative set of vectors.

        Args:
            config (RetroPreprocessingConfig): Retro preprocessing config.
        """

    @abc.abstractmethod
    def add(self, config: RetroPreprocessingConfig, text_dataset: GPTToTextDataset) -> None:
        """Add vectors to index.

        Args:
            config (RetroPreprocessingConfig): Retro preprocessing config.
            text_dataset (GPTToTextDataset): Text dataset that will be embedded and added to the index.
        """

    def embed_text_dataset_block(
        self, embedder: Embedder, text_dataset: GPTToTextDataset, _range: Tuple[int, int]
    ) -> np.ndarray:
        """Embed a range of a text dataset.

        Args:
            embedder (Embedder): Embedder used for embedding a text dataset.
            text_dataset (GPTToTextDataset): Text dataset that will be embedded.
            _range (Tuple[int, int]): Start/end sample indices within text dataset used for embedding.

        Returns:
            An array of embeddings, with shape (len(text_dataset), dimension(embedder)).
        """
        sub_dataset = torch.utils.data.Subset(text_dataset, range(*_range))
        return embedder.embed_text_dataset(sub_dataset)
