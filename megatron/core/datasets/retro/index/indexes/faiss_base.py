# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""
This class implements a simple, un-optimized wrapper around a Faiss index, that
implements the Index interface (see ..index.py). While this class is
instantiable, it is meant to be extended with optimizations in classes that
inherit from this class (see FaissParAddIndex, for an example).
"""

import os

import numpy as np
import torch
from tqdm import tqdm

from megatron.core.datasets.retro.config import RetroPreprocessingConfig
from megatron.core.datasets.retro.external_libs import faiss
from megatron.core.datasets.retro.index.index import Index
from megatron.core.datasets.retro.index.utils import (
    get_training_data_merged_path,
    num_samples_to_block_ranges,
)
from megatron.core.datasets.retro.utils import GPTToTextDataset, log_retro_rank_0


class FaissBaseIndex(Index):
    """Base class for Faiss-base indexes.

    This class wraps a Faiss index, and adds additional functionality for training
    and adding codes. This base class performs a naive sequential code adding,
    while the optimized FaissParallelAddIndex class performs a parallel
    index.add().
    """

    def _train(self, config: RetroPreprocessingConfig) -> None:
        """Train index (rank 0's method).

        Args:
            config (RetroPreprocessingConfig): Retro preprocessing config.
        """

        assert torch.distributed.get_rank() == 0

        # Set num threads (torch.distributed reset it to 1).
        faiss.omp_set_num_threads(64)

        empty_index_path = self.get_empty_index_path(config)

        # Index already exists? -> return.
        if os.path.isfile(empty_index_path):
            return

        # Load data.
        merged_path = get_training_data_merged_path(config)
        inp = np.memmap(merged_path, dtype="f4", mode="r").reshape((-1, config.hidden_size))

        # Init index.
        index = faiss.index_factory(config.hidden_size, config.retro_index_str)

        # Move to GPU.
        log_retro_rank_0("> move faiss index to gpu.")
        index_ivf = faiss.extract_index_ivf(index)
        clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
        index_ivf.clustering_index = clustering_index
        log_retro_rank_0("> finished moving to gpu.")
        self.make_object_verbose(index, True)
        self.make_object_verbose(index_ivf, True)
        self.make_object_verbose(index_ivf.quantizer, True)
        self.make_object_verbose(index_ivf.clustering_index, True)

        # Train index.
        index.train(inp)

        # Save index.
        faiss.write_index(index, empty_index_path)

    def train(self, config: RetroPreprocessingConfig) -> None:
        """Train index.

        Args:
            config (RetroPreprocessingConfig): Retro preprocessing config.
        """

        # Single process only.
        if torch.distributed.get_rank() == 0:
            self._train(config)

        torch.distributed.barrier()

    def _add(self, config: RetroPreprocessingConfig, text_dataset: GPTToTextDataset) -> None:
        """Add to index (rank 0's method).

        Args:
            config (RetroPreprocessingConfig): Retro preprocessing config.
            text_dataset (GPTToTextDataset): Text dataset that will be embedded and added to the index.
        """

        assert torch.distributed.get_rank() == 0

        dataset_sample_ranges = num_samples_to_block_ranges(len(text_dataset))

        # Set num threads (torch.distributed reset it to 1).
        faiss.omp_set_num_threads(64)

        # Bert embedder.
        embedder = config.bert_embedders.mem

        # Empty/added index paths.
        empty_index_path = self.get_empty_index_path()
        added_index_path = self.get_added_index_path()

        # Skip adding, if index exists.
        if os.path.isfile(added_index_path):
            return

        # Read trained index.
        index = faiss.read_index(empty_index_path)

        # Iterate data blocks & add.
        for sample_range in tqdm(dataset_sample_ranges, "faiss_base.add"):

            # Embed text.
            embeds = self.embed_text_dataset_block(embedder, text_dataset, sample_range)

            # Add to index.
            index.add(embeds)

        # Write index.
        faiss.write_index(index, added_index_path)

    def add(self, config: RetroPreprocessingConfig, text_dataset: GPTToTextDataset) -> str:
        """Add to index.

        Args:
            config (RetroPreprocessingConfig): Retro preprocessing config.
            text_dataset (GPTToTextDataset): Text dataset that will be embedded and added to the index.

        Returns:
            File path to the populated index.
        """

        # Single process only.
        if torch.distributed.get_rank() == 0:
            self._add(config, text_dataset)

        # Wait for rank 0.
        torch.distributed.barrier()

        # Get output index path, for return.
        return self.get_added_index_path(config)
