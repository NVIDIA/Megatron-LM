# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""
A RetroDataset wraps both:

  - A GPTDataset (which is nested as GPTChunkDataset -> MultiSplitGPTDataset ->
      GPTDataset).
  - Neighbor IDs of chunks in the chunk database, that were saved during
      preprocessing.

Both the GPT sample data and the neighbor IDs are returned within a sample from
this dataset.
"""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from megatron.core.datasets.retro.db.dataset import DBDataset
from megatron.core.datasets.retro.db.utils import get_merged_train_dataset as get_db_dataset
from megatron.core.datasets.retro.external_libs import h5py
from megatron.core.datasets.retro.utils import BlockPathMap, log_retro_rank_0
from megatron.core.models.retro import RetroConfig

from .gpt_chunk_dataset import GPTChunkDataset, build_gpt_chunk_datasets_from_gpt_datasets
from .utils import get_query_dir


class RetroDataset(torch.utils.data.Dataset):
    """Dataset of retro samples.

    Each sample contains the original GPT sample, along with the token IDs
    of each neighbor of each chunk within the sequence. Neighbor array has
    shape (num_chunks_per_sample, num_neighbors, num_retrieved_tokens).

    ** Note: chunk dataset wraps original GPT dataset (see gpt_chunk_dataset.py).

    Args:
        num_queried_samples (int): Total number of queried samples.
        num_neighbors (int): Total number of saved neighbors.
        num_retrieved_chunks (int): Number of retrieved chunks (e.g., 2 for neighbor + continuation).
        block_size (int): Number of neighbor entries per file.
        db_dataset (DBDataset): Chunk database used for retrieval.
        chunk_dataset (GPTChunkDataset): GPT chunk dataset, which is a wrapper around a standard GPT dataset that breaks each sample into chunks.
        neighbor_path_map (BlockPathMap): Mapping of neighbor ID to file path.
    """

    def __init__(
        self,
        num_queried_samples: int,
        num_neighbors: int,
        num_retrieved_chunks: int,
        block_size: int,
        db_dataset: DBDataset,
        chunk_dataset: GPTChunkDataset,
        neighbor_path_map: BlockPathMap,
    ):
        super().__init__()

        self.num_queried_samples = num_queried_samples
        self.num_neighbors = num_neighbors
        self.num_retrieved_chunks = num_retrieved_chunks
        self.block_size = block_size
        self.db_dataset = db_dataset
        self.chunk_dataset = chunk_dataset
        self.neighbor_path_map = neighbor_path_map

    def __len__(self) -> int:
        """Dataset length.

        Returns:
            Number of samples in dataset.
        """
        return len(self.chunk_dataset.sample_dataset)

    def __getitem__(self, sample_idx: int) -> dict:
        """Get dataset sample.

        Args:
            sample_idx (int): Index of sample in dataset.

        Returns:
            A dict consisting of GPT sample (attribute 'text') and corresponding neighbor chunk IDs ('neighbor_chunks', for indexing chunk database) and neighbor token IDs (corresponding chunk database GPT tokens).
        """
        n_chunks_per_sample = self.chunk_dataset.n_chunks_per_sample

        # Wrap sample idx around number of queried samples.
        sample_idx = sample_idx % self.num_queried_samples

        # Get standard sample.
        sample = self.chunk_dataset.sample_dataset[sample_idx]

        # Sample idx to chunk idxs.
        chunk_idxs = list(
            range(sample_idx * n_chunks_per_sample, (sample_idx + 1) * n_chunks_per_sample,)
        )

        # Collect retrieved tokens.
        all_retrieved_chunk_ids = []
        all_retrieved_token_ids = []
        for chunk_idx in chunk_idxs:

            # Neighbor chunk ids.
            neighbor_path = self.neighbor_path_map[chunk_idx]
            with h5py.File(neighbor_path, "r") as f:
                neighbor_chunk_ids = f["neighbors"][
                    chunk_idx % self.block_size, : self.num_neighbors
                ].tolist()

            # Retrieved (neighbor + continuation) token ids.
            retrieved_chunk_ids = []
            retrieved_token_ids = []
            for neighbor_chunk_id in neighbor_chunk_ids:
                current_chunk_ids = [
                    i % len(self.db_dataset)
                    for i in range(neighbor_chunk_id, neighbor_chunk_id + self.num_retrieved_chunks)
                ]
                current_token_ids = [self.db_dataset[ci]["text"] for ci in current_chunk_ids]
                retrieved_chunk_ids.append(current_chunk_ids)
                retrieved_token_ids.append(current_token_ids)

            # Collect retrieved tokens.
            all_retrieved_chunk_ids.append(retrieved_chunk_ids)
            all_retrieved_token_ids.append(retrieved_token_ids)

        # Reshape retrieved tokens.
        all_retrieved_chunk_ids = np.array(all_retrieved_chunk_ids).reshape(
            (n_chunks_per_sample, self.num_neighbors, -1)
        )
        all_retrieved_token_ids = np.array(all_retrieved_token_ids).reshape(
            (n_chunks_per_sample, self.num_neighbors, -1)
        )

        # Sample.
        sample: Dict[str, np.ndarray] = {
            **sample,
            "neighbor_chunks": all_retrieved_chunk_ids,
            "neighbor_tokens": all_retrieved_token_ids,
        }

        return sample


def get_retro_datasets(
    config: RetroConfig, gpt_datasets: dict, sample_length: int, eod_token_id: int,
) -> Tuple[Optional[RetroDataset], Optional[RetroDataset], Optional[RetroDataset]]:
    """Get train, valid, test retro datasets.

    Args:
        config (RetroConfig): Retro preprocessing config.
        gpt_datasets (dict): Mapping of data split key ('train', 'valid', or 'test') to the original sequence-length GPT dataset (i.e., not the chunk dataset).
        sample_length (int): Alias to `sequence_length`.
        eod_token_id (int): GPT EOD token ID.

    Returns:
        A tuple of 'train', 'valid', and 'test' `RetroDataset`s.
    """

    # DB dataset.
    db_dataset = get_db_dataset(
        project_dir=config.retro_project_dir,
        chunk_length=config.retro_chunk_length,
        eod_token_id=eod_token_id,
    )

    # GPT chunk datasets.
    chunk_ds_info_map = build_gpt_chunk_datasets_from_gpt_datasets(
        project_dir=config.retro_project_dir,
        gpt_datasets=gpt_datasets,
        sample_length=sample_length,
        chunk_length=config.retro_chunk_length,
    )

    # Retro datasets.
    retro_dataset_map: Dict[str, Optional[RetroDataset]] = {}
    query_dir = get_query_dir(config.retro_project_dir)
    for data_key, chunk_ds_info in chunk_ds_info_map.items():

        # Skip unused datasets.
        if chunk_ds_info is None:
            retro_dataset_map[data_key] = None
            continue

        # For consistency with preprocessing, the neighbor_dir is overwritten
        # (from its setting in `build_gpt_chunk_datasets_from_gpt_datasets()`
        # above). This is one piece -- along with setting data_path and
        # train_samples from config.json -- of ensuring consistency between
        # preprocessing and pretraining.
        chunk_dataset = chunk_ds_info["dataset"]
        chunk_ds_info["neighbor_dir"] = os.path.join(
            query_dir, config.retro_neighbor_dirs[data_key],
        )
        neighbor_dir = chunk_ds_info["neighbor_dir"]
        neighbor_path_map = BlockPathMap.from_dir(
            dir=neighbor_dir, block_size=config.retro_block_size
        )

        # Verify num chunks.
        n_active_chunks = chunk_ds_info["num_active_chunks"]
        n_neighbor_chunks = neighbor_path_map.max_idx

        if not os.path.isdir(neighbor_dir):
            if torch.distributed.get_rank() == 0:
                raise Exception(
                    "neighbor directory '%s' not found; please "
                    "compare --train-samples, --seq-length, --seed, "
                    "--eval-iters, and --eval-interval, with "
                    "retro preprocessing args." % neighbor_dir
                )
            torch.distributed.barrier()
            exit()

        if config.retro_verify_neighbor_count and n_active_chunks != n_neighbor_chunks:
            if torch.distributed.get_rank() == 0:
                log_retro_rank_0("neighbor_dir : %s" % neighbor_dir)
                log_retro_rank_0("neighbor_path_map : %s" % neighbor_path_map)
                raise Exception(
                    "num sampled chunks (%d) != num neighbor chunks "
                    "(%d); did you complete querying the entire "
                    "pretraining dataset?" % (n_active_chunks, n_neighbor_chunks)
                )
            torch.distributed.barrier()
            exit()

        # Retro dataset.
        retro_dataset_map[data_key] = RetroDataset(
            num_queried_samples=gpt_datasets[data_key][1],
            num_neighbors=config.retro_num_neighbors,
            num_retrieved_chunks=config.retro_num_retrieved_chunks,
            block_size=config.retro_block_size,
            db_dataset=db_dataset,
            chunk_dataset=chunk_dataset,
            neighbor_path_map=neighbor_path_map,
        )

    return (
        retro_dataset_map["train"],
        retro_dataset_map["valid"],
        retro_dataset_map["test"],
    )
