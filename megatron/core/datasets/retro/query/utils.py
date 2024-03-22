# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Utilities for querying the pretraining dataset."""

import os

from megatron.core.datasets.megatron_dataset import MegatronDataset


def get_query_dir(project_dir: str) -> str:
    """Get root directory of all saved query data.

    Args:
        project_dir (str): Retro project dir.

    Returns:
        Path to query sub-directory in Retro project.
    """
    return os.path.join(project_dir, "query")


def get_neighbor_dir(project_dir: str, key: str, dataset: MegatronDataset) -> str:
    """Get directory containing neighbor IDs for a dataset (i.e., train, valid, or test).

    Args:
        project_dir (str): Retro project dir.
        key (str): Dataset split key; 'train', 'valid', or 'test'.
        dataset (MegatronDataset): Dataset containing unique hash for finding corresponding neighbors.

    Returns:
        Path to directory containing this dataset's neighbors within Retro project.
    """
    return os.path.join(
        get_query_dir(project_dir), os.path.basename(f"{key}_{dataset.unique_description_hash}"),
    )
