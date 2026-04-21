# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Container dataclass for GPT chunk datasets (train, valid, and test)."""

from dataclasses import dataclass


@dataclass
class RetroGPTChunkDatasets:
    """Container dataclass for GPT chunk datasets."""

    # Each dict contains 'dataset', 'neighbor_dir', and 'num_active_chunks'.
    train: dict = None
    valid: dict = None
    test: dict = None
