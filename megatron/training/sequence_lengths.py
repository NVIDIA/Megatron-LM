# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Resolve dataset row widths independently of model input expansion."""

from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig


def get_dataset_input_length(dataset: BlendedMegatronDatasetConfig) -> int:
    """Return a dataset's output row width, not an SFT conversation truncation limit."""
    from megatron.training.datasets.sft_dataset import SFTDatasetConfig

    if isinstance(dataset, SFTDatasetConfig):
        return dataset.packed_sequence_length or dataset.sequence_length
    return dataset.sequence_length
