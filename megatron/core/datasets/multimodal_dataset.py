# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Callable, Dict

import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset


@dataclass
class MultimodalDatasetConfig(GPTDatasetConfig):
    """Configuration object for Megatron Core Multimodal datasets.

    Note: This is unused at the moment and may be missing features. Follow-up changes will use this.
    """

    image_h: int = None
    """Image height."""

    image_w: int = None
    """Image width."""

    # Function to preprocess the data sample to a format expected by a specific model. By default, do nothing.
    preprocess_func: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = lambda x: x
    """Optional function to preprocess data samples for a specific model."""

    def __post_init__(self) -> None:
        super().__post_init__()

        assert self.image_h is not None
        assert self.image_w is not None


class MockMultimodalDataset(MockGPTDataset):
    """Mock multimodal dataset.


    This is unused at the moment and may be missing features. Follow-up changes will use this.
    """

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a sample that contains a dummy image, text sequence and the associated labels and cost and attention masks.

        Args:
            idx (int): The integer seed for mock data generation.

        Returns:
            Dict[str, torch.Tensor]: The mock data.
        """
        # Get a text sample.
        sample = super().__getitem__(idx)

        # Add mock input image.
        sample["image"] = torch.zeros(
            (3, self.config.image_h, self.config.image_w), dtype=torch.float32
        )

        # Run optional data preprocessing.
        preprocess_func = self.config.preprocess_func

        return preprocess_func(sample)
