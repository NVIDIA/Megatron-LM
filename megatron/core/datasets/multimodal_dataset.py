# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Dict

import numpy
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset


@dataclass
class MultimodalDatasetConfig(GPTDatasetConfig):
    """Configuration object for Megatron Core Multimodal datasets.


    Note: This is unused at the moment and may be missing features. Follow-up changes will use this.

    Attributes:
        image_h (int): Image height.
        image_w (int): Image width.
    """

    image_h: int = None
    image_w: int = None

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
            Dict[str, numpy.ndarray]: The mock data.
        """
        # Get a text sample.
        sample = super().__getitem__(idx)

        # Add mock input image.
        sample["image"] = torch.zeros(
            (3, self.config.image_h, self.config.image_w), dtype=torch.float32
        )

        return sample
