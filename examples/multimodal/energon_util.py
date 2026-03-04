# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
import torch
from dataclasses import dataclass
from typing import Any, List

from megatron.energon import Sample


@dataclass
class SampleListSample(Sample):
    """Sample type for a list of samples of any type which needs to be packed together.

    This is useful for datasets which are packed offline.
    """

    #: The images of the sequence
    samples: List[Any]


@dataclass
class OfflineTargetAspectRatioSample(Sample):
    """Sample type for image + text samples with target aspect ratio computed offline."""

    #: The images of the sequence
    images: List[torch.Tensor]
    #: The texts of the sequence
    texts: List[str]
    target_aspect_ratio: List[List]

    if not hasattr(Sample, "__subflavor__"):
        __subflavor__: str
