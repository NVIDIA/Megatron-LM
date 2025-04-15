# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
import torch
import warnings
from dataclasses import dataclass
from typing import Any, List

from megatron.energon import Sample
from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.webdataset import DefaultDecoderWebdatasetFactory


@dataclass
class SampleListSample(Sample):
    """Sample type for a list of samples of any type which needs to be packed together.
    
    This is useful for datasets which are packed offline.
    """

    #: The images of the sequence
    samples: List[Any]


class SampleListWebdataset(DefaultDecoderWebdatasetFactory[SampleListSample]):
    __sample_type__ = SampleListSample

    def __init__(self, path: EPath, **kwargs):
        warnings.warn(
            f"{type(self)} is deprecated, use the default instead and set the sample_type:\n"
            f"To convert, update your {path}/.nv-meta/dataset.yaml to:\n"
            f"# remove top-level __module__ and __class__\n"
            f"sample_type:\n"
            f"  __module__: megatron.energon\n"
            f"  __class__: {self.__sample_type__.__name__}\n"
            f"# Keep the remaining content",
            DeprecationWarning,
        )
        super().__init__(path, **kwargs)


@dataclass
class OfflineTargetAspectRatioSample(Sample):
    """Sample type for image + text samples with target aspect ratio computed offline."""

    #: The images of the sequence
    images: List[torch.Tensor]
    #: The texts of the sequence
    texts: List[str]
    target_aspect_ratio: List[List]
