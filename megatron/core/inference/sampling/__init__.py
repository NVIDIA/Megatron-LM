# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core.inference.sampling.base import Sampling
from megatron.core.inference.sampling.flashinfer_sampling import FlashInferSampling
from megatron.core.inference.sampling.torch_sampling import TorchSampling

__all__ = ["Sampling", "TorchSampling", "FlashInferSampling"]
