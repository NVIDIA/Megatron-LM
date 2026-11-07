# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.training.models.base import ModelBuilder, ModelConfig, Serializable, compose_hooks
from megatron.training.models.dist_utils import (
    build_virtual_pipeline_stages,
    unimodal_build_distributed_models,
)
from megatron.training.models.distillation import DistillationConfig
from megatron.training.models.gpt import GPTModelBuilder, GPTModelConfig
from megatron.training.models.hybrid import HybridModelBuilder, HybridModelConfig
from megatron.training.models.transforms import ModelTransformConfig

MambaModelConfig = HybridModelConfig
MambaModelBuilder = HybridModelBuilder

__all__ = [
    "DistillationConfig",
    "ModelTransformConfig",
    "ModelBuilder",
    "ModelConfig",
    "Serializable",
    "compose_hooks",
    "build_virtual_pipeline_stages",
    "unimodal_build_distributed_models",
    "HybridModelConfig",
    "HybridModelBuilder",
    "MambaModelConfig",
    "MambaModelBuilder",
    "GPTModelConfig",
    "GPTModelBuilder",
]
