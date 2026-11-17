# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.training.models.base import ModelBuilder, ModelConfig, Serializable, compose_hooks
from megatron.training.models.bert import BertModelConfig
from megatron.training.models.dist_utils import (
    build_virtual_pipeline_stages,
    unimodal_build_distributed_models,
)
from megatron.training.models.hybrid import HybridModelBuilder, HybridModelConfig
from megatron.training.models.gpt import GPTModelBuilder, GPTModelConfig
from megatron.training.models.llava import LLaVAModelConfig
from megatron.training.models.mimo import MimoBuildConfig
from megatron.training.models.t5 import T5ModelConfig

MambaModelConfig = HybridModelConfig
MambaModelBuilder = HybridModelBuilder

__all__ = [
    "ModelBuilder",
    "ModelConfig",
    "BertModelConfig",
    "T5ModelConfig",
    "LLaVAModelConfig",
    "MimoBuildConfig",
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
