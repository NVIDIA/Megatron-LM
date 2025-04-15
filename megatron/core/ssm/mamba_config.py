# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass

from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class HybridConfig(TransformerConfig):
    """Configuration object for hybrid models."""

    mamba_state_dim: int = 128
    """The dimensionality of the state representation in Mamba layers."""

    mamba_head_dim: int = 64
    """The dimensionality of the heads in the Mamba layers."""

    mamba_num_groups: int = 8
    """The number of groups used in Mamba layers."""

    is_hybrid_model: bool = True
    """Indicates whether the model is a hybrid model."""
