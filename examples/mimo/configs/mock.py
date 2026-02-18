# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Mock configuration utilities for MIMO model with vision encoder.

This module provides functions to create test configurations for:
1. Language model (based on LLaMA architecture)
2. Vision encoder (based on CLIP ViT)
3. Vision projection (MLP)

These configurations are intended for testing and development purposes only.
"""

from typing import Optional

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


def get_mock_language_model_config(config: Optional[TransformerConfig] = None) -> TransformerConfig:
    """
    Create a mock language model configuration.

    Args:
        config: Optional base configuration to modify

    Returns:
        TransformerConfig: Mock configuration for a language model
    """

    config = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=4)

    if config is not None:
        for field_name, field_value in vars(config).items():
            setattr(config, field_name, field_value)

    return config

def get_mock_vision_model_config(config: Optional[TransformerConfig] = None) -> TransformerConfig:
    """
    Create a mock vision model configuration.

    Args:
        config: Optional base configuration to modify

    Returns:
        TransformerConfig: Mock configuration for a vision model
    """
    config = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=4)

    config.add_bias_linear = True
    config.add_qkv_bias = True
    config.hidden_dropout = 0.0
    config.attention_dropout = 0.0
    config.ffn_hidden_size = config.hidden_size * 4
    config.gated_linear_unit = False
    config.kv_channels = 64
    config.layernorm_zero_centered_gamma = False
    config.apply_query_key_layer_scaling = False
    config.bias_activation_fusion = False
    config.bias_dropout_fusion = False
    config.attention_softmax_in_fp32 = True
    config.normalization = 'LayerNorm'
    config.apply_rope_fusion = False
    return config


def get_mock_projection_config(hidden_size: int = 128) -> TransformerConfig:
    """
    Create a mock projection layer configuration.

    Args:
        hidden_size: Hidden dimension size (used as the vision projection output size)

    Returns:
        TransformerConfig: Mock configuration for a projection layer
    """
    config = TransformerConfig(num_layers=1, hidden_size=hidden_size, num_attention_heads=1)

    config.ffn_hidden_size = hidden_size * 4
    config.gated_linear_unit = False
    config.bias_activation_fusion = False
    config.add_bias_linear = False
    config.normalization = 'LayerNorm'

    return config


def get_mock_language_layer_spec():
    """
    Get the mock layer specification for the language model.

    Returns:
        ModuleSpec: Mock specification for language model layers
    """
    return get_gpt_layer_with_transformer_engine_spec()


def get_mock_vision_layer_spec():
    """
    Get the mock layer specification for the vision model.

    Args:
        normalization: Type of normalization to use

    Returns:
        ModuleSpec: Mock specification for vision model layers
    """
    return get_gpt_layer_with_transformer_engine_spec()


def get_mock_projection_layer_spec():
    """
    Get the mock layer specification for the projection layer.

    Returns:
        ModuleSpec: Mock specification for projection layers
    """
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear),
    )
