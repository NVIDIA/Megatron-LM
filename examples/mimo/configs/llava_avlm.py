# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Configuration utilities for the MIMO implementation of the LLaVA AVLM.
"""


from typing import Optional

import torch

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
from examples.mimo.configs.llava_vlm import get_vicuna_language_model_config


def get_llava_projection_config( 
    hidden_size: int = 4096,
    config: Optional[TransformerConfig] = None,
) -> TransformerConfig:
    """Return a TransformerConfig for the vision projection MLP."""

    cfg = TransformerConfig(num_layers=1, hidden_size=hidden_size, num_attention_heads=1)
    cfg.ffn_hidden_size = 4096
    cfg.bias_activation_fusion = True
    cfg.add_bias_linear = True
    cfg.activation_func = torch.nn.functional.gelu

    # Allow caller overrides.
    if config is not None:
        for field, value in vars(config).items():
            setattr(cfg, field, value)

    return cfg



def get_vicuna_language_layer_spec() -> ModuleSpec:
    """Layer spec for the language model (Transformer-Engine GPT block)."""
    return get_gpt_layer_with_transformer_engine_spec()

def get_llava_projection_layer_spec() -> ModuleSpec:
    """Layer spec for the vision-projection MLP."""

    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        ),
    )
