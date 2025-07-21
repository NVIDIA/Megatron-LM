# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Configuration utilities for the MIMO implementation of the LLaVA VLM.
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


def get_vicuna_language_model_config(  
    config: Optional[TransformerConfig] = None,
) -> TransformerConfig:
    """Return a TransformerConfig tuned for **Vicuna-7B**.

    The hyper-parameters follow the published Vicuna-7B weights (same sizes as
    Llama-7B).
    """

    cfg = TransformerConfig(num_layers=32, hidden_size=4096, num_attention_heads=32)

    # Feed-forward / MLP hidden size (11008 in original Vicuna).
    cfg.ffn_hidden_size = 11008

    # SwiGLU (SiLU-gate) activation.
    cfg.activation_func = torch.nn.functional.silu
    cfg.gated_linear_unit = True

    # Normalisation – RMSNorm
    cfg.normalization = "RMSNorm"
    cfg.rms_norm_eps = 1e-5

    # Positional embeddings – RoPE.
    cfg.position_embedding_type = "rope"
    cfg.rotary_base = 10000
    cfg.rotary_percent = 1.0

    # Sequence length.
    cfg.seq_length = 4096
    cfg.max_position_embeddings = 4096

    # Attention / dropout.
    cfg.attention_dropout = 0.0
    cfg.hidden_dropout = 0.0

    # GQA disabled (queries == heads).
    cfg.num_query_groups = 32

    # Bias usage.
    cfg.add_bias_linear = False

    # Weight sharing.
    cfg.untie_embeddings_and_output_weights = False

    # Kernel / TE fusions.
    cfg.bias_activation_fusion = True
    cfg.masked_softmax_fusion = True
    cfg.persist_layer_norm = True
    cfg.bias_dropout_fusion = True
    cfg.apply_rope_fusion = True

    # Apply user overrides last.
    if config is not None:
        for field, value in vars(config).items():
            setattr(cfg, field, value)

    return cfg

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
