# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
HLModel DSL Example: Simple 7B Dense Transformer

A straightforward dense transformer model designed for training on
a small cluster (<100 GPUs). Uses only tensor and data parallelism.

This example demonstrates:
- Simple layer pattern without pipeline stages
- Dense MLP layers (no MoE)
- Basic parallelism configuration (TP=8, SP enabled)
- CommonLayerConfig for shared settings with per-layer overrides
"""

from dataclasses import dataclass

from megatron.core.models.hl import (
    AttentionLayerConfig,
    CommonLayerConfig,
    CrossEntropyLayerConfig,
    EmbeddingLayerConfig,
    HLModelConfig,
    make_args_container,
    MLPLayerConfig,
)

import tyro

# =============================================================================
# ARGUMENTS
# =============================================================================

@dataclass
class ExtraArgs:
    num_attention_heads: int = 32


ArgsContainer = make_args_container(
    hl_model_config=HLModelConfig,
    common_layer_config=CommonLayerConfig,
    extra_args=ExtraArgs,
)

args = tyro.cli(ArgsContainer, default=ArgsContainer(hidden_size=4096))

# =============================================================================
# COMMON CONFIGURATION
# =============================================================================

# Shared settings inherited by all layers (can be overridden per-layer)
common_config = CommonLayerConfig(
    hidden_size=args.hidden_size,
    mixed_precision_dtype="bf16",
    sequence_parallel=True,
    normalization="RMSNorm",
    add_bias_linear=False,
    init_method_std=0.02,
)

# =============================================================================
# LAYER DEFINITIONS
# =============================================================================

Embedding = EmbeddingLayerConfig(
    common_config=common_config,
    vocab_size=128000,
    max_sequence_length=4096,
    position_embedding_type="rope",
    rotary_base=500000,
)

assert args.hidden_size % args.num_attention_heads == 0
Attention = AttentionLayerConfig(
    common_config=common_config,
    num_attention_heads=args.num_attention_heads,
    num_query_groups=8,
    kv_channels=args.hidden_size // args.num_attention_heads,
    use_flash_attention=True,
)

Mlp = MLPLayerConfig(common_config=common_config, ffn_hidden_size=14336, activation="swiglu")

Loss = CrossEntropyLayerConfig()

# =============================================================================
# LAYER PATTERN
# =============================================================================

# Simple pattern: 32 layers of Attention + MLP
# No pipeline parallelism needed for this small model
layer_pattern = [Embedding] + [Attention, Mlp] * 32 + [Loss]

# =============================================================================
# MODEL
# =============================================================================

simple_model = HLModelConfig(
    common_config=common_config,
    layer_pattern=layer_pattern,
    tensor_model_parallel_size=8,
    untie_embeddings_and_output_weights=False,
).build()


if __name__ == "__main__":
    print(f"Created simple 7B model: {simple_model}")
