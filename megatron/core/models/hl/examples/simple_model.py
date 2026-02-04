# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
HLModel DSL Example: Simple 7B Dense Transformer

A straightforward dense transformer model designed for training on
a small cluster (<100 GPUs). Uses only tensor and data parallelism.

This example demonstrates:
- Simple layer pattern without pipeline stages
- Dense MLP layers (no MoE)
- Basic parallelism configuration (TP=8, SP enabled)
- LayerConfig for shared settings with per-layer overrides
"""

from megatron.core.models.hl import (
    AttentionLayerConfig,
    EmbeddingLayerConfig,
    HLModel,
    LayerConfig,
    MLPLayerConfig,
)

# =============================================================================
# COMMON CONFIGURATION
# =============================================================================

# Shared settings inherited by all layers (can be overridden per-layer)
common_config = LayerConfig(
    hidden_size=4096, bf16=True, tensor_model_parallel_size=8, sequence_parallel=True
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

Attention = AttentionLayerConfig(
    common_config=common_config,
    num_attention_heads=32,
    num_query_groups=8,
    kv_channels=128,
    use_flash_attention=True,
)

Mlp = MLPLayerConfig(common_config=common_config, ffn_hidden_size=14336, activation="swiglu")

# =============================================================================
# LAYER PATTERN
# =============================================================================

# Simple pattern: 32 layers of Attention + MLP
# No pipeline parallelism needed for this small model
layer_pattern = [Attention, Mlp] * 32

# =============================================================================
# MODEL
# =============================================================================

simple_model = HLModel(
    common_config=common_config,
    embedding=Embedding,
    layer_pattern=layer_pattern,
    share_embeddings_and_output_weights=True,
    normalization="RMSNorm",
    disable_bias_linear=True,
    init_method_std=0.02,
)


if __name__ == "__main__":
    print(f"Created simple 7B model: {simple_model}")
