# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
HLModel DSL Example: Simple 7B Dense Transformer

A straightforward dense transformer model designed for training on
a small cluster (<100 GPUs). Uses only tensor and data parallelism.

This example demonstrates:
- Simple layer pattern without pipeline stages
- Dense MLP layers (no MoE)
- Basic parallelism configuration (TP=8, SP enabled)
- CommonConfig for shared settings with per-layer overrides
"""

from megatron.core.models.hl import (
    HLModel,
    HLModelConfig,
    CommonConfig,
    EmbeddingLayer,
    AttentionLayer,
    MLPLayer,
)

# =============================================================================
# COMMON CONFIGURATION
# =============================================================================

# Shared settings inherited by all layers (can be overridden per-layer)
common_config = CommonConfig(
    hidden_size=4096,
    bf16=True,
    tensor_model_parallel_size=8,
    sequence_parallel=True,
)

# =============================================================================
# LAYER DEFINITIONS
# =============================================================================

# Layers inherit hidden_size and parallelism settings from common_config

Embed = EmbeddingLayer(
    vocab_size=128000,
    max_sequence_length=4096,
    position_embedding_type="rope",
    rotary_base=500000,
)

A1 = AttentionLayer(
    num_attention_heads=32,
    num_query_groups=8,
    kv_channels=128,
    use_flash_attention=True,
)

F1 = MLPLayer(
    ffn_hidden_size=14336,
    activation="swiglu",
)

# =============================================================================
# LAYER PATTERN
# =============================================================================

# Simple pattern: 32 layers of Attention + MLP
# No pipeline parallelism needed for this small model
layer_pattern = [A1, F1] * 32

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

simple_config = HLModelConfig(
    common_config=common_config,
    embedding=Embed,
    layer_pattern=layer_pattern,

    # Model settings
    share_embeddings_and_output_weights=True,
    normalization="RMSNorm",
    disable_bias_linear=True,
    init_method_std=0.02,
)


def build_model() -> HLModel:
    """Build and return the simple 7B model."""
    return HLModel(simple_config)


if __name__ == "__main__":
    model = build_model()
    print(f"Created simple 7B model with config:\n{simple_config}")
