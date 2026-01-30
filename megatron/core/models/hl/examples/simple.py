# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
HLModel DSL Example: Simple 7B Dense Transformer

A straightforward dense transformer model designed for training on
a small cluster (<100 GPUs). Uses only tensor and data parallelism.

This example demonstrates:
- Simple layer pattern without pipeline stages
- Dense MLP layers (no MoE)
- Basic parallelism configuration (TP=8, SP enabled)
"""

from megatron.core.models.hl import (
    HLModel,
    HLModelConfig,
    AttentionLayer,
    MLPLayer,
    ParallelismConfig,
)

# =============================================================================
# SIMPLE 7B DENSE TRANSFORMER
# =============================================================================

simple_config = HLModelConfig(
    vocab_size=128000,
    max_sequence_length=4096,

    # Simple repeating pattern: Attention + MLP for each layer
    # 32 layers total, no pipeline stages
    layer_pattern="(A1 F1)x32",

    layer_configs={
        "A1": AttentionLayer(
            hidden_size=4096,
            num_attention_heads=32,
            num_query_groups=8,
            kv_channels=128,
            use_flash_attention=True,
            parallelism=ParallelismConfig(
                tensor_parallel_size=8,
                sequence_parallel=True,
            ),
        ),

        "F1": MLPLayer(
            hidden_size=4096,
            ffn_hidden_size=14336,
            activation="swiglu",
            parallelism=ParallelismConfig(
                tensor_parallel_size=8,
                sequence_parallel=True,
            ),
        ),
    },

    # Model settings
    share_embeddings_and_output_weights=True,
    position_embedding_type="rope",
    rotary_base=500000,
    normalization="RMSNorm",
    disable_bias_linear=True,
    init_method_std=0.02,
    dtype="bf16",
)


def build_model() -> HLModel:
    """Build and return the simple 7B model."""
    return HLModel(simple_config)


if __name__ == "__main__":
    model = build_model()
    print(f"Created simple 7B model with config:\n{simple_config}")
