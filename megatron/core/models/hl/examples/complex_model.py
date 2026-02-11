# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
HLModel DSL Example: Advanced Hybrid Model

Demonstrates using multiple layer configurations:
- GlobalAttention: Full sequence attention with FP8
- SlidingAttention: Local context attention (4096 window), bf16
- Mamba: State-space model layers
- LargeMoE: 128 experts (top-6), FP8, custom parallelism
- SmallMoE: 32 experts (top-2), bf16, custom parallelism
"""

import torch

from megatron.core.models.hl import (
    AttentionLayerConfig,
    CommonLayerConfig,
    CrossEntropyLayerConfig,
    EmbeddingLayerConfig,
    HLModelConfig,
    LinearLayerConfig,
    MambaLayerConfig,
    MTPLayerConfig,
    MoELayerConfig,
    PipelineSplit,
)

# =============================================================================
# COMMON CONFIGURATION
# =============================================================================

# Shared settings inherited by all layers (can be overridden per-layer)
common_config = CommonLayerConfig(
    hidden_size=2688,
    mixed_precision_dtype="bf16",
    sequence_parallel=True,
    normalization="RMSNorm",
    add_bias_linear=False,
    init_method_std=0.0173,
)

# MoE-specific configuration (copy of common_config with expert parallelism)
moe_common_config = common_config.update(
    expert_model_parallel_size=16, expert_tensor_parallel_size=8
)

# =============================================================================
# LAYER DEFINITIONS
# =============================================================================

Embedding = EmbeddingLayerConfig(
    common_config=common_config,
    vocab_size=131072,
    max_sequence_length=8192,
    position_embedding_type="none",
)

Mamba = MambaLayerConfig(
    common_config=common_config, num_heads=64, head_dim=64, state_size=128, conv_kernel_size=4
)

GlobalAttention = AttentionLayerConfig(
    common_config=common_config,
    num_attention_heads=32,
    num_query_groups=2,
    kv_channels=128,
    use_flash_attention=True,
    dtype=torch.float8_e4m3fn,  # Override: use FP8 for faster attention
)

SlidingAttention = AttentionLayerConfig(
    common_config=common_config,
    num_attention_heads=32,
    num_query_groups=2,
    kv_channels=128,
    use_flash_attention=True,
    sliding_window_size=4096,
)

LargeMoE = MoELayerConfig(
    common_config=moe_common_config,
    ffn_hidden_size=1856,
    num_experts=128,
    top_k=6,
    router_score_function="sigmoid",
    router_load_balancing_type="seq_aux_loss",
    router_topk_scaling_factor=2.5,
    router_enable_expert_bias=True,
    router_dtype="fp32",
    moe_aux_loss_coeff=1e-4,
    shared_expert_intermediate_size=3712,
    activation="squared_relu",
    token_dispatcher_type="allgather",
    grouped_gemm=True,
    dtype=torch.float8_e4m3fn,  # Override: FP8 for expert computation
)

SmallMoE = MoELayerConfig(
    common_config=moe_common_config.update(expert_model_parallel_size=4),
    ffn_hidden_size=3712,
    num_experts=32,
    top_k=2,
    router_score_function="softmax",
    router_load_balancing_type="aux_loss",
    router_dtype="fp32",
    moe_aux_loss_coeff=1e-2,
    activation="swiglu",
    token_dispatcher_type="alltoall",
    grouped_gemm=True,
)

MTP = MTPLayerConfig(
    common_config=common_config,
    enorm="RMSNorm",
    hnorm="RMSNorm",
    eh_proj=LinearLayerConfig(common_config=common_config),
    embedding=Embedding,
)

Loss = CrossEntropyLayerConfig()

# =============================================================================
# LAYER PATTERN
# =============================================================================

# Pipeline split marker
PS = PipelineSplit()

# Define each pipeline stage
# Hybrid pattern with multiple layer types
Stage1 = [Embedding, Mamba, [LargeMoE, Mamba] * 2, GlobalAttention, [SmallMoE, Mamba] * 3, SlidingAttention]
Stage2 = [[LargeMoE, Mamba] * 3, SlidingAttention, [SmallMoE, Mamba] * 3, GlobalAttention]
Stage3 = [[LargeMoE, Mamba] * 3, SlidingAttention, [SmallMoE, Mamba] * 4, GlobalAttention]
Stage4 = [[LargeMoE, Mamba] * 4, SmallMoE, MTP, Loss]

layer_pattern = [Stage1, PS, Stage2, PS, Stage3, PS, Stage4]

# =============================================================================
# MODEL
# =============================================================================

complex_model = HLModelConfig(
    common_config=common_config,
    layer_pattern=layer_pattern,
    tensor_model_parallel_size=8,
    untie_embeddings_and_output_weights=True,
).build()


if __name__ == "__main__":
    print(f"Created complex hybrid model: {complex_model}")
