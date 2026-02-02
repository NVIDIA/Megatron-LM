# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
HLModel DSL Example: Complex (Imagined) Model

Demonstrates using multiple layer configurations:
- A1: Global attention (full sequence), FP8 override
- A2: Sliding window attention (local context), inherits bf16
- E1: Large MoE (128 experts, top-6), FP8 override, custom moe_config
- E2: Small MoE (32 experts, top-2), inherits bf16, custom moe_config
- CommonConfig for shared settings with per-layer overrides
"""

from dataclasses import replace

import torch

from megatron.core.models.hl import (
    HLModel,
    CommonConfig,
    EmbeddingLayerConfig,
    AttentionLayerConfig,
    MambaLayerConfig,
    MoELayerConfig,
    PipelineSplit,
)

# =============================================================================
# COMMON CONFIGURATION
# =============================================================================

# Shared settings inherited by all layers (can be overridden per-layer)
common_config = CommonConfig(
    hidden_size=2688,
    bf16=True,
    tensor_model_parallel_size=8,
    sequence_parallel=True,
)

# MoE-specific configuration (copy of common_config with expert parallelism)
moe_common_config = replace(
    common_config,
    expert_model_parallel_size=16,
    expert_tensor_parallel_size=8,
)

# =============================================================================
# LAYER DEFINITIONS
# =============================================================================

# Layers inherit hidden_size and parallelism settings from common_config

Embed = EmbeddingLayerConfig(
    vocab_size=131072,
    max_sequence_length=8192,
    position_embedding_type="none",
)

# Defaults to using common_config
M1 = MambaLayerConfig(
    num_heads=64,
    head_dim=64,
    state_size=128,
    conv_kernel_size=4,
)

# Defaults to using common_config
A1 = AttentionLayerConfig(
    num_attention_heads=32,
    num_query_groups=2,
    kv_channels=128,
    use_flash_attention=True,
    dtype=torch.float8_e4m3fn,  # Override: use FP8 for faster attention
)

# Defaults to using common_config
A2 = AttentionLayerConfig(
    num_attention_heads=32,
    num_query_groups=2,
    kv_channels=128,
    use_flash_attention=True,
    sliding_window_size=4096,
)

E1 = MoELayerConfig(
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

E2 = MoELayerConfig(
    common_config=replace(moe_common_config, expert_model_parallel_size=4),
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

# =============================================================================
# LAYER PATTERN
# =============================================================================

# Pipeline split marker
PS = PipelineSplit()

# Define each pipeline stage
# Hybrid pattern with multiple layer types
P1 = [M1, [E1, M1] * 2, A1, [E2, M1] * 3, A2]
P2 = [[E1, M1] * 3, A2, [E2, M1] * 3, A1]
P3 = [[E1, M1] * 3, A2, [E2, M1] * 4, A1]
P4 = [[E1, M1] * 4, E2]

layer_pattern = [P1, PS, P2, PS, P3, PS, P4]

# =============================================================================
# MODEL
# =============================================================================

complex_model = HLModel(
    common_config=common_config,
    embedding=Embed,
    layer_pattern=layer_pattern,
    share_embeddings_and_output_weights=False,
    normalization="RMSNorm",
    disable_bias_linear=True,
    init_method_std=0.0173,
)


if __name__ == "__main__":
    print(f"Created complex hybrid model: {complex_model}")
