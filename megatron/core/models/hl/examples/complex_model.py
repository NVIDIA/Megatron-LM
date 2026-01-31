# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
HLModel DSL Example: Complex (Imagined) Model

Demonstrates using multiple layer configurations:
- A1: Global attention (full sequence)
- A2: Sliding window attention (local context)
- E1: Large MoE (128 experts, top-6)
- E2: Small MoE (32 experts, top-2)
"""

from megatron.core.models.hl import (
    HLModel,
    HLModelConfig,
    AttentionLayer,
    MambaLayer,
    MoELayer,
    ParallelismConfig,
    PipelineSplit,
)

# =============================================================================
# LAYER DEFINITIONS
# =============================================================================

M1 = MambaLayer(
    hidden_size=2688,
    num_heads=64,
    head_dim=64,
    state_size=128,
    conv_kernel_size=4,
    parallelism=ParallelismConfig(
        tensor_parallel_size=8,
        sequence_parallel=True,
    ),
)

A1 = AttentionLayer(
    hidden_size=2688,
    num_attention_heads=32,
    num_query_groups=2,
    kv_channels=128,
    use_flash_attention=True,
    parallelism=ParallelismConfig(
        tensor_parallel_size=8,
        sequence_parallel=True,
    ),
)

A2 = AttentionLayer(
    hidden_size=2688,
    num_attention_heads=32,
    num_query_groups=2,
    kv_channels=128,
    use_flash_attention=True,
    sliding_window_size=4096,
    parallelism=ParallelismConfig(
        tensor_parallel_size=8,
        sequence_parallel=True,
    ),
)

E1 = MoELayer(
    hidden_size=2688,
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
    parallelism=ParallelismConfig(
        tensor_parallel_size=8,
        sequence_parallel=True,
        expert_parallel_size=16,
        expert_tensor_parallel_size=8,
    ),
)

E2 = MoELayer(
    hidden_size=2688,
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
    parallelism=ParallelismConfig(
        tensor_parallel_size=8,
        sequence_parallel=True,
        expert_parallel_size=4,
    ),
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
# MODEL CONFIGURATION
# =============================================================================

model_config = HLModelConfig(
    vocab_size=131072,
    max_sequence_length=8192,
    layer_pattern=layer_pattern,

    # Model settings
    share_embeddings_and_output_weights=False,
    position_embedding_type="none",
    normalization="RMSNorm",
    disable_bias_linear=True,
    init_method_std=0.0173,
    dtype="bf16",
)


def build_model() -> HLModel:
    """Build and return the complex hybrid model."""
    return HLModel(model_config)


if __name__ == "__main__":
    model = build_model()
    print(f"Created complex hybrid model with config:\n{model_config}")
