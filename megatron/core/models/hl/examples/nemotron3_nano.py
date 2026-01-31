# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
HLModel DSL Example: Nemotron-3 Nano (30B total, 3B active)
Using `examples/post_training/modelopt/conf/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.sh` as reference.

This example demonstrates:
- Hybrid architecture with Mamba (M), Attention (A), and MoE (E) layers
- Pipeline parallelism across 4 stages
- Expert parallelism for MoE layers
"""

from megatron.core.models.hl import (
    HLModel,
    HLModelConfig,
    EmbeddingLayer,
    AttentionLayer,
    MambaLayer,
    MoELayer,
    ParallelismConfig,
    PipelineSplit,
)

# =============================================================================
# LAYER DEFINITIONS
# =============================================================================

Embed = EmbeddingLayer(
    vocab_size=131072,
    hidden_size=2688,
    max_sequence_length=8192,
    position_embedding_type="none",
    parallelism=ParallelismConfig(
        tensor_parallel_size=8,
        sequence_parallel=True,
    ),
)

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

# =============================================================================
# LAYER PATTERN
# =============================================================================

# Pipeline split marker
PS = PipelineSplit()

# Define each pipeline stage
# Pattern: MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME
# 52 layers total, split across 4 pipeline stages
P1 = [M1, [E1, M1] * 2, A1, [E1, M1] * 3, A1]
P2 = [[E1, M1] * 3, A1, [E1, M1] * 3, A1]
P3 = [[E1, M1] * 3, A1, [E1, M1] * 4, A1]
P4 = [[E1, M1] * 4, E1]

layer_pattern = [P1, PS, P2, PS, P3, PS, P4]

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

nemotron_config = HLModelConfig(
    embedding=Embed,
    layer_pattern=layer_pattern,

    # Model settings
    share_embeddings_and_output_weights=False,
    normalization="RMSNorm",
    disable_bias_linear=True,
    init_method_std=0.0173,
)


def build_model() -> HLModel:
    """Build and return the Nemotron-3 Nano model."""
    return HLModel(nemotron_config)


if __name__ == "__main__":
    model = build_model()
    print(f"Created Nemotron-3 Nano model with config:\n{nemotron_config}")
