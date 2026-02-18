# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
HLModel DSL Example: Nemotron-3 Nano (30B total, 3B active)
Using `examples/post_training/modelopt/conf/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.sh` as reference.

This example demonstrates:
- Hybrid architecture with Mamba (M), Attention (A), and MoE (E) layers
- Pipeline parallelism across 4 stages
- Expert parallelism for MoE layers
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
    MambaLayerConfig,
    MoELayerConfig,
    PipelineSplit,
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

args = tyro.cli(ArgsContainer, default=ArgsContainer(hidden_size=2688))

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

Attention = AttentionLayerConfig(
    common_config=common_config,
    num_attention_heads=32,
    num_query_groups=2,
    kv_channels=128,
    use_flash_attention=True,
)

MoE = MoELayerConfig(
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
)

Loss = CrossEntropyLayerConfig()

# =============================================================================
# LAYER PATTERN
# =============================================================================

# Pipeline split marker
PS = PipelineSplit()

# Define each pipeline stage (52 layers total, split across 4 stages)
Stage1 = [Embedding, Mamba, [MoE, Mamba] * 2, Attention, [MoE, Mamba] * 3, Attention]
Stage2 = [[MoE, Mamba] * 3, Attention, [MoE, Mamba] * 3, Attention]
Stage3 = [[MoE, Mamba] * 3, Attention, [MoE, Mamba] * 4, Attention]
Stage4 = [[MoE, Mamba] * 4, MoE, Loss]

layer_pattern = [Stage1, PS, Stage2, PS, Stage3, PS, Stage4]

# =============================================================================
# MODEL
# =============================================================================

nemotron_model = HLModelConfig(
    common_config=common_config,
    layer_pattern=layer_pattern,
    tensor_model_parallel_size=8,
    untie_embeddings_and_output_weights=True,
).build()


if __name__ == "__main__":
    print(f"Created Nemotron-3 Nano model: {nemotron_model}")
