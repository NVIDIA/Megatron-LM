# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
HLModel DSL Example: Ablation-Ready Nemotron-3 Nano

Based on Nemotron-3 Nano (30B total, 3B active), restructured as a factory
function so that individual settings can be overridden from the command line
via `pretrain_hl.py --override`.

Usage:
    # Default configuration (identical to nemotron3_nano.py)
    python pretrain_hl.py --hlmodel-path megatron/core/models/hl/examples/ablations.py

    # Override hidden size for a scaling ablation
    python pretrain_hl.py --hlmodel-path megatron/core/models/hl/examples/ablations.py \
        --override hidden_size=4096

    # Sweep multiple values from a shell loop
    for hs in 1024 2048 4096; do
        python pretrain_hl.py --hlmodel-path megatron/core/models/hl/examples/ablations.py \
            --override hidden_size=$hs num_experts=64
    done
"""

from megatron.core.models.hl import (
    AttentionLayerConfig,
    CommonLayerConfig,
    CrossEntropyLayerConfig,
    EmbeddingLayerConfig,
    HLModelConfig,
    MambaLayerConfig,
    MoELayerConfig,
    PipelineSplit,
)

# =============================================================================
# DEFAULTS — these are the knobs researchers most commonly want to vary.
# Each one maps to a specific location in the config tree below.
# =============================================================================

DEFAULTS = dict(
    # CommonLayerConfig
    hidden_size=2688,
    mixed_precision_dtype="bf16",
    normalization="RMSNorm",
    init_method_std=0.0173,
    # Attention
    num_attention_heads=32,
    num_query_groups=2,
    kv_channels=128,
    # MoE
    ffn_hidden_size=1856,
    num_experts=128,
    top_k=6,
    moe_aux_loss_coeff=1e-4,
    # Model parallelism
    tensor_model_parallel_size=8,
    expert_model_parallel_size=16,
    expert_tensor_parallel_size=8,
)


def build_model(**overrides):
    """Build a Nemotron-3 Nano model, optionally overriding default settings.

    Any key in DEFAULTS can be overridden. Unknown keys raise an error so
    typos are caught early.

    Returns:
        An HLModelConfig instance (not yet built).
    """
    cfg = {**DEFAULTS, **overrides}

    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise ValueError(
            f"Unknown override(s): {unknown}. " f"Valid keys: {sorted(DEFAULTS)}"
        )

    # =========================================================================
    # COMMON CONFIGURATION
    # =========================================================================

    common_config = CommonLayerConfig(
        hidden_size=cfg["hidden_size"],
        mixed_precision_dtype=cfg["mixed_precision_dtype"],
        sequence_parallel=True,
        normalization=cfg["normalization"],
        add_bias_linear=False,
        init_method_std=cfg["init_method_std"],
    )

    moe_common_config = common_config.update(
        expert_model_parallel_size=cfg["expert_model_parallel_size"],
        expert_tensor_parallel_size=cfg["expert_tensor_parallel_size"],
    )

    # =========================================================================
    # LAYER DEFINITIONS
    # =========================================================================

    Embedding = EmbeddingLayerConfig(
        common_config=common_config,
        vocab_size=131072,
        max_sequence_length=8192,
        position_embedding_type="none",
    )

    Mamba = MambaLayerConfig(
        common_config=common_config,
        num_heads=64,
        head_dim=64,
        state_size=128,
        conv_kernel_size=4,
    )

    Attention = AttentionLayerConfig(
        common_config=common_config,
        num_attention_heads=cfg["num_attention_heads"],
        num_query_groups=cfg["num_query_groups"],
        kv_channels=cfg["kv_channels"],
        use_flash_attention=True,
    )

    MoE = MoELayerConfig(
        common_config=moe_common_config,
        ffn_hidden_size=cfg["ffn_hidden_size"],
        num_experts=cfg["num_experts"],
        top_k=cfg["top_k"],
        router_score_function="sigmoid",
        router_load_balancing_type="seq_aux_loss",
        router_topk_scaling_factor=2.5,
        router_enable_expert_bias=True,
        router_dtype="fp32",
        moe_aux_loss_coeff=cfg["moe_aux_loss_coeff"],
        shared_expert_intermediate_size=3712,
        activation="squared_relu",
        token_dispatcher_type="allgather",
        grouped_gemm=True,
    )

    Loss = CrossEntropyLayerConfig()

    # =========================================================================
    # LAYER PATTERN
    # =========================================================================

    PS = PipelineSplit()

    Stage1 = [Embedding, Mamba, [MoE, Mamba] * 2, Attention, [MoE, Mamba] * 3, Attention]
    Stage2 = [[MoE, Mamba] * 3, Attention, [MoE, Mamba] * 3, Attention]
    Stage3 = [[MoE, Mamba] * 3, Attention, [MoE, Mamba] * 4, Attention]
    Stage4 = [[MoE, Mamba] * 4, MoE, Loss]

    layer_pattern = [Stage1, PS, Stage2, PS, Stage3, PS, Stage4]

    # =========================================================================
    # MODEL
    # =========================================================================

    return HLModelConfig(
        common_config=common_config,
        layer_pattern=layer_pattern,
        tensor_model_parallel_size=cfg["tensor_model_parallel_size"],
        untie_embeddings_and_output_weights=True,
    )


# Allow direct execution for quick inspection
if __name__ == "__main__":
    model = build_model()
    print(f"Created Nemotron-3 Nano model: {model}")
