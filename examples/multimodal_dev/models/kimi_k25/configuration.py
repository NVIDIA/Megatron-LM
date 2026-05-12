# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Configuration helpers for Kimi K2.5 VL vision-language model.

Provides MLATransformerConfig builders for the language decoder.
The vision encoder is dynamically loaded from HuggingFace (MoonViT3d),
so no vision TransformerConfig is needed here.

The language backbone uses MoE with Multi-Latent Attention (MLA),
sharing architecture with DeepSeek V2/V3 and Kimi K2.

Supported language variants:
    ``proxy``    4 layers, 16 experts — single-node testing
    ``full``     61 layers, 256 experts — production Kimi K2.5 VL
"""

import torch

from megatron.core.transformer.transformer_config import MLATransformerConfig

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# Token IDs from Kimi K2.5 HF config defaults
KIMI_K25_IMAGE_TOKEN_ID: int = 163605  # media_placeholder_token_id
KIMI_K25_BOS_TOKEN_ID: int = 163584
KIMI_K25_EOS_TOKEN_ID: int = 163585
KIMI_K25_PAD_TOKEN_ID: int = 163839

# Vocabulary size from HF config (163840), rounded up for divisibility
KIMI_K25_VOCAB_SIZE: int = 163840

# MoonViT3d vision geometry (224 px input)
VISION_PATCH_SIZE: int = 14
VISION_TEMPORAL: int = 1
VISION_HEIGHT_PATCHES: int = 16       # 224 // 14
VISION_WIDTH_PATCHES: int = 16        # 224 // 14
VISION_MERGE_SIZE: int = 2
VISION_IN_CHANNELS: int = 3
VISION_TOTAL_RAW_PATCHES: int = (
    VISION_TEMPORAL * VISION_HEIGHT_PATCHES * VISION_WIDTH_PATCHES
)  # 256
VISION_MERGED_PATCHES: int = (
    VISION_TEMPORAL
    * (VISION_HEIGHT_PATCHES // VISION_MERGE_SIZE)
    * (VISION_WIDTH_PATCHES // VISION_MERGE_SIZE)
)  # 64
VISION_PER_PATCH_DIM: int = (
    VISION_IN_CHANNELS * VISION_PATCH_SIZE * VISION_PATCH_SIZE
)  # 588

# ---------------------------------------------------------------------------
# Language config variants
# ---------------------------------------------------------------------------

# Kimi K2 / K2.5 architecture values (from HF config: moonshotai/Kimi-K2.5).
#
# HF config → MCore mapping:
#   hidden_size=7168, intermediate_size=18432, num_attention_heads=64,
#   num_key_value_heads=64, q_lora_rank=1536, kv_lora_rank=512,
#   qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
#   n_routed_experts=384, num_experts_per_tok=8, moe_intermediate_size=2048,
#   n_shared_experts=1, first_k_dense_replace=1, vocab_size=163840,
#   rope_theta=50000, rope_scaling=yarn(factor=64).
#
# The "full" variant matches the Kimi K2.5 VL production model.
# The "proxy" variant shrinks layers/experts for quick testing while
# keeping the same hidden_size and MLA config.
_VARIANT_CONFIGS = {
    "proxy": {
        "num_layers": 4,
        "hidden_size": 7168,
        "ffn_hidden_size": 1024,
        "num_attention_heads": 64,
        "num_query_groups": 64,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_head_dim": 128,
        "qk_pos_emb_head_dim": 64,
        "v_head_dim": 128,
        "num_moe_experts": 16,
        "moe_router_topk": 8,
        "moe_ffn_hidden_size": 64,
        "moe_shared_expert_intermediate_size": 2048,
        "first_k_dense_replace": 1,
    },
    "full": {
        "num_layers": 61,
        "hidden_size": 7168,
        "ffn_hidden_size": 18432,
        "num_attention_heads": 64,
        "num_query_groups": 64,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_head_dim": 128,
        "qk_pos_emb_head_dim": 64,
        "v_head_dim": 128,
        "num_moe_experts": 384,
        "moe_router_topk": 8,
        "moe_ffn_hidden_size": 2048,
        "moe_shared_expert_intermediate_size": 2048,
        "first_k_dense_replace": 1,
    },
}


def get_kimi_k25_language_config(
    variant: str = "proxy",
    **overrides,
) -> MLATransformerConfig:
    """MLATransformerConfig for the Kimi K2.5 VL language decoder.

    Args:
        variant: One of ``proxy``, ``full``.
        **overrides: Override any MLATransformerConfig field.

    Returns:
        Fully-populated MLATransformerConfig.
    """
    if variant not in _VARIANT_CONFIGS:
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Choose from {list(_VARIANT_CONFIGS.keys())}"
        )

    v = _VARIANT_CONFIGS[variant]

    # Build moe_layer_freq: first_k_dense_replace dense layers, rest MoE
    first_k = v["first_k_dense_replace"]
    total = v["num_layers"]
    moe_layer_freq = [0] * first_k + [1] * (total - first_k)

    kwargs = dict(
        # Architecture
        num_layers=v["num_layers"],
        hidden_size=v["hidden_size"],
        ffn_hidden_size=v["ffn_hidden_size"],
        num_attention_heads=v["num_attention_heads"],
        num_query_groups=v["num_query_groups"],
        # MLA-specific
        multi_latent_attention=True,
        q_lora_rank=v["q_lora_rank"],
        kv_lora_rank=v["kv_lora_rank"],
        qk_head_dim=v["qk_head_dim"],
        qk_pos_emb_head_dim=v["qk_pos_emb_head_dim"],
        v_head_dim=v["v_head_dim"],
        # Normalization & activation
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        # Attention
        qk_layernorm=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_bias_linear=False,
        # RoPE — YaRN params from HF config rope_scaling dict
        # NOTE: position_embedding_type is set on GPTModel, not TransformerConfig
        rope_type="yarn",
        rotary_base=50000,
        rotary_scaling_factor=64,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1.0,
        mscale_all_dim=1.0,
        # MoE
        num_moe_experts=v["num_moe_experts"],
        moe_router_topk=v["moe_router_topk"],
        moe_ffn_hidden_size=v["moe_ffn_hidden_size"],
        moe_shared_expert_intermediate_size=v["moe_shared_expert_intermediate_size"],
        moe_layer_freq=moe_layer_freq,
        moe_grouped_gemm=True,
        moe_router_pre_softmax=True,
        moe_token_dispatcher_type="alltoall",
        moe_router_load_balancing_type="seq_aux_loss",
        moe_shared_expert_overlap=True,
        moe_router_enable_expert_bias=True,
        moe_router_score_function="sigmoid",
        moe_router_topk_scaling_factor=2.827,
        moe_router_dtype="fp32",
        moe_aux_loss_coeff=1e-3,
        moe_router_bias_update_rate=1e-3,
        # MoE routing extra fields (match Bridge defaults)
        moe_router_num_groups=1,
        moe_router_group_topk=1,
        # Kernel / TE fusions (must match Bridge for bitwise parity)
        apply_rope_fusion=False,
        bias_activation_fusion=True,
        masked_softmax_fusion=True,
        persist_layer_norm=True,
        bias_dropout_fusion=True,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="te",
        moe_permute_fusion=True,
        gradient_accumulation_fusion=True,
        # Misc
        attention_softmax_in_fp32=False,
        # Precision
        bf16=True,
        params_dtype=torch.bfloat16,
    )

    kwargs.update(overrides)
    return MLATransformerConfig(**kwargs)
