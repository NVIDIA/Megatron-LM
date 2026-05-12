# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Configuration helpers for Qwen3.5-VL vision-language model.

Provides TransformerConfig builders for the vision encoder and all language
decoder variants.  Both the standalone ``multimodal_dev`` training path and the
MIMO path import from here — this is the single source of truth.

Supported language variants (HuggingFace Qwen3.5 series):
    ``0.8b``          Dense 0.8B
    ``2b``            Dense 2B
    ``4b``            Dense 4B
    ``9b``            Dense 9B
    ``27b``           Dense 27B
    ``35b_a3b``       MoE 35B-A3B (256 experts, top-8)
    ``122b_a10b``     MoE 122B-A10B (256 experts, top-8)
    ``397b_a17b``     MoE 397B-A17B (512 experts, top-10)
    ``35b_a3b_light`` Reduced 35B-A3B for testing
    ``proxy``         Reduced proxy based on 397B for single-node testing
"""

from typing import Optional

import torch

from megatron.core.transformer.transformer_config import TransformerConfig

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

QWEN35_VL_IMAGE_TOKEN_ID: int = 248056
QWEN35_VL_VIDEO_TOKEN_ID: int = 248057
QWEN35_VL_VISION_START_TOKEN_ID: int = 248053
QWEN35_VL_VISION_END_TOKEN_ID: int = 248054
QWEN35_VL_VOCAB_SIZE: int = 248320

ROTARY_BASE: int = 10_000_000
ROTARY_PERCENT: float = 0.25
MROPE_SECTION: list = [11, 11, 10]

# ---------------------------------------------------------------------------
# Vision config
# ---------------------------------------------------------------------------

VISION_KWARGS = {
    "in_channels": 3,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "out_hidden_size": 3584,
    "max_num_positions": 2304,
}

# Three distinct vision encoder architectures in the Qwen3.5 family.
_VISION_SMALL = {
    "num_layers": 12, "hidden_size": 768, "num_attention_heads": 12,
    "kv_channels": 64, "ffn_hidden_size": 3072,
}
_VISION_MEDIUM = {
    "num_layers": 24, "hidden_size": 1024, "num_attention_heads": 16,
    "kv_channels": 64, "ffn_hidden_size": 4096,
}
_VISION_LARGE = {
    "num_layers": 27, "hidden_size": 1152, "num_attention_heads": 16,
    "kv_channels": 72, "ffn_hidden_size": 4304,
}

# Per-variant vision config.  ``out_hidden_size`` equals the language model's
# hidden_size and controls the merger projection output dimension.
_VISION_VARIANT_CONFIGS = {
    "0.8b":       {**_VISION_SMALL,  "out_hidden_size": 1024},
    "2b":         {**_VISION_MEDIUM, "out_hidden_size": 2048},
    "4b":         {**_VISION_MEDIUM, "out_hidden_size": 2560},
    "9b":         {**_VISION_LARGE,  "out_hidden_size": 4096},
    "27b":        {**_VISION_LARGE,  "out_hidden_size": 5120},
    "35b_a3b":    {**_VISION_LARGE,  "out_hidden_size": 2048},
    "122b_a10b":  {**_VISION_LARGE,  "out_hidden_size": 3072},
    "397b_a17b":  {**_VISION_LARGE,  "out_hidden_size": 4096},
}

# Fallback for proxy/unknown variants (large ViT, generic out_hidden_size).
_VISION_DEFAULT = {**_VISION_LARGE, "out_hidden_size": 3584}


def get_qwen35_vl_vision_config(
    num_layers_override: Optional[int] = None,
    variant: Optional[str] = None,
) -> TransformerConfig:
    """TransformerConfig for the Qwen3.5-VL vision encoder.

    Three ViT architectures are used across the family:
    - Small  (0.8b): depth 12, 768-dim, 12 heads
    - Medium (2b, 4b): depth 24, 1024-dim, 16 heads
    - Large  (9b, 27b, MoE variants): depth 27, 1152-dim, 16 heads

    Args:
        num_layers_override: Override vision backbone depth for proxy runs.
        variant: Language model variant name.  When set, selects the
            matching vision config from ``_VISION_VARIANT_CONFIGS`` if one
            exists; otherwise the default large-ViT config is used.
    """
    vcfg = _VISION_VARIANT_CONFIGS.get(variant, _VISION_DEFAULT)
    num_layers = vcfg["num_layers"]
    if num_layers_override is not None:
        num_layers = num_layers_override

    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=vcfg["hidden_size"],
        num_attention_heads=vcfg["num_attention_heads"],
        kv_channels=vcfg["kv_channels"],
        ffn_hidden_size=vcfg["ffn_hidden_size"],
        hidden_dropout=0.0,
        attention_dropout=0.0,
        layernorm_epsilon=1e-6,
        normalization="LayerNorm",
        gated_linear_unit=False,
        activation_func=lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
        bias_activation_fusion=False,
        apply_query_key_layer_scaling=False,
        apply_rope_fusion=False,
        bf16=False,
    )


# ---------------------------------------------------------------------------
# Language config variants
# ---------------------------------------------------------------------------

_VARIANT_CONFIGS = {
    "0.8b": {
        "num_layers": 24,
        "hidden_size": 1024,
        "ffn_hidden_size": 3584,
        "num_attention_heads": 8,
        "num_query_groups": 2,
        "kv_channels": 256,
        "linear_num_value_heads": 16,
        "num_moe_experts": None,
        "moe_router_topk": None,
        "moe_ffn_hidden_size": None,
        "moe_shared_expert_intermediate_size": None,
    },
    "2b": {
        "num_layers": 24,
        "hidden_size": 2048,
        "ffn_hidden_size": 6144,
        "num_attention_heads": 8,
        "num_query_groups": 2,
        "kv_channels": 256,
        "linear_num_value_heads": 16,
        "num_moe_experts": None,
        "moe_router_topk": None,
        "moe_ffn_hidden_size": None,
        "moe_shared_expert_intermediate_size": None,
    },
    "4b": {
        "num_layers": 32,
        "hidden_size": 2560,
        "ffn_hidden_size": 9216,
        "num_attention_heads": 16,
        "num_query_groups": 4,
        "kv_channels": 256,
        "linear_num_value_heads": 32,
        "num_moe_experts": None,
        "moe_router_topk": None,
        "moe_ffn_hidden_size": None,
        "moe_shared_expert_intermediate_size": None,
    },
    "9b": {
        "num_layers": 32,
        "hidden_size": 4096,
        "ffn_hidden_size": 12288,
        "num_attention_heads": 16,
        "num_query_groups": 4,
        "kv_channels": 256,
        "linear_num_value_heads": 32,
        "num_moe_experts": None,
        "moe_router_topk": None,
        "moe_ffn_hidden_size": None,
        "moe_shared_expert_intermediate_size": None,
    },
    "27b": {
        "num_layers": 64,
        "hidden_size": 5120,
        "ffn_hidden_size": 17408,
        "num_attention_heads": 24,
        "num_query_groups": 4,
        "kv_channels": 256,
        "linear_num_value_heads": 48,
        "num_moe_experts": None,
        "moe_router_topk": None,
        "moe_ffn_hidden_size": None,
        "moe_shared_expert_intermediate_size": None,
    },
    "35b_a3b": {
        "num_layers": 40,
        "hidden_size": 2048,
        "ffn_hidden_size": 4096,
        "num_attention_heads": 16,
        "num_query_groups": 2,
        "kv_channels": 256,
        "linear_num_value_heads": 32,
        "num_moe_experts": 256,
        "moe_router_topk": 8,
        "moe_ffn_hidden_size": 512,
        "moe_shared_expert_intermediate_size": 512,
    },
    "35b_a3b_light": {
        "num_layers": 20,
        "hidden_size": 2048,
        "ffn_hidden_size": 4096,
        "num_attention_heads": 16,
        "num_query_groups": 2,
        "kv_channels": 256,
        "linear_num_value_heads": 32,
        "num_moe_experts": 256,
        "moe_router_topk": 8,
        "moe_ffn_hidden_size": 512,
        "moe_shared_expert_intermediate_size": 512,
    },
    "122b_a10b": {
        "num_layers": 48,
        "hidden_size": 3072,
        "ffn_hidden_size": 8192,
        "num_attention_heads": 32,
        "num_query_groups": 2,
        "kv_channels": 256,
        "linear_num_value_heads": 64,
        "num_moe_experts": 256,
        "moe_router_topk": 8,
        "moe_ffn_hidden_size": 1024,
        "moe_shared_expert_intermediate_size": 1024,
    },
    "397b_a17b": {
        "num_layers": 60,
        "hidden_size": 4096,
        "ffn_hidden_size": 10240,
        "num_attention_heads": 32,
        "num_query_groups": 2,
        "kv_channels": 256,
        "linear_num_value_heads": 64,
        "num_moe_experts": 512,
        "moe_router_topk": 10,
        "moe_ffn_hidden_size": 1024,
        "moe_shared_expert_intermediate_size": 1024,
    },
    "proxy": {
        "num_layers": 4,
        "hidden_size": 4096,
        "ffn_hidden_size": 10240,
        "num_attention_heads": 32,
        "num_query_groups": 2,
        "kv_channels": 256,
        "linear_num_value_heads": 64,
        "num_moe_experts": 16,
        "moe_router_topk": 2,
        "moe_ffn_hidden_size": 1024,
        "moe_shared_expert_intermediate_size": 1024,
    },
}


def get_qwen35_vl_language_config(
    variant: str = "proxy",
    **overrides,
) -> TransformerConfig:
    """TransformerConfig for the Qwen3.5-VL language decoder.

    The ``397b_a17b`` variant reproduces the MIMO
    ``get_qwen35_language_model_config()`` output exactly.

    Args:
        variant: One of ``0.8b``, ``2b``, ``4b``, ``9b``, ``27b``,
            ``35b_a3b``, ``122b_a10b``, ``397b_a17b``,
            ``35b_a3b_light``, ``proxy``.
        **overrides: Override any TransformerConfig field.

    Returns:
        Fully-populated TransformerConfig.
    """
    if variant not in _VARIANT_CONFIGS:
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Choose from {list(_VARIANT_CONFIGS.keys())}"
        )

    v = _VARIANT_CONFIGS[variant]

    kwargs = dict(
        # Architecture
        num_layers=v["num_layers"],
        hidden_size=v["hidden_size"],
        ffn_hidden_size=v["ffn_hidden_size"],
        num_attention_heads=v["num_attention_heads"],
        num_query_groups=v["num_query_groups"],
        kv_channels=v["kv_channels"],
        # Normalization & activation
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        layernorm_zero_centered_gamma=True,
        apply_residual_connection_post_layernorm=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        # MRoPE section (interleaved T/H/W layout, Qwen3.5-VL style)
        mrope_section=list(MROPE_SECTION),
        mrope_interleaved=True,
        rotary_interleaved=False,
        # Attention
        qk_layernorm=True,
        attention_output_gate=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_bias_linear=False,
        # Hybrid attention (GatedDeltaNet)
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=v["linear_num_value_heads"],
        # Kernel / TE fusions
        bias_activation_fusion=True,
        masked_softmax_fusion=True,
        persist_layer_norm=True,
        bias_dropout_fusion=True,
        apply_rope_fusion=False,
        # Precision
        bf16=True,
    )

    # MoE config (only for MoE variants)
    if v["num_moe_experts"] is not None:
        kwargs.update(
            num_moe_experts=v["num_moe_experts"],
            moe_router_topk=v["moe_router_topk"],
            moe_ffn_hidden_size=v["moe_ffn_hidden_size"],
            moe_shared_expert_intermediate_size=v[
                "moe_shared_expert_intermediate_size"
            ],
            moe_shared_expert_gate=True,
            moe_layer_freq=1,
            moe_router_pre_softmax=False,
            moe_router_load_balancing_type="global_aux_loss",
            moe_permute_fusion=True,
            moe_aux_loss_coeff=1e-3,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            moe_router_dtype="fp32",
        )

    kwargs.update(overrides)
    return TransformerConfig(**kwargs)
