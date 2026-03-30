# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Configuration helpers for Qwen3.5-VL vision-language model.

Provides TransformerConfig builders for the vision encoder and all language
decoder variants.  Both the standalone ``multimodal_v2`` training path and the
MIMO path import from here — this is the single source of truth.

Supported language variants:
    ``9b``            Dense 9B (Qwen3.5-9B)
    ``35b_a3b``       MoE 35B-A3B (Qwen3.5-35B-A3B)
    ``35b_a3b_light`` Reduced 35B-A3B (20 decoder layers, 14 ViT layers)
    ``397b_a17b``     MoE 397B-A17B (Qwen3.5-397B-A17B)
    ``proxy``         Reduced proxy based on 397B for single-node testing
"""

from typing import Optional

import torch
import torch.nn.functional as F

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


def get_qwen35_vl_vision_config(
    num_layers_override: Optional[int] = None,
    params_dtype: torch.dtype = torch.bfloat16,
) -> TransformerConfig:
    """TransformerConfig for the Qwen3.5-VL vision encoder.

    Values match the HF ``Qwen3VLVisionConfig`` defaults: depth 27,
    hidden_size 1152, 16 heads, kv_channels 72, ffn 4304,
    ``gelu_pytorch_tanh`` activation, bias enabled, LayerNorm.

    Args:
        num_layers_override: Override vision backbone depth for proxy runs.
        params_dtype: Parameter dtype (inherited from language config).
    """
    return TransformerConfig(
        num_layers=(
            27 if num_layers_override is None else num_layers_override
        ),
        hidden_size=1152,
        num_attention_heads=16,
        kv_channels=72,
        num_query_groups=16,
        ffn_hidden_size=4304,
        add_bias_linear=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        normalization="LayerNorm",
        layernorm_epsilon=1e-6,
        gated_linear_unit=False,
        activation_func=lambda x: F.gelu(x, approximate='tanh'),
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        apply_query_key_layer_scaling=False,
        apply_rope_fusion=False,
        params_dtype=params_dtype,
    )


# ---------------------------------------------------------------------------
# Language config variants
# ---------------------------------------------------------------------------

_VARIANT_CONFIGS = {
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
        variant: One of ``9b``, ``35b_a3b``, ``35b_a3b_light``,
            ``397b_a17b``, ``proxy``.
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
        # MRoPE section
        mrope_section=list(MROPE_SECTION),
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
        apply_rope_fusion=True,
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
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=1e-3,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            moe_router_dtype="fp32",
        )

    kwargs.update(overrides)
    return TransformerConfig(**kwargs)
