# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Configuration utilities for the MIMO implementation of Qwen3.5-397B-A17B VLM.

The decoder is a Qwen3-Next architecture with:
  - 60 layers, hidden_size 4096, 32 attention heads, 2 KV heads (GQA)
  - 512 experts with top-10 routing + shared expert (MoE in every layer)
  - Hybrid attention: 3 Gated-Delta-Net layers + 1 full-attention layer, repeating
  - Attention output gating, QK LayerNorm, partial RoPE (25%)

The vision encoder uses the HuggingFace Qwen3.5 vision model which outputs at
out_hidden_size=4096, matching the decoder hidden_size directly. No additional
projection MLP is needed (the PatchMerger IS the projection).
"""

from typing import Optional

import torch

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.transformer.transformer_config import TransformerConfig


def get_qwen35_language_model_config(
    config: Optional[TransformerConfig] = None,
) -> TransformerConfig:
    """Return a TransformerConfig for the Qwen3.5-397B-A17B decoder.

    All values match the HuggingFace text_config for Qwen3.5-397B-A17B.
    """

    cfg = TransformerConfig(num_layers=60, hidden_size=4096, num_attention_heads=32)

    # GQA: 2 KV heads, head_dim=256
    cfg.num_query_groups = 2
    cfg.kv_channels = 256

    # FFN — placeholder for dense MLP (unused since all layers are MoE)
    cfg.ffn_hidden_size = 10240
    cfg.activation_func = torch.nn.functional.silu
    cfg.gated_linear_unit = True  # SwiGLU

    # Normalization — Zero-Centered RMSNorm
    cfg.normalization = "RMSNorm"
    cfg.rms_norm_eps = 1e-6
    cfg.layernorm_zero_centered_gamma = True

    # Positional embeddings — partial RoPE (25%)
    cfg.position_embedding_type = "rope"
    cfg.rotary_base = 10_000_000
    cfg.rotary_percent = 0.25

    # Sequence length
    cfg.seq_length = 4096
    cfg.max_position_embeddings = 262144

    # Attention
    cfg.attention_dropout = 0.0
    cfg.hidden_dropout = 0.0
    cfg.qk_layernorm = True
    cfg.attention_output_gate = True
    cfg.add_bias_linear = False

    # Weight sharing
    cfg.untie_embeddings_and_output_weights = True

    # MoE — 512 experts, top-10 routing, every layer
    cfg.num_moe_experts = 512
    cfg.moe_ffn_hidden_size = 1024
    cfg.moe_shared_expert_intermediate_size = 1024
    cfg.moe_shared_expert_gate = True
    cfg.moe_router_topk = 10
    cfg.moe_router_pre_softmax = False
    cfg.moe_router_load_balancing_type = "aux_loss"
    cfg.moe_aux_loss_coeff = 1e-3
    cfg.moe_grouped_gemm = True
    cfg.moe_layer_freq = 1
    cfg.moe_router_dtype = "fp32"

    # Gated Delta Net — 3 linear-attention + 1 full-attention, repeating
    cfg.experimental_attention_variant = "gated_delta_net"
    cfg.linear_attention_freq = 4
    cfg.linear_conv_kernel_dim = 4
    cfg.linear_key_head_dim = 128
    cfg.linear_value_head_dim = 128
    cfg.linear_num_key_heads = 16
    cfg.linear_num_value_heads = 64

    # Kernel / TE fusions
    cfg.bias_activation_fusion = True
    cfg.masked_softmax_fusion = True
    cfg.persist_layer_norm = True
    cfg.bias_dropout_fusion = True
    cfg.apply_rope_fusion = True

    if config is not None:
        for field, value in vars(config).items():
            setattr(cfg, field, value)

    return cfg


def get_qwen35_language_layer_spec(config: TransformerConfig):
    """Layer spec for the Qwen3.5 decoder (heterogeneous GDN + full-attention + MoE).

    Must use get_transformer_block_with_experimental_attention_variant_spec
    because the model mixes linear-attention and full-attention layers.
    """
    return get_transformer_block_with_experimental_attention_variant_spec(config)
