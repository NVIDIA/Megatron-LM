# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Configuration utilities for the MIMO implementation of the LLaVA VLM.
"""


from typing import Optional

import torch

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
# from bagel.modeling.bagel import Qwen2Config


def get_bagel_language_model_config(
    config: Optional[TransformerConfig] = None,
    hf_config = None,
    use_moe_mlp: bool = False,
) -> TransformerConfig:
    """Return a TransformerConfig tuned for **Qwen2-7B**.

    The hyper-parameters follow the published Qwen2-7B weights..
    https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT/blob/main/llm_config.json
    """

    cfg = TransformerConfig(num_layers=hf_config.num_hidden_layers, hidden_size=hf_config.hidden_size, num_attention_heads=hf_config.num_attention_heads)

    # Feed-forward / MLP hidden size
    cfg.ffn_hidden_size = hf_config.intermediate_size

    # SwiGLU (SiLU-gate) activation.
    cfg.activation_func = torch.nn.functional.silu
    cfg.gated_linear_unit = True

    # Normalisation – RMSNorm
    cfg.normalization = "RMSNorm"
    cfg.rms_norm_eps = hf_config.rms_norm_eps

    # Positional embeddings – RoPE.
    cfg.position_embedding_type = "rope"
    cfg.rotary_base = 1000000.0
    cfg.rotary_percent = 1.0

    # Sequence length.
    cfg.seq_length = 4096
    cfg.max_position_embeddings = 32768

    # Attention / dropout.
    cfg.attention_dropout = hf_config.attention_dropout
    cfg.hidden_dropout = 0.0

    # GQA
    cfg.num_query_groups = hf_config.num_key_value_heads

    # Bias usage.
    cfg.add_bias_linear = False
    cfg.add_qkv_bias = True

    # Weight sharing.
    cfg.untie_embeddings_and_output_weights = hf_config.tie_word_embeddings

    # Kernel / TE fusions.
    cfg.bias_activation_fusion = True
    cfg.masked_softmax_fusion = True
    cfg.persist_layer_norm = True
    cfg.bias_dropout_fusion = True
    cfg.apply_rope_fusion = True

    cfg.bf16 = True

    if use_moe_mlp:
        # cfg.num_moe_experts = 128
        cfg.num_moe_experts = 16
        cfg.moe_router_load_balancing_type = "aux_loss"
        cfg.moe_aux_loss_coeff = 1e-3
        cfg.moe_router_topk = 8
        cfg.moe_router_pre_softmax = False
        cfg.moe_grouped_gemm = True
        cfg.moe_token_dispatcher_type = "alltoall"
        cfg.moe_permute_fusion = True
        cfg.moe_ffn_hidden_size = int(0.375 * hf_config.hidden_size) # 0.375 is from Qwen3-30B

    # Apply user overrides last.
    if config is not None:
        for field, value in vars(config).items():
            setattr(cfg, field, value)

    return cfg


def get_bagel_projection_config(
    hidden_size: int = 896,
    ffn_hidden_size: int = 896,
    config: Optional[TransformerConfig] = None,
) -> TransformerConfig:
    """Return a TransformerConfig for the vision projection MLP."""

    cfg = TransformerConfig(num_layers=1, hidden_size=hidden_size, num_attention_heads=1)
    cfg.ffn_hidden_size = ffn_hidden_size
    cfg.bias_activation_fusion = True
    cfg.add_bias_linear = True
    cfg.activation_func = torch.nn.functional.gelu

    cfg.bf16 = True

    # Allow caller overrides.
    if config is not None:
        for field, value in vars(config).items():
            setattr(cfg, field, value)

    return cfg


def get_bagel_language_layer_spec(num_experts: Optional[int] = None, 
                                  moe_grouped_gemm: Optional[bool] = None, 
                                  use_flex_attention: bool = False) -> ModuleSpec:
    """Layer spec for the language model (Transformer-Engine GPT block)."""
    return get_gpt_layer_with_transformer_engine_spec(num_experts=num_experts, 
                                                      moe_grouped_gemm=moe_grouped_gemm, 
                                                      use_flex_attention=use_flex_attention)


def get_bagel_projection_layer_spec() -> ModuleSpec:
    """Layer spec for the vision-projection MLP."""

    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        ),
    )
