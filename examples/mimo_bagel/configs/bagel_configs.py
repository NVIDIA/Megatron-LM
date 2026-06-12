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
from megatron.core.models.bagel.flex_attention import FlexAttention
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


def get_bagel_language_model_config_qwen3_30b(
    config: Optional[TransformerConfig] = None,
    hf_config=None,
) -> TransformerConfig:
    """Return a TransformerConfig for **Qwen3-30B-A3B** (MoE backbone).

    Qwen3-30B-A3B-Instruct-2507 specifics vs Qwen2-7B:
      - hidden_size: 2048 (vs 3584)
      - num_hidden_layers: 48 (vs 28)
      - num_attention_heads: 32 (vs 28)
      - head_dim: 128 explicit (not 2048/32=64 — must set kv_channels)
      - num_key_value_heads: 4 (same)
      - ALL layers are MoE (decoder_sparse_step=1): 128 experts, top-8
      - moe_intermediate_size: 768
      - rope_theta: 10_000_000 (10× larger than Qwen2-7B)
      - max_position_embeddings: 262144
      - vocab_size: 151936 (same as Qwen2-7B)

    Reference: https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507
    """
    cfg = TransformerConfig(
        num_layers=hf_config.num_hidden_layers,
        hidden_size=hf_config.hidden_size,
        num_attention_heads=hf_config.num_attention_heads,
    )

    # Qwen3 sets head_dim=128 explicitly, independent of hidden_size/num_heads.
    # Without this, MCore would compute kv_channels = 2048/32 = 64 (wrong).
    head_dim = getattr(hf_config, 'head_dim', None)
    if head_dim is not None:
        cfg.kv_channels = head_dim

    # Feed-forward / MLP hidden size (dense fallback; all layers are MoE in practice).
    cfg.ffn_hidden_size = hf_config.intermediate_size

    # SwiGLU (SiLU-gate) activation.
    cfg.activation_func = torch.nn.functional.silu
    cfg.gated_linear_unit = True

    # Normalisation – RMSNorm.
    cfg.normalization = "RMSNorm"
    cfg.rms_norm_eps = hf_config.rms_norm_eps

    # Positional embeddings – RoPE.
    # Qwen3 uses rope_theta=10_000_000 (10× larger than Qwen2's 1_000_000).
    cfg.position_embedding_type = "rope"
    cfg.rotary_base = getattr(hf_config, 'rope_theta', 10_000_000.0)
    cfg.rotary_percent = 1.0

    # Sequence length.
    cfg.seq_length = 4096
    cfg.max_position_embeddings = getattr(hf_config, 'max_position_embeddings', 262144)

    # Attention / dropout.
    cfg.attention_dropout = hf_config.attention_dropout
    cfg.hidden_dropout = 0.0

    # GQA
    cfg.num_query_groups = hf_config.num_key_value_heads

    # Bias usage – Qwen3 has no bias on attention or linear layers.
    cfg.add_bias_linear = False
    cfg.add_qkv_bias = False

    # Weight sharing.
    cfg.untie_embeddings_and_output_weights = hf_config.tie_word_embeddings

    # Kernel / TE fusions.
    cfg.bias_activation_fusion = True
    cfg.masked_softmax_fusion = True
    cfg.persist_layer_norm = True
    cfg.bias_dropout_fusion = True
    cfg.apply_rope_fusion = True

    cfg.bf16 = True

    # Build the model on meta-device (zero memory) so TEGroupedMLP never
    # allocates all 128-expert tensors on GPU before FSDP shards them.
    # MegatronFSDP calls reset_parameters() per shard after sharding, which
    # TE's GroupedLinear supports via defer_init=True.  This also avoids the
    # "CPU RNG state changed within GPU RNG context" warning that
    # use_cpu_initialization triggers.
    cfg.init_model_with_meta_device = True

    # ── MoE — Qwen3-30B-A3B architectural / training facts (do NOT tune) ──
    # All 48 layers are sparse (decoder_sparse_step=1). These fields are
    # determined by the model architecture or by Qwen3's training recipe
    # and must match HF / the checkpoint for correctness.
    cfg.num_moe_experts = getattr(hf_config, 'num_experts', 128)           # arch: 128 experts
    cfg.moe_router_topk = getattr(hf_config, 'num_experts_per_tok', 8)     # arch: top-8
    cfg.moe_router_pre_softmax = False                                     # arch: Qwen3 = post-softmax
    cfg.moe_aux_loss_coeff = getattr(hf_config, 'router_aux_loss_coef', 1e-3)
    cfg.moe_ffn_hidden_size = getattr(hf_config, 'moe_intermediate_size',
                                      int(0.375 * hf_config.hidden_size))
    # Qwen3 training-recipe / no-feature facts. Pinned because they are
    # part of how Qwen3 was trained, NOT perf knobs to sweep:
    cfg.moe_router_load_balancing_type = "aux_loss"        # Qwen3 trained with aux_loss
    cfg.moe_router_enable_expert_bias = False              # Qwen3 not aux-loss-free (no DeepSeek-style bias)
    cfg.moe_router_bias_update_rate = 1e-3                 # default (irrelevant since bias is off)
    cfg.moe_expert_capacity_factor = None                  # Qwen3 is dropless
    cfg.moe_pad_expert_input_to_capacity = False           # Qwen3 is dropless
    cfg.moe_token_drop_policy = "probs"                    # default (irrelevant since dropless)
    cfg.moe_shared_expert_overlap = False                  # Qwen3 has no shared experts
    # ── User-tunable MoE knobs left UNSET here so the CLI flag values can
    # take effect. The propagation block in `model_providers/bagel.py`
    # (mirroring the existing recompute_* propagation) copies args.moe_*
    # into language_config after construction. Knobs handled there:
    #   moe_grouped_gemm, moe_use_legacy_grouped_gemm,
    #   moe_token_dispatcher_type, moe_flex_dispatcher_backend,
    #   moe_deepep_num_sms, moe_hybridep_num_sms,
    #   moe_permute_fusion,
    #   moe_router_fusion, moe_router_dtype, moe_router_padding_for_fp8,
    #   moe_z_loss_coeff, moe_input_jitter_eps,
    #   moe_layer_recompute

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
    spec =  get_gpt_layer_with_transformer_engine_spec(num_experts=num_experts, 
                                                      moe_grouped_gemm=moe_grouped_gemm
                                                      )
    if use_flex_attention:
        spec.submodules.self_attention.submodules.core_attention = FlexAttention
    return spec


def get_bagel_projection_layer_spec() -> ModuleSpec:
    """Layer spec for the vision-projection MLP."""

    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        ),
    )
