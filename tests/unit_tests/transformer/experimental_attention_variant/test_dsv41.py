# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tiny V4.1 test configurations covering every attention mode."""

from typing import Any

import torch
import torch.nn.functional as F

from megatron.core.transformer.transformer_config import MLATransformerConfig


def test_factory_selects_csa2():
    """The public attention factory must not instantiate a V4 compressor for V4.1."""
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.transformer.experimental_attention_variant import (
        deepseek_v4_hybrid_attention_module_specs as specs,
    )
    from megatron.core.transformer.experimental_attention_variant.csa2 import (
        CompressedSparseAttention2,
    )

    spec = specs.get_dsv4_hybrid_module_spec_for_backend(_make_config(), TESpecProvider())
    assert spec.submodules.core_attention.func is CompressedSparseAttention2


def _make_config(
    *, params_dtype: torch.dtype = torch.float32, **overrides: Any
) -> MLATransformerConfig:
    """Create six layers covering SWA, Full2, Reuse2, Full1, Reindex1, and Reuse1."""
    values = dict(
        experimental_attention_variant="dsv4_hybrid",
        dsv4_version="v4.1",
        dsa_kernel_backend="none",
        gradient_accumulation_fusion=False,
        transformer_impl="transformer_engine",
        use_cpu_initialization=True,
        params_dtype=params_dtype,
        bf16=params_dtype == torch.bfloat16,
        num_layers=6,
        hidden_size=32,
        num_attention_heads=4,
        q_lora_rank=16,
        v_head_dim=16,
        qk_pos_emb_head_dim=8,
        output_projection_groups=2,
        output_projection_lora_rank=8,
        qk_layernorm=True,
        layernorm_epsilon=1e-20,
        normalization="RMSNorm",
        rotary_base=10000,
        csa_compress_rotary_base=160000,
        original_max_position_embeddings=65536,
        rotary_scaling_factor=16,
        beta_fast=32,
        beta_slow=1,
        mscale=0,
        mscale_all_dim=0,
        csa_window_size=4,
        csa_compress_ratios=[0, 2, 2, 1, 1, 1],
        csa2_kv_source_layers=[1, 3],
        csa2_index_source_layers=[1, 3, 4],
        csa2_candidate_source_layer=3,
        csa2_candidate_topk_blocks=2,
        csa2_candidate_block_size=2,
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=8,
        dsa_indexer_topk=4,
        dsa_indexer_rotate_activation=False,
        enable_mhc_connections=True,
        # This recipe opts in explicitly; model-version selection does not enable it.
        mhc_num_residual_streams=4,
        mhc_sinkhorn_iterations=20,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_ffn_hidden_size=48,
        moe_shared_expert_intermediate_size=48,
        moe_router_score_function="sqrtsoftplus",
        moe_router_dtype="fp32",
        moe_router_topk_scaling_factor=1.5,
        moe_router_enable_expert_bias=True,
        activation_func=F.silu,
        gated_linear_unit=True,
        activation_func_clamp_value=10,
        add_bias_linear=False,
        hidden_dropout=0,
        attention_dropout=0,
    )
    values.update(overrides)
    return MLATransformerConfig(**values)
