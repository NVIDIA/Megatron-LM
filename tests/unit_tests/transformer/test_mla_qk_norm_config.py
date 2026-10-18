# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from megatron.core.models.backends import get_backend
from megatron.core.transformer.mla_qk_norm_config import QKNormConfigResolver
from megatron.core.transformer.multi_latent_attention import MLASelfAttentionSubmodules
from megatron.core.transformer.transformer_config import MLATransformerConfig


def test_plain_q_proj_without_q_lora_is_kept():
    config = MLATransformerConfig(
        num_layers=1, hidden_size=64, num_attention_heads=4, q_lora_rank=None, qk_layernorm=True
    )
    linear = get_backend(config.transformer_impl).column_parallel_linear()
    submodules = MLASelfAttentionSubmodules(
        linear_proj=None, q_layernorm=None, kv_layernorm=None, linear_q_proj=linear
    )
    assert QKNormConfigResolver(config, submodules).resolve()["linear_q_proj"] is linear
