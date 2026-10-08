# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.models.backends import get_backend
from megatron.core.transformer.mla_qk_norm_config import QKNormConfigResolver
from megatron.core.transformer.multi_latent_attention import MLASelfAttentionSubmodules
from megatron.core.transformer.transformer_config import MLATransformerConfig


@pytest.mark.parametrize("fused", [False, True])
def test_q_proj_without_q_lora_follows_spec(fused):
    config = MLATransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        q_lora_rank=None,
        kv_lora_rank=32,
        qk_head_dim=16,
        qk_pos_emb_head_dim=16,
        v_head_dim=16,
        qk_layernorm=True,
        rope_type="rope",
    )
    backend = get_backend(config.transformer_impl)
    q_proj = (
        backend.column_parallel_layer_norm_linear() if fused else backend.column_parallel_linear()
    )
    submodules = MLASelfAttentionSubmodules(
        linear_proj=backend.row_parallel_linear(),
        q_layernorm=None,
        kv_layernorm=None,
        linear_q_proj=q_proj,
        linear_kv_down_proj=backend.linear(),
        linear_kv_up_proj=backend.column_parallel_linear(),
    )
    resolved = QKNormConfigResolver(config, submodules).resolve()
    assert resolved["linear_q_proj"] is q_proj
    assert resolved["linear_kv_up_proj"] is backend.column_parallel_layer_norm_linear()
