# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tiny MiniMax-M3 configurations shared by the minimax_m3 unit tests.

Both use the on-disk HF ``config.json`` spelling so the real ``MiniMaxM3Config._from_hf_dict``
mapping is exercised. Layers 0-1 are dense attention + dense MLP, layers 2-3 are MSA + MoE,
mirroring the real model's L0-2 / L3-59 split.

* ``tiny_hf_kwargs``: small heads (8/2, top-4) for the flex backend; sparse past 4 KV blocks (512 tokens).
* ``magi_hf_kwargs``: the msa_v1 kernel shapes (64/4 heads, 4x128 index heads, top-16); sparse past 2048 tokens.
"""

from __future__ import annotations

import pytest

_NUM_LAYERS = 4
_LAYER_IS_MOE = [0, 0, 1, 1]


def _hf_kwargs(*, hidden_size: int, num_attention_heads: int, num_key_value_heads: int, topk_blocks: int) -> dict:
    return dict(
        model_type="minimax_m3_vl_text",
        vocab_size=256,
        hidden_size=hidden_size,
        num_hidden_layers=_NUM_LAYERS,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=128,
        dense_intermediate_size=2 * hidden_size,
        intermediate_size=hidden_size // 2,
        shared_intermediate_size=hidden_size // 2,
        n_shared_experts=1,
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_layer_freq=list(_LAYER_IS_MOE),
        scoring_func="sigmoid",
        use_routing_bias=True,
        routed_scaling_factor=2.0,
        hidden_act="swigluoai",
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        rms_norm_eps=1e-6,
        use_gemma_norm=True,
        use_qk_norm=True,
        qk_norm_type="per_head",
        rope_theta=5_000_000,
        rotary_dim=64,
        partial_rotary_factor=0.5,
        max_position_embeddings=65536,
        tie_word_embeddings=False,
        attention_output_gate=False,
        sparse_attention_config=dict(
            use_sparse_attention=True,
            sparse_index_dim=128,
            sparse_num_index_heads=num_key_value_heads,
            sparse_topk_blocks=topk_blocks,
            sparse_block_size=128,
            sparse_score_type="max",
            sparse_init_block=0,
            sparse_local_block=1,
            sparse_disable_index_value=list(_LAYER_IS_MOE),
            sparse_attention_freq=list(_LAYER_IS_MOE),
        ),
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )


@pytest.fixture
def tiny_hf_kwargs() -> dict:
    return _hf_kwargs(hidden_size=256, num_attention_heads=8, num_key_value_heads=2, topk_blocks=4)


@pytest.fixture
def magi_hf_kwargs() -> dict:
    return _hf_kwargs(hidden_size=512, num_attention_heads=64, num_key_value_heads=4, topk_blocks=16)
