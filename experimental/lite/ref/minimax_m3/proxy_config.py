# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Proxy-M3: randomly initialised, structurally isomorphic small MiniMax-M3 text model.

Two views of the same configuration:

* ``hf_proxy_text_config_kwargs()`` -> kwargs for
  ``transformers.MiniMaxM3VLTextConfig`` (transformers 5.16.1). The kwargs use
  the *checkpoint* spelling (``moe_layer_freq``, ``sparse_attention_config``,
  ``rope_theta`` / ``partial_rotary_factor``) so the same code path as the real
  ``config.json`` is exercised.
* ``MLITE_PROXY_FIELDS`` -> the intended Megatron Lite / TransformerConfig field
  values, one-to-one with the HF fields. Consumed by
  ``model_compose/minimax_m3/config_mapping.py`` in P2; kept here so the two
  views are reviewed side by side.

Design notes
* ``index_topk_blocks`` is shrunk from 16 to 4 so a 4K-token sequence
  (32 blocks) is already sparse; with top-16 the proxy would only become sparse
  past 2K tokens.
* Layers 0-2 are dense (full attention + dense MLP), layers 3-5 are MoE + MSA,
  reproducing the real model's L0-2 / L3-59 split.
"""

from __future__ import annotations

from typing import Any

PROXY_NUM_LAYERS = 6
PROXY_DENSE_LAYERS = 3
_LAYER_IS_MOE = [0] * PROXY_DENSE_LAYERS + [1] * (PROXY_NUM_LAYERS - PROXY_DENSE_LAYERS)


def hf_proxy_text_config_kwargs(*, magi: bool = False) -> dict[str, Any]:
    """``magi=True``: the msa_v1 kernel shapes (64 Q heads / 4 KV heads / top-16), otherwise the small proxy.

    The magi proxy is only sparse past 16 KV blocks (2048 tokens), so its tests use S >= 4096.
    """
    kwargs = dict(
        vocab_size=8192,
        hidden_size=1024,
        num_hidden_layers=PROXY_NUM_LAYERS,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=128,
        # MLP / MoE
        dense_intermediate_size=4096,
        intermediate_size=512,
        shared_intermediate_size=512,
        n_shared_experts=1,
        num_local_experts=16,
        num_experts_per_tok=2,
        moe_layer_freq=list(_LAYER_IS_MOE),
        scoring_func="sigmoid",
        use_routing_bias=True,
        routed_scaling_factor=2.0,
        hidden_act="swigluoai",
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        # norms / rope
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
        # MSA
        sparse_attention_config=dict(
            use_sparse_attention=True,
            sparse_index_dim=128,
            sparse_num_index_heads=4,
            sparse_topk_blocks=4,
            sparse_block_size=128,
            sparse_score_type="max",
            sparse_init_block=0,
            sparse_local_block=1,
            sparse_disable_index_value=list(_LAYER_IS_MOE),
            sparse_attention_freq=list(_LAYER_IS_MOE),
        ),
        # misc from the real config.json, kept so field coverage matches
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    if magi:
        kwargs["num_attention_heads"] = 64
        kwargs["sparse_attention_config"]["sparse_topk_blocks"] = 16
    return kwargs


def hf_proxy_text_config():
    """Build the HF config object (requires transformers >= 5.16 with ``minimax_m3_vl``)."""
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig

    return MiniMaxM3VLTextConfig(**hf_proxy_text_config_kwargs())


# Target Megatron Lite fields (P2 will turn this into the real config_mapping).
MLITE_PROXY_FIELDS: dict[str, Any] = dict(
    num_layers=PROXY_NUM_LAYERS,
    hidden_size=1024,
    num_attention_heads=16,
    num_query_groups=4,
    kv_channels=128,
    ffn_hidden_size=4096,  # dense layers
    moe_ffn_hidden_size=512,
    moe_shared_expert_intermediate_size=512,
    num_moe_experts=16,
    moe_router_topk=2,
    moe_layer_freq=list(_LAYER_IS_MOE),
    moe_router_score_function="sigmoid",
    moe_router_enable_expert_bias=True,
    moe_router_topk_scaling_factor=2.0,
    moe_router_pre_softmax=False,  # HF normalises the top-k sigmoid weights after selection
    activation="swiglu_clamped",  # alpha=1.702, limit=7.0, (up + 1) * glu  -- see P0 findings
    normalization="RMSNorm",
    layernorm_zero_centered_gamma=True,  # Gemma (1 + w) form
    layernorm_epsilon=1e-6,
    qk_layernorm=True,
    rotary_percent=0.5,
    rotary_interleaved=False,
    rotary_base=5_000_000,
    untie_embeddings_and_output_weights=True,
    vocab_size=8192,
    # MSA
    sparse_attention_start_layer=PROXY_DENSE_LAYERS,
    msa_block_size=128,
    msa_topk_blocks=4,
    msa_index_heads=4,
    msa_index_head_dim=128,
    msa_local_blocks=1,
)
