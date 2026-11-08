# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared model FLOP formulas used by training and config-defined architectures.

Layer helpers return forward FLOPs including multiply/add as two operations, except
``dsv4_attention_flops``, which retains its existing forward/backward expansion.
These preserve training's established approximations; they are not kernel profiles.
"""


def mlp_layer_flops(total_tokens, hidden_size, expansion=4.0, swiglu=False):
    """Calculate FLOPs for an MLP layer."""
    scale_factor = 3.0 / 2.0 if swiglu else 1.0
    return 4 * expansion * scale_factor * total_tokens * hidden_size**2


def moe_layer_flops(
    total_tokens,
    hidden_size,
    moe_ffn_hidden_size,
    shared_expert_ffn_hidden_size,
    num_experts_routed_to,
    moe_latent_size=None,
    swiglu=False,
):
    """Calculate FLOPs for an MoE layer."""
    scale_factor = 3.0 / 2.0 if swiglu else 1.0
    if moe_latent_size is None:
        routed_flops = (
            4
            * total_tokens
            * hidden_size
            * moe_ffn_hidden_size
            * num_experts_routed_to
            * scale_factor
        )
    else:
        # Routed experts run on moe_latent_size.
        routed_flops = (
            4
            * total_tokens
            * moe_latent_size
            * moe_ffn_hidden_size
            * num_experts_routed_to
            * scale_factor
        )
        # Up proj and down proj.
        routed_flops += 4 * total_tokens * hidden_size * moe_latent_size
    shared_flops = 4 * total_tokens * hidden_size * shared_expert_ffn_hidden_size * scale_factor
    return routed_flops + shared_flops


def attn_layer_flops(
    total_tokens,
    seqlen_squared_sum,
    hidden_size,
    num_heads,
    gqa=True,
    gqa_groups=8,
    kv_channels=None,
    attention_output_gate=False,
):
    """Calculate FLOPs for an attention layer.

    Two parts:
      - Token-linear (QKV + output projections): scales with ``total_tokens``
        (= ``sum_i(L_i)``). Padding tokens are excluded by passing the real
        (unpadded) token count.
      - Core attention (``QK^T`` and ``softmax(QK^T) V``): scales with
        ``seqlen_squared_sum`` (= ``sum_i(L_i ** 2)``).
    """
    p = (kv_channels * num_heads / hidden_size) if kv_channels else 1
    g = gqa_groups if gqa else num_heads
    # QKV + output projections, forward only with multiply/add counted as two operations.
    # 2 * sum(L^2) * h * p: core attention (causal mask -> /2 cancels with FMA *2).
    return (
        4 * total_tokens * hidden_size * p * (hidden_size + hidden_size * (g / num_heads))
        + 2 * seqlen_squared_sum * hidden_size * p
        + (
            2
            * total_tokens
            * hidden_size
            * (kv_channels * num_heads if kv_channels else hidden_size)
            if attention_output_gate
            else 0
        )
    )


def mamba_layer_flops(
    total_tokens, hidden_size, state_dim=16, head_dim=64, num_groups=1, num_heads=128
):
    """Calculate FLOPs for a Mamba layer."""
    # Note (rwaleffe): flops estimate for scan should be updated based on new SSD kernels,
    # but small percent of overall layer flops
    d_in = 2 * hidden_size
    if num_heads:
        nheads = num_heads
    else:
        nheads = d_in // head_dim
    return (
        (
            2 * total_tokens * hidden_size * (2 * d_in + 2 * num_groups * state_dim + nheads)
        )  # in_proj
        + (7 * total_tokens * d_in * state_dim)  # scan
        + (2 * total_tokens * d_in * hidden_size)  # out_proj
    )


def gated_delta_product_layer_flops(
    total_tokens,
    hidden_size,
    num_householder,
    state_dim=128,
    head_dim=64,
    num_groups=8,
    num_heads=None,
    conv_kernel_dim=4,
):
    """Calculate FLOPs for a Gated Delta Product (GDP) layer."""
    if num_heads is None:
        d_inner = 2 * hidden_size
        num_heads = d_inner // head_dim
    else:
        d_inner = num_heads * head_dim
    in_proj_dim = (
        d_inner * (1 + num_householder)
        + num_groups * state_dim * (1 + num_householder)
        + num_heads * (1 + num_householder)
    )
    conv_dim = d_inner * num_householder + num_groups * state_dim * (1 + num_householder)
    non_core_flops = (
        2
        * total_tokens
        * (hidden_size * in_proj_dim + conv_kernel_dim * conv_dim + d_inner * hidden_size)
    )
    # Best-case recurrent GDP core estimate. The FLA chunk kernel may do additional
    # score/solve/WY work, but this keeps the implementation-agnostic lower bound explicit.
    core_flops = (4 * num_householder + 3) * total_tokens * d_inner * state_dim
    return non_core_flops + core_flops


def gdn_layer_flops(
    total_tokens,
    hidden_size,
    qk_head_dim=128,
    v_head_dim=128,
    num_qk_heads=16,
    num_v_heads=32,
    conv_kernel_dim=4,
    use_gdn2=False,
):
    """Calculate FLOPs for a Gated Delta Net (GDN) layer."""
    qk_dim = qk_head_dim * num_qk_heads
    v_dim = v_head_dim * num_v_heads
    if use_gdn2:
        # GDN2 in_proj: hidden_size -> (4*qk_dim + 3*v_dim) for q, k, v, z, f, b, w
        in_proj_dim = 4 * qk_dim + 3 * v_dim
    else:
        # GDN in_proj: hidden_size -> (2*qk_dim + 2*v_dim + 2*num_v_heads)
        in_proj_dim = 2 * qk_dim + 2 * v_dim + 2 * num_v_heads
    return (
        2
        * total_tokens
        * (
            # in_proj
            hidden_size * in_proj_dim
            # conv1d
            + conv_kernel_dim * (2 * qk_dim + v_dim)
            # gated delta rule: KK^T, VK^T, S(a(I-bKK^T)), and SQ
            + num_v_heads * (v_head_dim**2) * 4
            # out_proj: v_dim -> hidden_size
            + hidden_size * v_dim
        )
    )


def mla_layer_flops(
    total_tokens,
    seqlen_squared_sum,
    hidden_size,
    num_attention_heads,
    q_lora_rank,
    kv_lora_rank,
    qk_head_dim,
    qk_pos_emb_head_dim,
    v_head_dim,
):
    """Return forward MLA FLOPs using training's projection and causal-pair convention."""
    q_head_dim = qk_head_dim + qk_pos_emb_head_dim
    q_term = (
        hidden_size * num_attention_heads * q_head_dim
        if q_lora_rank is None
        else q_lora_rank * (hidden_size + num_attention_heads * q_head_dim + 1)
    )
    token_linear = 2 * (
        q_term
        + kv_lora_rank * (hidden_size + num_attention_heads * (qk_head_dim + v_head_dim) + 1)
        + hidden_size * qk_pos_emb_head_dim
        + num_attention_heads * v_head_dim * hidden_size
    )
    core = num_attention_heads * (q_head_dim + v_head_dim)
    return total_tokens * token_linear + seqlen_squared_sum * core


def kda_layer_flops(
    total_tokens,
    hidden_size,
    qk_head_dim,
    v_head_dim,
    num_qk_heads,
    num_v_heads,
    conv_kernel_dim=4,
    two_stage_gates=False,
):
    """Estimate KDA projections/conv and recurrent work for either gate architecture.

    KDA has no existing pattern estimate. Like GDN, its recurrent estimate excludes
    additional chunk-kernel work, elementwise gates, and normalization.
    """
    qk_dim, v_dim = qk_head_dim * num_qk_heads, v_head_dim * num_v_heads
    if two_stage_gates:
        in_proj_dim = 2 * qk_dim + v_dim
        gate_projections = (
            hidden_size * (qk_head_dim + v_head_dim) + qk_head_dim * qk_dim + v_head_dim * v_dim
        )
    else:
        in_proj_dim, gate_projections = 3 * qk_dim + 2 * v_dim, 0
    return (
        2
        * total_tokens
        * (
            hidden_size * (in_proj_dim + num_qk_heads)  # Includes the separate beta projection.
            + gate_projections
            + conv_kernel_dim * (2 * qk_dim + v_dim)
            + 4 * num_v_heads * qk_head_dim * v_head_dim
            + hidden_size * v_dim
        )
    )


def dsv4_attention_flops(transformer_cfg, layer_counts, total_tokens, seqlen_squared_sum):
    """Estimate DSv4 W/C/H attention FLOPs, including projections and compressors.

    Uses the model-FLOPs convention: multiply/add counts as two
    operations and forward plus backward as three passes. Window and capped
    Top-K work are token-linear; HCA and indexer scoring use the real sequence
    length squared sum. Packed Top-K saturation uses the token-weighted mean
    length, since these batch statistics do not retain individual document lengths.
    """
    from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols

    if total_tokens == 0:
        return 0
    window_layers = layer_counts[Symbols.WINDOW]
    csa_layers = layer_counts[Symbols.CSA]
    hca_layers = layer_counts[Symbols.HCA]
    attention_layers = window_layers + csa_layers + hca_layers
    hidden = transformer_cfg.hidden_size
    heads = transformer_cfg.num_attention_heads
    head_dim = transformer_cfg.v_head_dim
    q_rank = transformer_cfg.q_lora_rank
    output_rank = transformer_cfg.output_projection_lora_rank
    output_groups = transformer_cfg.output_projection_groups
    window = transformer_cfg.csa_window_size
    sequence_length = seqlen_squared_sum / total_tokens

    # Read v_head_dim directly: DSv4 derives qk_head_dim on the layer config,
    # so the stack config may still carry the ordinary MLA default.
    query_projection = q_rank * (hidden + heads * head_dim + 1)
    kv_projection = hidden * head_dim + head_dim
    output_projection = heads * head_dim * output_rank + output_groups * output_rank * hidden
    token_term = attention_layers * (query_projection + kv_projection + output_projection)
    token_term += attention_layers * heads * window * head_dim * 2

    # Ratio-4 compression overlaps two windows; ratio-128 uses one.
    token_term += csa_layers * hidden * (2 * head_dim) * 2
    token_term += hca_layers * hidden * head_dim * 2
    core_term = hca_layers * heads * head_dim / 128

    if csa_layers:
        if transformer_cfg.csa_dense_mode:
            # Dense CSA attends to all compressed keys and has no indexer.
            core_term += csa_layers * heads * head_dim / 4
        else:
            indexer_heads = transformer_cfg.dsa_indexer_n_heads
            indexer_dim = transformer_cfg.dsa_indexer_head_dim
            topk = min(transformer_cfg.dsa_indexer_topk, sequence_length // 4)
            average_compressed_keys = topk * (1 - topk * 4 / (2 * sequence_length))
            token_term += csa_layers * heads * average_compressed_keys * head_dim * 2
            token_term += csa_layers * (
                hidden * (2 * indexer_dim) * 2
                + q_rank * indexer_heads * indexer_dim
                + hidden * indexer_heads
            )
            core_term += csa_layers * indexer_heads * indexer_dim / 4

    return 6 * (token_term * total_tokens + core_term * seqlen_squared_sum)
