def mla_layer_flops(
    batch_size,
    seq_len,
    hidden_size,
    num_attention_heads,
    qk_head_dim,
    v_head_dim,
    qk_pos_emb_head_dim,
    kv_lora_rank,
    q_lora_rank=None,
):
    """Calculate FLOPs for MLA attention layer."""
    if q_lora_rank is None:
        mla_q_proj = hidden_size * num_attention_heads * (qk_head_dim + qk_pos_emb_head_dim)
    else:
        mla_q_a_proj = hidden_size * q_lora_rank
        mla_q_b_proj = q_lora_rank * (
            num_attention_heads * (qk_head_dim + qk_pos_emb_head_dim) + 1
        )  # + norm
        mla_q_proj = 2 * batch_size * seq_len * (mla_q_a_proj + mla_q_b_proj)

    mla_kv_a_proj = hidden_size * (kv_lora_rank + qk_pos_emb_head_dim)
    mla_kv_b_proj = kv_lora_rank * (
        num_attention_heads * (qk_head_dim + v_head_dim) + 1
    )  # + norm
    mla_kv_proj = 2 * batch_size * seq_len * (mla_kv_a_proj + mla_kv_b_proj)

    return (
        mla_q_proj
        + mla_kv_proj
        + 2
        * batch_size
        * (seq_len * seq_len / 2)
        * (qk_head_dim + qk_pos_emb_head_dim)
        * num_attention_heads  # qk score
        + 2 * batch_size * (seq_len * seq_len / 2) * v_head_dim * num_attention_heads  # v value
        + 2 * batch_size * seq_len * hidden_size * (num_attention_heads * v_head_dim)  # o proj
    )


def lighting_linear_layer_flops(
    batch_size,
    seq_len,
    hidden_size,
    num_attention_heads,
    gqa=True,
    num_query_groups=8,
    kv_channels=None,
):
    """Calculate FLOPs for an attention layer.
    flops = (2 * batch_size * seq_length * hidden_size * kv_channels * num_attention_heads # q proj
            + 2 * 2 * batch_size * seq_length * hidden_size * kv_channels * num_query_groups # kv proj
            + 2 * batch_size * seq_length * num_attention_heads * kv_channels**2 # kv + S
            + 2 * batch_size * seq_length * num_attention_heads * kv_channels**2 # Q与KV相乘
            + 2 * batch_size * seq_length * (hidden_size + 1) * num_attention_heads * kv_channels  # gated + norm
            + 2 * batch_size * seq_length * hidden_size * kv_channels * num_attention_heads) # o proj
    """
    kv_channels = kv_channels if kv_channels else hidden_size / num_attention_heads
    g = num_query_groups if gqa else num_attention_heads
    return (
        2
        * batch_size
        * seq_len
        * hidden_size
        * (
            3 * kv_channels * num_attention_heads  # q proj + gated + o proj
            + 2 * kv_channels * g  # kv proj
            + 2 * num_attention_heads * kv_channels**2 / hidden_size  # qk score + v value
        )
    )

def KDA_linear_layer_flops(
    batch_size,
    seq_len,
    hidden_size,
    num_attention_heads,
    gqa=True,
    num_query_groups=8,
    kv_channels=None,
    no_kda_lora=True,
):
    """Calculate FLOPs for an attention layer.
    flops = (2 * batch_size * seq_length * hidden_size * kv_channels * num_attention_heads # q proj
            + 2 * 2 * batch_size * seq_length * hidden_size * kv_channels * num_query_groups # kv proj
            + 2 * batch_size * seq_length * num_attention_heads * kv_channels**2 # kv + S
            + 5 * batch_size * seq_length * num_attention_heads * kv_channels**2 # (I-kk)*Diag*S 注意
            + 2 * batch_size * seq_length * num_attention_heads * kv_channels**2 # Q与KV相乘
            + 2 * 2 * batch_size * seq_length * (hidden_size * num_attention_heads * kv_channels) # alpha + gated, without lora
            # or + 2 * 2 * batch_size * seq_length * (hidden_size * kv_channels + kv_channels * num_attention_heads * kv_channels) # alpha + gated, with lora
            + 2 * batch_size * seq_length * hidden_size * kv_channels * num_attention_heads) # o proj
    """
    kv_channels = kv_channels if kv_channels else hidden_size / num_attention_heads
    g = num_query_groups if gqa else num_attention_heads
    alpha_gated = num_attention_heads * kv_channels if no_kda_lora else (kv_channels + num_attention_heads * kv_channels**2 / hidden_size)  # alpha + gate
    return (
        2
        * batch_size
        * seq_len
        * hidden_size
        * (
            2 * kv_channels * num_attention_heads  # q proj + o proj
            + 2 * kv_channels * g  # kv proj
            + 2 * num_attention_heads * kv_channels**2 / hidden_size  # qk score + v value
            + 2.5 * num_attention_heads * kv_channels**2 / hidden_size  # (I-kk)*Diag*S
            + 2 * alpha_gated  # alpha + gated
        )
    )

def moe_layer_flops(
    batch_size,
    seq_len,
    hidden_size,
    moe_ffn_hidden_size,
    moe_router_topk,
    moe_shared_expert_intermediate_size,
    swiglu=True,
):
    """Calculate FLOPs for an MoE layer."""
    scale_factor = 3.0 / 2.0 if swiglu else 1.0
    return (
        4
        * scale_factor
        * batch_size
        * seq_len
        * hidden_size
        * (moe_ffn_hidden_size * moe_router_topk + moe_shared_expert_intermediate_size)
    )


def mlp_layer_flops(batch_size, seq_len, hidden_size, expansion=4.0, swiglu=False):
    """Calculate FLOPs for an MLP layer."""
    scale_factor = 3.0 / 2.0 if swiglu else 1.0
    return 4 * expansion * scale_factor * batch_size * seq_len * hidden_size**2


def moe_expert_flops(batch_size, seq_len, hidden_size, moe_ffn_hidden_size, moe_router_topk, swiglu=True):
    """Calculate FLOPs for routed experts only (TEGroupedMLP), excluding shared experts."""
    scale_factor = 3.0 / 2.0 if swiglu else 1.0
    return 4 * scale_factor * batch_size * seq_len * hidden_size * moe_ffn_hidden_size * moe_router_topk


def shared_expert_flops(batch_size, seq_len, hidden_size, moe_shared_expert_intermediate_size, swiglu=True):
    """Calculate FLOPs for shared expert MLP."""
    scale_factor = 3.0 / 2.0 if swiglu else 1.0
    return 4 * scale_factor * batch_size * seq_len * hidden_size * moe_shared_expert_intermediate_size


def router_flops(batch_size, seq_len, hidden_size, num_experts):
    """Calculate FLOPs for TopK router (linear projection to num_experts)."""
    return 2 * batch_size * seq_len * hidden_size * num_experts


def loss_flops(batch_size, seq_len, vocab_size):
    """Calculate FLOPs for cross-entropy loss (output projection + softmax + loss).

    Includes:
    - Output projection: 2 * B * S * H * V (counted separately if in the module)
    - Softmax: ~5 * B * S * V (exp, sum, div, sub, exp)
    - Cross-entropy: ~3 * B * S * V (log, gather, negate)
    """
    return batch_size * seq_len * vocab_size * 8