# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Release configuration lowered to existing Core components, without Core patches."""
import torch
import torch.nn.functional as F

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.lite.model.deepseek_v41.topology import validate_build


def core_config(release, parallel, *, initialize=True):
    """Validate unsupported topology before groups or large tensors are created."""
    validate_build(release, context_parallel_size=parallel.cp)
    if parallel.tp != 1 or parallel.etp not in (None, 1):
        raise ValueError("DS4.1 native Hybrid attention requires TP=ETP=1")
    return MLATransformerConfig(
        num_layers=release.num_hidden_layers,
        hidden_size=release.hidden_size,
        num_attention_heads=release.num_attention_heads,
        num_query_groups=1,
        ffn_hidden_size=release.moe_intermediate_size,
        tensor_model_parallel_size=1,
        context_parallel_size=parallel.cp,
        pipeline_model_parallel_size=parallel.pp,
        expert_model_parallel_size=parallel.ep,
        expert_tensor_parallel_size=1,
        params_dtype=torch.bfloat16,
        bf16=True,
        perform_initialization=initialize,
        normalization="RMSNorm",
        layernorm_epsilon=release.rms_norm_eps,
        attention_latent_norm_epsilon=release.rms_norm_eps,
        qk_layernorm=True,
        multi_latent_attention=True,
        q_lora_rank=release.q_lora_rank,
        kv_lora_rank=release.head_dim,
        qk_head_dim=release.head_dim - release.qk_rope_head_dim,
        qk_pos_emb_head_dim=release.qk_rope_head_dim,
        v_head_dim=release.head_dim,
        o_lora_rank=release.o_lora_rank,
        o_groups=release.o_groups,
        experimental_attention_variant="dsv4_hybrid",
        csa_compress_ratios=list(release.compress_ratios[: release.num_hidden_layers])
        or ([0, 0] + [2] * 18 + [1] * 20)[: release.num_hidden_layers],
        csa_window_size=release.sliding_window,
        csa_compress_rotary_base=release.compress_rope_theta,
        rotary_base=release.rope_theta,
        rotary_scaling_factor=release.rotary_scaling_factor,
        original_max_position_embeddings=release.original_max_position_embeddings,
        beta_fast=release.beta_fast,
        beta_slow=release.beta_slow,
        apply_rope_fusion=False,
        dsa_kernel_backend="none",
        dsa_indexer_n_heads=release.index_n_heads,
        dsa_indexer_head_dim=release.index_head_dim,
        dsa_indexer_topk=release.index_topk,
        dsa_indexer_loss_coeff=0.0,
        dsa_indexer_precision="bf16",
        dsa_cp_balance_indexer=False,
        enable_hyper_connections=True,
        num_residual_streams=release.hc_mult,
        mhc_sinkhorn_iterations=release.hc_sinkhorn_iters,
        use_fused_mhc=False,
        num_moe_experts=release.n_routed_experts,
        moe_router_topk=release.num_experts_per_tok,
        moe_ffn_hidden_size=release.moe_intermediate_size,
        moe_shared_expert_intermediate_size=release.moe_intermediate_size
        * release.n_shared_experts,
        moe_router_score_function=release.scoring_func,
        moe_router_pre_softmax=False,
        moe_router_topk_scaling_factor=release.routed_scaling_factor,
        moe_router_enable_expert_bias=True,
        moe_router_dtype="fp32",
        moe_router_fusion=False,
        moe_router_load_balancing_type="none",
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
        activation_func=F.silu,
        gated_linear_unit=True,
        activation_func_clamp_value=release.swiglu_limit,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
    )


def core_groups(state):
    """Use the caller's existing groups; never create duplicate collectives here."""
    return ProcessGroupCollection(
        tp=state.tp_group,
        cp=state.cp_group,
        pp=state.pp_group,
        ep=state.ep_group,
        expt_tp=state.etp_group,
        tp_ep=state.tp_ep_group,
        dp=state.dp_group,
        dp_cp=state.dp_cp_group,
        expt_dp=state.ep_dp_group,
    )
