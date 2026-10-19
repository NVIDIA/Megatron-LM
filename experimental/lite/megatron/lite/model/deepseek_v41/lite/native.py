# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Release configuration lowered to existing Core components, without Core patches."""
import torch
import torch.nn.functional as F

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import _validate_dsa_kernel_backend_dependencies
from megatron.lite.model.deepseek_v41.topology import validate_build


def reject_native_recompute(granularity, method):
    if granularity is not None or method is not None:
        raise ValueError(
            "DS4.1 layer recomputation must use KVSourceState.checkpoint; "
            "Core recompute_granularity/recompute_method bypass shared-source tensor inputs"
        )


def require_native_hadamard():
    """CUDA indexer keys/queries need the real fast_hadamard_transform.

    Core's rotate_activation asserts only that some transform is installed;
    a CPU test stand-in must never reach a GPU run.
    """
    from megatron.core.transformer.experimental_attention_variant import dsa

    module = getattr(dsa.hadamard_transform, "__module__", None) or ""
    if not module.startswith("fast_hadamard_transform"):
        raise ValueError(
            "DS4.1 CUDA index selection requires the native fast_hadamard_transform; "
            f"found {dsa.hadamard_transform!r}"
        )


def core_config(
    release,
    parallel,
    *,
    initialize=True,
    recompute_granularity=None,
    recompute_method=None,
    attention_backend="none",
):
    """Validate unsupported topology before groups or large tensors are created.

    ``attention_backend`` is Core's dsa_kernel_backend: "none" runs the unfused
    sparse attention, "cudnn"/"tilelang" FlashMLA. It is set at construction so
    Core validates the backend and its dependencies.
    """
    reject_native_recompute(recompute_granularity, recompute_method)
    _validate_dsa_kernel_backend_dependencies(attention_backend)
    validate_build(release, context_parallel_size=parallel.cp)
    if parallel.tp != 1 or parallel.etp not in (None, 1):
        raise ValueError("DS4.1 native Hybrid attention requires TP=ETP=1")
    release.cp_size = parallel.cp
    config = MLATransformerConfig(
        num_layers=release.num_hidden_layers,
        hidden_size=release.hidden_size,
        num_attention_heads=release.num_attention_heads,
        num_query_groups=1,
        ffn_hidden_size=release.moe_intermediate_size,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        cp_partition_mode="contiguous",
        sequence_packing_scheduler="dp_balanced",
        pipeline_model_parallel_size=parallel.pp,
        pipeline_dtype=torch.bfloat16 if parallel.pp > 1 else None,
        pipeline_model_parallel_layout=getattr(parallel, "pp_layout", None),
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
        # DS4.1 composes Core component specs directly. The stock DSv4
        # selector admits only 0/4/128 and is not the DS4.1 ratio grammar.
        # Keep literal ratios; do not select or remap to that different model.
        experimental_attention_variant=None,
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
        dsa_kernel_backend=attention_backend,
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

    # DS4.1 composes components directly, not the stock DSv4 model selector.
    # Real CP belongs to release.cp_size / runtime groups and PackedSeqParams;
    # Core's config declares CP1. See the source-guard/MRO/rope contracts.
    config.ds41_candidate_blocks = release.candidate_topk_blocks
    config.ds41_candidate_block_size = release.candidate_block_size
    if torch.cuda.is_available() and set(config.csa_compress_ratios) & {1, 2}:
        require_native_hadamard()
    return config


def validate_runtime_cp(release, group, params=None):
    if group is None or group.size() != release.cp_size:
        raise ValueError("DS4.1 configured CP size differs from runtime process group")
    if params is not None:
        if params.qkv_format != "thd" or params.cp_partition_mode != "contiguous":
            raise ValueError("DS4.1 runtime CP requires contiguous packed THD")
        if params.cp_group is not group:
            raise ValueError("DS4.1 PackedSeqParams CP group differs from model group")


def core_groups(state, release):
    """Use the caller's existing groups; never create duplicate collectives here."""
    validate_runtime_cp(release, state.cp_group)
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


def attach_ratio_two_sources(attention, *, compressor, indexer):
    """Attach DS4.1 sources that Core dev 7b150acf does not construct at ratio 2.

    Core CompressedSparseAttention.forward/_forward_thd/_forward_thd_cp
    consume these module attributes for ratio > 1. Its constructor only builds
    an indexer at ratio 4. Keep the literal DS4.1 ratio and native forward;
    do not select the unrelated DSv4 ratio-4 pooling or duplicate CP code.
    Ratio 1 is deliberately excluded: Core's forward cannot consume it.
    """
    if attention.compress_ratio != 2:
        raise ValueError("DS4.1 source attachment currently supports only literal ratio 2")
    if compressor is None or indexer is None:
        raise ValueError("DS4.1 ratio 2 requires both compressor and indexer sources")
    if compressor.compress_ratio != 2 or indexer.compress_ratio != 2:
        raise ValueError("DS4.1 attached sources must preserve literal ratio 2")
    if attention.config.csa_dense_mode:
        raise ValueError("DS4.1 learned index selection cannot use csa_dense_mode")
    freeze_indexer_sources(indexer, loss_coefficient=attention.config.dsa_indexer_loss_coeff)
    attention.compressor, attention.indexer = compressor, indexer
    return attention


def freeze_indexer_sources(indexer, *, loss_coefficient):
    """Register intentionally unused indexer gradients in standard post-training.

    Discrete index selection contributes no differentiable LP loss and this
    recipe has no auxiliary indexer loss. Only the indexer's registered source
    parameters are frozen; a main-KV owner must be a non-owning reference.
    This is an explicit post-training policy, not an inferred official trainer.
    """
    if loss_coefficient != 0:
        raise ValueError("DS4.1 standard post-training does not implement indexer auxiliary loss")
    names = tuple(name for name, _ in indexer.named_parameters())
    indexer.requires_grad_(False)
    indexer.ds41_frozen_parameters = names
    return names
