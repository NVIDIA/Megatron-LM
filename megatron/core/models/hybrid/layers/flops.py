# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model FLOP estimates for built-in hybrid layers, independent of pipeline placement.

These count multiply/add as two operations and ordinary forward/backward as three
passes. Elementwise operations, communication, optimizer work, and recomputation
are excluded except for the small normalization terms retained by existing MLA/MTP
estimates. Recurrent kernels use recurrent-operation estimates, not chunk-kernel
instruction counts. Sparse/compressed attention uses the length-weighted mean
sum(L**2)/sum(L), since batch aggregates cannot recover individual document lengths.
"""

import math
from typing import Sequence

from megatron.core.models.hybrid.hybrid_layer_allocation import (
    HybridLayerEntry,
    MTPSplit,
    PipelineSplit,
)
from megatron.core.models.hybrid.layers.dsa_flops import (
    _dsa_indexer_flops,
    _dsa_kpool_attention_scale,
    _dsa_sparse_core_scale,
)
from megatron.core.ssm.gdn_layer_config import GDNLayerConfig
from megatron.core.ssm.kda_layer_config import KDALayerConfig
from megatron.core.ssm.mamba_layer_config import MambaLayerConfig
from megatron.core.ssm.mlp_layer_config import MLPLayerConfig
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.experimental_attention_variant.dsa_layer_config import DSALayerConfig
from megatron.core.transformer.experimental_attention_variant.dsv4_layer_config import (
    CSALayerConfig,
)
from megatron.core.transformer.mla_layer_config import MLALayerConfig
from megatron.core.transformer.moe.moe_layer_config import MoELayerConfig
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


def _attention_layer_flops(
    config: AttentionLayerConfig, total_tokens: float, seqlen_squared_sum: float
) -> float:
    projection_ratio = (
        config.kv_channels * config.num_attention_heads / config.hidden_size
        if config.kv_channels
        else 1
    )
    # QKV/output projections are token-linear; causal core attention scales with sum(L^2).
    flops = (
        4
        * total_tokens
        * config.hidden_size
        * projection_ratio
        * (
            config.hidden_size
            + config.hidden_size * (config.num_query_groups / config.num_attention_heads)
        )
        + 2 * seqlen_squared_sum * config.hidden_size * projection_ratio
    )
    if config.attention_output_gate:
        query_projection_size = config.kv_channels * config.num_attention_heads
        flops += 2 * total_tokens * config.hidden_size * query_projection_size
    return flops


def _mla_projection_flops(config: MLALayerConfig | DSALayerConfig, total_tokens: float) -> float:
    q_head_dim = config.qk_head_dim + config.qk_pos_emb_head_dim
    if config.q_lora_rank is None:
        q_term = config.hidden_size * config.num_attention_heads * q_head_dim
    else:
        q_term = config.q_lora_rank * (
            config.hidden_size + config.num_attention_heads * q_head_dim + 1
        )
    projection_term = (
        q_term
        + config.kv_lora_rank
        * (
            config.hidden_size
            + config.num_attention_heads * (config.qk_head_dim + config.v_head_dim)
            + 1
        )
        + config.hidden_size * config.qk_pos_emb_head_dim
        + config.num_attention_heads * config.v_head_dim * config.hidden_size
    )
    return 2 * total_tokens * projection_term


def _mla_layer_flops(
    config: MLALayerConfig, total_tokens: float, seqlen_squared_sum: float
) -> float:
    q_head_dim = config.qk_head_dim + config.qk_pos_emb_head_dim
    core_term = config.num_attention_heads * (q_head_dim + config.v_head_dim)
    return _mla_projection_flops(config, total_tokens) + seqlen_squared_sum * core_term


def _dsa_layer_flops(
    config: DSALayerConfig,
    total_tokens: float,
    seqlen_squared_sum: float,
    layer_number: int,
    hybrid_stack_spec: ModuleSpec,
) -> tuple[float, float]:
    from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
        AbsorbedMLASelfAttention,
    )
    from megatron.core.transformer.experimental_attention_variant.dsa import is_dsa_skip_topk_layer

    dsa_layer_spec = getattr(hybrid_stack_spec.submodules, 'dsa_layer', None)
    attention_spec = getattr(getattr(dsa_layer_spec, 'submodules', None), 'self_attention', None)
    if attention_spec is not None and attention_spec.module is AbsorbedMLASelfAttention:
        core_dim = 2 * config.kv_lora_rank + config.qk_pos_emb_head_dim
    else:
        core_dim = config.qk_head_dim + config.qk_pos_emb_head_dim + config.v_head_dim
    sparse_scale = (
        _dsa_kpool_attention_scale(
            total_tokens,
            seqlen_squared_sum,
            config.dsa_indexer_topk,
            config.dsa_indexer_kpool,
            config.dsa_indexer_kpool_always_select_tail,
        )
        if config.dsa_indexer_kpool > 1
        else _dsa_sparse_core_scale(total_tokens, seqlen_squared_sum, config.dsa_indexer_topk)
    )
    flops = (
        _mla_projection_flops(config, total_tokens)
        + seqlen_squared_sum * sparse_scale * config.num_attention_heads * core_dim
    )

    executes_indexer = not is_dsa_skip_topk_layer(
        layer_number, config.dsa_indexer_skip_topk_offset, config.dsa_indexer_topk_freq
    )
    token_linear, core = _dsa_indexer_flops(
        total_tokens=total_tokens,
        seqlen_squared_sum=seqlen_squared_sum,
        hidden_size=config.hidden_size,
        q_lora_rank=config.q_lora_rank,
        n_heads=config.dsa_indexer_n_heads,
        head_dim=config.dsa_indexer_head_dim,
        num_indexer_layers=int(executes_indexer),
        dsa_indexer_loss_enabled=(config.dsa_indexer_loss_coeff or 0) > 0,
        dsa_indexer_use_sparse_loss=config.dsa_indexer_use_sparse_loss,
        dsa_indexer_topk=config.dsa_indexer_topk,
        dsa_indexer_kpool=config.dsa_indexer_kpool,
    )
    # Indexer inputs are detached; its loss-dependent backward factors are already included.
    return flops, total_tokens * token_linear + seqlen_squared_sum * core


def _csa_layer_flops(
    config: CSALayerConfig, total_tokens: float, seqlen_squared_sum: float
) -> tuple[float, float]:
    """Return forward attention FLOPs and separately expanded indexer training FLOPs.

    W/C/H use the DSv4 grouped output projection and shared KV representation.
    As in the pattern estimator, window work is token-linear and compressed
    causal pairs use a continuous approximation. Compressor pooling, top-k,
    teacher recomputation, and KL arithmetic are excluded.
    """
    ratio = config.compress_ratio
    if ratio not in (0, 4, 128):
        raise ValueError(f"Unsupported CSA compression ratio: {ratio}")
    if total_tokens == 0:
        return 0.0, 0.0
    hidden, heads, dim = config.hidden_size, config.num_attention_heads, config.v_head_dim
    if config.q_lora_rank is None:
        raise ValueError("CSA FLOPs require q_lora_rank, as does DSv4 attention.")
    token_term = (
        config.q_lora_rank * (hidden + heads * dim + 1)
        + hidden * dim
        + dim
        + heads * dim * config.output_projection_lora_rank
        + config.output_projection_groups * config.output_projection_lora_rank * hidden
        + 2 * heads * config.csa_window_size * dim
    )
    if ratio == 0:
        return 2 * total_tokens * token_term, 0.0

    mean_length = seqlen_squared_sum / total_tokens
    if mean_length < ratio:
        return 2 * total_tokens * token_term, 0.0
    overlap = 2 if ratio == 4 else 1
    token_term += 2 * hidden * overlap * dim
    compressed_pairs = seqlen_squared_sum / (2 * ratio)
    if ratio == 128 or config.csa_dense_mode:
        return 2 * total_tokens * token_term + 4 * compressed_pairs * heads * dim, 0.0

    sparse_scale = _dsa_sparse_core_scale(
        total_tokens / ratio, seqlen_squared_sum / ratio**2, config.dsa_indexer_topk
    )
    forward = 2 * total_tokens * token_term + 4 * compressed_pairs * sparse_scale * heads * dim
    index_heads, index_dim = config.dsa_indexer_n_heads, config.dsa_indexer_head_dim
    index_projection = (
        2
        * total_tokens
        * (
            2 * hidden * overlap * index_dim
            + config.q_lora_rank * index_heads * index_dim
            + hidden * index_heads
        )
    )
    index_scores = 2 * compressed_pairs * index_heads * index_dim
    if (config.dsa_indexer_loss_coeff or 0) > 0:
        # Projection inputs are detached; score gradients can be restricted to top-k.
        # Unlike the legacy DSv4 pattern approximation, these paths do not all pay x3.
        index_projection *= 2
        index_scores *= 1 + 2 * (sparse_scale if config.dsa_indexer_use_sparse_loss else 1)
    return forward, index_projection + index_scores


def _gated_delta_product_layer_flops(config: MambaLayerConfig, total_tokens: float) -> float:
    num_heads = config.mamba_num_heads
    if num_heads is None:
        d_inner = 2 * config.hidden_size
        num_heads = d_inner // config.mamba_head_dim
    else:
        d_inner = num_heads * config.mamba_head_dim
    num_householder = config.gdp_num_householder
    in_proj_dim = (
        d_inner * (1 + num_householder)
        + config.mamba_num_groups * config.mamba_state_dim * (1 + num_householder)
        + num_heads * (1 + num_householder)
    )
    conv_dim = d_inner * num_householder + config.mamba_num_groups * config.mamba_state_dim * (
        1 + num_householder
    )
    conv_kernel_dim = 4
    non_core_flops = (
        2
        * total_tokens
        * (
            config.hidden_size * in_proj_dim
            + conv_kernel_dim * conv_dim
            + d_inner * config.hidden_size
        )
    )
    # Recurrent lower bound; chunk kernels may perform additional score/solve/WY work.
    core_flops = (4 * num_householder + 3) * total_tokens * d_inner * config.mamba_state_dim
    return non_core_flops + core_flops


def _mamba_layer_flops(
    config: MambaLayerConfig, total_tokens: float, hybrid_stack_spec: ModuleSpec
) -> float:
    from megatron.core.ssm.gated_delta_product import GatedDeltaProductMixer

    mamba_layer_spec = getattr(hybrid_stack_spec.submodules, 'mamba_layer', None)
    mixer_spec = getattr(getattr(mamba_layer_spec, 'submodules', None), 'mixer', None)
    if mixer_spec is not None and mixer_spec.module is GatedDeltaProductMixer:
        return _gated_delta_product_layer_flops(config, total_tokens)

    d_inner = (
        config.mamba_num_heads * config.mamba_head_dim
        if config.mamba_num_heads is not None
        else 2 * config.hidden_size
    )
    num_heads = d_inner // config.mamba_head_dim
    # Retain the scan estimate used by the args-based estimator; SSD kernels may differ.
    return (
        2
        * total_tokens
        * config.hidden_size
        * (2 * d_inner + 2 * config.mamba_num_groups * config.mamba_state_dim + num_heads)
        + 7 * total_tokens * d_inner * config.mamba_state_dim
        + 2 * total_tokens * d_inner * config.hidden_size
    )


def _gdn_layer_flops(config: GDNLayerConfig, total_tokens: float) -> float:
    qk_head_dim = config.linear_key_head_dim or 128
    v_head_dim = config.linear_value_head_dim or 128
    num_qk_heads = config.linear_num_key_heads or 16
    num_v_heads = config.linear_num_value_heads or 32
    conv_kernel_dim = config.linear_conv_kernel_dim or 4
    qk_dim = qk_head_dim * num_qk_heads
    v_dim = v_head_dim * num_v_heads
    if config.experimental_attention_variant == 'gdn2':
        in_proj_dim = 4 * qk_dim + 3 * v_dim
    else:
        in_proj_dim = 2 * qk_dim + 2 * v_dim + 2 * num_v_heads
    return (
        2
        * total_tokens
        * (
            config.hidden_size * in_proj_dim
            + conv_kernel_dim * (2 * qk_dim + v_dim)
            + num_v_heads * qk_head_dim * v_head_dim * 4
            + config.hidden_size * v_dim
        )
    )


def _kda_layer_flops(config: KDALayerConfig, total_tokens: float) -> float:
    """Count KDA projections/conv plus a recurrent-operation estimate for its core."""
    config.validate_kda()
    heads, key_dim = config.linear_num_key_heads, config.linear_key_head_dim
    value_dim = config.linear_value_head_dim
    qk_width, value_width = heads * key_dim, heads * value_dim
    if config.kda_two_stage_gates:
        input_width = 2 * qk_width + value_width
        gates = (
            config.hidden_size * (key_dim + value_dim)
            + key_dim * qk_width
            + value_dim * value_width
        )
    else:
        input_width = 3 * qk_width + 2 * value_width
        gates = 0
    return (
        2
        * total_tokens
        * (
            config.hidden_size * (input_width + heads)  # QKV/gates and separate beta projection.
            + gates
            + config.linear_conv_kernel_dim * (2 * qk_width + value_width)
            + 4 * heads * key_dim * value_dim
            + config.hidden_size * value_width
        )
    )


def _mlp_layer_flops(config: MLPLayerConfig, total_tokens: float) -> float:
    expansion = config.ffn_hidden_size / config.hidden_size
    scale_factor = 3.0 / 2.0 if config.gated_linear_unit else 1.0
    return 4 * expansion * scale_factor * total_tokens * config.hidden_size**2


def _moe_layer_flops(config: MoELayerConfig, total_tokens: float) -> float:
    scale_factor = 3.0 / 2.0 if config.gated_linear_unit else 1.0
    ffn_hidden_size = config.moe_ffn_hidden_size or config.ffn_hidden_size
    if config.moe_latent_size is None:
        routed_flops = (
            4
            * total_tokens
            * config.hidden_size
            * ffn_hidden_size
            * config.moe_router_topk
            * scale_factor
        )
    else:
        routed_flops = (
            4
            * total_tokens
            * config.moe_latent_size
            * ffn_hidden_size
            * config.moe_router_topk
            * scale_factor
        )
        routed_flops += 4 * total_tokens * config.hidden_size * config.moe_latent_size
    shared_flops = (
        4
        * total_tokens
        * config.hidden_size
        * (config.moe_shared_expert_intermediate_size or 0)
        * scale_factor
    )
    return routed_flops + shared_flops


def estimate_hybrid_config_list_flops(
    layer_config_list: Sequence[HybridLayerEntry],
    hybrid_stack_spec: ModuleSpec,
    config: TransformerConfig,
    vocab_size: int,
    total_real_tokens_in_batch: float,
    seqlen_squared_sum_in_batch: float,
) -> float:
    """Estimate global-batch forward/backward FLOPs from a validated config sequence.

    Args:
        layer_config_list: Full unsplit model sequence, including pipeline and MTP markers.
            Every MTP head is counted even when its physical module is shared.
        hybrid_stack_spec: Layer implementations used to construct the model.
        config: Model-level configuration for MTP input projections and output hidden size.
        vocab_size: Model output vocabulary size.
        total_real_tokens_in_batch: Sum of unpadded sequence lengths in the global batch.
        seqlen_squared_sum_in_batch: Sum of squared unpadded sequence lengths in the global batch.

    Returns:
        Estimated forward/backward FLOPs for the whole model, independent of PP/VPP placement.
    """
    if any(
        not math.isfinite(value) or value < 0
        for value in (total_real_tokens_in_batch, seqlen_squared_sum_in_batch)
    ):
        raise ValueError("Hybrid FLOP token statistics must be finite and nonnegative.")
    forward_flops = 0
    indexer_training_flops = 0
    mtp_num_depths = 0
    layer_number = 0
    for entry in layer_config_list:
        if entry is PipelineSplit:
            continue
        if entry is MTPSplit:
            mtp_num_depths += 1
            layer_number = 0
            continue

        layer_number += 1
        if type(entry) is AttentionLayerConfig:
            forward_flops += _attention_layer_flops(
                entry, total_real_tokens_in_batch, seqlen_squared_sum_in_batch
            )
        elif type(entry) is MLALayerConfig:
            forward_flops += _mla_layer_flops(
                entry, total_real_tokens_in_batch, seqlen_squared_sum_in_batch
            )
        elif type(entry) is DSALayerConfig:
            attention_flops, indexer_flops = _dsa_layer_flops(
                entry,
                total_real_tokens_in_batch,
                seqlen_squared_sum_in_batch,
                layer_number,
                hybrid_stack_spec,
            )
            forward_flops += attention_flops
            indexer_training_flops += indexer_flops
        elif type(entry) is CSALayerConfig:
            attention_flops, indexer_flops = _csa_layer_flops(
                entry, total_real_tokens_in_batch, seqlen_squared_sum_in_batch
            )
            forward_flops += attention_flops
            indexer_training_flops += indexer_flops
        elif type(entry) is MambaLayerConfig:
            forward_flops += _mamba_layer_flops(
                entry, total_real_tokens_in_batch, hybrid_stack_spec
            )
        elif type(entry) is GDNLayerConfig:
            forward_flops += _gdn_layer_flops(entry, total_real_tokens_in_batch)
        elif type(entry) is KDALayerConfig:
            forward_flops += _kda_layer_flops(entry, total_real_tokens_in_batch)
        elif type(entry) is MLPLayerConfig:
            forward_flops += _mlp_layer_flops(entry, total_real_tokens_in_batch)
        elif type(entry) is MoELayerConfig:
            forward_flops += _moe_layer_flops(entry, total_real_tokens_in_batch)
        else:
            raise ValueError(
                f"Unexpected hybrid layer config type in FLOPs calculation: {type(entry).__name__}"
            )

    # Every prediction depth executes its input norms/projection and logits, even for shared MTP.
    forward_flops += (
        2
        * total_real_tokens_in_batch
        * mtp_num_depths
        * (3 * config.hidden_size + 2 * config.hidden_size**2)
    )
    forward_flops += (
        2 * total_real_tokens_in_batch * config.hidden_size * vocab_size * (1 + mtp_num_depths)
    )
    return forward_flops * 3 + indexer_training_flops
