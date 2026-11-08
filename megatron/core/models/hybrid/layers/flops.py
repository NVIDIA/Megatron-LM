# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Config-list dispatch and aggregation over shared Core FLOP formulas."""

import math
from numbers import Real
from typing import Sequence

from megatron.core import flops
from megatron.core.models.hybrid.hybrid_layer_allocation import (
    HybridLayerEntry,
    MTPSplit,
    PipelineSplit,
)
from megatron.core.models.hybrid.layers.dsa_flops import _dsa_indexer_flops, dsa_layer_flops
from megatron.core.models.hybrid.layers.utils import Symbols, get_custom_layer_type
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


def _builtin_layer_flops(config, total_tokens, squared_lengths, layer_number, stack_spec):
    """Return forward FLOPs and any separately expanded training FLOPs."""
    if type(config) is AttentionLayerConfig:
        return (
            flops.attn_layer_flops(
                total_tokens,
                squared_lengths,
                config.hidden_size,
                config.num_attention_heads,
                gqa_groups=config.num_query_groups,
                kv_channels=config.kv_channels,
                attention_output_gate=config.attention_output_gate,
            ),
            0,
        )
    if type(config) is MLALayerConfig:
        return (
            flops.mla_layer_flops(
                total_tokens,
                squared_lengths,
                config.hidden_size,
                config.num_attention_heads,
                config.q_lora_rank,
                config.kv_lora_rank,
                config.qk_head_dim,
                config.qk_pos_emb_head_dim,
                config.v_head_dim,
            ),
            0,
        )
    if type(config) is DSALayerConfig:
        from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
            AbsorbedMLASelfAttention,
        )
        from megatron.core.transformer.experimental_attention_variant.dsa import (
            is_dsa_skip_topk_layer,
        )

        layer_spec = stack_spec.submodules.dsa_layer
        attention_spec = layer_spec.submodules.self_attention
        forward = dsa_layer_flops(
            total_tokens,
            squared_lengths,
            config.hidden_size,
            config.num_attention_heads,
            config.q_lora_rank,
            config.kv_lora_rank,
            config.qk_head_dim,
            config.qk_pos_emb_head_dim,
            config.v_head_dim,
            config.dsa_indexer_topk,
            dsa_indexer_kpool=config.dsa_indexer_kpool,
            dsa_indexer_kpool_always_select_tail=config.dsa_indexer_kpool_always_select_tail,
            absorbed_mla=attention_spec.module is AbsorbedMLASelfAttention,
        )
        token_linear, core = _dsa_indexer_flops(
            total_tokens=total_tokens,
            seqlen_squared_sum=squared_lengths,
            hidden_size=config.hidden_size,
            q_lora_rank=config.q_lora_rank,
            n_heads=config.dsa_indexer_n_heads,
            head_dim=config.dsa_indexer_head_dim,
            num_indexer_layers=int(
                not is_dsa_skip_topk_layer(
                    layer_number, config.dsa_indexer_skip_topk_offset, config.dsa_indexer_topk_freq
                )
            ),
            dsa_indexer_loss_enabled=(config.dsa_indexer_loss_coeff or 0) > 0,
            dsa_indexer_use_sparse_loss=config.dsa_indexer_use_sparse_loss,
            dsa_indexer_topk=config.dsa_indexer_topk,
            dsa_indexer_kpool=config.dsa_indexer_kpool,
        )
        return forward, total_tokens * token_linear + squared_lengths * core
    if type(config) is CSALayerConfig:
        if config.compress_ratio not in (0, 4, 128):
            raise ValueError(f"Unsupported CSA compression ratio: {config.compress_ratio}")
        counts = {
            Symbols.WINDOW: int(config.compress_ratio == 0),
            Symbols.CSA: int(config.compress_ratio == 4),
            Symbols.HCA: int(config.compress_ratio == 128),
        }
        return 0, flops.dsv4_attention_flops(config, counts, total_tokens, squared_lengths)
    if type(config) is MambaLayerConfig:
        from megatron.core.ssm.gated_delta_product import GatedDeltaProductMixer

        mixer_spec = stack_spec.submodules.mamba_layer.submodules.mixer
        if mixer_spec.module is GatedDeltaProductMixer:
            return (
                flops.gated_delta_product_layer_flops(
                    total_tokens,
                    config.hidden_size,
                    config.gdp_num_householder,
                    config.mamba_state_dim,
                    config.mamba_head_dim,
                    config.mamba_num_groups,
                    config.mamba_num_heads,
                ),
                0,
            )
        return (
            flops.mamba_layer_flops(
                total_tokens,
                config.hidden_size,
                config.mamba_state_dim,
                config.mamba_head_dim,
                config.mamba_num_groups,
                config.mamba_num_heads,
            ),
            0,
        )
    if type(config) is GDNLayerConfig:
        return (
            flops.gdn_layer_flops(
                total_tokens,
                config.hidden_size,
                config.linear_key_head_dim or 128,
                config.linear_value_head_dim or 128,
                config.linear_num_key_heads or 16,
                config.linear_num_value_heads or 32,
                config.linear_conv_kernel_dim or 4,
                use_gdn2=config.experimental_attention_variant == "gdn2",
            ),
            0,
        )
    if type(config) is KDALayerConfig:
        config.validate_kda()
        return (
            flops.kda_layer_flops(
                total_tokens,
                config.hidden_size,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
                config.linear_num_key_heads,
                config.linear_num_value_heads,
                config.linear_conv_kernel_dim,
                config.kda_two_stage_gates,
            ),
            0,
        )
    if type(config) is MLPLayerConfig:
        return (
            flops.mlp_layer_flops(
                total_tokens,
                config.hidden_size,
                config.ffn_hidden_size / config.hidden_size,
                config.gated_linear_unit,
            ),
            0,
        )
    if type(config) is MoELayerConfig:
        return (
            flops.moe_layer_flops(
                total_tokens,
                config.hidden_size,
                config.moe_ffn_hidden_size or config.ffn_hidden_size,
                config.moe_shared_expert_intermediate_size or 0,
                config.moe_router_topk,
                config.moe_latent_size,
                config.gated_linear_unit,
            ),
            0,
        )
    raise ValueError(
        f"Unexpected hybrid layer config type in FLOPs calculation: {type(config).__name__}"
    )


def estimate_hybrid_config_list_flops(
    layer_config_list: Sequence[HybridLayerEntry],
    hybrid_stack_spec: ModuleSpec,
    config: TransformerConfig,
    vocab_size: int,
    total_real_tokens_in_batch: float,
    seqlen_squared_sum_in_batch: float,
) -> float:
    """Estimate global-batch training FLOPs from the full, validated config sequence.

    Pipeline splits do not change the count. Every MTP depth executes its layers
    and output head even when weights are shared. Built-ins reuse training's
    established formulas; custom layers supply forward FLOPs through their class.
    DSA indexers and CSA already include their own backward expansion.
    """
    forward_flops = 0
    separately_expanded_flops = 0
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
        custom_layer_type = get_custom_layer_type(entry)
        if custom_layer_type is not None:
            layer_flops = custom_layer_type.estimate_flops(
                entry,
                total_real_tokens_in_batch=total_real_tokens_in_batch,
                seqlen_squared_sum_in_batch=seqlen_squared_sum_in_batch,
            )
            if (
                not isinstance(layer_flops, Real)
                or isinstance(layer_flops, bool)
                or not math.isfinite(layer_flops)
                or layer_flops < 0
            ):
                raise ValueError(
                    f"{custom_layer_type.__name__}.estimate_flops must return a finite, "
                    f"nonnegative real number, got {layer_flops!r}."
                )
            forward_flops += layer_flops
        else:
            layer_flops, training_flops = _builtin_layer_flops(
                entry,
                total_real_tokens_in_batch,
                seqlen_squared_sum_in_batch,
                layer_number,
                hybrid_stack_spec,
            )
            forward_flops += layer_flops
            separately_expanded_flops += training_flops

    forward_flops += (
        2
        * total_real_tokens_in_batch
        * mtp_num_depths
        * (3 * config.hidden_size + 2 * config.hidden_size**2)
    )
    forward_flops += (
        2 * total_real_tokens_in_batch * config.hidden_size * vocab_size * (1 + mtp_num_depths)
    )
    return forward_flops * 3 + separately_expanded_flops
