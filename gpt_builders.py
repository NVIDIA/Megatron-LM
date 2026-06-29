# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_with_inference_spec,
    get_gpt_mtp_block_spec,
    get_gpt_decoder_layer_specs,
)
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
    get_transformer_layer_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml


def _apply_yarn_config_from_args(config, args) -> None:
    """Populate YaRN fields on config from args when not already set.

    Preserves values already present on ``config`` (e.g. from YAML or a caller-
    supplied config). YaRN-specific hyperparameters must be supplied via CLI
    when ``position_embedding_type == 'yarn'`` (see functional test configs).
    """
    if args.position_embedding_type != 'yarn':
        return

    def _set_if_missing(attr: str, value) -> None:
        if value is None:
            return
        if not hasattr(config, attr):
            setattr(config, attr, value)

    _set_if_missing('yarn_rotary_scaling_factor', args.rotary_scaling_factor)
    _set_if_missing(
        'yarn_original_max_position_embeddings', args.yarn_original_max_position_embeddings
    )
    _set_if_missing('yarn_beta_fast', args.yarn_beta_fast)
    _set_if_missing('yarn_beta_slow', args.yarn_beta_slow)
    _set_if_missing('yarn_mscale', args.mscale)
    _set_if_missing('yarn_mscale_all_dim', args.mscale_all_dim)
    _set_if_missing('yarn_correction_range_round_to_int', args.yarn_correction_range_round_to_int)


def _get_transformer_layer_spec(use_te, config):
    """Get transformer layer specification based on configuration.

    Args:
        use_te (bool): Whether to use Transformer Engine
        config: Model configuration

    Returns:
        transformer_layer_spec: The transformer layer specification
    """
    if use_te:
        return get_gpt_layer_with_transformer_engine_spec(
            config.num_moe_experts,
            config.moe_grouped_gemm,
            config.qk_layernorm,
            config.multi_latent_attention,
            config.experimental_attention_variant,
            qk_l2_norm=config.qk_l2_norm,
            use_kitchen=config.use_kitchen,
            use_te_activation_func=config.use_te_activation_func,
            use_kitchen_attention=config.use_kitchen_attention,
            kitchen_attention_backend=config.kitchen_attention_backend,
            mla_down_proj_fusion=getattr(config, "mla_down_proj_fusion", False),
            use_grouped_gemm_for_dense_mlp=config.use_grouped_gemm_for_dense_mlp,
        )
    elif config.transformer_impl == "inference_optimized":
        return get_gpt_layer_with_inference_spec(
            config.qk_layernorm, config.multi_latent_attention, qk_l2_norm=config.qk_l2_norm
        )
    else:
        return get_gpt_layer_local_spec(
            config.num_moe_experts,
            config.moe_grouped_gemm,
            config.qk_layernorm,
            config.multi_latent_attention,
            config.experimental_attention_variant,
            normalization=config.normalization,
            use_kitchen=config.use_kitchen,
            use_kitchen_attention=config.use_kitchen_attention,
            kitchen_attention_backend=config.kitchen_attention_backend,
        )
