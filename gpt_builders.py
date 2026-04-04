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

import megatron.legacy.model  # isort: skip

# NOTE: Loading `megatron.legacy.model` earlier fails due to circular import


def gpt_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    print_rank_0('building GPT model ...')
    if config is None:
        if args.yaml_cfg is not None:
            config = core_transformer_config_from_yaml(args, "language_model")
        else:
            config = core_transformer_config_from_args(args)
    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else:  # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            use_te = args.transformer_impl == "transformer_engine"

            if args.experimental_attention_variant is not None:
                transformer_layer_spec = (
                    get_transformer_block_with_experimental_attention_variant_spec(
                        config=config, vp_stage=vp_stage
                    )
                )
            elif args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(
                    config,
                    use_transformer_engine=use_te,
                    normalization=args.normalization,
                    qk_l2_norm=args.qk_l2_norm,
                    vp_stage=vp_stage,
                )
            elif args.heterogeneous_layers_config_path is not None:
                assert not (config.transformer_impl == "inference_optimized")
                transformer_layer_spec = get_gpt_heterogeneous_layer_spec(config, use_te)
            else:
                # Define the decoder layer spec
                transformer_layer_spec = _get_transformer_layer_spec(use_te, config)
        mtp_block_spec = None
        if args.mtp_num_layers is not None:
            assert not (config.transformer_impl == "inference_optimized")
            # Get GPT decoder layer specs for the model.
            if args.spec is not None:
                mtp_transformer_layer_spec = import_module(args.spec)
            else:
                # Define the decoder block spec
                if args.experimental_attention_variant is not None:
                    decoder_layer_specs = (
                        get_transformer_layer_with_experimental_attention_variant_spec(
                            config=config
                        )
                    )
                else:
                    decoder_layer_specs = get_gpt_decoder_layer_specs(
                        config,
                        use_transformer_engine=use_te,
                        normalization=args.normalization,
                        qk_l2_norm=args.qk_l2_norm,
                    )
                mtp_transformer_layer_spec = decoder_layer_specs[-1]
            # Use spec of the last layer in decoder block as spec of the transformer layer in MTP
            mtp_block_spec = get_gpt_mtp_block_spec(
                config,
                mtp_transformer_layer_spec,
                use_transformer_engine=use_te,
                vp_stage=vp_stage,
            )

        model_cls = GPTModel
        model_kwargs = {}
        if getattr(args, "use_engram", False):
            from megatron.core.models.engram.engram_model import EngramGPTModel
            from megatron.core.models.engram.plugin import (
                apply_engram_to_transformer_layer_spec,
                build_engram_config_from_args,
            )

            engram_config = build_engram_config_from_args(args)
            transformer_layer_spec = apply_engram_to_transformer_layer_spec(
                transformer_layer_spec, engram_config
            )
            model_cls = EngramGPTModel
            model_kwargs["engram_config"] = engram_config

        model = model_cls(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
            pg_collection=pg_collection,
            **model_kwargs,
        )

    return model


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
            fallback_to_eager_attn=config.fallback_to_eager_attn,
            enable_hyper_connection=config.enable_hyper_connections,
            mla_down_proj_fusion=getattr(config, "mla_down_proj_fusion", False),
        )
    elif config.transformer_impl == "inference_optimized":
        return get_gpt_layer_with_inference_spec(
            config.qk_layernorm,
            config.multi_latent_attention,
            qk_l2_norm=config.qk_l2_norm,
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
            enable_hyper_connection=config.enable_hyper_connections,
        )
