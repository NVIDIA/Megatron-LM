# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from model_provider import count_parameters_in_layer
from megatron.core.models.mamba import MambaModel
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.spec_utils import import_module
from megatron.training import print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.models.mamba.mamba_layer_specs import mamba_inference_stack_spec

def mamba_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    print_rank_0('building MAMBA model ...')
    if config is None:
        config = core_transformer_config_from_args(args, TransformerConfig)
    assert args.use_legacy_models is False, "Mamba only supported in Mcore!"

    if config.transformer_impl == "inference_optimized":
        mamba_stack_spec = mamba_inference_stack_spec 
    elif args.spec is not None:
        mamba_stack_spec = import_module(args.spec)
    else:
        raise ValueError("You must provide a valid Mamba layer spec via --spec")

    model = MambaModel(
        config=config,
        mamba_stack_spec=mamba_stack_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        hybrid_attention_ratio=args.hybrid_attention_ratio,
        hybrid_mlp_ratio=args.hybrid_mlp_ratio,
        hybrid_override_pattern=args.hybrid_override_pattern,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        pg_collection=pg_collection,
    )

    for l in range(model.decoder.num_layers_per_pipeline_rank):
        layer_params = count_parameters_in_layer(model, f'decoder.layers.{l}.')
        print_rank_0(f" == params layer {l}: {layer_params}")

    return model
