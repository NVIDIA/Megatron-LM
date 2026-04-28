# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_inference_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.spec_utils import import_module
from megatron.training import print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from model_provider import count_parameters_in_layer


def hybrid_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    print_rank_0('building Hybrid model ...')
    if config is None:
        config = core_transformer_config_from_args(args, TransformerConfig)
    assert args.use_legacy_models is False, "Hybrid model only supported in Mcore!"

    if config.transformer_impl == "inference_optimized":
        hybrid_stack_spec = hybrid_inference_stack_spec
        assert (
            not config.inference_fuse_tp_communication
        ), "inference_fuse_tp_communication is not supported for HybridModel"
    elif args.spec is not None:
        hybrid_stack_spec = import_module(args.spec)
    else:
        raise ValueError("You must provide a valid hybrid layer spec via --spec")

    model = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        hybrid_layer_pattern=args.hybrid_layer_pattern,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        pg_collection=pg_collection,
        vp_stage=vp_stage,
    )

    for l in range(model.decoder.num_layers_per_pipeline_rank):
        layer_params = count_parameters_in_layer(model, f'decoder.layers.{l}.')
        print_rank_0(f" == params layer {l}: {layer_params}")

    return model


# Backward-compatible alias
mamba_builder = hybrid_builder
