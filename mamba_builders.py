# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
"""Backward-compatible re-export of hybrid_builders.

Deprecated. Use hybrid_builders instead.
"""
import warnings

warnings.warn(
    "mamba_builders has been deprecated. Use hybrid_builders instead.",
    DeprecationWarning,
    stacklevel=2,
)

def mamba_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    print_rank_0('building MAMBA model ...')
    if config is None:
        config = core_transformer_config_from_args(args, TransformerConfig)
    assert args.use_legacy_models is False, "Mamba only supported in Mcore!"

    if config.transformer_impl == "inference_optimized":
        mamba_stack_spec = mamba_inference_stack_spec
        assert (
            not config.inference_fuse_tp_communication
        ), "inference_fuse_tp_communication is not supported for Mamba"
    elif args.spec is not None:
        mamba_stack_spec = import_module(args.spec)
    else:
        raise ValueError("You must provide a valid Mamba layer spec via --spec")

    model = MambaModel(
        config=config,
        mamba_stack_spec=mamba_stack_spec,
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
        rope_scaling=args.use_rope_scaling,
        rope_scaling_factor=args.rope_scaling_factor,
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
        pg_collection=pg_collection,
        vp_stage=vp_stage,
    )

    for l in range(model.decoder.num_layers_per_pipeline_rank):
        layer_params = count_parameters_in_layer(model, f'decoder.layers.{l}.')
        print_rank_0(f" == params layer {l}: {layer_params}")

    return model
