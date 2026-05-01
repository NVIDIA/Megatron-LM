# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_inference_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.hybrid.layer_pattern import load_recipe
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.spec_utils import import_module
from megatron.training import print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from model_provider import count_parameters_in_layer


def hybrid_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    """Construct a HybridModel from CLI args, dispatching on ``--model-recipe``."""
    print_rank_0('building Hybrid model ...')
    assert args.use_legacy_models is False, "Hybrid model only supported in Mcore!"

    # --model-recipe path: the recipe is the source of truth for the model
    # architecture (config + layer pattern). Skip args→config conversion entirely.
    if getattr(args, 'model_recipe', None) is not None:
        if config is not None:
            print_rank_0(
                "Note: --model-recipe was provided; ignoring caller-supplied config "
                "and using the recipe's compiled TransformerConfig."
            )
        recipe = getattr(args, '_compiled_model_recipe', None)
        if recipe is None:
            recipe = load_recipe(args.model_recipe)

        # Spec selection precedence: explicit recipe.stack_spec wins over
        # transformer_impl-based auto-pick. ``None`` falls through to
        # HybridModel's default hybrid_stack_spec.
        if recipe.stack_spec is not None:
            recipe_stack_spec = import_module(recipe.stack_spec)
        elif recipe.config.transformer_impl == "inference_optimized":
            assert (
                not recipe.config.inference_fuse_tp_communication
            ), "inference_fuse_tp_communication is not supported for HybridModel"
            recipe_stack_spec = hybrid_inference_stack_spec
        else:
            recipe_stack_spec = None

        model = HybridModel(
            config=recipe.config,
            hybrid_stack_spec=recipe_stack_spec,
            vocab_size=recipe.vocab_size,
            max_sequence_length=recipe.max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=recipe.fp16_lm_cross_entropy,
            parallel_output=recipe.parallel_output,
            share_embeddings_and_output_weights=recipe.share_embeddings_and_output_weights,
            position_embedding_type=recipe.position_embedding_type,
            rotary_percent=recipe.rotary_percent,
            rotary_base=recipe.rotary_base,
            seq_len_interpolation_factor=recipe.seq_len_interpolation_factor,
            scatter_embedding_sequence_parallel=recipe.scatter_embedding_sequence_parallel,
            layer_type_list=recipe.layer_type_list,
            layer_config_list=recipe.layer_config_list,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )
    else:
        if config is None:
            config = core_transformer_config_from_args(args, TransformerConfig)
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
            hybrid_layer_pattern=args.hybrid_layer_pattern,
        )

    for l in range(model.decoder.num_layers_per_pipeline_rank):
        layer_params = count_parameters_in_layer(model, f'decoder.layers.{l}.')
        print_rank_0(f" == params layer {l}: {layer_params}")

    return model


# Backward-compatible alias
mamba_builder = hybrid_builder
