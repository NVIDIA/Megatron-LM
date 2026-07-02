# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model builder for two-tower Mamba diffusion training.

Provides the ``two_tower_mamba_builder`` function used as the
``model_builder`` callback in ``pretrain_mamba_tt_diffusion.py``.
It translates Megatron CLI arguments into a fully configured
:class:`TwoTowerMambaModel` instance.
"""

from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.spec_utils import import_module
from megatron.training import print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from model_provider import count_parameters_in_layer


def two_tower_mamba_builder(
    args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None
):
    """Construct a :class:`TwoTowerMambaModel` from Megatron arguments.

    Reads ``--tt-diffusion-*`` flags from *args* and passes them to
    :class:`TwoTowerMambaModel`.  After construction, logs per-layer
    parameter counts for both the context and denoiser towers.

    Args:
        args (argparse.Namespace): Parsed Megatron command-line arguments.
        pre_process (bool): Whether this rank handles the first pipeline stage
            (embeddings).
        post_process (bool): Whether this rank handles the last pipeline stage
            (output layer + loss).
        vp_stage (Optional[int]): Virtual pipeline stage index, if using
            interleaved pipeline parallelism.
        config (Optional[TransformerConfig]): Transformer config override.
            Built from *args* when ``None``.
        pg_collection (Optional[ProcessGroupCollection]): Model communication
            process groups.

    Returns:
        TwoTowerMambaModel: Fully initialised model ready for training.
    """
    from megatron.diffusion.two_tower import TwoTowerMambaModel

    print_rank_0('building Two-Tower MAMBA model ...')
    if config is None:
        config = core_transformer_config_from_args(args, TransformerConfig)
    assert args.use_legacy_models is False, "Mamba only supported in Mcore!"

    if args.spec is not None:
        mamba_stack_spec = import_module(args.spec)
    else:
        raise ValueError("You must provide a valid Mamba layer spec via --spec")

    model = TwoTowerMambaModel(
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
        freeze_context=not getattr(args, 'tt_diffusion_no_freeze_context', False),
        tied_towers=getattr(args, 'tt_diffusion_tied_towers', False),
        use_time_conditioning=getattr(args, 'tt_diffusion_time_conditioning', False),
        bidirectional_mamba=getattr(args, 'tt_diffusion_bidirectional_mamba', False),
        context_ar_loss=getattr(args, 'tt_diffusion_context_ar_loss', False),
        mask_token_id=getattr(args, 'tt_diffusion_mask_token_id', 0),
        pg_collection=pg_collection,
        vp_stage=vp_stage,
    )

    model.block_size = getattr(args, 'tt_diffusion_block_size', 1)

    block_size = model.block_size
    from megatron.core.ssm.mamba_mixer import MambaMixer

    for tower in (model.context_tower, model.denoiser_tower):
        for layer in tower.layers:
            for module in layer.modules():
                if isinstance(module, MambaMixer) and module.chunk_size != block_size:
                    module.chunk_size = block_size
    print_rank_0(f"  Block size: {block_size} (Mamba chunk_size overridden to match)")

    for l in range(model.context_tower.num_layers_per_pipeline_rank):
        layer_params = count_parameters_in_layer(model, f'context_tower.layers.{l}.')
        print_rank_0(f" == params context layer {l}: {layer_params}")
    for l in range(model.denoiser_tower.num_layers_per_pipeline_rank):
        layer_params = count_parameters_in_layer(model, f'denoiser_tower.layers.{l}.')
        print_rank_0(f" == params denoiser layer {l}: {layer_params}")

    return model
