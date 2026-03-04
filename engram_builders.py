# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from model_provider import count_parameters_in_layer
from megatron.core.models.engram import EngramGPTModel
from megatron.core.models.engram.engram_module import EngramConfig, NgramHashMapping
from megatron.core.models.engram.engram_layer_specs import get_engram_layer_local_spec
from megatron.core.transformer import TransformerConfig
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args


def _build_engram_config(args) -> EngramConfig:
    """Build EngramConfig from training arguments."""
    engram_layer_ids = getattr(args, 'engram_layer_ids', [1, 15])
    if isinstance(engram_layer_ids, str):
        engram_layer_ids = [int(x) for x in engram_layer_ids.split(',')]

    return EngramConfig(
        engram_vocab_size=getattr(
            args, 'engram_vocab_size', [129280 * 5, 129280 * 5]
        ),
        max_ngram_size=getattr(args, 'engram_max_ngram_size', 3),
        n_embed_per_ngram=getattr(args, 'engram_n_embed_per_ngram', 512),
        n_head_per_ngram=getattr(args, 'engram_n_head_per_ngram', 8),
        engram_layer_ids=engram_layer_ids,
        pad_id=getattr(args, 'engram_pad_id', 2),
        seed=getattr(args, 'engram_seed', 0),
        kernel_size=getattr(args, 'engram_kernel_size', 4),
        hc_mult=getattr(args, 'engram_hc_mult', 4),
        tokenizer_name_or_path=getattr(
            args, 'engram_tokenizer', 'deepseek-ai/DeepSeek-V3'
        ),
    )


def engram_builder(
    args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None
):
    """Build an Engram-augmented GPT model.

    Constructs the EngramConfig, pre-computes the n-gram vocabulary sizes via
    NgramHashMapping, builds the layer spec with Engram support, and returns
    an EngramGPTModel instance.
    """
    print_rank_0('building Engram GPT model ...')
    if config is None:
        config = core_transformer_config_from_args(args, TransformerConfig)
    assert args.use_legacy_models is False, "Engram only supported in Mcore!"

    engram_config = _build_engram_config(args)

    # Pre-compute vocab sizes for the hash embedding tables.
    # NgramHashMapping also initializes the CompressedTokenizer, which is
    # moderately expensive, so we do it once here and pass the results through.
    ngram_hash_mapping = NgramHashMapping(
        engram_vocab_size=engram_config.engram_vocab_size,
        max_ngram_size=engram_config.max_ngram_size,
        n_embed_per_ngram=engram_config.n_embed_per_ngram,
        n_head_per_ngram=engram_config.n_head_per_ngram,
        layer_ids=engram_config.engram_layer_ids,
        tokenizer_name_or_path=engram_config.tokenizer_name_or_path,
        pad_id=engram_config.pad_id,
        seed=engram_config.seed,
    )
    vocab_size_across_layers = ngram_hash_mapping.vocab_size_across_layers

    transformer_layer_spec = get_engram_layer_local_spec(
        engram_config=engram_config,
        vocab_size_across_layers=vocab_size_across_layers,
        num_experts=getattr(args, 'num_experts', None),
        moe_grouped_gemm=getattr(args, 'moe_grouped_gemm', False),
        qk_layernorm=getattr(args, 'qk_layernorm', False),
        normalization=getattr(args, 'normalization', None),
        qk_l2_norm=getattr(args, 'qk_l2_norm', False),
    )

    model = EngramGPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        engram_config=engram_config,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        vp_stage=vp_stage,
        pg_collection=pg_collection,
    )

    for l in range(model.decoder.num_layers_per_pipeline_rank):
        layer_params = count_parameters_in_layer(model, f'decoder.layers.{l}.')
        print_rank_0(f" == params layer {l}: {layer_params}")

    return model
