# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_inference_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer import MLATransformerConfig, TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec, import_module
from megatron.training import print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from model_provider import count_parameters_in_layer


def hybrid_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    print_rank_0('building Hybrid model ...')
    if config is None:
        config = core_transformer_config_from_args(args, TransformerConfig)
    # MLA (and DSv4 hybrid) require MLATransformerConfig so that its __post_init__ runs
    # the dsv4_hybrid derivation. The hybrid pretrain path can hand us a plain
    # TransformerConfig, which silently skips that derivation; rebuild as MLA to match GPT.
    if args.multi_latent_attention and not isinstance(config, MLATransformerConfig):
        config = core_transformer_config_from_args(args)
    # DSv4-hybrid head-dim contract: qk_head_dim and kv_lora_rank are derived from
    # v_head_dim and qk_pos_emb_head_dim (MLATransformerConfig.__post_init__ does this for the
    # GPT path). The hybrid config can reach here without that derivation applied, which breaks
    # the MLA up-proj / fused-rope contract (q head dim must equal qk_head_dim + qk_pos_emb_head
    # _dim == v_head_dim). Apply it for any DSv4 MLA attention: experimental_attention_variant
    # == dsv4_hybrid, OR the layer pattern uses a DSv4 attention symbol (D/C/H). Idempotent.
    _pattern = getattr(args, "hybrid_layer_pattern", None) or ""
    _uses_dsv4_attn = (
        getattr(args, "experimental_attention_variant", None) == "dsv4_hybrid"
        or any(sym in _pattern for sym in ("C", "H"))
    )
    if _uses_dsv4_attn:
        derived = config.v_head_dim - config.qk_pos_emb_head_dim
        if config.qk_head_dim != derived or config.kv_lora_rank != derived:
            print_rank_0(
                f"[hybrid dsv4] deriving qk_head_dim/kv_lora_rank = {config.v_head_dim} - "
                f"{config.qk_pos_emb_head_dim} = {derived} (was qk_head_dim={config.qk_head_dim}, "
                f"kv_lora_rank={config.kv_lora_rank})"
            )
            config.qk_head_dim = derived
            config.kv_lora_rank = derived
        # 'C'/'H' layers carry their compress ratio via the spec, but array-driven 'D' layers
        # AND the indexer-loss logger (which counts ratio==4 layers) read
        # config.csa_compress_ratios. When not given explicitly, derive it from the pattern
        # symbols (C->4, H->128, D/other->0) so the array is consistent with the symbols and
        # the indexer loss is normalized correctly; pad MTP depths with 0. An explicit
        # --csa-compress-ratios is always respected.
        if config.csa_compress_ratios is None:
            ratio_map = {"C": 4, "H": 128}
            # One entry per ACTUAL layer: main layers, then every MTP layer of every MTP depth
            # (a depth can hold multiple hybrid layers, e.g. "/MD-E"), mirroring the arguments.py
            # derivation. Padding by mtp_num_layers (depth count) would be too short and an MTP
            # attention that isn't first would IndexError at num_layers + layer_number - 1.
            sections = _pattern.split("/")
            ratios = [ratio_map.get(c, 0) for c in sections[0].replace("|", "")]
            for mtp_sec in sections[1:]:
                ratios += [ratio_map.get(c, 0) for c in mtp_sec.replace("|", "")]
            config.csa_compress_ratios = ratios
            print_rank_0(
                f"[hybrid dsv4] derived csa_compress_ratios from pattern symbols: {ratios}"
            )

    if config.transformer_impl == "inference_optimized":
        hybrid_stack_spec = hybrid_inference_stack_spec
        assert (
            not config.inference_fuse_tp_communication
        ), "inference_fuse_tp_communication is not supported for HybridModel"
    elif args.spec is not None:
        hybrid_stack_spec = import_module(args.spec)
        # Allow config-aware specs: if --spec resolves to a callable (not a ModuleSpec),
        # call it with config to build the stack spec (e.g. hybrid_dsv4_stack_spec, which
        # wires the DSv4 CompressedSparseAttention into the 'D' layer per config).
        if not isinstance(hybrid_stack_spec, ModuleSpec) and callable(hybrid_stack_spec):
            hybrid_stack_spec = hybrid_stack_spec(config)
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
