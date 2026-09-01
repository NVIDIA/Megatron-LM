# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Factory functions for Qwen3.5-VL model construction.

Encapsulates all Qwen3.5-VL-specific logic needed by ``pretrain_multimodal.py``
so that the training entry point remains model-agnostic.
"""

from examples.multimodal_dev.models.qwen35_vl.configuration import VISION_KWARGS


def post_language_config(language_config, args):
    """Apply Qwen3.5-VL-specific settings to the language TransformerConfig.

    Called after ``core_transformer_config_from_args`` to inject model-specific
    fields that cannot be expressed via CLI args alone.

    ``mrope_section`` is deliberately not set here: it is a ``TransformerConfig``
    field fed by ``--mrope-section``, so overriding it would let the recipe and
    the constructed model disagree.

    ``mrope_interleaved`` goes the other way. ``--mrope-interleaved`` also exists
    as a generated flag, but it is ``store_true`` with a ``False`` default, so a
    recipe that forgot it would silently build a non-interleaved decoder.
    Qwen3.5 always interleaves the T/H/W sections, so it is pinned here as an
    architectural constant, alongside ``ROTARY_PERCENT`` / ``ROTARY_BASE`` in
    ``model.py``. ``mrope_section`` cannot get the same treatment because
    ``validate_args`` requires it on the CLI before the model is built.
    """
    language_config.mrope_interleaved = True


def set_vision_flops_metadata(args, language_config, vision_config):
    """Expose Qwen3.5-VL vision-model dimensions for FLOPs estimation."""
    args.count_vision_model_flops = True
    args.vision_flops_variant = "qwen35_vl_v2"
    args.vision_num_layers = vision_config.num_layers
    args.vision_hidden_size = vision_config.hidden_size
    args.vision_ffn_hidden_size = vision_config.ffn_hidden_size
    args.vision_num_attention_heads = vision_config.num_attention_heads
    args.vision_kv_channels = vision_config.kv_channels
    args.vision_in_channels = VISION_KWARGS["in_channels"]
    args.vision_patch_size = VISION_KWARGS["patch_size"]
    args.vision_temporal_patch_size = VISION_KWARGS["temporal_patch_size"]
    args.vision_spatial_merge_size = VISION_KWARGS["spatial_merge_size"]
    args.vision_out_hidden_size = language_config.hidden_size


def build_model(args, language_config, vision_config, **kwargs):
    """Build a complete Qwen3.5-VL model instance.

    Selects the HybridModel stack spec and instantiates the model with the
    unified decoder/MTP layer pattern parsed from the CLI.

    Args:
        args: Megatron parsed arguments.
        language_config: ``TransformerConfig`` for the language decoder
            (already post-processed by :func:`post_language_config`).
        vision_config: ``TransformerConfig`` for the vision encoder.
        **kwargs: Extra keyword arguments (e.g. ``vp_stage``).

    Returns:
        A :class:`Qwen35VLModel` instance.
    """
    hybrid_layer_pattern = getattr(args, "hybrid_layer_pattern", None)
    if hybrid_layer_pattern is None:
        raise ValueError(
            "Qwen3.5-VL uses HybridModel and requires --hybrid-layer-pattern. "
            "Use GEGEGE*E per four MoE blocks (or G-G-G-*- for dense blocks), "
            "and append /*E or /*- for each MTP depth."
        )

    from examples.multimodal_dev.models.qwen35_vl.model import Qwen35VLModel
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec

    # When --untie-embeddings-and-output-weights is NOT passed, Megatron
    # defaults to tied embeddings (share_embeddings_and_output_weights=True).
    # The 0.8B variant uses tied embeddings, while larger variants untie them.
    share_embeddings = not getattr(
        args, "untie_embeddings_and_output_weights", False
    )

    return Qwen35VLModel(
        language_config=language_config,
        hybrid_stack_spec=hybrid_stack_spec,
        hybrid_layer_pattern=hybrid_layer_pattern,
        vision_config=vision_config,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        image_token_id=getattr(args, "image_token_id", 248056),
        position_embedding_type=args.position_embedding_type,
        parallel_output=True,
        share_embeddings_and_output_weights=share_embeddings,
    )
