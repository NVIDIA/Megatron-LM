# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Factory functions for Qwen3.5-VL model construction.

Encapsulates all Qwen3.5-VL-specific logic needed by ``pretrain_multimodal.py``
so that the training entry point remains model-agnostic.
"""

from examples.multimodal_dev.models.qwen35_vl.configuration import (
    MROPE_SECTION,
    VISION_KWARGS,
)


def post_language_config(language_config, args):
    """Apply Qwen3.5-VL-specific settings to the language TransformerConfig.

    Called after ``core_transformer_config_from_args`` to inject model-specific
    fields that cannot be expressed via CLI args alone.
    """
    language_config.mrope_section = list(MROPE_SECTION)


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

    Handles language spec construction, optional MTP block spec, and
    model instantiation with Qwen3.5-VL-specific parameters.

    Args:
        args: Megatron parsed arguments.
        language_config: ``TransformerConfig`` for the language decoder
            (already post-processed by :func:`post_language_config`).
        vision_config: ``TransformerConfig`` for the vision encoder.
        **kwargs: Extra keyword arguments (e.g. ``vp_stage``).

    Returns:
        A :class:`Qwen35VLModel` instance.
    """
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_mtp_block_spec,
    )

    from examples.multimodal_dev.models.qwen35_vl.model import Qwen35VLModel
    from examples.multimodal_dev.models.qwen35_vl.specs import (
        get_qwen35_vl_language_spec,
    )

    language_spec = get_qwen35_vl_language_spec(
        config=language_config,
        vp_stage=kwargs.get("vp_stage", None),
        pp_rank=None,
    )

    mtp_block_spec = None
    if getattr(args, "mtp_num_layers", None):
        mtp_block_spec = get_gpt_mtp_block_spec(
            config=language_config,
            spec=language_spec,
            use_transformer_engine=(
                args.transformer_impl == "transformer_engine"
            ),
            vp_stage=kwargs.get("vp_stage", None),
            pp_rank=None,
        )

    return Qwen35VLModel(
        language_config=language_config,
        language_spec=language_spec,
        vision_config=vision_config,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        image_token_id=getattr(args, "image_token_id", 248056),
        mtp_block_spec=mtp_block_spec,
        parallel_output=True,
    )
