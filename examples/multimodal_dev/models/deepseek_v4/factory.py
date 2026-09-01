# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Factory functions for DeepSeek-V4-Flash-Vision construction."""

from examples.multimodal_dev.models.deepseek_v4.configuration import (
    DEEPSEEK_V4_VOCAB_SIZE,
    VISION_DOWNSAMPLE_RATIO,
    VISION_PATCH_SIZE,
)
from megatron.core.models.hybrid.hybrid_layer_allocation import parse_hybrid_pattern


def post_language_config(language_config, args) -> None:
    """Apply decoder fields that are specific to DSv4's multimodal checkpoint."""
    if not language_config.moe_router_enable_expert_bias:
        raise ValueError(
            "DeepSeek-V4-Vision requires --moe-router-enable-expert-bias for separate "
            "text and vision-language routing biases."
        )
    language_config.actual_vocab_size = DEEPSEEK_V4_VOCAB_SIZE
    language_config.moe_router_enable_vl_bias = True
    if language_config.experimental_attention_variant != "dsv4_hybrid":
        raise ValueError(
            "DeepSeek-V4-Vision requires experimental_attention_variant='dsv4_hybrid'."
        )
    if not language_config.multi_latent_attention:
        raise ValueError("DeepSeek-V4-Vision requires --multi-latent-attention.")


def set_vision_flops_metadata(args, language_config, vision_config) -> None:
    """Expose the native DSv4 vision dimensions to throughput accounting."""
    args.count_vision_model_flops = True
    args.vision_flops_variant = "deepseek_v4"
    args.vision_num_layers = vision_config.num_layers
    args.vision_hidden_size = vision_config.hidden_size
    args.vision_ffn_hidden_size = vision_config.ffn_hidden_size
    args.vision_num_attention_heads = vision_config.num_attention_heads
    args.vision_kv_channels = vision_config.kv_channels
    args.vision_in_channels = 3
    args.vision_patch_size = VISION_PATCH_SIZE
    args.vision_temporal_patch_size = 1
    args.vision_spatial_merge_size = VISION_DOWNSAMPLE_RATIO
    args.vision_out_hidden_size = language_config.hidden_size


def build_model(args, language_config, vision_config, **kwargs):
    """Build DeepSeek-V4-Flash-Vision on the PR-6315-style HybridModel path."""
    from examples.multimodal_dev.models.deepseek_v4.model import DeepSeekV4VisionModel
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_dsv4_stack_spec

    pattern = getattr(args, "hybrid_layer_pattern", None)
    if not pattern:
        raise ValueError("DeepSeek-V4-Vision requires --hybrid-layer-pattern.")
    parsed = parse_hybrid_pattern(pattern)
    if parsed.mtp_num_depths or getattr(args, "mtp_num_layers", 0):
        raise ValueError("DeepSeek-V4-Vision phase one does not support MTP.")
    if getattr(args, "position_embedding_type", "rope") != "rope":
        raise ValueError("DeepSeek-V4-Vision requires --position-embedding-type rope.")

    return DeepSeekV4VisionModel(
        language_config=language_config,
        hybrid_stack_spec=hybrid_dsv4_stack_spec(language_config),
        hybrid_layer_pattern=pattern,
        vision_config=vision_config,
        vocab_size=args.padded_vocab_size,
        actual_vocab_size=DEEPSEEK_V4_VOCAB_SIZE,
        max_sequence_length=args.max_position_embeddings,
        build_vision_encoder=not getattr(args, "use_external_vision_embeddings", False),
        parallel_output=True,
        share_embeddings_and_output_weights=not getattr(
            args, "untie_embeddings_and_output_weights", False
        ),
    )
