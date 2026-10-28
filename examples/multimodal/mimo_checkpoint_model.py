# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Build a LLaVAModel for inference from a MIMO VLM checkpoint.

The language model is the hybrid model created with the standard model arguments. The vision
encoder is the --vision-model-type registry entry, with sizes from the --vision-* arguments where
given, followed by a two-layer MLP projection at the language model width. The model's sharded
state dict uses the checkpoint's MIMO key names, so it loads with the standard checkpoint loader.
"""

from dataclasses import replace

import torch

from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.models.vision.encoder_registry import get_spec
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import get_args, print_rank_0
from megatron.training.argument_utils import hybrid_config_from_args
from megatron.training.models.hybrid import resolve_hybrid_stack_spec

# Encoder sizes that the --vision-<size> arguments override.
_VISION_SIZE_FIELDS = (
    "num_layers",
    "hidden_size",
    "ffn_hidden_size",
    "num_attention_heads",
    "num_query_groups",
    "kv_channels",
)


def _unfused_fast_gelu(x: torch.Tensor) -> torch.Tensor:
    """Fast GELU without the global MCore torch.compile decorator.

    The projection input's token count varies with the number and size of images, so a compiled
    activation would recompile for each new shape.
    """
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


_PROJECTION_ACTIVATIONS = {"gelu": torch.nn.functional.gelu, "fast_gelu": _unfused_fast_gelu}


class MimoCheckpointLLaVAModel(LLaVAModel):
    """LLaVAModel whose sharded state dict uses a MIMO checkpoint's key names.

    Args:
        checkpoint_prefix_map: Map from LLaVAModel key prefixes (language_model., vision_model.,
            vision_projection.) to the MIMO checkpoint prefixes they load from.
    """

    def __init__(self, *args, checkpoint_prefix_map: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_prefix_map = checkpoint_prefix_map
        # Checkpoint tensor keys of the latest sharded state dict, i.e. the ones the last load
        # requested (built with that checkpoint's metadata).
        self.sharded_checkpoint_keys = None

    def sharded_state_dict(self, prefix: str = '', sharded_offsets: tuple = (), metadata=None):
        """Build the sharded state dict with keys renamed to the checkpoint's."""
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        apply_prefix_mapping(
            sharded_state_dict,
            {f'{prefix}{key}': value for key, value in self.checkpoint_prefix_map.items()},
        )
        self.sharded_checkpoint_keys = {
            value.key
            for value in nested_values(sharded_state_dict)
            if isinstance(value, (ShardedTensor, ShardedTensorFactory))
        }
        return sharded_state_dict


def _tower_config(language_config: TransformerConfig, **kwargs) -> TransformerConfig:
    """Return a dense config sharing only the language model's parallelism and precision.

    Starting from defaults rather than a copy of the language config keeps language-only
    features (MoE, wide residuals, inference-optimized layers, ...) out of the vision modules.
    """
    return TransformerConfig(
        tensor_model_parallel_size=language_config.tensor_model_parallel_size,
        params_dtype=language_config.params_dtype,
        bf16=language_config.bf16,
        fp16=language_config.fp16,
        use_cpu_initialization=language_config.use_cpu_initialization,
        perform_initialization=language_config.perform_initialization,
        **kwargs,
    )


def vision_config(args, language_config: TransformerConfig) -> TransformerConfig:
    """Build the vision encoder config from its registry entry and the --vision-* sizes."""
    overrides = {}
    for field in _VISION_SIZE_FIELDS:
        value = getattr(args, f"vision_{field}", None)
        if value is not None:
            overrides[field] = value
    # Registry head layouts belong to the registry sizes; fall back to the TransformerConfig
    # defaults (one group per head, hidden size / heads) when the sizes change.
    derived = {}
    if "num_attention_heads" in overrides:
        derived["num_query_groups"] = None
    if "hidden_size" in overrides or "num_attention_heads" in overrides:
        derived["kv_channels"] = None
    spec = replace(get_spec(args.vision_model_type), **{**derived, **overrides})
    config = _tower_config(
        language_config,
        num_layers=spec.num_layers,
        hidden_size=spec.hidden_size,
        num_attention_heads=spec.num_attention_heads,
    )
    spec.apply_to_config(config)
    config.vision_model_type = args.vision_model_type
    return config


def vision_projection_config(args, language_config: TransformerConfig, add_bias: bool):
    """Build W2(act(W1(z))) with both hidden widths matching the language model."""
    return _tower_config(
        language_config,
        num_layers=1,
        hidden_size=language_config.hidden_size,
        ffn_hidden_size=language_config.hidden_size,
        num_attention_heads=1,
        activation_func=_PROJECTION_ACTIVATIONS[args.vision_projection_activation],
        gated_linear_unit=False,
        add_bias_linear=add_bias,
        bias_activation_fusion=False,
    )


def model_provider(
    pre_process=True,
    post_process=True,
    add_encoder=True,
    add_decoder=True,
    parallel_output=True,
    vp_stage=None,
    config=None,
    pg_collection=None,
) -> LLaVAModel:
    """Build a LLaVAModel that loads the MIMO checkpoint described by the arguments."""
    args = get_args()
    print_rank_0('building a LLaVA model for a MIMO checkpoint ...')

    language_model_config = hybrid_config_from_args(args, config)
    language_config = language_model_config.transformer
    language_config.is_hybrid_model = True
    encoder_config = vision_config(args, language_config)
    image_token_id = args.image_token_id
    if image_token_id is None:
        raise ValueError("MIMO checkpoint inference requires --image-token-id")

    return MimoCheckpointLLaVAModel(
        checkpoint_prefix_map=args.mimo_checkpoint_prefix_map,
        language_transformer_config=language_config,
        language_transformer_layer_spec=resolve_hybrid_stack_spec(language_model_config),
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=args.max_position_embeddings,
        vision_transformer_config=encoder_config,
        vision_transformer_layer_spec=get_vit_layer_with_transformer_engine_spec(),
        drop_vision_class_token=args.disable_vision_class_token,
        vision_projection_config=vision_projection_config(
            args, language_config, encoder_config.add_bias_linear
        ),
        vision_projection_layer_spec=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear
        ),
        parallel_output=parallel_output,
        share_embeddings_and_output_weights=language_model_config.share_embeddings_and_output_weights,
        language_position_embedding_type=language_model_config.position_embedding_type,
        language_rotary_percent=language_model_config.rotary_percent,
        pre_process=pre_process,
        post_process=post_process,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        img_h=args.img_h,
        img_w=args.img_w,
        patch_dim=args.patch_dim,
        language_rotary_base=language_model_config.rotary_base,
        hybrid_layer_pattern=language_model_config.hybrid_layer_pattern,
        fp16_lm_cross_entropy=language_model_config.fp16_lm_cross_entropy,
        logit_dtype=language_model_config.logit_dtype,
        image_token_index=image_token_id,
        pixel_shuffle=args.pixel_shuffle,
        conv_merging=args.conv_merging,
        dynamic_resolution=args.dynamic_resolution,
        max_num_tiles=args.max_num_tiles,
        tokenizer_type=args.tokenizer_prompt_format,
        pg_collection=pg_collection,
        vp_stage=vp_stage,
    )
