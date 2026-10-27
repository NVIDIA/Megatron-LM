# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""RADIO vision encoder for hetero MIMO examples: wrapper, vision config, encoder spec, and args."""

from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Optional

import torch

from megatron.core.activations import fast_gelu

# Keep the example import paths compatible for downstream providers.
from megatron.core.models.mimo.submodules.radio_encoder import (
    RADIO_ENCODER_MODULE_NAME,
    RADIOEncoderWrapper,
    _pixel_shuffle_dynamic_res,
)
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


def add_radio_encoder_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the RADIO-encoder-specific CLI args (stock owns img/patch/hidden)."""
    group = parser.add_argument_group("radio vision encoder")
    group.add_argument(
        "--class-token-len",
        type=int,
        default=8,
        help="Number of class tokens prepended by RADIO per tile.",
    )
    group.add_argument(
        "--pixel-shuffle", action="store_true", help="Apply pixel shuffle to the RADIO features."
    )
    group.add_argument(
        "--disable-vision-class-token",
        action="store_true",
        help="Drop the RADIO class tokens from the emitted features.",
    )
    group.add_argument(
        "--dynamic-resolution",
        action="store_true",
        help="Patchify each image at native aspect ratio with a token budget.",
    )
    return parser


def _dtype(args: argparse.Namespace):
    """Resolve params/pipeline dtype from stock Megatron precision args."""
    dtype = getattr(args, "params_dtype", None)
    if dtype is None:
        if getattr(args, "bf16", False):
            dtype = torch.bfloat16
        elif getattr(args, "fp16", False):
            dtype = torch.float16
        else:
            dtype = torch.float32
    return bool(getattr(args, "bf16", False)), dtype


def _base_config(args: argparse.Namespace) -> TransformerConfig:
    """Stock config from CLI args; the per-tower override helpers deepcopy this."""
    from megatron.training.argument_utils import core_transformer_config_from_args

    return core_transformer_config_from_args(args)


def _make_dense_non_hybrid(config: TransformerConfig) -> None:
    """Strip language-only architecture and precision settings from the base config."""
    config.activation_func_tanh_clamp_scale = None
    config.activation_func_tanh_clamp_scale_linear = None
    # FP32 residual accumulation is a language-model policy. Preserve the pretrained
    # modality math and restore the communication dtype promoted by TransformerConfig.
    config.fp32_residual_connection = False
    config.pipeline_dtype = config.params_dtype
    config.num_moe_experts = None
    config.moe_ffn_hidden_size = None
    config.moe_shared_expert_intermediate_size = None
    config.moe_grouped_gemm = False
    config.moe_router_fusion = False
    config.moe_permute_fusion = False
    config.moe_shared_expert_overlap = False
    config.moe_shortcut_connection = False
    config.moe_shortcut_parallel = False
    config.moe_shortcut_post_norm = False
    config.is_hybrid_model = False
    config.use_fused_weighted_squared_relu = False
    if config.recompute_modules is not None:
        config.recompute_modules = [
            module for module in config.recompute_modules if module != "shortcut_pre_mlp_layernorm"
        ]
    if getattr(config, "offload_modules", None) is not None:
        config.offload_modules = [
            module for module in config.offload_modules if module != "shortcut_post_norm"
        ]
    config.wide_residual = None
    config.residual_stream_recompute_num_layers = None
    if config.recompute_modules is not None:
        config.recompute_modules = [
            module for module in config.recompute_modules if module != "residual_stream"
        ]


def _disable_gtp(config: TransformerConfig) -> None:
    """Keep this module replicated across any LLM GTP axes."""
    config.tensor_parallel_num_weight_shards = config.tensor_model_parallel_size
    config.gtp_weight_remat_size = 1
    config.tensor_parallel_num_sequence_shards = None
    expert_tp = config.expert_tensor_parallel_size or config.tensor_model_parallel_size
    config.expert_tensor_parallel_num_weight_shards = expert_tp
    config.expert_gtp_weight_remat_size = 1


def radio_vision_config(args: argparse.Namespace, tp_size: int, pp_size: int) -> TransformerConfig:
    """RADIO vision config: stock from-args base + RADIO-specific overrides."""
    config = deepcopy(_base_config(args))
    bf16, dtype = _dtype(args)
    config.num_layers = 32
    config.hidden_size = 1280
    config.num_attention_heads = 16
    config.kv_channels = 80
    config.num_query_groups = 16
    config.ffn_hidden_size = 5120
    config.gated_linear_unit = False
    config.activation_func = fast_gelu
    config.add_bias_linear = True
    config.add_qkv_bias = True
    config.normalization = "LayerNorm"
    config.layernorm_epsilon = 1.0e-6
    config.layernorm_zero_centered_gamma = False
    config.apply_rope_fusion = False
    config.qk_layernorm = False
    config.bias_activation_fusion = False
    config.bias_dropout_fusion = False
    config.attention_softmax_in_fp32 = True
    config.attention_dropout = 0.0
    config.hidden_dropout = 0.0
    config.mtp_num_layers = 0  # Trigger TransformerBlock's final_layernorm allocation.
    _make_dense_non_hybrid(config)  # ViT inherits no MoE/Mamba/hybrid settings.
    config.params_dtype = dtype
    config.pipeline_dtype = dtype
    config.bf16 = bf16
    config.tensor_model_parallel_size = tp_size
    config.pipeline_model_parallel_size = pp_size
    config.context_parallel_size = 1
    _disable_gtp(config)
    config.sequence_parallel = False
    return config


def radio_vision_encoder_spec(
    args: argparse.Namespace,
    vision_config: TransformerConfig,
    pg_collection: Optional[ProcessGroupCollection],
) -> ModuleSpec:
    """Build the RADIO encoder ``ModuleSpec``, reading the RADIO knobs off ``args``."""
    return ModuleSpec(
        module=RADIOEncoderWrapper,
        params={
            "transformer_config": vision_config,
            "transformer_layer_spec": get_vit_layer_with_transformer_engine_spec(),
            "pg_collection": pg_collection,
            "img_h": args.img_h,
            "img_w": args.img_w,
            "patch_dim": args.patch_dim,
            "class_token_len": args.class_token_len,
            "drop_class_token": args.disable_vision_class_token,
            "apply_pixel_shuffle": args.pixel_shuffle,
            "force_eval_mode": args.freeze_vit,
            "dynamic_resolution": bool(getattr(args, "dynamic_resolution", False)),
        },
    )
