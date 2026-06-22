# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""RADIO vision encoder for hetero MIMO examples: wrapper, vision config, encoder spec, and args."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from typing import Optional

import torch

from megatron.core.activations import fast_gelu
from megatron.core.models.multimodal.llava_model import pixel_shuffle
from megatron.core.models.vision.radio import RADIOViTModel
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import sharded_state_dict_default

# Canonical RADIO encoder module name (shared by the provider key + topology default).
RADIO_ENCODER_MODULE_NAME = "radio_encoder"


def add_radio_encoder_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the RADIO-encoder-specific CLI args (stock owns img/patch/hidden)."""
    group = parser.add_argument_group("radio vision encoder")
    group.add_argument("--class-token-len", type=int, default=8,
                       help="Number of class tokens prepended by RADIO per tile.")
    group.add_argument("--pixel-shuffle", action="store_true",
                       help="Apply pixel shuffle to the RADIO features.")
    group.add_argument("--disable-vision-class-token", action="store_true",
                       help="Drop the RADIO class tokens from the emitted features.")
    group.add_argument("--dynamic-resolution", action="store_true",
                       help="Patchify each image at native aspect ratio with a token budget.")
    return parser


def _dtype(args: argparse.Namespace):
    """Resolve params/pipeline dtype: bf16 unless --fp32/--fp16."""
    bf16 = not getattr(args, "fp32", False) and not getattr(args, "fp16", False)
    return bf16, (torch.bfloat16 if bf16 else torch.float32)


def _base_config(args: argparse.Namespace) -> TransformerConfig:
    """Stock config from CLI args; the per-tower override helpers deepcopy this."""
    from megatron.training.argument_utils import core_transformer_config_from_args

    return core_transformer_config_from_args(args)


def _make_dense_non_hybrid(config: TransformerConfig) -> None:
    """Strip language-only MoE/Mamba/hybrid settings inherited from the base config."""
    config.num_moe_experts = None
    config.moe_ffn_hidden_size = None
    config.moe_shared_expert_intermediate_size = None
    config.moe_grouped_gemm = False
    config.moe_router_fusion = False
    config.moe_permute_fusion = False
    config.moe_shared_expert_overlap = False
    config.is_hybrid_model = False
    config.use_fused_weighted_squared_relu = False


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
    config.sequence_parallel = False
    return config


def _pixel_shuffle_dynamic_res(x, imgs_sizes, patch_dim, scale_factor=0.5, version=2):
    """Pixel shuffle for dynamic resolution (variable tile sizes).

    Splits the packed sequence by per-tile lengths, applies pixel shuffle to each
    tile, then re-concatenates. Element ordering intentionally differs from core
    ``pixel_shuffle`` (e2e-validated); do not swap to match it.
    """
    seq_lens = torch.prod(imgs_sizes // patch_dim, dim=-1)
    splits = torch.split(x, seq_lens.tolist(), dim=-2)

    out = []
    for i, sv in enumerate(splits):
        h = imgs_sizes[i][0] // patch_dim
        w = imgs_sizes[i][1] // patch_dim
        sv = sv.reshape(sv.shape[0], h, w, -1)

        n, h, w, c = sv.size()
        sv = sv.view(n, h, int(w * scale_factor), int(c / scale_factor))
        sv = sv.permute(0, 2, 1, 3).contiguous()
        sv = sv.view(
            n,
            int(w * scale_factor),
            int(h * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )

        if version == 2:
            sv = sv.permute(0, 2, 1, 3).contiguous()

        sv = sv.reshape(sv.shape[0], -1, sv.shape[-1])
        out.append(sv)

    return torch.cat(out, dim=-2)


class RADIOEncoderWrapper(MegatronModule):
    """RADIO encoder wrapper matching the Nemotron6-MoE VLM provider."""

    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        pg_collection: Optional[ProcessGroupCollection],
        img_h: int,
        img_w: int,
        patch_dim: int,
        class_token_len: int,
        drop_class_token: bool = True,
        apply_pixel_shuffle: bool = True,
        force_eval_mode: bool = False,
        dynamic_resolution: bool = False,
    ) -> None:
        super().__init__(config=transformer_config)
        self.class_token_len = class_token_len
        self.drop_class_token = drop_class_token
        self.apply_pixel_shuffle = apply_pixel_shuffle
        self.force_eval_mode = force_eval_mode
        self.dynamic_resolution = dynamic_resolution
        self.radio_model = RADIOViTModel(
            transformer_config=transformer_config,
            transformer_layer_spec=transformer_layer_spec,
            patch_dim=patch_dim,
            img_h=img_h,
            img_w=img_w,
            class_token_len=class_token_len,
            add_class_token=True,
            max_img_h=2048,
            max_img_w=2048,
            has_cpe=True,
            embedder_bias=False,
            dynamic_resolution=dynamic_resolution,
            force_eval_mode=force_eval_mode,
            pg_collection=pg_collection,
        )

    def forward(
        self,
        x: torch.Tensor,
        imgs_sizes: Optional[torch.Tensor] = None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        """Run RADIO, drop class tokens, and apply pixel shuffle."""
        context = torch.no_grad() if self.force_eval_mode else nullcontext()
        with context:
            x = x.to(dtype=self.radio_model.embedder.weight.dtype)
            embeddings = self.radio_model(
                x, imgs_sizes=imgs_sizes, packed_seq_params=packed_seq_params
            )
        if self.drop_class_token:
            if self.dynamic_resolution and imgs_sizes is not None and self.class_token_len > 0:
                # Class tokens are interleaved between tiles; build mask to remove them.
                remove_mask = torch.full(
                    (embeddings.shape[-2],), True, dtype=torch.bool, device=embeddings.device
                )
                patch_dim = self.radio_model.patch_dim
                if torch.is_tensor(imgs_sizes):
                    seq_lens = torch.prod(imgs_sizes // patch_dim, dim=-1)
                else:
                    seq_lens = torch.tensor(
                        [(h // patch_dim) * (w // patch_dim) for h, w in imgs_sizes]
                    )
                current_length = 0
                for sl in seq_lens:
                    remove_mask[current_length : current_length + self.class_token_len] = False
                    current_length += int(sl) + self.class_token_len
                embeddings = embeddings[:, remove_mask, :]
            else:
                embeddings = embeddings[:, self.class_token_len :, :]
        if self.apply_pixel_shuffle:
            if self.dynamic_resolution and imgs_sizes is not None:
                embeddings = _pixel_shuffle_dynamic_res(
                    embeddings, imgs_sizes, self.radio_model.patch_dim
                )
            else:
                embeddings = pixel_shuffle(embeddings, scale_factor=0.5)
        return embeddings

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        # Param-less wrapper: delegate straight to the child so checkpoint keys keep
        # the ``radio_model.`` prefix without the base-class tp/dp_cp_group machinery.
        sharded_sd = {}
        for name, child in self.named_children():
            sharded_sd.update(
                sharded_state_dict_default(child, f"{prefix}{name}.", sharded_offsets, metadata)
            )
        return sharded_sd


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
