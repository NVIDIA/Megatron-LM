# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Assemble the Nemotron Omni encoder stack and provider from a config and a checkpoint."""

import json
import os
from typing import Any, Optional

import torch

from megatron.core.inference.multimodal.nemotron_omni.checkpoint import NemotronOmniWeightMapper
from megatron.core.inference.multimodal.nemotron_omni.config import NemotronOmniConfig
from megatron.core.inference.multimodal.nemotron_omni.encoder_stack import NemotronOmniEncoderStack
from megatron.core.inference.multimodal.nemotron_omni.provider import NemotronOmniEmbeddingProvider
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.radio import RADIOViTModel
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig


def load_omni_config(path: str) -> NemotronOmniConfig:
    """Read a Nemotron Omni `config.json` (file or directory) into a `NemotronOmniConfig`."""
    if os.path.isdir(path):
        path = os.path.join(path, "config.json")
    with open(path, "r", encoding="utf-8") as handle:
        return NemotronOmniConfig.from_hf(json.load(handle))


def build_vision_tower(
    config: NemotronOmniConfig,
    base_config: TransformerConfig,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> RADIOViTModel:
    """Build the RADIO tower with the Omni checkpoint's geometry.

    `class_token_len` is the trap here: it must be `num_cls_tokens + num_registers` (10 for the
    shipped checkpoint), not `RADIOViTModel`'s default of 8. A wrong value strips the wrong
    number of rows per tile and shifts every image's embeddings without raising.
    """
    vision = config.vision
    return RADIOViTModel(
        transformer_config=vision.to_transformer_config(base_config),
        transformer_layer_spec=get_vit_layer_with_transformer_engine_spec(),
        patch_dim=vision.patch_size,
        img_h=vision.preferred_resolution[0],
        img_w=vision.preferred_resolution[1],
        class_token_len=vision.class_token_len,
        add_class_token=True,
        max_img_h=vision.max_img_h,
        max_img_w=vision.max_img_w,
        has_cpe=True,
        embedder_bias=False,
        dynamic_resolution=True,
        temporal_patch_dim=vision.video_temporal_patch_size,
        separate_video_embedder=vision.separate_video_embedder,
        force_eval_mode=True,
        pg_collection=pg_collection,
    )


def build_vision_projector(
    config: NemotronOmniConfig,
    base_config: TransformerConfig,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MultimodalProjector:
    """Build the `RMSNorm -> Linear -> ReLU^2 -> Linear` vision projector.

    The leading RMSNorm is folded into fc1 via `TELayerNormColumnParallelLinear`;
    `MultimodalProjector`'s usual `ColumnParallelLinear` fc1 carries no norm, and the checkpoint
    has one (`mlp1.0.weight`).
    """
    from megatron.core.extensions.transformer_engine import (
        TELayerNormColumnParallelLinear,
        TERowParallelLinear,
    )

    input_size = config.vision.projector_input_size
    return MultimodalProjector(
        config=config.projector_transformer_config(base_config, input_size),
        submodules=MLPSubmodules(
            linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
        ),
        projector_type="mlp",
        input_size=input_size,
        pg_collection=pg_collection,
    )


def build_provider(
    config: NemotronOmniConfig,
    base_config: TransformerConfig,
    tokenizer: Any,
    checkpoint_weights: Optional[Any] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
    with_audio: bool = False,
) -> NemotronOmniEmbeddingProvider:
    """Build a ready-to-serve `NemotronOmniEmbeddingProvider`.

    Args:
        config (NemotronOmniConfig): Resolved model configuration.
        base_config (TransformerConfig): Language model config, for dtype and parallelism.
        tokenizer: Tokenizer used for placeholder expansion.
        checkpoint_weights (Optional[Any]): Iterable of `(key, tensor)` pairs from the HF
            checkpoint. When None the towers are left randomly initialized, which is only
            useful for shape and graph-safety tests.
        pg_collection (Optional[ProcessGroupCollection]): Process groups for the towers.
        with_audio (bool): Whether to build the Parakeet tower. Requires
            transformers >= 5.5.3.

    Return:
        (NemotronOmniEmbeddingProvider) Provider to pass to `DynamicInferenceEngine`.
    """
    vision_model = build_vision_tower(config, base_config, pg_collection)
    vision_projection = build_vision_projector(config, base_config, pg_collection)

    audio_model = None
    if with_audio:
        from megatron.core.models.audio.parakeet_model import ProjectedParakeet

        audio_model = ProjectedParakeet(
            config=config.sound,
            llm_hidden_size=config.hidden_size,
            projector_hidden_size=config.projector_hidden_size,
            dtype=base_config.params_dtype,
            projector_norm_eps=config.projector_norm_eps,
        )

    if checkpoint_weights is not None:
        mapper = NemotronOmniWeightMapper(config)
        state_dicts = mapper.convert(checkpoint_weights)
        vision_model.load_state_dict(state_dicts["vision_model"], strict=False)
        vision_projection.load_state_dict(state_dicts["vision_projection"], strict=False)
        if audio_model is not None:
            audio_model.load_state_dict(state_dicts["audio_model"], strict=False)

    device = torch.cuda.current_device()
    encoder_stack = NemotronOmniEncoderStack(
        config=config,
        vision_model=vision_model.to(device).eval(),
        vision_projection=vision_projection.to(device).eval(),
        audio_model=audio_model.to(device).eval() if audio_model is not None else None,
    )

    return NemotronOmniEmbeddingProvider(
        config=config, encoder_stack=encoder_stack, tokenizer=tokenizer
    )
