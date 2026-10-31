# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Model provider for MIMO model with vision encoder.

This module provides a model provider function to create a MIMO model
with language model, vision encoder, and projection components.
"""



from examples.mimo.configs.mock import (
    get_mock_language_layer_spec,
    get_mock_language_model_config,
    get_mock_projection_config,
    get_mock_projection_layer_spec,
    get_mock_vision_layer_spec,
    get_mock_vision_model_config,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo import MimoModel, MimoModelConfig
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.process_groups_config import (
    ProcessGroupCollection,
    global_process_group_fallback,
)
from megatron.core.transformer.spec_utils import ModuleSpec


def model_provider_mock_vlm_single_encoder(
    pre_process: bool = True,
    post_process: bool = True,
    add_encoder=True,
    add_decoder=True,
    special_token_id: int = 32000,
    pg_collection=None,
):
    """
    Build a MIMO model with a vision encoder.
    """
    if pg_collection is None:
        # Callers that omit the process groups get those of the global parallel grid; that
        # fallback is deprecated.
        with global_process_group_fallback(
            "examples.mimo.model_providers.mock.model_provider_mock_vlm_single_encoder",
            deprecated_in="0.21",
            removed_in="0.23",
        ):
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    # PP not supported, so add_encoder/add_decoder are ignored
    # Get configs for each component
    vision_config = get_mock_vision_model_config()
    language_config = get_mock_language_model_config()

    # Create encoder config for vision
    vision_encoder = ModuleSpec(
        module=CLIPViTModel,
        params={
            "transformer_config": vision_config,
            "transformer_layer_spec": get_mock_vision_layer_spec(),
            "patch_dim": 16,
            "img_h": 224,
            "img_w": 224,
            "pg_collection": pg_collection,
        },
    )

    # Create projection config for vision to language
    vision_projection = ModuleSpec(
        module=MultimodalProjector,
        params={
            "config": get_mock_projection_config(),
            "submodules": get_mock_projection_layer_spec().submodules,
            "projector_type": "mlp",
            "input_size": 128,
            "pg_collection": pg_collection,
        },
    )

    # Create modality config for vision
    vision_submodule_spec = ModuleSpec(
        module=VisionModalitySubmodules,
        params={"pg_collection": pg_collection},
        submodules={
            "encoders": {'clip_encoder': vision_encoder},
            "input_projections": [vision_projection],
        }
    )

    # Create language model config
    language_model_spec  = ModuleSpec(
        module=GPTModel,
        params={
            "config": language_config,
            "transformer_layer_spec": get_mock_language_layer_spec(),
            "vocab_size": 50304,
            "max_sequence_length": 2048,
            "pre_process": pre_process,
            "post_process": post_process,
            "pg_collection": pg_collection,
        },
    )

    # Create MIMO model config
    mimo_model_config = MimoModelConfig(
        language_model_spec=language_model_spec,
        modality_submodules_spec={"images": vision_submodule_spec},
        special_token_ids={"images": special_token_id}
    )

    # Create MIMO model
    mimo_model = MimoModel(
        mimo_model_config,
        cp_group=pg_collection.cp,
        tp_group=pg_collection.tp,
        tp_cp_group=pg_collection.tp_cp,
    )
    # Wrappers such as Float16Module and the training loop read the groups from the model.
    mimo_model.pg_collection = pg_collection

    return mimo_model
