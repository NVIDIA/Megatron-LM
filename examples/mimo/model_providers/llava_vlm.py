# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Model provider for a LLaVA-style Vision-Language Model.

This provider assembles a MIMO model that consists of:
• Vicuna-7B language model (Llama-based) built with Transformer-Engine GPT blocks.
• CLIP ViT-L/14 visual encoder (336 px) that produces image patch embeddings.
• A 2-layer MLP projector that maps vision embeddings into Vicuna hidden size.
"""


import torch
from configs.llava_vlm import (
    get_llava_projection_config,
    get_llava_projection_layer_spec,
    get_vicuna_language_layer_spec,
    get_vicuna_language_model_config,
)

from examples.mimo.model_providers.hf_clip_encoder import HFCLIPEncoderWrapper
from examples.mimo.utils.logging import print_mimo_structure
from examples.mimo.utils.model_helpers import load_submodule_ckpt
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo import MimoModel, MimoModelConfig
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.transformer.spec_utils import ModuleSpec


def model_provider_llava_vlm(
    pre_process: bool = True,
    post_process: bool = True,
    add_encoder=True,
    add_decoder=True,
    image_special_token_id: int = 32000,
    is_video_input: bool = False
):
    """
    Build a LLaVA-style Vision-Language MIMO model composed of:
    • Vicuna language model.
    • CLIP ViT-L/14 vision encoder.
    • 2-layer MLP vision→language projector.
    """
    # NOTE: Pipeline parallelism for the encoder/decoder is not yet supported in this
    # MIMO path, therefore *add_encoder* and *add_decoder* are currently ignored.

    # Language (Vicuna-7B)
    language_config = get_vicuna_language_model_config()

    # Vision→language projection MLP – hidden size follows Vicuna (4096)
    projection_config = get_llava_projection_config(
        hidden_size=language_config.hidden_size
    )

    # Sync precision flags from global args (if we're running under Megatron training loop)
    try:
        from megatron.training import get_args  # late import to avoid circular deps

        _args = get_args()
        if getattr(_args, "bf16", False):
            language_config.bf16 = True
            projection_config.bf16 = True
        if getattr(_args, "fp16", False):
            language_config.fp16 = True
            projection_config.fp16 = True
    except (ModuleNotFoundError, AssertionError):
        pass

    # HF encoder
    vision_encoder = ModuleSpec(
        module=HFCLIPEncoderWrapper,
        params={"is_video_input" : is_video_input},
    )

    # Create projection config for vision to language
    vision_projection = ModuleSpec(
        module=MultimodalProjector,
        params={
            "config": projection_config,
            "submodules": get_llava_projection_layer_spec().submodules,
            "projector_type": "mlp",
            "input_size": 1024,
        },
    )

    # Create modality config for vision
    vision_submodule_spec = ModuleSpec(
        module=VisionModalitySubmodules,
        params={},
        submodules={
            "encoders": {"clip_encoder": vision_encoder},
            "input_projections": [vision_projection],
        },
    )

    # Create language model config
    language_model_spec = ModuleSpec(
        module=GPTModel,
        params={
            "config": language_config,
            "transformer_layer_spec": get_vicuna_language_layer_spec(),
            "vocab_size": 32256,
            "max_sequence_length": 4096,
            "pre_process": pre_process,
            "post_process": post_process,
            "position_embedding_type": "rope",
        },
    )

    # Create MIMO model config
    mimo_model_config = MimoModelConfig(
        language_model_spec=language_model_spec,
        modality_submodules_spec={"images": vision_submodule_spec},
        special_token_ids={"images": image_special_token_id}
    )

    # Create MIMO model
    mimo_model = MimoModel(mimo_model_config)
    print("*"*100)
    print_mimo_structure(mimo_model)
    print("*"*100)

    # load the checkpoint
    try:
        from megatron.training import get_args  # late import to avoid circular deps

        _args = get_args()
        if  _args.language_model_checkpoint is not None:
            load_submodule_ckpt(mimo_model.language_model, _args.language_model_checkpoint)
            print(f"Successfully loaded LLaVA pretrained checkpoint from {_args.language_model_checkpoint}")
    except (ModuleNotFoundError, AssertionError):
        pass

    # TODO: ykarnati make these configurable and have an API to freeze/unfreeze   
    # freeze vision encoder and LLM parameters
    modules_to_freeze = [mimo_model.modality_submodules.images.encoders.clip_encoder, mimo_model.language_model]
    for module in modules_to_freeze:
        for param in module.parameters():
            param.requires_grad = False

    return mimo_model