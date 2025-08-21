# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Model provider for a LLaVA-style Vision-Language Model.

This provider assembles a MIMO model that consists of:
• Vicuna-7B language model (Llama-based) built with Transformer-Engine GPT blocks.
• CLIP ViT-L/14 visual encoder (336 px) that produces image patch embeddings.
• A 2-layer MLP projector that maps vision embeddings into Vicuna hidden size.
"""


import torch
from configs.llava_avlm import (
    get_llava_projection_config,
    get_llava_projection_layer_spec,
    get_vicuna_language_layer_spec,
    get_vicuna_language_model_config,
)

from examples.mimo.model_providers.hf_clip_encoder import HFCLIPEncoderWrapper
from examples.mimo.model_providers.hf_whisper_encoder import HFWhisperEncoderWrapper
from examples.mimo.utils.logging import print_mimo_structure
from examples.mimo.utils.model_helpers import load_submodule_ckpt
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo import MimoModel, MimoModelConfig
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.mimo.submodules.audio import AudioModalitySubmodules
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.transformer.spec_utils import ModuleSpec


def model_provider_llava_avlm(
    pre_process: bool = True,
    post_process: bool = True,
    add_encoder=True,
    add_decoder=True,
    image_special_token_id: int = 32000,
    audio_special_token_id: int = 32002,
):
    """
    Build a LLaVA-style Audio-Vision-Language MIMO model composed of:
    • Vicuna language model.
    • Whisper audio encoder.
    • CLIP ViT-L/14 vision encoder.
    • 2-layer MLP vision→language projector.
    • 2-layer MLP audio→language projector.
    """
    # NOTE: Pipeline parallelism for the encoder/decoder is not yet supported in this
    # MIMO path, therefore *add_encoder* and *add_decoder* are currently ignored.


    # Language (Vicuna-7B)
    language_config = get_vicuna_language_model_config()


    # Vision→language and audio→language projection MLP – hidden size follows Vicuna (4096)
    vision_projection_config = get_llava_projection_config(
        hidden_size=language_config.hidden_size
    )
    audio_projection_config = get_llava_projection_config(
        hidden_size=language_config.hidden_size
    )


    # Sync precision flags from global args (if we're running under Megatron training loop)
    try:
        from megatron.training import get_args  # late import to avoid circular deps
        _args = get_args()
        if getattr(_args, "bf16", False):
            language_config.bf16 = True
            vision_projection_config.bf16 = True
            audio_projection_config.bf16 = True
        if getattr(_args, "fp16", False):
            language_config.fp16 = True
            vision_projection_config.fp16 = True
            audio_projection_config.fp16 = True
    except (ModuleNotFoundError, AssertionError):
        pass


    # HF vision encoder
    vision_encoder_params = {"is_video_input" : False}
    vision_encoder = ModuleSpec(
        module=HFCLIPEncoderWrapper,
        params=vision_encoder_params,
    )
    # HF audio encoder
    audio_encoder_params = {"model_name" : "openai/whisper-base"}
    audio_encoder = ModuleSpec(
        module=HFWhisperEncoderWrapper,
        params=audio_encoder_params,
    )


    # Create projection config for vision and audio to language
    vision_projection = ModuleSpec(
        module=MultimodalProjector,
        params={
            "config": vision_projection_config,
            "submodules": get_llava_projection_layer_spec().submodules,
            "projector_type": "mlp",
            "input_size": 1024,
        },
    )
    audio_projection = ModuleSpec(
        module=MultimodalProjector,
        params={
            "config": audio_projection_config,
            "submodules": get_llava_projection_layer_spec().submodules,
            "projector_type": "mlp",
            "input_size": 512,
        },
    )


    # Create modality config for vision and audio
    vision_submodule_spec = ModuleSpec(
        module=VisionModalitySubmodules,
        params={},
        submodules={
            "encoders": {"clip_encoder": vision_encoder},
            "input_projections": [vision_projection],
        },
    )
    audio_submodule_spec = ModuleSpec(
        module=AudioModalitySubmodules,
        params={},
        submodules={
            "encoders": {"whisper_encoder": audio_encoder},
            "input_projections": [audio_projection],
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
        modality_submodules_spec={"images": vision_submodule_spec, "audios": audio_submodule_spec},
        special_token_ids={"images": image_special_token_id, "audios": audio_special_token_id}
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
    modules_to_freeze = [
        mimo_model.modality_submodules.images.encoders.clip_encoder,
        mimo_model.modality_submodules.audios.encoders.whisper_encoder,
        mimo_model.language_model
    ]
    for module in modules_to_freeze:
        for param in module.parameters():
            param.requires_grad = False

    return mimo_model