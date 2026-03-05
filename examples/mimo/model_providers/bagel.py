# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Model provider for a LLaVA-style Vision-Language Model.

This provider assembles a MIMO model that consists of:
• Vicuna-7B language model (Llama-based) built with Transformer-Engine GPT blocks.
• CLIP ViT-L/14 visual encoder (336 px) that produces image patch embeddings.
• A 2-layer MLP projector that maps vision embeddings into Vicuna hidden size.
"""

import os
import torch
from configs.bagel import (
    get_bagel_projection_config,
    get_bagel_projection_layer_spec,
    get_bagel_language_layer_spec,
    get_bagel_language_model_config,
)

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_src_rank,
)

from examples.mimo.diffusion.diffusion_modality_submodule import DiffusionModalitySubmodules
from examples.mimo.diffusion.embeddings import TimestepEmbedder, PositionEmbedding

from examples.mimo.model_providers.hf_bagel_vision_encoder import HFBagelVisionEncoderWrapper
from examples.mimo.model_providers.hf_bagel_llm import BagelLLMHuggingFaceModel
from examples.mimo.model_providers.mcore_bagel_llm import BagelMCoreModel
from examples.mimo.utils.logging import print_mimo_structure
from examples.mimo.utils.model_helpers import load_submodule_ckpt, load_submodule_ckpt_for_mot
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo import MimoModel, MimoModelConfig
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.transformer.spec_utils import ModuleSpec

from bagel.modeling.bagel import Qwen2Config, SiglipVisionConfig, BagelConfig

from examples.mimo.diffusion.diffusion_wrapper import build_diffusion_wrapper


def model_provider_bagel(
    pre_process: bool = True,
    post_process: bool = True,
    add_encoder=True,
    add_decoder=True,
    image_special_token_id: int = 32000,
    is_video_input: bool = False,
    freeze_vit: bool = False,
    freeze_llm: bool = False,
    model_path: str = None,
    llm_path: str = None,
    vit_path: str = None,
    language_use_mcore: bool = False,  # Whether to use Megatron Core GPTModel
    decoder_layer_module: str = "Qwen2DecoderLayer",
    use_moe_mlp: bool = True,
):
    """
    Build a LLaVA-style Vision-Language MIMO model composed of:
    • Qwen2 language model.
    • Bagel vision encoder.
    • 2-layer MLP vision→language projector.

    Args:
        pre_process: Whether to pre-process the model
        post_process: Whether to post-process the model
        add_encoder: Whether to add an encoder (not used)
        add_decoder: Whether to add a decoder (not used)
        image_special_token_id: Special token ID for images
        is_video_input: Whether input is video
        freeze_vit: Whether to freeze ViT parameters
        freeze_llm: Whether to freeze LLM parameters
        model_path: Path to model checkpoints
        language_use_mcore: If True, use Megatron Core BagelMCoreModel;
                           If False, use HuggingFace BagelLLMHuggingFaceModel
    """
    # NOTE: Pipeline parallelism for the encoder/decoder is not yet supported in this
    # MIMO path, therefore *add_encoder* and *add_decoder* are currently ignored.
    finetune_from_hf = False
    if finetune_from_hf:
        llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
        vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    else:
        print("llm_path", llm_path, "vit_path", vit_path)
        llm_config = Qwen2Config.from_pretrained(llm_path)
        vit_config = SiglipVisionConfig.from_pretrained(vit_path)
    llm_config.layer_module = decoder_layer_module
    # vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
    vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 - 2
    vit_config.rope = False
    bagel_config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=None,
    )

    # Language
    language_config = get_bagel_language_model_config(hf_config=llm_config, use_moe_mlp=use_moe_mlp)
    print("language_config.hidden_size", language_config.hidden_size)

    # Vision→language projection MLP – hidden size follows Qwen2
    projection_config = get_bagel_projection_config(
        hidden_size=language_config.hidden_size,
        ffn_hidden_size=language_config.hidden_size,
    )


    # Sync precision flags from global args (if we're running under Megatron training loop)
    dtype = torch.float32
    try:
        from megatron.training import get_args  # late import to avoid circular deps

        _args = get_args()
        if getattr(_args, "bf16", False):
            language_config.bf16 = True
            projection_config.bf16 = True
            dtype = torch.bfloat16
        if getattr(_args, "fp16", False):
            language_config.fp16 = True
            projection_config.fp16 = True
            dtype = torch.float16
    except (ModuleNotFoundError, AssertionError):
        pass

    #we build diffusion wrapper here for bagel model needs latent channels. 
    #but actually we don't want vae on all devices, so we will remove diffusion wrapper on the devices which 
    #has no visual data.
    _args.diffusion_wrapper = build_diffusion_wrapper()
    latent_channels = _args.diffusion_wrapper.vae_params.z_channels
    if get_tensor_model_parallel_rank() != 0:
        _args.diffusion_wrapper = None

    ### Diffusion modality submodule
    timestep_embedder = ModuleSpec(
        module=TimestepEmbedder,
        params={
            "hidden_size": language_config.hidden_size,
        },
    )
    pos_embedder = ModuleSpec(
        module=PositionEmbedding,
        params={
            "max_num_patch_per_side": _args.max_latent_size, 
            "hidden_size": language_config.hidden_size,
        }
        
    )
    
    #projector for diffusion
    vae2llm = ModuleSpec(
        module=torch.nn.Linear,
        params={
            "in_features": _args.latent_patch_size**2 * latent_channels,
            "out_features": language_config.hidden_size,
            "dtype": dtype,
        }
    )

    llm2vae = ModuleSpec(
        module=torch.nn.Linear,
        params={
            "in_features": language_config.hidden_size,
            "out_features": _args.latent_patch_size**2 * latent_channels,
            "dtype": dtype,
        }
    )

    # Create modality config for vision
    diffusion_submodule_spec = ModuleSpec(
        module=DiffusionModalitySubmodules,
        params={'timestep_shift': _args.timestep_shift, 'dtype': dtype},
        submodules={
            "encoders": {
                         "timestep": timestep_embedder, 
                         "latent_position_ids": pos_embedder
                         },
            "input_projections": [vae2llm],
            "output_projections": [llm2vae],
        },
    )

    # HF encoder
    vision_encoder = ModuleSpec(
        module=HFBagelVisionEncoderWrapper,
        params={"bagel_config": bagel_config, "vit_path": vit_path, "dtype": dtype},
    )

    # Create projection config for vision to language
    vision_projection = ModuleSpec(
        module=MultimodalProjector,
        params={
            "config": projection_config,
            "submodules": get_bagel_projection_layer_spec().submodules,
            "projector_type": "mlp",
            "input_size": vit_config.hidden_size,  # vision hidden size
        },
    )

    # Create modality config for vision
    vision_submodule_spec = ModuleSpec(
        module=VisionModalitySubmodules,
        params={},
        submodules={
            "encoders": {"vision_encoder": vision_encoder},
            "input_projections": [vision_projection],
        },
    )

    # Create language model config (Qwen2-based)
    use_mo = "Mo" in llm_config.layer_module if hasattr(llm_config, 'layer_module') else False
    if language_use_mcore:
        print(f"use_flex_attention={_args.use_flex_attention}")
        if use_mo:
            from megatron.core.transformer.transformer_mot_block import get_mot_layer_spec
            transformer_layer_spec = get_mot_layer_spec(num_experts=language_config.num_moe_experts, 
                                                        moe_grouped_gemm=language_config.moe_grouped_gemm, 
                                                        use_flex_attention=_args.use_flex_attention, 
                                                        qk_layernorm=True
                                                        )
        else:
            transformer_layer_spec = get_bagel_language_layer_spec(num_experts=language_config.num_moe_experts, 
                                                                   moe_grouped_gemm=language_config.moe_grouped_gemm, 
                                                                   use_flex_attention=_args.use_flex_attention
                                                                   )


        # Use Megatron Core based BagelMCoreModel
        language_model_spec = ModuleSpec(
            module=BagelMCoreModel,
            params={
                "config": language_config,
                "transformer_layer_spec": transformer_layer_spec,
                "vocab_size": 151936,  # Qwen2 vocab size
                "max_sequence_length": 32768,  # Qwen2 max sequence length
                "pre_process": pre_process,
                "post_process": post_process,
                "position_embedding_type": "rope",
                "rotary_base": 1000000,  # Qwen2 RoPE base
                "llm_config": llm_config,  # Bagel-specific config
                "use_flex_attention": _args.use_flex_attention,  # Pass flex attention flag
            },
        )
    else:
        # Use HuggingFace based BagelLLMHuggingFaceModel
        language_model_spec = ModuleSpec(
            module=BagelLLMHuggingFaceModel,
            params={"config": language_config, "llm_config": llm_config, "llm_path": llm_path},
        )

    # Create MIMO model config
    mimo_model_config = MimoModelConfig(
        language_model_spec=language_model_spec,
        modality_submodules_spec={"images": vision_submodule_spec,
                                  "diffusion": diffusion_submodule_spec},
        special_token_ids={"images": image_special_token_id,
                           }
    )

    # Create MIMO model
    mimo_model = MimoModel(mimo_model_config)
    print("*" * 100)
    print(f"Using {'Megatron Core BagelMCoreModel' if language_use_mcore else 'HuggingFace BagelLLMHuggingFaceModel'}")
    print_mimo_structure(mimo_model)
    print("*" * 100)

    # load the checkpoint
    try:
        from megatron.training import get_args  # late import to avoid circular deps

        _args = get_args()
        if _args.language_model_checkpoint is not None:
            print(f"[model_provider_bagel] Attempting to load checkpoint from {_args.language_model_checkpoint}")
            try:
                if use_mo:
                    # For MoT models, use the special loader that handles _gen parameters
                    load_submodule_ckpt_for_mot(
                        mimo_model.language_model,
                        _args.language_model_checkpoint,
                        init_gen_from_und=True
                    )
                    print(f"Successfully loaded checkpoint (MoT mode) from {_args.language_model_checkpoint}")
                else:
                    load_submodule_ckpt(mimo_model.language_model, _args.language_model_checkpoint)
                    print(f"Successfully loaded checkpoint from {_args.language_model_checkpoint}")
            except Exception as e:
                print(f"[model_provider_bagel] WARNING: Failed to load checkpoint: {e}")
                print("[model_provider_bagel] Continuing with randomly initialized weights.")
    except (ModuleNotFoundError, AssertionError):
        pass

    # TODO: ykarnati make these configurable and have an API to freeze/unfreeze
    # freeze vision encoder and LLM parameters
    modules_to_freeze = []
    if freeze_vit:
        modules_to_freeze.append(mimo_model.modality_submodules.images.encoders.vision_encoder)
    if freeze_llm:
        modules_to_freeze.append(mimo_model.language_model)
    for module in modules_to_freeze:
        for param in module.parameters():
            param.requires_grad = False

    return mimo_model
