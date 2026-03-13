# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Model provider for a Qwen3.5-397B-A17B style Vision-Language Model.

This provider assembles a MIMO model that consists of:
  - Qwen3.5 vision encoder (ViT + PatchMerger) from HuggingFace, outputting at 4096-dim.
  - Qwen3-Next language decoder with MoE (512 experts, top-10) and hybrid GDN/full attention.
  - No separate projection MLP — the vision encoder's PatchMerger already projects
    to out_hidden_size=4096 which matches the decoder hidden_size.

Architecture reference:
  https://huggingface.co/Qwen/Qwen3.5-397B-A17B
"""

import torch
from configs.qwen35_vlm import get_qwen35_language_layer_spec, get_qwen35_language_model_config

from examples.mimo.model_providers.hf_qwen35_vision_encoder import HFQwen35VisionEncoderWrapper
from examples.mimo.utils.logging import print_mimo_structure
from examples.mimo.utils.model_helpers import load_submodule_ckpt
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo import MimoModel, MimoModelConfig
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.transformer.spec_utils import ModuleSpec

# Qwen3.5-397B-A17B special token IDs from HuggingFace config
QWEN35_IMAGE_TOKEN_ID = 248056
QWEN35_VIDEO_TOKEN_ID = 248057
QWEN35_VISION_START_TOKEN_ID = 248053
QWEN35_VISION_END_TOKEN_ID = 248054
QWEN35_VOCAB_SIZE = 248320


def _apply_args_to_language_config(cfg):
    """Selectively override language config fields from megatron command-line args.

    Only the fields that correspond to actual command-line arguments are
    overridden; everything else retains the 397B defaults from
    get_qwen35_language_model_config().  Returns the megatron args namespace
    (or None if unavailable).
    """
    try:
        from megatron.training import get_args

        args = get_args()
    except (ModuleNotFoundError, AssertionError):
        return None

    # Core architecture
    cfg.num_layers = args.num_layers
    cfg.hidden_size = args.hidden_size
    cfg.num_attention_heads = args.num_attention_heads
    cfg.ffn_hidden_size = args.ffn_hidden_size
    cfg.kv_channels = getattr(args, "kv_channels", None) or (
        args.hidden_size // args.num_attention_heads
    )
    cfg.num_query_groups = getattr(args, "num_query_groups", None) or args.num_attention_heads
    cfg.seq_length = args.seq_length
    cfg.max_position_embeddings = args.max_position_embeddings

    # MoE
    if getattr(args, "num_experts", None) is not None:
        cfg.num_moe_experts = args.num_experts
    if getattr(args, "moe_ffn_hidden_size", None) is not None:
        cfg.moe_ffn_hidden_size = args.moe_ffn_hidden_size
    if getattr(args, "moe_shared_expert_intermediate_size", None) is not None:
        cfg.moe_shared_expert_intermediate_size = args.moe_shared_expert_intermediate_size
    if getattr(args, "moe_router_topk", None) is not None:
        cfg.moe_router_topk = args.moe_router_topk

    # Gated Delta Net
    if getattr(args, "linear_attention_freq", None) is not None:
        cfg.linear_attention_freq = args.linear_attention_freq
    if getattr(args, "linear_key_head_dim", None) is not None:
        cfg.linear_key_head_dim = args.linear_key_head_dim
    if getattr(args, "linear_value_head_dim", None) is not None:
        cfg.linear_value_head_dim = args.linear_value_head_dim
    if getattr(args, "linear_num_key_heads", None) is not None:
        cfg.linear_num_key_heads = args.linear_num_key_heads
    if getattr(args, "linear_num_value_heads", None) is not None:
        cfg.linear_num_value_heads = args.linear_num_value_heads

    # Parallelism (from ModelParallelConfig, parent of TransformerConfig)
    cfg.tensor_model_parallel_size = getattr(args, "tensor_model_parallel_size", 1)
    cfg.pipeline_model_parallel_size = getattr(args, "pipeline_model_parallel_size", 1)
    cfg.expert_model_parallel_size = getattr(args, "expert_model_parallel_size", 1)
    cfg.sequence_parallel = getattr(args, "sequence_parallel", False)
    cfg.context_parallel_size = getattr(args, "context_parallel_size", 1)

    # Precision
    if getattr(args, "bf16", False):
        cfg.bf16 = True
    if getattr(args, "fp16", False):
        cfg.fp16 = True

    return args


def model_provider_qwen35_vlm(
    pre_process: bool = True,
    post_process: bool = True,
    add_encoder: bool = True,
    add_decoder: bool = True,
    image_special_token_id: int = QWEN35_IMAGE_TOKEN_ID,
):
    """Build a Qwen3.5-397B-A17B style Vision-Language MIMO model.

    Components:
      - HuggingFace Qwen3.5 vision encoder (frozen, outputs at decoder hidden_size)
      - Qwen3-Next MoE decoder (hybrid GDN/full attention)

    The decoder architecture is controlled by megatron command-line args
    (--num-layers, --hidden-size, --num-experts, etc.).  The full 397B defaults
    in get_qwen35_language_model_config are overridden accordingly.
    """

    language_config = get_qwen35_language_model_config()
    _apply_args_to_language_config(language_config)

    # HF Qwen3.5 vision encoder — includes PatchMerger, outputs at out_hidden_size=4096
    vision_encoder = ModuleSpec(
        module=HFQwen35VisionEncoderWrapper,
        params={
            "pretrained_model_name": "Qwen/Qwen3.5-397B-A17B",
            "load_pretrained_weights": False,
        },
    )

    # Vision modality submodules — no separate projection needed since the
    # PatchMerger already maps to the decoder's hidden_size (4096).
    vision_submodule_spec = ModuleSpec(
        module=VisionModalitySubmodules,
        params={},
        submodules={
            "encoders": {"qwen35_vision": vision_encoder},
            "input_projections": [],
        },
    )

    # Language model (Qwen3-Next decoder with MoE + GDN)
    layer_spec = get_qwen35_language_layer_spec(language_config)

    language_model_spec = ModuleSpec(
        module=GPTModel,
        params={
            "config": language_config,
            "transformer_layer_spec": layer_spec,
            "vocab_size": QWEN35_VOCAB_SIZE,
            "max_sequence_length": language_config.seq_length,
            "pre_process": pre_process,
            "post_process": post_process,
            "position_embedding_type": "rope",
        },
    )

    # Assemble MIMO model
    mimo_model_config = MimoModelConfig(
        language_model_spec=language_model_spec,
        modality_submodules_spec={"images": vision_submodule_spec},
        special_token_ids={"images": image_special_token_id},
    )

    mimo_model = MimoModel(mimo_model_config)
    print("*" * 100)
    print_mimo_structure(mimo_model)
    print("*" * 100)

    # Optionally load a pretrained language model checkpoint
    try:
        from megatron.training import get_args

        _args = get_args()
        if _args.language_model_checkpoint is not None:
            load_submodule_ckpt(mimo_model.language_model, _args.language_model_checkpoint)
            print(
                f"Successfully loaded language model checkpoint from {_args.language_model_checkpoint}"
            )
    except (ModuleNotFoundError, AssertionError):
        pass

    # Freeze vision encoder (the decoder is trainable by default)
    for param in mimo_model.modality_submodules.images.encoders.qwen35_vision.parameters():
        param.requires_grad = False

    return mimo_model
