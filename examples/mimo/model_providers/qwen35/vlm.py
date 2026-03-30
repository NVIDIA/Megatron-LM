# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Model provider for Qwen3.5-VL Vision-Language MIMO model.

Assembles a ``MimoModel`` using components from the local
``examples/mimo/model_providers/qwen35/`` package (a self-contained
duplicate of the qwen35_vl model code).  This file contains only
MIMO-specific assembly logic (PP flags, arg overrides, checkpoint
loading, encoder freezing).

Supported variants (via ``--model-variant`` CLI arg):
  ``proxy``       4 layers, 16 experts — single-node bring-up
  ``397b_a17b``   60 layers, 512 experts — production (default)
  ``9b``          Dense 32-layer 9B model
  ``35b_a3b``     MoE 35B-A3B model
  ``35b_a3b_light``  Reduced 35B-A3B for single-node testing
"""

from typing import Any, Dict, Optional

import torch

from examples.mimo.utils.logging import print_mimo_structure
from examples.mimo.utils.model_helpers import load_submodule_ckpt
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo import MimoModel, MimoModelConfig
from megatron.core.models.mimo.submodules.vision import (
    VisionModalitySubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec

from . import (
    QWEN35_VL_IMAGE_TOKEN_ID,
    QWEN35_VL_VOCAB_SIZE,
    ROTARY_BASE,
    ROTARY_PERCENT,
    VISION_KWARGS,
    Qwen35VLVisionEncoder,
    get_qwen35_vl_language_config,
    get_qwen35_vl_language_spec,
    get_qwen35_vl_vision_config,
    get_qwen35_vl_vision_spec,
)
from .mrope import compute_mrope_position_ids


def _apply_args_to_language_config(cfg):
    """Apply Megatron CLI args to the language TransformerConfig.

    Overrides precision, parallelism, and training-time settings from
    ``get_args()`` onto the architecture config returned by
    ``get_qwen35_vl_language_config``.  This ensures that flags such as
    ``--sequence-parallel``, ``--use-distributed-optimizer``, and
    ``--use-megatron-fsdp`` are reflected in the TransformerConfig that
    GPTModel reads during construction.
    """
    try:
        from megatron.training import get_args

        args = get_args()
    except (ModuleNotFoundError, AssertionError):
        return cfg

    # Precision
    if getattr(args, "bf16", False):
        cfg.bf16 = True
    if getattr(args, "fp16", False):
        cfg.fp16 = True

    # Parallelism
    cfg.tensor_model_parallel_size = getattr(args, "tensor_model_parallel_size", 1)
    cfg.pipeline_model_parallel_size = getattr(args, "pipeline_model_parallel_size", 1)
    cfg.expert_model_parallel_size = getattr(args, "expert_model_parallel_size", 1)
    cfg.sequence_parallel = getattr(args, "sequence_parallel", False)
    cfg.context_parallel_size = getattr(args, "context_parallel_size", 1)

    # Distributed optimizer (needed for FSDP)
    if getattr(args, "use_distributed_optimizer", False):
        cfg.use_distributed_optimizer = True

    # Meta-device init (needed for FSDP to skip .to(device) on meta tensors)
    if getattr(args, "init_model_with_meta_device", False):
        cfg.init_model_with_meta_device = True

    # MoE overrides from CLI (dispatcher type may differ per variant/FSDP)
    if getattr(args, "moe_token_dispatcher_type", None) is not None:
        cfg.moe_token_dispatcher_type = args.moe_token_dispatcher_type
    if getattr(args, "moe_grouped_gemm", None) is not None:
        cfg.moe_grouped_gemm = args.moe_grouped_gemm

    return cfg


def _get_variant() -> str:
    """Read model variant from CLI args, defaulting to ``'397b_a17b'``."""
    try:
        from megatron.training import get_args

        return getattr(get_args(), "model_variant", "397b_a17b") or "397b_a17b"
    except (ModuleNotFoundError, AssertionError):
        return "397b_a17b"


class Qwen35VLMimoModel(MimoModel):
    """MimoModel with end-to-end MRoPE for Qwen3.5-VL.

    Overrides ``forward()`` to compute MRoPE position IDs ``[3, B, S]``
    from ``input_ids`` and the image grid dimensions (extracted from
    ``modality_inputs``) before delegating to the parent class.  The
    parent's ``language_model`` call then receives the correct
    ``position_ids`` for ``MultimodalRotaryEmbedding``.

    If ``position_ids`` is already supplied by the caller it is used
    as-is (allows external override for testing).
    """

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        modality_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        if position_ids is None:
            grid_thw = None
            if modality_inputs is not None:
                grid_thw = (
                    modality_inputs
                    .get("images", {})
                    .get("qwen35_vision", {})
                    .get("grid_thw")
                )
                if grid_thw is not None and grid_thw.ndim == 3:
                    grid_thw = grid_thw.reshape(-1, grid_thw.shape[-1])
            image_token_id = self.special_token_ids.get(
                "images", QWEN35_VL_IMAGE_TOKEN_ID
            )
            position_ids = compute_mrope_position_ids(
                input_ids=input_ids,
                image_grid_thw=grid_thw,
                image_token_id=image_token_id,
                spatial_merge_size=VISION_KWARGS["spatial_merge_size"],
            )
        return super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            labels=labels,
            modality_inputs=modality_inputs,
        )


def model_provider_qwen35_vlm(
    pre_process: bool = True,
    post_process: bool = True,
    add_encoder: bool = True,
    add_decoder: bool = True,
    image_special_token_id: int = QWEN35_VL_IMAGE_TOKEN_ID,
):
    """Build a Qwen3.5-VL Vision-Language MIMO model.

    The model variant is controlled by ``--model-variant`` CLI arg:
      - ``proxy``     : 4 layers, 16 experts  (single-node testing)
      - ``397b_a17b`` : 60 layers, 512 experts (production)

    Components:
      - Megatron-native Qwen3.5 vision encoder (frozen, language-dim output)
      - Qwen3-Next MoE decoder (variant-controlled depth and expert count)
      - End-to-end MRoPE via ``Qwen35VLMimoModel`` subclass
    """
    variant = _get_variant()

    # --- Language config & spec ---
    language_config = get_qwen35_vl_language_config(variant=variant)
    _apply_args_to_language_config(language_config)

    layer_spec = get_qwen35_vl_language_spec(language_config)

    # --- Vision config & encoder ---
    vision_config = get_qwen35_vl_vision_config(
        params_dtype=language_config.params_dtype,
    )
    vision_config.tensor_model_parallel_size = language_config.tensor_model_parallel_size
    vision_config.sequence_parallel = language_config.sequence_parallel
    vision_spec = get_qwen35_vl_vision_spec()

    vision_encoder = ModuleSpec(
        module=Qwen35VLVisionEncoder,
        params={
            "config": vision_config,
            "transformer_layer_spec": vision_spec,
            "spatial_merge_size": 2,
            "out_hidden_size": language_config.hidden_size,
        },
    )

    vision_submodule_spec = ModuleSpec(
        module=VisionModalitySubmodules,
        params={},
        submodules={
            "encoders": {"qwen35_vision": vision_encoder},
            "input_projections": [],
        },
    )

    # --- Language model spec ---
    seq_length = 4096
    try:
        from megatron.training import get_args

        seq_length = getattr(get_args(), "seq_length", 4096)
    except (ModuleNotFoundError, AssertionError):
        pass

    language_model_spec = ModuleSpec(
        module=GPTModel,
        params={
            "config": language_config,
            "transformer_layer_spec": layer_spec,
            "vocab_size": QWEN35_VL_VOCAB_SIZE,
            "max_sequence_length": seq_length,
            "pre_process": pre_process,
            "post_process": post_process,
            "position_embedding_type": "mrope",
            "rotary_percent": ROTARY_PERCENT,
            "rotary_base": ROTARY_BASE,
        },
    )

    # --- Assemble MIMO model ---
    mimo_model_config = MimoModelConfig(
        language_model_spec=language_model_spec,
        modality_submodules_spec={"images": vision_submodule_spec},
        special_token_ids={"images": image_special_token_id},
    )

    mimo_model = Qwen35VLMimoModel(mimo_model_config)
    print("*" * 100)
    print_mimo_structure(mimo_model)
    print("*" * 100)

    # --- Load language model checkpoint (optional) ---
    try:
        from megatron.training import get_args

        _args = get_args()
        if _args.language_model_checkpoint is not None:
            load_submodule_ckpt(
                mimo_model.language_model,
                _args.language_model_checkpoint,
            )
            print(
                "Successfully loaded language model checkpoint "
                f"from {_args.language_model_checkpoint}"
            )
    except (ModuleNotFoundError, AssertionError):
        pass

    # --- Freeze vision encoder ---
    for param in (
        mimo_model.modality_submodules.images.encoders.qwen35_vision.parameters()
    ):
        param.requires_grad = False

    return mimo_model
