# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Standalone entry point for multimodal_dev model training (FSDP + EP).

This entry point is **model-agnostic**.  All model-specific logic (layer
specs, model construction, FLOPs metadata, dataset generation) is
delegated to factory functions registered in
:data:`multimodal_dev.models.MODEL_REGISTRY`.

Adding a new architecture only requires:

1. Creating a new model package under ``multimodal_dev/models/<arch>/``
   with the appropriate factory functions.
2. Registering an entry in ``MODEL_REGISTRY``.

No changes to this file are necessary.

Usage::

    torchrun --nproc_per_node=8 multimodal_dev/pretrain_multimodal.py \\
        --model-arch qwen35_vl \\
        --dataset-provider mock \\
        ... (other megatron args)
"""

import importlib
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
)

from megatron.core.enums import ModelType
from megatron.training import get_args, pretrain
from megatron.training.arguments import core_transformer_config_from_args

from examples.multimodal_dev.arguments import add_multimodal_args
from examples.multimodal_dev.forward_step import forward_step


def model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    **kwargs,
):
    """Build a multimodal model from ``--model-arch``.

    The language ``TransformerConfig`` is built from CLI args so that
    parallelism settings, precision, and fusion flags are inherited.
    Model-specific post-processing and construction are delegated to the
    registry factory functions.
    """
    args = get_args()
    model_arch = getattr(args, "model_arch", "qwen35_vl")

    from examples.multimodal_dev.models import MODEL_REGISTRY

    if model_arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model arch '{model_arch}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    registry = MODEL_REGISTRY[model_arch]

    # --- language config (generic + model-specific post-processing) ---
    language_config = core_transformer_config_from_args(args)
    post_language_config_fn = registry.get("post_language_config_fn")
    if post_language_config_fn is not None:
        post_language_config_fn(language_config, args)

    # --- vision config ---
    vision_config = registry["vision_config_fn"](
        num_layers_override=getattr(args, "vision_num_layers", None),
    )
    vision_config.bf16 = language_config.bf16
    vision_config.fp16 = language_config.fp16

    if getattr(args, "recompute_vision", False):
        vision_config.recompute_granularity = "full"
        vision_config.recompute_method = "uniform"
        vision_config.recompute_num_layers = 1

    # --- vision FLOPs metadata ---
    vision_flops_fn = registry.get("vision_flops_fn")
    if vision_flops_fn is not None:
        vision_flops_fn(args, language_config, vision_config)

    # --- build model (fully delegated to the arch factory) ---
    model = registry["model_factory_fn"](
        args=args,
        language_config=language_config,
        vision_config=vision_config,
        **kwargs,
    )

    return model


def _resolve_provider_fn(provider_fn):
    """Resolve a provider that may be a dotted import path string."""
    if isinstance(provider_fn, str):
        module_path, func_name = provider_fn.rsplit(".", 1)
        provider_fn = getattr(
            importlib.import_module(module_path), func_name,
        )
    return provider_fn


def datasets_provider(train_val_test_num_samples):
    """Dataset provider dispatcher.

    Routes to the dataset factory registered for the current
    ``(--model-arch, --dataset-provider)`` combination.
    """
    args = get_args()
    model_arch = getattr(args, "model_arch", "qwen35_vl")
    provider = getattr(args, "dataset_provider", "mock")

    from examples.multimodal_dev.models import MODEL_REGISTRY

    if model_arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model arch '{model_arch}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    registry = MODEL_REGISTRY[model_arch]
    available = registry.get("dataset_providers", {})

    if provider not in available:
        raise ValueError(
            f"Unknown dataset provider '{provider}' for arch "
            f"'{model_arch}'. Available: {list(available.keys())}"
        )

    provider_fn = _resolve_provider_fn(available[provider])
    return provider_fn(train_val_test_num_samples)


if __name__ == "__main__":
    datasets_provider.is_distributed = True

    pretrain(
        datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={},
        extra_args_provider=add_multimodal_args,
    )
