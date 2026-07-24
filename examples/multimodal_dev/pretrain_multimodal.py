# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

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

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from examples.multimodal_dev.arguments import add_multimodal_args
from examples.multimodal_dev.forward_step import forward_step
from megatron.core.enums import ModelType
from megatron.training import get_args, pretrain
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import core_transformer_config_from_args, parse_and_validate_args


def model_provider(pre_process: bool = True, post_process: bool = True, **kwargs):
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
            f"Unknown model arch '{model_arch}'. " f"Available: {list(MODEL_REGISTRY.keys())}"
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
        variant=getattr(args, "model_variant", None),
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
        args=args, language_config=language_config, vision_config=vision_config, **kwargs
    )

    # Chunked vision-encoder execution (Phase B runtime). The Megatron-FSDP
    # wrapper is installed AFTER this provider returns, so the lockstep
    # group cannot be queried from it here; inject the expected dp-cp
    # sharding group from parallel_state instead (the GPU prototype asserts
    # it matches wrapper.dist_index.get_fsdp_group()).
    model.vision_encoder_chunk_patches = getattr(args, "vision_encoder_chunk_patches", 0)
    if torch.distributed.is_initialized():
        from megatron.core import parallel_state

        model.vision_lockstep_group = parallel_state.get_data_parallel_group(
            with_context_parallel=True
        )

    # Synthetic-streaming pixels (Phase B): ONE read-only noise pool of
    # exactly one chunk; every vision chunk input is a view into it, so
    # autograd-saved conv inputs alias a single storage (retained-input
    # memory is O(pool), independent of the window payload). Registered as
    # a non-persistent buffer: never checkpointed, moves with the model.
    if getattr(args, "mock_synthetic_streaming_pixels", False):
        chunk_patches = int(getattr(args, "vision_encoder_chunk_patches", 0) or 0)
        if chunk_patches <= 0:
            raise ValueError(
                "--mock-synthetic-streaming-pixels requires " "--vision-encoder-chunk-patches > 0."
            )
        pixel_dim = (
            int(getattr(args, "vision_in_channels", 3))
            * int(getattr(args, "vision_temporal_patch_size", 2))
            * int(getattr(args, "vision_patch_size", 16)) ** 2
        )
        pool_dtype = torch.bfloat16 if args.bf16 else torch.half if args.fp16 else torch.float32
        # Placeholder only: under meta-device init this lands on meta, and
        # to_empty-style materialization may wipe buffer contents either
        # way. The model regenerates the content deterministically from
        # vision_noise_pool_seed at first streaming use, on the real device.
        model.register_buffer(
            "vision_noise_pool",
            torch.empty(chunk_patches, pixel_dim, dtype=pool_dtype),
            persistent=False,
        )
        model.vision_noise_pool_seed = int(getattr(args, "seed", 1234))

    return model


def _resolve_provider_fn(provider_fn):
    """Resolve a provider that may be a dotted import path string."""
    if isinstance(provider_fn, str):
        module_path, func_name = provider_fn.rsplit(".", 1)
        provider_fn = getattr(importlib.import_module(module_path), func_name)
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
            f"Unknown model arch '{model_arch}'. " f"Available: {list(MODEL_REGISTRY.keys())}"
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

    args = parse_and_validate_args(extra_args_provider=add_multimodal_args, args_defaults={})
    # multimodal_dev's model_provider builds the full model on every rank and
    # does not honor pre_process / post_process pipeline-stage flags. PP>1
    # would silently violate Megatron's pipeline-parallel contract.
    if args.pipeline_model_parallel_size > 1:
        raise ValueError(
            "multimodal_dev does not support pipeline_model_parallel_size > 1 "
            f"(got {args.pipeline_model_parallel_size}). The model provider "
            "builds the full model on every rank; pipeline-stage splitting is "
            "not wired through. Run with --pipeline-model-parallel-size 1."
        )
    full_config = pretrain_cfg_container_from_args(args)
    pretrain(
        full_config,
        datasets_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        model_provider=model_provider,
    )
