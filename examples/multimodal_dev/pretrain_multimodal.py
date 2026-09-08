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

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
)

from examples.multimodal_dev.arguments import add_multimodal_args
from examples.multimodal_dev.forward_step import forward_step
from megatron.core.enums import ModelType
from megatron.training import get_args, pretrain
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import core_transformer_config_from_args, parse_and_validate_args
from megatron.training.utils import start_memory_history_recording


def configure_vision_recompute(vision_config, *, whole_tower: bool = False) -> None:
    """--recompute-vision: full activation recompute for the vision tower.

    The block size is the whole trade-off, and it is payload-dependent, so it
    stays opt-in rather than becoming a silent change of what --recompute-vision
    has always meant:

    - per-layer blocks (default): every layer's input is saved, and backward
      re-materializes one layer at a time — a bounded spike.
    - one whole-tower block (--recompute-vision-whole-tower): only the block
      input (the patch-embed output) is saved, but backward re-materializes ALL
      layers' internal activations simultaneously.

    Recompute FLOPs are identical either way (any full recompute re-runs the
    tower in backward). Whole-tower is the measured winner for this stack's
    long-window envelope, where the per-layer saves
    (raw_patches x vision_hidden x num_layers) dominate vision memory; it was
    validated by the 128K qualification with allocation-point margin forensics.
    A different model or a lighter payload can just as easily be dominated by
    the backward spike instead, which is why the default is unchanged.
    """
    vision_config.recompute_granularity = "full"
    vision_config.recompute_method = "uniform"
    vision_config.recompute_num_layers = vision_config.num_layers if whole_tower else 1


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
        variant=getattr(args, "model_variant", None),
    )
    vision_config.bf16 = language_config.bf16
    vision_config.fp16 = language_config.fp16
    vision_config.apply_rope_fusion = language_config.apply_rope_fusion

    if getattr(args, "recompute_vision", False):
        configure_vision_recompute(
            vision_config, whole_tower=getattr(args, "recompute_vision_whole_tower", False)
        )

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


def validate_entry_args(args) -> None:
    """Reject statically-decidable misconfigurations before any model
    construction (fail in seconds, not after multi-node setup)."""
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
    # MTP itself IS wired through (models/base.py passes mtp_block_spec to the
    # language model). The conflict is narrower: this entry sets
    # scatter_embedding_sequence_parallel=False and scatters later, so under an
    # EFFECTIVE sequence-parallel layout the decoder embedding keeps its full
    # [S, B, D] shape while MTP expects the scattered hidden states. Without SP
    # (or at TP=1, where SP does nothing) the two agree and MTP is supported.
    if (
        getattr(args, "mtp_num_layers", 0)
        and getattr(args, "sequence_parallel", False)
        and args.tensor_model_parallel_size > 1
    ):
        raise ValueError(
            "MTP is not supported together with sequence parallelism on this entry: "
            "the deferred embedding scatter (scatter_embedding_sequence_parallel=False) "
            "leaves the decoder embedding unscattered while MTP consumes scattered "
            "hidden states. Run with --mtp-num-layers 0, or drop --sequence-parallel."
        )
    # Block-size without the feature is a no-op the operator cannot see: the
    # run starts with NO vision recompute at all and, for a long-window recipe,
    # only says so as an OOM at the GPU allocation point.
    if getattr(args, "recompute_vision_whole_tower", False) and not getattr(
        args, "recompute_vision", False
    ):
        raise ValueError(
            "--recompute-vision-whole-tower selects the recompute BLOCK SIZE and "
            "does nothing on its own; pass --recompute-vision as well, or drop it."
        )
    # The fixed-shape providers size their samples from --total-seq-length
    # while pack_or_pad_batch caps at --seq-length, so this combination always
    # aborts at step 1. It is decidable here, before the run costs anything.
    total_seq_length = getattr(args, "total_seq_length", None)
    if (
        getattr(args, "dataset_provider", "mock") != "mock_varlen"
        and total_seq_length is not None
        and total_seq_length > args.seq_length
    ):
        raise ValueError(
            f"--total-seq-length {total_seq_length} exceeds --seq-length "
            f"{args.seq_length}: the fixed-shape providers would emit samples the "
            "packer refuses to truncate. Lower --total-seq-length or raise "
            "--seq-length."
        )
    # Statically decidable misconfig: the multimodal packed THD path does not
    # support CUDA graphs (forward_step keeps a runtime guard as defense in
    # depth); reject the combination at startup instead of at the first step.
    if getattr(args, "use_packed_sequence", False) and getattr(
        args, "cuda_graph_impl", "none"
    ) not in (None, "none"):
        raise ValueError(
            "--use-packed-sequence is incompatible with "
            f"--cuda-graph-impl {args.cuda_graph_impl}: the multimodal packed "
            "THD path does not support CUDA Graph. Run with "
            "--cuda-graph-impl none."
        )


if __name__ == "__main__":
    datasets_provider.is_distributed = True

    args = parse_and_validate_args(
        extra_args_provider=add_multimodal_args,
        args_defaults={},
    )
    validate_entry_args(args)
    full_config = pretrain_cfg_container_from_args(args)
    # training.py enables allocator history only on the config-container MODEL
    # flow; this entry uses model_provider, so it enables recording itself, and
    # must stay ahead of pretrain(), which constructs the model.
    start_memory_history_recording(getattr(full_config, "profiling", None))
    pretrain(
        full_config,
        datasets_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        model_provider=model_provider,
    )
