# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Canonical settings shared with the native BAGEL training recipe."""

from argparse import Namespace


def apply_reference_llm_overrides(config, decoder_layer_module: str):
    """Apply the overrides made by BAGEL's ``pretrain_unified_navit.py``."""

    config.layer_module = decoder_layer_module
    config.qk_norm = True
    config.tie_word_embeddings = False
    config.freeze_und = False
    return config


def get_reference_data_seed(
    global_seed: int, data_parallel_world_size: int, data_parallel_rank: int
) -> int:
    """Return BAGEL's rank-local seed: ``global_seed * world_size + rank``."""

    return global_seed * data_parallel_world_size + data_parallel_rank


def validate_reference_training_args(args: Namespace) -> None:
    """Validate options used to align MCore training with native BAGEL."""

    if args.native_model_checkpoint is not None:
        incompatible_options = [
            option
            for option, value in (
                ("--load", args.load),
                ("--pretrained-checkpoint", args.pretrained_checkpoint),
            )
            if value is not None
        ]
        if incompatible_options:
            raise ValueError(
                "--native-model-checkpoint cannot be combined with "
                + " or ".join(incompatible_options)
            )

    if args.reset_reference_training_rng and args.native_model_checkpoint is None:
        raise ValueError(
            "--reset-reference-training-rng requires --native-model-checkpoint"
        )

    if args.reset_reference_training_rng:
        topology = (
            args.tensor_model_parallel_size,
            args.pipeline_model_parallel_size,
            args.context_parallel_size,
        )
        if topology != (1, 1, 1):
            tp_size, pp_size, cp_size = topology
            raise ValueError(
                "--reset-reference-training-rng is strictly equivalent to native BAGEL only "
                f"with TP=PP=CP=1; got TP={tp_size}, PP={pp_size}, CP={cp_size}"
            )
