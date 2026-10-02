# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Checkpoint loading for inference entry points."""

import torch

from megatron.training.checkpointing import load_checkpoint


def load_checkpoint_for_inference(
    model: list[torch.nn.Module], *, strict: bool = True, load_arg: str = 'load'
) -> tuple[int, float]:
    """Load model weights without resuming a training run.

    Checkpoint format handling and model compatibility checks are shared with
    training. RNG restoration retains the existing ``no_load_rng``/``finetune``
    policy; optimizer, scheduler, rerun and consumed-sample state are not restored.
    The returned iteration/FLOP count are checkpoint metadata only.

    Args:
        model: Model partitions to restore.
        strict: Whether to require an exact model state-dict match.
        load_arg: Argument naming the checkpoint directory.

    Returns:
        Checkpoint iteration and cumulative floating-point operation count.
    """
    return load_checkpoint(
        model, None, None, load_arg=load_arg, strict=strict, restore_training_state=False
    )
