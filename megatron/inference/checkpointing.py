# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Checkpoint loading for inference entry points."""

from typing import Callable

import torch

from megatron.training.checkpointing import load_checkpoint


def load_checkpoint_for_inference(
    model: list[torch.nn.Module],
    *,
    strict: bool = True,
    load_arg: str = 'load',
    model_sharded_state_dict_modifier: Callable[[dict], None] | None = None,
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
        model_sharded_state_dict_modifier: Optional in-place model-key remapping
            for distributed checkpoints.

    Returns:
        Checkpoint iteration and cumulative floating-point operation count.
    """
    return load_checkpoint(
        model,
        None,
        None,
        load_arg=load_arg,
        strict=strict,
        model_sharded_state_dict_modifier=model_sharded_state_dict_modifier,
        restore_training_state=False,
    )
