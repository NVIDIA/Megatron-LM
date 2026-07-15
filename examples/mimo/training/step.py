# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Forward step and per-token loss for MIMO training."""

from __future__ import annotations

from contextlib import nullcontext
from functools import partial

import torch

from examples.mimo.training.encoder_prefetch import (
    PREFETCHED_FEATURES_KEY,
    PROJECTION_TIMER_KEY,
    move_batch_to_cuda,
)


def loss_func(output_tensor: torch.Tensor, *, loss_mask: torch.Tensor):
    """Return summed per-token loss, integer local token count, and logging tensors."""
    if not isinstance(output_tensor, torch.Tensor):
        raise TypeError(
            "loss_func expects the terminal language stage to return a per-token loss tensor, "
            f"got {type(output_tensor).__name__}"
        )

    if not isinstance(loss_mask, torch.Tensor) or output_tensor.shape != loss_mask.shape:
        raise RuntimeError(
            "MIMO per-token loss requires a loss_mask with the same shape as the model output"
        )

    output = output_tensor.float()
    mask = loss_mask.float()
    masked = output * mask
    num_tokens = mask.sum().to(torch.int)
    loss_sum = masked.sum()
    return (
        loss_sum,
        num_tokens,
        {"lm loss": torch.stack((loss_sum.detach(), num_tokens.detach().float()))},
    )


def mimo_forward_step(data_iterator, model):
    """Run a raw-input or prefetched-feature MIMO microbatch for the pipeline schedule.

    On the last pipeline stage, the schedule passes ``output_tensor`` to the returned loss closure.
    """
    batch = next(data_iterator) if data_iterator is not None else {"input_ids": None}
    prefetched = batch.pop(PREFETCHED_FEATURES_KEY, None)
    projection_timer = batch.pop(PROJECTION_TIMER_KEY, None)

    if prefetched is None:
        if projection_timer is not None:
            raise RuntimeError("encoder prefetch timer has no prefetched features")
        batch = move_batch_to_cuda(batch)
        output_tensor, loss_mask = model(**batch)
        return output_tensor, partial(loss_func, loss_mask=loss_mask)

    if batch.get("modality_inputs"):
        raise ValueError("prefetched features cannot be combined with raw modality inputs")

    projection_context = projection_timer if projection_timer is not None else nullcontext()
    with projection_context:
        output_tensor = model._forward_encoders(
            batch.get("input_ids"), modality_inputs=None, input_tensors=prefetched
        )
    # Encoder ranks never evaluate the language-model loss closure.
    return output_tensor, partial(loss_func, loss_mask=None)
