# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Utilities for clipping gradients at the model output layer."""

import torch


def clip_output_grad_hook(
    module: torch.nn.Module,
    grad_input: tuple[torch.Tensor | None, ...],
    grad_output: tuple[torch.Tensor | None, ...],
    *,
    scale: float = 3.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor | None, ...]:
    """Clip the first input gradient using its local median absolute deviation.

    The threshold is shared across all elements of this rank's input gradient;
    it is not computed independently for each token. Other input gradients are
    returned unchanged.
    """
    d_input = grad_input[0]
    if d_input is None:
        return grad_input

    flat = d_input.view(-1).abs()
    median = flat.median()
    # Median absolute deviation or spread.
    spread = (flat - median).abs().median().clamp_min(eps)
    threshold = median + scale * spread

    clipped = d_input.clamp(min=-threshold, max=threshold)
    return (clipped,) + grad_input[1:]
