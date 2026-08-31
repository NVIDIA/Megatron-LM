# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Test helpers shared across the multimodal_dev test suite."""


def grad_norm(model):
    """L2 norm of all populated parameter gradients on this rank."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.float().norm(2).item() ** 2
    return total**0.5


def mean_loss(per_token_loss, loss_mask):
    """Mean per-token loss over valid (mask>0) positions on this rank."""
    flat = per_token_loss.float().view(-1)
    mask = loss_mask.float().view(-1)
    return (flat * mask).sum() / mask.sum().clamp(min=1)
