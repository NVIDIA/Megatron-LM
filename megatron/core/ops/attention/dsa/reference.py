# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""PyTorch score postprocessing used by the existing KPool selection path."""

import torch


def torch_kpool_score_postprocess(
    scores: torch.Tensor, weights: torch.Tensor, use_relu: bool
) -> torch.Tensor:
    """Reuse detached scores for the original optional-ReLU, multiply and head sum."""
    if torch.is_grad_enabled() and (scores.requires_grad or weights.requires_grad):
        raise ValueError("KPool score postprocessing requires detached selection inputs")
    if use_relu:
        scores.relu_()
    scores.mul_(weights.unsqueeze(-1))
    return scores.sum(dim=2)
