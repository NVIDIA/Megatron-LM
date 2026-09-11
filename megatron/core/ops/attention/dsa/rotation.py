# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Hadamard rotation used by sparse-attention indexers."""

import torch


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard rotation activation.
    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L424-L428

    Args:
        x: Input tensor (must be bfloat16).

    Returns:
        Rotated tensor.
    """
    assert (
        x.dtype == torch.bfloat16
    ), f"rotate_activation only support bf16 input, but got {x.dtype}"
    from fast_hadamard_transform import hadamard_transform

    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size**-0.5)
