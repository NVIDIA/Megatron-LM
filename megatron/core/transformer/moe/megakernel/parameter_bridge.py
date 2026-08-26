# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bridge in-kernel weight-gradient accumulation to MCore DDP parameters."""

from __future__ import annotations

import torch
from torch import nn


def dummy_weight_gradient(param: nn.Parameter) -> torch.Tensor:
    """Return a storage-free gradient sentinel that triggers MCore DDP hooks.

    A megakernel has already accumulated the numerical gradient into main_grad.
    Autograd only needs to invoke the post-accumulate hook; that hook does not
    read this value after grad_added_to_main_grad is set.
    """
    return param.detach()


def main_grad_buffer(param: nn.Parameter) -> torch.Tensor:
    """Return and validate an optimizer-visible FP32 or BF16 gradient buffer."""
    main_grad = getattr(param, "main_grad", None)
    if main_grad is None:
        raise RuntimeError(
            "Megakernel gradient accumulation requires DDP to assign param.main_grad"
        )
    if main_grad.shape != param.shape:
        raise RuntimeError(
            "Megakernel weight-gradient shape mismatch: "
            f"main_grad={tuple(main_grad.shape)}, param={tuple(param.shape)}"
        )
    if main_grad.dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError("Megakernel direct accumulation requires FP32 or BF16 main_grad")
    if not main_grad.is_contiguous():
        raise RuntimeError("Megakernel direct accumulation requires contiguous main_grad")
    if getattr(param, "zero_out_wgrad", False):
        raise RuntimeError("Megakernel does not support zero_out_wgrad parameters")
    if main_grad.device != param.device:
        raise RuntimeError("Megakernel main_grad must be on the parameter device")
    return main_grad


def finish_weight_gradient(param: nn.Parameter) -> torch.Tensor:
    """Mark an in-kernel accumulation complete and return a DDP hook-only grad."""
    param.grad_added_to_main_grad = True
    return dummy_weight_gradient(param)
