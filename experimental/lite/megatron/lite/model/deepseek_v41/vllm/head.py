# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native deployment head logits and log-softmax; Lite owns model execution."""

import torch


def log_softmax(logits):
    if not logits.is_cuda:
        return torch.log_softmax(logits.float(), -1)  # CPU derivative reference only.
    from vllm.model_executor.kernels.batch_invariant import native_log_softmax

    return native_log_softmax(logits.float(), -1)


def head_logits(hidden, weight, *, dtype="float32"):
    """The whole FP32 deployment logits, same arithmetic as one head block."""
    if dtype not in ("float32", "bfloat16"):
        raise ValueError(f"Unsupported head dtype {dtype!r}")
    if hidden.is_cuda:
        from .linear import linear

        logits = linear(hidden, weight, fp8=False)
    else:
        logits = torch.nn.functional.linear(hidden.float(), weight.float())
    return _RoundBF16.apply(logits) if dtype == "bfloat16" else logits


class _RoundBF16(torch.autograd.Function):
    """BF16 logits of the deployment head; identity (STE) derivative."""

    @staticmethod
    def forward(ctx, x):
        return x.to(torch.bfloat16)

    @staticmethod
    def backward(ctx, grad):
        return grad.float()
