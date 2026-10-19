# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native deployment head logits and log-softmax; Lite owns model execution."""

import torch


def log_softmax(logits):
    if not logits.is_cuda:
        return torch.log_softmax(logits.float(), -1)  # CPU derivative reference only.
    from vllm.model_executor.kernels.batch_invariant import native_log_softmax

    return native_log_softmax(logits.float(), -1)


def head_logits(hidden, weight):
    """The whole FP32 deployment logits, same arithmetic as one head block."""
    if hidden.is_cuda:
        from .linear import linear

        return linear(hidden, weight, fp8=False)
    return torch.nn.functional.linear(hidden.float(), weight.float())
