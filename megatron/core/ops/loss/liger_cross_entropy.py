# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Liger's vocab-parallel cross entropy."""

from __future__ import annotations

from functools import cache

import torch


@cache
def _cross_entropy():
    # Imported on first use, so this module imports without liger-kernel installed.
    from liger_kernel.megatron import LigerMegatronCrossEntropy  # type: ignore[import-untyped]

    return LigerMegatronCrossEntropy(ignore_index=-100, reduction="none", label_smoothing=0.0)


def liger_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    """Return FP32 per-token loss without modifying logits.

    The supplied group owns all vocabulary reductions; ``None`` means local vocabulary.
    """
    return _cross_entropy()(logits, labels, tp_group=tp_group)
