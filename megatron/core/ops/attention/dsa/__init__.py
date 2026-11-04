# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""DSA score postprocessing: detached [query, batch, head, key] to [query, batch, key].

The operation weights optional-ReLU FP32 GEMM scores and reduces the head axis.
It owns no state, masks or process groups. The caller owns chunking, pooling,
packed/CP bounds and ordered top-k. The torch path reuses its input score buffer;
the Triton path preserves both inputs and supports 32 heads with dense head/key axes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch


class KPoolScorePostprocess(Protocol):
    """Callable bound once by the indexer for detached pool selection."""

    def __call__(self, scores: torch.Tensor, weights: torch.Tensor, use_relu: bool) -> torch.Tensor:
        """Reduce weighted head scores without changing the preceding scoring GEMM."""
        ...
