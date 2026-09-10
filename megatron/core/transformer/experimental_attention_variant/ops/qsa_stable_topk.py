# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The canonical PyTorch definition of the QSA Top-K tie contract.

QSA Top-K is defined by one lexicographic key:

1. score descending (bitwise FP32 comparison, no tolerance);
2. on bitwise-equal score, block ID ascending.

That contract exists because QSA's score is a sum of ReLUs. Many blocks can be
tied at exactly zero, so "exact Top-K" is not a single set unless a tie-break is
defined. ``torch.topk`` does not define which tied elements are returned, so it
must not decide a QSA route, a KL support mask, or a test expectation.

A stable descending sort of a block-ID-ascending axis *is* that key, and this
module is the single PyTorch definition that optimized backends must match.
"""

from __future__ import annotations

import torch


def qsa_stable_topk_indices(scores: torch.Tensor, topk: int) -> torch.Tensor:
    """Return canonical Top-K indices along the last dimension.

    Args:
        scores: Tensor of shape ``[..., P]``. Excluded candidates must carry
            ``-inf`` so they sort below every live candidate while keeping their
            relative index order.
        topk: Number of indices to keep.  Must not exceed ``P``.

    Returns:
        int64 indices ``[..., topk]`` ordered score descending, index ascending.
    """
    if topk < 0 or topk > scores.size(-1):
        raise ValueError(f"topk={topk} is outside [0, {scores.size(-1)}]")
    # The sort input axis is index-ascending by construction, and ``stable=True``
    # preserves that order inside every group of bitwise-equal scores, so the
    # result is exactly (score descending, index ascending).
    order = torch.sort(scores, dim=-1, descending=True, stable=True).indices
    return order[..., :topk]


__all__ = ["qsa_stable_topk_indices"]
