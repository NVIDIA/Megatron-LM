# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from typing import Any

import torch


def expand_mhc_hidden_for_pipeline(hidden: torch.Tensor, *, hc_mult: int) -> torch.Tensor:
    if hidden.dim() == 3:
        return hidden.unsqueeze(2).expand(-1, -1, hc_mult, -1).contiguous()
    return hidden


def fold_mhc_hidden_for_pipeline(hidden: torch.Tensor) -> torch.Tensor:
    """Fold the 4-D hc streams [B, S, hc_mult, H] into [B, S, hc_mult * H] for PP P2P.

    The pipeline P2P recv buffer is 3-D, so the hc_mult parallel residual streams must be
    flattened into the hidden dimension before crossing a stage boundary. No-op for 3-D input.
    """
    if hidden.dim() == 4:
        b, s, m, h = hidden.shape
        return hidden.reshape(b, s, m * h).contiguous()
    return hidden


def unfold_mhc_hidden_from_pipeline(hidden: torch.Tensor, *, hc_mult: int) -> torch.Tensor:
    """Inverse of :func:`fold_mhc_hidden_for_pipeline`: [B, S, hc_mult * H] -> [B, S, hc_mult, H].

    Used on non-first PP stages to restore the hc_mult streams received over P2P. No-op when the
    tensor is already 4-D.
    """
    if hidden.dim() == 4:
        return hidden
    b, s, mh = hidden.shape
    return hidden.reshape(b, s, hc_mult, mh // hc_mult).contiguous()


def contract_mhc_hidden_for_pipeline(
    hidden: torch.Tensor,
    *,
    norm: Any,
    head: Any,
    return_source: bool = False,
):
    if head is None or norm is None:
        if return_source:
            return hidden, None
        return hidden
    source = hidden
    contracted = norm(head(hidden))
    if return_source:
        return contracted, source
    return contracted
