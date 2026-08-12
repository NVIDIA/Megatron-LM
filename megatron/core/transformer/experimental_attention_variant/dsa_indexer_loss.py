# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Backend-independent helpers for the DSA indexer KL objective."""

from typing import Optional

import torch

INDEXER_LOSS_EPS = 1e-10


def normalize_indexer_target(target: torch.Tensor) -> torch.Tensor:
    """L1-normalize non-negative indexer target scores along the key dimension."""
    return target / target.sum(dim=-1, keepdim=True).clamp_min(INDEXER_LOSS_EPS)


def normalize_indexer_target_(target: torch.Tensor) -> torch.Tensor:
    """L1-normalize non-negative indexer target scores in place."""
    return target.div_(target.sum(dim=-1, keepdim=True).clamp_min(INDEXER_LOSS_EPS))


def _indexer_kl_terms(
    target: torch.Tensor, predict_log_probs: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Return elementwise ``KL(target || predict)`` contributions."""
    kl_terms = target * (torch.log(target.clamp_min(INDEXER_LOSS_EPS)) - predict_log_probs)
    if valid_mask is not None:
        kl_terms = kl_terms.masked_fill(~valid_mask, 0.0)
    return kl_terms


def indexer_kl_per_row(
    target: torch.Tensor, predict_log_probs: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Return ``KL(target || predict)`` reduced over the key dimension."""
    kl_terms = _indexer_kl_terms(target, predict_log_probs, valid_mask)
    return kl_terms.sum(dim=-1)


def indexer_kl_sum(
    target: torch.Tensor, predict_log_probs: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Return ``KL(target || predict)`` reduced over every dimension."""
    return _indexer_kl_terms(target, predict_log_probs, valid_mask).sum()


def reduce_indexer_kl_sum(
    kl_sum: torch.Tensor,
    *,
    num_rows: int,
    calculate_per_token_loss: bool,
    valid_row_count: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reduce an already-summed KL value using DSA token-loss semantics."""
    if calculate_per_token_loss:
        return kl_sum
    if valid_row_count is not None:
        return kl_sum / valid_row_count.to(dtype=torch.float32, device=kl_sum.device).clamp_min(1.0)
    return kl_sum / max(num_rows, 1)


def indexer_loss_from_target(
    target: torch.Tensor,
    predict_log_probs: torch.Tensor,
    loss_coeff: float,
    query_valid_rows: Optional[torch.Tensor] = None,
    calculate_per_token_loss: bool = False,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute scaled DSA indexer KL loss from normalized target probabilities."""
    kl_per_row = indexer_kl_per_row(target, predict_log_probs, valid_mask)
    valid_row_count = None
    if query_valid_rows is not None:
        row_mask = query_valid_rows.to(dtype=torch.float32, device=kl_per_row.device)
        kl_per_row = kl_per_row * row_mask
        valid_row_count = row_mask.sum()
    kl_div = reduce_indexer_kl_sum(
        kl_per_row.sum(),
        num_rows=kl_per_row.numel(),
        calculate_per_token_loss=calculate_per_token_loss,
        valid_row_count=valid_row_count,
    )
    return kl_div * loss_coeff
