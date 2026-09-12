# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""PyTorch reference implementation of the QSA indexer KL objective.

The teacher is detached from core attention. Gradients reach only the indexer
logits, while Top-K support stays non-differentiable. These helpers materialize
the teacher distribution and are intended for correctness tests and short
sequences; optimized long-context backends should fuse this reduction.
"""

from __future__ import annotations

import torch

from megatron.core.transformer.experimental_attention_variant import dsa_indexer_loss, dsa_masking


def qsa_teacher_token_distribution(
    attention_probabilities: torch.Tensor,
    valid_key_mask: torch.Tensor | None = None,
    *,
    tensor_parallel_size: int = 1,
) -> torch.Tensor:
    """Build the detached token-level QSA teacher distribution.

    Args:
        attention_probabilities: Per-head attention probabilities
            ``[B, H, Sq, Skv]``.
        valid_key_mask: Optional attendable-key mask ``[B, Sq, Skv]``.
        tensor_parallel_size: Teacher head-sharding factor. This materialized
            reference supports only one because it has no process group for
            the required cross-rank head sum.

    Returns:
        Detached token distribution ``[B, Sq, Skv]`` in FP32, unless the input
        is FP64.

    Raises:
        NotImplementedError: If attention heads are tensor-parallel sharded.
    """
    if tensor_parallel_size != 1:
        raise NotImplementedError(
            "The materialized QSA teacher must sum over every attention head. "
            "Use tensor_parallel_size=1 for this reference implementation."
        )
    if attention_probabilities.ndim != 4:
        raise ValueError(
            "attention_probabilities must be [B, H, Sq, Skv], "
            f"got {tuple(attention_probabilities.shape)}."
        )
    probabilities = attention_probabilities.detach()
    compute_dtype = torch.float64 if probabilities.dtype == torch.float64 else torch.float32
    summed = probabilities.to(compute_dtype).sum(dim=1)
    if valid_key_mask is not None:
        if valid_key_mask.shape != summed.shape:
            raise ValueError("valid_key_mask must be [B, Sq, Skv].")
        summed = summed.masked_fill(~valid_key_mask, 0.0)
    return dsa_indexer_loss.normalize_indexer_target(summed)


def qsa_maxpool_teacher_to_blocks(
    token_distribution: torch.Tensor, *, compress_ratio: int, num_blocks: int
) -> torch.Tensor:
    """Max-pool token teacher mass onto complete compressed blocks.

    Args:
        token_distribution: Teacher token mass ``[B, Sq, Skv]``.
        compress_ratio: Tokens represented by one block.
        num_blocks: Complete block count ``P``.

    Returns:
        Unnormalized block teacher mass ``[B, Sq, P]``. Incomplete tail tokens
        are excluded.
    """
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}.")
    if num_blocks < 0:
        raise ValueError(f"num_blocks must be non-negative, got {num_blocks}.")
    usable = num_blocks * compress_ratio
    if token_distribution.size(-1) < usable:
        raise ValueError(
            f"token_distribution covers {token_distribution.size(-1)} keys, "
            f"fewer than the {usable} tokens spanned by {num_blocks} complete blocks."
        )
    complete = token_distribution[..., :usable]
    return complete.unflatten(-1, (num_blocks, compress_ratio)).amax(dim=-1)


def qsa_indexer_kl_loss(
    block_logits: torch.Tensor,
    teacher_token_distribution: torch.Tensor,
    support_mask: torch.Tensor,
    *,
    compress_ratio: int,
    loss_coeff: float,
    query_valid_rows: torch.Tensor | None = None,
    calculate_per_token_loss: bool = False,
) -> torch.Tensor:
    """Compute the scaled QSA indexer KL on a fixed block support.

    Args:
        block_logits: Differentiable scaled indexer logits ``[B, Sq, P]``.
        teacher_token_distribution: Detached teacher token mass
            ``[B, Sq, Skv]``.
        support_mask: Blocks entering the KL ``[B, Sq, P]``. Use every causal
            block for dense warmup or the indexer's detached Top-K for sparse
            training.
        compress_ratio: Tokens represented by one block.
        loss_coeff: Objective multiplier.
        query_valid_rows: Optional valid-query mask ``[B, Sq]``.
        calculate_per_token_loss: Return the valid-row sum instead of its mean.

    Returns:
        Scalar loss carrying gradient into ``block_logits`` only.
    """
    if block_logits.shape != support_mask.shape:
        raise ValueError("block_logits and support_mask must have the same shape.")
    query_valid_rows = dsa_masking.normalize_query_valid_rows(
        query_valid_rows,
        b=block_logits.size(0),
        sq=block_logits.size(1),
        device=block_logits.device,
    )
    pooled = qsa_maxpool_teacher_to_blocks(
        teacher_token_distribution.detach(),
        compress_ratio=compress_ratio,
        num_blocks=block_logits.size(-1),
    )
    pooled = pooled.masked_fill(~support_mask, 0.0).to(block_logits.dtype)
    target = dsa_indexer_loss.normalize_indexer_target(pooled)
    student_log_probs = dsa_masking.masked_log_softmax(block_logits, support_mask)
    return dsa_indexer_loss.indexer_loss_from_target(
        target,
        student_log_probs,
        loss_coeff,
        query_valid_rows=query_valid_rows,
        calculate_per_token_loss=calculate_per_token_loss,
        valid_mask=support_mask,
    )


__all__ = ["qsa_indexer_kl_loss", "qsa_maxpool_teacher_to_blocks", "qsa_teacher_token_distribution"]
