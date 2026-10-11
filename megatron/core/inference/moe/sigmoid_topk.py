# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Single-kernel sigmoid top-k routing for inference.

Replaces the compiled ``topk_routing_with_score_function`` sequence for sigmoid
scores with an expert bias (an inductor sigmoid+bias kernel, aten ``topk``, and
two inductor gather/normalize kernels) with one launch that produces the same
bits: the same expert set in the same order as ``torch.topk(..., sorted=False)``
and the same probability arithmetic.
"""

from typing import Optional
from unittest.mock import MagicMock

import torch

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()


@triton.jit
def _sigmoid_topk_kernel(
    logits_ptr,
    expert_bias_ptr,
    probs_ptr,
    indices_ptr,
    num_tokens,
    scaling_factor,
    NUM_EXPERTS: tl.constexpr,
    EXPERTS_POW2: tl.constexpr,
    TOPK: tl.constexpr,
    APPLY_SCALING: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Top-k of sigmoid(logits) + bias, probabilities from sigmoid(logits).

    Matches the compiled reference bit for bit:
      * selection and order follow aten topk(sorted=False): a row's k-th largest
        key under radix (total float) order, then every column above it in
        ascending column order, then columns equal to it in ascending order;
      * probabilities are (s_j / (s_0 + ... + s_{k-1} + 1e-20)) * scaling_factor
        with the sum folded left in output order, as inductor generates it.
    """
    rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    cols = tl.arange(0, EXPERTS_POW2)
    row_mask = rows < num_tokens
    col_mask = cols < NUM_EXPERTS
    mask = row_mask[:, None] & col_mask[None, :]
    rows64 = rows.to(tl.int64)

    logits = tl.load(logits_ptr + rows64[:, None] * NUM_EXPERTS + cols[None, :], mask=mask)
    bias = tl.load(expert_bias_ptr + cols, mask=col_mask)
    scores = tl.sigmoid(logits)
    routing_scores = scores + bias[None, :]

    # Radix-order keys: a signed-int compare of these equals aten's unsigned
    # compare of its converted floats (-0.0 < +0.0). Padding columns get the
    # smallest key so they are never selected.
    bits = routing_scores.to(tl.int32, bitcast=True)
    keys = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
    lowest = tl.full([BLOCK_T, EXPERTS_POW2], -2147483648, tl.int32)
    keys = tl.where(col_mask[None, :], keys, lowest)

    # k-th largest key, counting repeated keys with multiplicity.
    remaining = keys
    kth = tl.full([BLOCK_T], -2147483648, tl.int32)
    found = tl.zeros([BLOCK_T], dtype=tl.int32)
    for _ in tl.static_range(TOPK):
        top = tl.max(remaining, axis=1)
        is_top = remaining == top[:, None]
        count = tl.sum(is_top.to(tl.int32), axis=1)
        kth = tl.where((found < TOPK) & (found + count >= TOPK), top, kth)
        found += count
        remaining = tl.where(is_top, lowest, remaining)

    above = (keys > kth[:, None]) & col_mask[None, :]
    tied = (keys == kth[:, None]) & col_mask[None, :]
    num_above = tl.sum(above.to(tl.int32), axis=1)
    tied_rank = tl.cumsum(tied.to(tl.int32), axis=1)
    selected_tied = tied & (tied_rank <= TOPK - num_above[:, None])
    slot = tl.where(
        above, tl.cumsum(above.to(tl.int32), axis=1) - 1, num_above[:, None] + tied_rank - 1
    )
    selected = above | selected_tied

    out_base = rows64 * TOPK
    tl.store(
        indices_ptr + out_base[:, None] + slot,
        tl.broadcast_to(cols[None, :], (BLOCK_T, EXPERTS_POW2)).to(tl.int64),
        mask=selected & row_mask[:, None],
    )

    # Selected sigmoid scores in output order; exactly one column per slot, so
    # the masked sums are exact. Sigmoid scores are >= +0.0, so folding from
    # 0.0 reproduces the reference's left fold s_0 + s_1 + ... exactly.
    total = tl.zeros([BLOCK_T], dtype=tl.float32)
    for j in tl.static_range(TOPK):
        total = total + tl.sum(tl.where(selected & (slot == j), scores, 0.0), axis=1)
    denom = total + 1e-20
    for j in tl.static_range(TOPK):
        s_j = tl.sum(tl.where(selected & (slot == j), scores, 0.0), axis=1)
        p_j = s_j / denom
        if APPLY_SCALING:
            p_j = p_j * scaling_factor
        tl.store(probs_ptr + out_base + j, p_j, mask=row_mask)


def sigmoid_topk_routing(
    logits: torch.Tensor,
    expert_bias: torch.Tensor,
    topk: int,
    scaling_factor: Optional[float] = None,
    block_tokens: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense sigmoid top-k routing with an expert bias.

    Bitwise equivalent to ``topk_routing_with_score_function(logits, topk,
    score_function="sigmoid", expert_bias=expert_bias, scaling_factor=...,
    dense_output=True)`` evaluated by torch.compile with gradients disabled
    (which selects with ``torch.topk(sorted=False)``) for finite fp32 logits.

    Args:
        logits: [num_tokens, num_experts] fp32 router logits.
        expert_bias: [num_experts] fp32 routing bias (selection only).
        topk: experts per token, > 1.
        scaling_factor: optional multiplier applied to the normalized probabilities.

    Returns:
        probs [num_tokens, topk] fp32 and top_indices [num_tokens, topk] int64.
    """
    assert logits.dim() == 2 and logits.dtype == torch.float32 and logits.is_contiguous()
    assert expert_bias.dtype == torch.float32 and expert_bias.numel() == logits.shape[1]
    num_tokens, num_experts = logits.shape
    assert 1 < topk <= num_experts
    probs = torch.empty(num_tokens, topk, dtype=torch.float32, device=logits.device)
    indices = torch.empty(num_tokens, topk, dtype=torch.int64, device=logits.device)
    if num_tokens == 0:
        return probs, indices
    _sigmoid_topk_kernel[(triton.cdiv(num_tokens, block_tokens),)](
        logits,
        expert_bias,
        probs,
        indices,
        num_tokens,
        float(scaling_factor) if scaling_factor else 1.0,
        NUM_EXPERTS=num_experts,
        EXPERTS_POW2=triton.next_power_of_2(num_experts),
        TOPK=topk,
        APPLY_SCALING=bool(scaling_factor),
        BLOCK_T=block_tokens,
    )
    return probs, indices
