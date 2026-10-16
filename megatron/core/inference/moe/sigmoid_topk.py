# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Single-kernel sigmoid top-k routing for inference.

Replaces the compiled ``topk_routing_with_score_function`` sequence for sigmoid
scores with an expert bias (an inductor sigmoid+bias kernel, aten ``topk``, and
two inductor gather/normalize kernels) with one launch. Expert selection and
ordering match ``torch.topk(..., sorted=False)``. Small top-k also matches the
compiled probability bits; large top-k uses a fixed fp32 reduction tree and can
differ in rounding from Inductor's autotuned reduction. Batch-invariant routing
does not use this kernel.
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

HAVE_PARALLEL_TOPK = HAVE_TRITON and hasattr(tl, "topk")


def is_sigmoid_topk_supported(num_experts: int, topk: int) -> bool:
    """Whether the fused path covers this expert count and top-k width."""
    return (
        1 < topk <= min(num_experts, 32) and num_experts <= 512 and (topk < 8 or HAVE_PARALLEL_TOPK)
    )


@triton.jit
def _sigmoid_topk_parallel_kernel(
    logits_ptr,
    expert_bias_ptr,
    probs_ptr,
    indices_ptr,
    num_tokens,
    scaling_factor,
    NUM_EXPERTS: tl.constexpr,
    EXPERTS_POW2: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_POW2: tl.constexpr,
    APPLY_SCALING: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Select large top-k in parallel and normalize only the selected slots.

    Repeated full-row maxima scale poorly with top-k, particularly when a decode
    step has only one or two tokens. Two integer top-k operations find the cutoff
    and restore aten's unsorted output order without a loop over selected experts.
    """
    rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    rows64 = rows.to(tl.int64)
    cols = tl.arange(0, EXPERTS_POW2)
    mask = (rows[:, None] < num_tokens) & (cols[None, :] < NUM_EXPERTS)
    logits = tl.load(logits_ptr + rows64[:, None] * NUM_EXPERTS + cols[None, :], mask, other=0)
    bias = tl.load(expert_bias_ptr + cols, cols < NUM_EXPERTS, other=0)
    scores = tl.sigmoid(logits)
    bits = (scores + bias[None, :]).to(tl.int32, bitcast=True)
    keys = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
    keys = tl.where(cols[None, :] < NUM_EXPERTS, keys, -2147483648)

    largest = tl.topk(keys, TOPK_POW2, dim=1)
    kth = tl.gather(largest, tl.full((BLOCK_T, 1), TOPK - 1, tl.int32), axis=1)
    # aten emits experts strictly above the cutoff in ascending column order,
    # then fills the remaining slots with ties, also in ascending column order.
    priority = tl.where(
        keys > kth,
        2 * NUM_EXPERTS - cols[None, :],
        tl.where(keys == kth, NUM_EXPERTS - cols[None, :], 0),
    )
    priority = tl.where(cols[None, :] < NUM_EXPERTS, priority, 0)
    ordered = tl.topk(priority, TOPK_POW2, dim=1)
    indices = tl.where(ordered > NUM_EXPERTS, 2 * NUM_EXPERTS - ordered, NUM_EXPERTS - ordered)
    # Unselected power-of-two padding slots may decode to NUM_EXPERTS.
    indices = tl.minimum(indices, NUM_EXPERTS - 1)
    slots = tl.arange(0, TOPK_POW2)
    selected = tl.gather(scores, indices, axis=1)
    selected = tl.where(slots[None, :] < TOPK, selected, 0.0)

    # Use a fixed butterfly tree, matching Inductor's one-row persistent reduction.
    # A plain tl.sum can change the local sum order when topk's result layout puts
    # multiple slots in one thread. Inductor may autotune a different reduction
    # layout for larger batches, so equivalence to it is numerical, not bitwise.
    folded = selected
    for level in tl.static_range(triton.next_power_of_2(TOPK).bit_length() - 1):
        partner = tl.broadcast_to(
            (slots ^ (TOPK_POW2 >> (level + 1)))[None, :], (BLOCK_T, TOPK_POW2)
        )
        folded = folded + tl.gather(folded, partner, axis=1)
    total = tl.gather(folded, tl.full((BLOCK_T, 1), 0, tl.int32), axis=1)
    probs = selected / (total + 1e-20)
    if APPLY_SCALING:
        probs *= scaling_factor
    out_offsets = rows64[:, None] * TOPK + slots[None, :]
    out_mask = (rows[:, None] < num_tokens) & (slots[None, :] < TOPK)
    tl.store(probs_ptr + out_offsets, probs, out_mask)
    tl.store(indices_ptr + out_offsets, indices.to(tl.int64), out_mask)


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
    block_tokens: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense sigmoid top-k routing with an expert bias.

    Equivalent to ``topk_routing_with_score_function(logits, topk,
    score_function="sigmoid", expert_bias=expert_bias, scaling_factor=...,
    dense_output=True)`` evaluated by torch.compile with gradients disabled
    (which selects with ``torch.topk(sorted=False)``) for finite fp32 logits in
    the shape range covered by ``is_sigmoid_topk_supported``. Small top-k uses
    the compiled left fold and matches bitwise. Top-k 8--32 uses a fixed butterfly
    reduction: indices match exactly, but probabilities may differ by fp32 rounding
    from the compiled path's autotuned sum order. Repeated calls on identical inputs
    produce identical bits. This path is not used in batch-invariant mode.

    Args:
        logits: [num_tokens, num_experts] fp32 router logits.
        expert_bias: [num_experts] fp32 routing bias (selection only).
        topk: experts per token, in [2, 32] and no greater than num_experts.
        scaling_factor: optional multiplier applied to the normalized probabilities.
        block_tokens: optional rows per CTA; defaults to four for small top-k and
            one for large top-k so tiny decode batches expose independent CTAs.

    Returns:
        probs [num_tokens, topk] fp32 and top_indices [num_tokens, topk] int64.
    """
    assert logits.dim() == 2 and logits.dtype == torch.float32 and logits.is_contiguous()
    assert expert_bias.dtype == torch.float32 and expert_bias.numel() == logits.shape[1]
    num_tokens, num_experts = logits.shape
    assert is_sigmoid_topk_supported(num_experts, topk)
    if block_tokens is None:
        block_tokens = 4 if topk < 8 else 1
    assert block_tokens > 0 and block_tokens & (block_tokens - 1) == 0
    probs = torch.empty(num_tokens, topk, dtype=torch.float32, device=logits.device)
    indices = torch.empty(num_tokens, topk, dtype=torch.int64, device=logits.device)
    if num_tokens == 0:
        return probs, indices
    if topk >= 8:
        _sigmoid_topk_parallel_kernel[(triton.cdiv(num_tokens, block_tokens),)](
            logits,
            expert_bias,
            probs,
            indices,
            num_tokens,
            float(scaling_factor) if scaling_factor else 1.0,
            NUM_EXPERTS=num_experts,
            EXPERTS_POW2=triton.next_power_of_2(num_experts),
            TOPK=topk,
            TOPK_POW2=triton.next_power_of_2(topk),
            APPLY_SCALING=bool(scaling_factor),
            BLOCK_T=block_tokens,
            num_warps=4,
        )
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
