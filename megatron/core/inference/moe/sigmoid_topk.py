# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Sigmoid top-k routing kernel for inference.

Replaces the compiled ``topk_routing_with_score_function`` sequence for sigmoid
scores with an expert bias (an inductor sigmoid+bias kernel, aten ``topk``, and
two inductor gather/normalize kernels) with one launch. Expert selection and
ordering match ``torch.topk(..., sorted=False)``. Probabilities match the
compiled left-fold normalization bitwise for top-k below eight; for larger top-k
the normalization sum is reduced in a different order, so a probability can
differ in its last bit. With ``batch_invariant=True`` the launch configuration does
not depend on the number of tokens, so every row's result is independent of the
batch; batch-invariant routing uses the kernel this way in training and inference.
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


def is_sigmoid_topk_supported(num_experts: int, topk: int) -> bool:
    """Whether the fused kernel supports this routing shape."""
    return 1 < topk <= min(32, num_experts) and num_experts <= 512


@triton.jit
def _sigmoid_topk_kernel(
    logits_ptr,
    expert_bias_ptr,
    probs_ptr,
    indices_ptr,
    num_tokens,
    scaling_factor,
    NUM_EXPERTS: tl.constexpr,
    BIAS_STRIDE: tl.constexpr,
    EXPERTS_POW2: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_POW2: tl.constexpr,
    USE_TL_TOPK: tl.constexpr,
    APPLY_SCALING: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Top-k of sigmoid(logits) + bias, probabilities from sigmoid(logits).

    Matches the compiled reference:
      * selection and order follow aten topk(sorted=False): a row's k-th largest
        key under radix (total float) order, then every column above it in
        ascending column order, then columns equal to it in ascending order;
      * probabilities are (s_j / (s_0 + ... + s_{k-1} + 1e-20)) * scaling_factor.
        For top-k below eight the sum is folded left in output order, as inductor
        generates it, so probabilities match bit for bit. Larger top-k uses a tree
        reduction, which can differ from the compiled sum in the last bit.
    """
    rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    cols = tl.arange(0, EXPERTS_POW2)
    row_mask = rows < num_tokens
    col_mask = cols < NUM_EXPERTS
    mask = row_mask[:, None] & col_mask[None, :]
    rows64 = rows.to(tl.int64)

    logits = tl.load(logits_ptr + rows64[:, None] * NUM_EXPERTS + cols[None, :], mask=mask)
    bias = tl.load(expert_bias_ptr + cols * BIAS_STRIDE, mask=col_mask)
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
    if USE_TL_TOPK:
        largest = tl.topk(keys, TOPK_POW2, dim=1)
        kth = tl.sum(tl.where(tl.arange(0, TOPK_POW2)[None, :] == TOPK - 1, largest, 0), axis=1)
    else:
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

    if TOPK < 8:
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
    else:
        denom = tl.sum(tl.where(selected, scores, 0.0), axis=1) + 1e-20
        probs = scores / denom[:, None]
        if APPLY_SCALING:
            probs = probs * scaling_factor
        tl.store(probs_ptr + out_base[:, None] + slot, probs, mask=selected & row_mask[:, None])


def sigmoid_topk_routing(
    logits: torch.Tensor,
    expert_bias: torch.Tensor,
    topk: int,
    scaling_factor: Optional[float] = None,
    block_tokens: Optional[int] = None,
    batch_invariant: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense sigmoid top-k routing with an expert bias.

    Equivalent to ``topk_routing_with_score_function(logits, topk,
    score_function="sigmoid", expert_bias=expert_bias, scaling_factor=...,
    dense_output=True)`` evaluated by torch.compile with gradients disabled
    (which selects with ``torch.topk(sorted=False)``) for finite fp32 logits and bias in
    the shape range covered by ``is_sigmoid_topk_supported``. Indices always match
    bitwise. Probabilities match bitwise for top-k below eight; for larger top-k the
    compiled reduction order depends on autotuning and prior batch shapes, so a
    probability can differ in its last bit.

    Args:
        logits: [num_tokens, num_experts] fp32 router logits.
        expert_bias: [num_experts] fp32 routing bias on the same device as logits
            (selection only). Strided views are supported.
        topk: experts per token, in [2, 32] and no greater than num_experts.
        scaling_factor: optional multiplier applied to the normalized probabilities.
        block_tokens: optional rows per CTA; defaults to one for top-k of eight or
            more or up to 128 experts, and four otherwise.
        batch_invariant: keep the launch configuration independent of the number of
            tokens, so each row's probabilities and indices do not depend on the batch.

    Returns:
        probs [num_tokens, topk] fp32 and top_indices [num_tokens, topk] int64.
    """
    assert logits.dim() == 2 and logits.dtype == torch.float32 and logits.is_contiguous()
    assert expert_bias.dim() == 1, "expert_bias must be one-dimensional"
    assert expert_bias.dtype == torch.float32 and expert_bias.numel() == logits.shape[1]
    assert expert_bias.device == logits.device, "expert_bias and logits must be on the same device"
    num_tokens, num_experts = logits.shape
    assert is_sigmoid_topk_supported(num_experts, topk)
    num_warps = 4
    if block_tokens is None:
        if topk >= 8:
            # One row per CTA. Four warps hide the top-k latency on decode batches;
            # one warp exposes more independent CTAs once prefill batches are wide.
            block_tokens = 1
            num_warps = 1 if num_tokens >= 1024 and not batch_invariant else 4
        else:
            # One warp per short row avoids idle rows on decode batches and exposes
            # more independent CTAs on prefill. Wider rows retain the existing layout.
            block_tokens = 1 if num_experts <= 128 else 4
            num_warps = 1 if num_experts <= 128 else 4
    assert block_tokens > 0 and block_tokens & (block_tokens - 1) == 0
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
        BIAS_STRIDE=expert_bias.stride(0),
        EXPERTS_POW2=triton.next_power_of_2(num_experts),
        TOPK=topk,
        TOPK_POW2=triton.next_power_of_2(topk),
        USE_TL_TOPK=topk >= 8 and hasattr(tl, "topk"),
        APPLY_SCALING=bool(scaling_factor),
        BLOCK_T=block_tokens,
        num_warps=num_warps,
    )
    return probs, indices
