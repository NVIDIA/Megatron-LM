# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Sigmoid top-k routing kernels for inference.

Replaces the compiled ``topk_routing_with_score_function`` sequence for sigmoid
scores with an expert bias (an inductor sigmoid+bias kernel, aten ``topk``, and
two inductor gather/normalize kernels) with one launch for top-k below eight.
Expert selection and ordering match ``torch.topk(..., sorted=False)``, and
probabilities match the compiled left-fold normalization bitwise. Larger top-k
uses parallel expert selection while preserving the compiled sigmoid and
normalization graph. Batch-invariant routing does not use these kernels.
"""

from typing import Optional
from unittest.mock import MagicMock

import torch
from torch.utils._python_dispatch import TorchDispatchMode

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
    """Whether the fused path preserves the compiled router's normalization order."""
    return 1 < topk < 8 and topk <= num_experts <= 512


def is_topk_scores_supported(num_experts: int, topk: int) -> bool:
    """Whether parallel top-k selection is available for this inference shape."""
    return (
        HAVE_TRITON
        and hasattr(tl, "topk")
        and hasattr(TorchDispatchMode, "ignore_compile_internals")
        and 8 <= topk <= min(32, num_experts)
        and num_experts <= 512
    )


@triton.jit
def _topk_scores_kernel(
    scores_ptr,
    values_ptr,
    indices_ptr,
    NUM_EXPERTS: tl.constexpr,
    EXPERTS_POW2: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_POW2: tl.constexpr,
):
    """Parallel cutoff selection with aten's unsorted expert order."""
    row = tl.program_id(0)
    cols = tl.arange(0, EXPERTS_POW2)
    valid = cols < NUM_EXPERTS
    scores = tl.load(scores_ptr + row.to(tl.int64) * NUM_EXPERTS + cols, valid, other=0.0)
    bits = scores.to(tl.int32, bitcast=True)
    keys = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
    keys = tl.where(valid, keys, -2147483648)
    largest = tl.topk(keys, TOPK_POW2, dim=0)
    kth = tl.sum(tl.where(tl.arange(0, TOPK_POW2) == TOPK - 1, largest, 0))

    # Use only the cutoff from topk. Emit above-cutoff columns first, followed
    # by enough cutoff ties to reach k; both groups follow input column order.
    above = (keys > kth) & valid
    tied = (keys == kth) & valid
    num_above = tl.sum(above.to(tl.int32), axis=0)
    tied_rank = tl.cumsum(tied.to(tl.int32), axis=0)
    selected_tied = tied & (tied_rank <= TOPK - num_above)
    slot = tl.where(above, tl.cumsum(above.to(tl.int32), axis=0) - 1, num_above + tied_rank - 1)
    offsets = row.to(tl.int64) * TOPK + slot
    tl.store(indices_ptr + offsets, cols.to(tl.int64), above | selected_tied)
    tl.store(values_ptr + offsets, scores, above | selected_tied)


def topk_scores(scores: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Unsorted top-k for contiguous finite fp32 inference scores on CUDA.

    Called at execution time by ``ParallelTopKMode`` so the original compiled
    router, including its cached normalization kernels, stays unchanged.
    """
    assert scores.is_cuda and scores.ndim == 2 and scores.dtype == torch.float32
    assert scores.is_contiguous()
    num_tokens, num_experts = scores.shape
    assert is_topk_scores_supported(num_experts, topk)
    values = scores.new_empty((num_tokens, topk))
    indices = scores.new_empty((num_tokens, topk), dtype=torch.int64)
    if num_tokens:
        _topk_scores_kernel[(num_tokens,)](
            scores,
            values,
            indices,
            NUM_EXPERTS=num_experts,
            EXPERTS_POW2=triton.next_power_of_2(num_experts),
            TOPK=topk,
            TOPK_POW2=triton.next_power_of_2(topk),
            # Wide prefill batches expose enough row-level parallelism to use
            # fewer warps per CTA. Tiny decode batches benefit from four warps.
            num_warps=1 if num_experts == 512 and num_tokens >= 1024 else 4,
        )
    return values, indices


class ParallelTopKMode(TorchDispatchMode):
    """Replace aten top-k only while executing the original compiled router.

    Replacing top-k in the FX graph changes its cache key. A new graph can
    autotune normalization differently from a cached reference, changing
    probability bits even if its mathematical expressions are unchanged.
    This mode leaves tracing, compilation and cache lookup untouched; only
    the runtime top-k call is replaced. The selected values and indices are
    identical, so the same compiled probability kernels see the same inputs.
    """

    @classmethod
    def ignore_compile_internals(cls):
        """Keep this mode out of Dynamo tracing and cache keys; intercept only at runtime."""
        return True

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.ops.aten.topk.default:
            scores, topk, *options = args
            dim = options[0] if options else kwargs.get("dim", -1)
            largest = options[1] if len(options) > 1 else kwargs.get("largest", True)
            sorted_output = options[2] if len(options) > 2 else kwargs.get("sorted", True)
            if (
                scores.is_cuda
                and scores.ndim == 2
                and scores.dtype == torch.float32
                and scores.is_contiguous()
                and dim in (1, -1)
                and largest
                and not sorted_output
                and is_topk_scores_supported(scores.shape[1], topk)
            ):
                return topk_scores(scores, topk)
        return func(*args, **kwargs)


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
    (which selects with ``torch.topk(sorted=False)``) for finite fp32 logits and bias in
    the shape range covered by ``is_sigmoid_topk_supported``. Both indices and
    probabilities match bitwise. Top-k of eight or more is unsupported: its compiled
    reduction order can depend on autotuning and prior batch shapes. The inference
    router replaces only top-k selection for those calls, retaining compiled
    sigmoid and normalization. This kernel is not used in batch-invariant mode.

    Args:
        logits: [num_tokens, num_experts] fp32 router logits.
        expert_bias: [num_experts] fp32 routing bias on the same device as logits
            (selection only). Strided views are supported.
        topk: experts per token, in [2, 7] and no greater than num_experts.
        scaling_factor: optional multiplier applied to the normalized probabilities.
        block_tokens: optional rows per CTA; defaults to one for up to 128 experts
            and four for wider rows.

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
        APPLY_SCALING=bool(scaling_factor),
        BLOCK_T=block_tokens,
        num_warps=num_warps,
    )
    return probs, indices
