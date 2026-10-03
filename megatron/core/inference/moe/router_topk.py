# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused softmax + top-k router selection for decode-shaped MoE inference.

The eager path runs ``torch.softmax`` over the expert dimension and then
``torch.topk``. At decode shapes (``[num_tokens, 128]`` fp32) both kernels are
launch/latency dominated and ``topk`` in particular runs a multi-pass radix
select. One CTA per token can do the whole thing out of registers.
"""

import os
from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

# Fuse the router softmax and top-k into one Triton kernel. Env-toggleable; default off.
USE_FUSED_ROUTER_TOPK: bool = os.environ.get("MCORE_ROUTER_FUSED_TOPK", "0") == "1"

# Above this token count the two-kernel torch path amortizes its launch cost and the
# one-CTA-per-token kernel stops being the right shape, so the fused path is decode-only.
FUSED_ROUTER_TOPK_MAX_TOKENS: int = 1024


if HAVE_TRITON:

    @triton.jit
    def _softmax_topk_kernel(
        logits_ptr,
        probs_ptr,
        indices_ptr,
        num_experts,
        TOPK: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        """One CTA per token: softmax over experts, then TOPK selection passes.

        Selection is max-then-mask, which yields the top-k in descending score
        order with ties broken toward the lower expert id. ``torch.topk`` is
        called with ``sorted=False`` during inference, so its own order is
        unspecified and any consistent order is an equally valid answer.
        """
        token_id = tl.program_id(0)
        offs = tl.arange(0, BLOCK_E)
        mask = offs < num_experts

        x = tl.load(logits_ptr + token_id * num_experts + offs, mask=mask, other=-float("inf"))
        x = x.to(tl.float32)
        m = tl.max(x, axis=0)
        e = tl.exp(x - m)
        e = tl.where(mask, e, 0.0)
        p = e / tl.sum(e, axis=0)

        cur = tl.where(mask, p, -float("inf"))
        for k in tl.static_range(TOPK):
            best = tl.max(cur, axis=0)
            is_best = cur == best
            best_idx = tl.min(tl.where(is_best, offs, num_experts), axis=0)
            tl.store(probs_ptr + token_id * TOPK + k, best)
            tl.store(indices_ptr + token_id * TOPK + k, best_idx)
            cur = tl.where(offs == best_idx, -float("inf"), cur)


def fused_softmax_topk(logits: torch.Tensor, topk: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-softmax top-k routing in a single kernel.

    Args:
        logits: ``[num_tokens, num_experts]``, any float dtype.
        topk: number of experts per token.

    Returns:
        ``(probs, top_indices)`` with shapes ``[num_tokens, topk]``. ``probs``
        matches the dtype of ``logits`` (the softmax itself is fp32, as in the
        eager path); ``top_indices`` is int64 to match ``torch.topk``.
    """
    assert logits.dim() == 2, f"Expected 2D logits, got {logits.dim()}D"
    num_tokens, num_experts = logits.shape
    probs = torch.empty(num_tokens, topk, dtype=logits.dtype, device=logits.device)
    indices = torch.empty(num_tokens, topk, dtype=torch.int64, device=logits.device)
    _softmax_topk_kernel[(num_tokens,)](
        logits,
        probs,
        indices,
        num_experts,
        TOPK=topk,
        BLOCK_E=triton.next_power_of_2(num_experts),
        num_warps=1,
    )
    return probs, indices


def can_use_fused_softmax_topk(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool,
    num_groups,
    group_topk,
    scaling_factor,
    score_function: str,
    expert_bias,
    router_replay,
) -> bool:
    """Whether the fused kernel reproduces this exact routing contract."""
    return (
        HAVE_TRITON
        and USE_FUSED_ROUTER_TOPK
        and logits.dim() == 2
        and logits.is_cuda
        and logits.shape[0] <= FUSED_ROUTER_TOPK_MAX_TOKENS
        and score_function == "softmax"
        and use_pre_softmax
        and num_groups is None
        and group_topk is None
        and not scaling_factor
        and expert_bias is None
        and router_replay is None
        # topk == 1 is excluded because the reference path ends in
        # `top_indices.squeeze(1)`, which drops the k dimension only at k == 1.
        # This kernel always returns [num_tokens, topk], so honoring k == 1 here
        # would hand the dispatcher a differently-ranked tensor than the path it
        # replaces.
        and 1 < topk <= logits.shape[1]
    )
