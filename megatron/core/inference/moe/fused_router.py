# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Single-launch router for decode: gating GEMM, softmax, and top-k in one kernel.

The decode router runs as three kernels, and the profile shows why that is expensive out
of proportion to its arithmetic. The gating GEMM is ``[num_tokens, hidden] x
[hidden, num_experts]`` -- 256 x 2048 x 128 here -- whose output is a couple of tiles, so
cuBLASLt splits K to find any parallelism at all and emits a separate ``splitKreduce``
pass to finish the sum. That reduce alone accounts for 48 of the 82 splitK launches per
step, a work category the reference engine has none of, and it exists only because the
GEMM's result has to land in memory before the next kernel can read it.

Fusing removes the boundary rather than the split: one CTA takes a block of tokens, walks
K accumulating the whole ``[BLOCK_M, num_experts]`` logit tile in registers, and then does
the softmax and the top-k selection on it in place. Nothing round-trips through memory.

The trade this makes, and the reason the shape matters: a fused kernel *cannot* split K,
because the softmax needs a token's entire logit row. Parallelism is therefore capped at
``num_tokens / BLOCK_M`` CTAs, and every CTA reads the whole weight matrix. That is a good
trade at decode batch sizes and a bad one at prefill, so the fused path is gated to
decode-sized token counts and declines above them.

Numerics match the reference in kind, not bitwise: ``te_general_gemm`` multiplies bf16 and
accumulates fp32, and so does ``tl.dot``. Only the K-reduction order differs, so the
selected expert *set* is the invariant to check rather than bitwise equality of the
probabilities.
"""

import os
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

# Default off; the reference three-kernel path stays untouched unless this is set.
USE_FUSED_ROUTER: bool = os.environ.get("MCORE_FUSED_ROUTER", "0") == "1"

# Above this the fused shape is the wrong one: with parallelism capped at
# num_tokens / BLOCK_M and every CTA reading the whole weight, a splitting GEMM wins.
FUSED_ROUTER_MAX_TOKENS: int = int(os.environ.get("MCORE_FUSED_ROUTER_MAX_TOKENS", "256"))

# Tile shape, from a BLOCK_M x BLOCK_K x warps sweep at the decode shape; env-overridable
# so the sweep can be repeated on other hardware or other expert counts. A narrow BLOCK_M
# with the widest K tile wins: it maximises the CTA count, and the whole-weight read every
# CTA does is then what hides the K loop.
BLOCK_M: int = int(os.environ.get("MCORE_FUSED_ROUTER_BLOCK_M", "16"))
BLOCK_K: int = int(os.environ.get("MCORE_FUSED_ROUTER_BLOCK_K", "256"))
NUM_WARPS: int = int(os.environ.get("MCORE_FUSED_ROUTER_WARPS", "4"))


if HAVE_TRITON:

    @triton.jit
    def _fused_router_kernel(
        x_ptr,
        w_ptr,
        probs_ptr,
        idx_ptr,
        n_tokens,
        K,
        x_rs,
        w_rs,
        eps_unused,
        N: tl.constexpr,
        TOPK: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Gating GEMM, softmax, and top-k for one block of tokens against all experts.

        The expert axis is a compile-time constant and is never tiled: the whole logit row
        must be live for the softmax and for the selection passes, which is exactly the
        property that forbids splitting K here.
        """
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        ns = tl.arange(0, N)
        row_live = rows < n_tokens

        acc = tl.zeros((BLOCK_M, N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            k_live = ks < K
            x = tl.load(
                x_ptr + rows[:, None] * x_rs + ks[None, :],
                mask=row_live[:, None] & k_live[None, :],
                other=0.0,
            )
            w = tl.load(w_ptr + ns[:, None] * w_rs + ks[None, :], mask=k_live[None, :], other=0.0)
            acc += tl.dot(x, tl.trans(w))

        # Pre-softmax routing: softmax over every expert, then take the top-k of that.
        p = tl.exp(acc - tl.max(acc, axis=1)[:, None])
        p = p / tl.sum(p, axis=1)[:, None]

        # Max-then-mask selection, which yields the top-k in descending order with ties
        # broken toward the lower expert id -- the same order the standalone fused top-k
        # produces, and torch.topk is called with sorted=False here anyway.
        cur = p
        for t in tl.static_range(TOPK):
            best = tl.max(cur, axis=1)
            best_idx = tl.min(tl.where(cur == best[:, None], ns[None, :], N), axis=1)
            tl.store(probs_ptr + rows * TOPK + t, best, mask=row_live)
            tl.store(idx_ptr + rows * TOPK + t, best_idx.to(tl.int64), mask=row_live)
            cur = tl.where(ns[None, :] == best_idx[:, None], -float("inf"), cur)


def fused_router(
    hidden_states: torch.Tensor, weight: torch.Tensor, topk: int, out_dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gating GEMM + softmax + top-k in one launch.

    Args:
        hidden_states: ``[num_tokens, hidden]``; last dim must be contiguous.
        weight: router weight ``[num_experts, hidden]``, last dim contiguous.
        topk: experts per token.
        out_dtype: dtype for the returned probabilities, matching what the reference
            path produces for this model's ``moe_router_dtype``.

    Returns:
        ``(probs, top_indices)``, shapes ``[num_tokens, topk]``; indices are int64 to
        match ``torch.topk``.
    """
    n_tokens, k = hidden_states.shape
    n_experts = weight.shape[0]
    probs = torch.empty(n_tokens, topk, dtype=out_dtype, device=hidden_states.device)
    idx = torch.empty(n_tokens, topk, dtype=torch.int64, device=hidden_states.device)
    _fused_router_kernel[(triton.cdiv(n_tokens, BLOCK_M),)](
        hidden_states,
        weight,
        probs,
        idx,
        n_tokens,
        k,
        hidden_states.stride(0),
        weight.stride(0),
        0.0,
        N=n_experts,
        TOPK=topk,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_warps=NUM_WARPS,
    )
    return probs, idx


def can_use_fused_router(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    topk: int,
) -> bool:
    """Whether the fused kernel reproduces this router's contract at this shape.

    The routing-semantics guards (pre-softmax, no expert groups, no scaling factor, no
    expert bias, no router replay) are the caller's to check, since it owns the config;
    this checks only what the kernel itself requires of the tensors.
    """
    if not (HAVE_TRITON and USE_FUSED_ROUTER):
        return False
    if bias is not None:  # the kernel has no bias epilogue
        return False
    if hidden_states.ndim != 2 or weight.ndim != 2:
        return False
    if not (hidden_states.is_cuda and weight.is_cuda):
        return False
    if hidden_states.dtype != weight.dtype or hidden_states.dtype not in (
        torch.bfloat16,
        torch.float16,
    ):
        return False
    if hidden_states.stride(1) != 1 or weight.stride(1) != 1:
        return False
    if hidden_states.shape[1] != weight.shape[1]:
        return False
    n_experts = weight.shape[0]
    if n_experts & (n_experts - 1):  # tl.arange needs a power of two
        return False
    if topk > n_experts:
        return False
    return hidden_states.shape[0] <= FUSED_ROUTER_MAX_TOKENS
