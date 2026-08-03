# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Single-launch fused q/k RMSNorm for decode-shaped attention.

Qwen3-style attention applies a per-head RMSNorm to the query and to the key
separately, right after the QKV split and before RoPE. In the
``inference_optimized`` decode path these are two distinct ``TENorm`` (TE
``RMSNorm``) module calls, so they show up as two ``rmsnorm_fwd_general``
kernels per layer. Each normalizes a tiny ``[.., head_dim]`` row (head_dim=128
here), so both are launch/latency dominated rather than bandwidth bound —
merging the two launches into one recovers a graph node and its dispatch gap
per layer with no change to the math.

The two source tensors have different shapes and different weights, but both
reduce to a ``[num_rows, head_dim]`` view with a *uniform* row stride (the key
view's leading strides are exact multiples of its head stride), so a single
kernel can process the concatenation of query rows and key rows, selecting the
right weight per row. The normalization itself is identical to TE's RMSNorm:
fp32 mean-of-squares over ``head_dim``, ``rsqrt(var + eps)``, then the
(optionally 1-centered) gamma — so the result matches the two-call path to
bf16 rounding.
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

# Fuse the two QK RMSNorm launches into one. Env-toggleable; default off so the
# untouched two-call path stays byte-identical.
USE_FUSED_QK_NORM: bool = os.environ.get("MCORE_FUSED_QK_NORM", "0") == "1"

# The one-row-per-CTA kernel wins only in the decode regime; above ~256 tokens
# TE's multi-row norm is faster (measured 1.25x at 256 tokens, 0.83x at 384).
# Restrict to decode; prefill / large batches keep the two-call path.
FUSED_QK_NORM_MAX_TOKENS: int = int(os.environ.get("MCORE_FUSED_QK_NORM_MAX_TOKENS", "256"))

# Read the query out of the grouped QKV output rather than out of a repacked copy,
# removing one full-query strided copy per layer per step. Requires the fused norm.
USE_GROUPED_QK_NORM: bool = os.environ.get("MCORE_GROUPED_QK_NORM", "0") == "1"


if HAVE_TRITON:

    @triton.jit
    def _fused_qk_rmsnorm_kernel(
        q_ptr,
        k_ptr,
        qo_ptr,
        ko_ptr,
        wq_ptr,
        wk_ptr,
        n_q_rows,
        q_in_rs,
        k_in_rs,
        q_out_rs,
        k_out_rs,
        eps,
        q_grp_rs,
        HN: tl.constexpr,
        ZERO_CENTERED: tl.constexpr,
        Q_GROUPED: tl.constexpr,
        NPG: tl.constexpr,
        HEADS: tl.constexpr,
    ):
        """One CTA per row across the concatenated [q_rows; k_rows] space.

        Programs with ``row < n_q_rows`` normalize a query row with ``wq``;
        the rest normalize a key row with ``wk``. Both candidate loads are
        issued (the off-path one clamped to row 0 and masked out of the store),
        which keeps the kernel branch-free on the hot path for a 128-wide row.

        With ``Q_GROUPED`` the query is read straight out of the QKV projection's
        output instead of from a repacked copy. There, q heads are grouped with the
        k and v head of their group, so a q row's address needs two strides -- one
        per group (``q_grp_rs``) and one per head inside it -- and no single row
        stride exists. See ``fused_qk_rmsnorm_grouped`` for why that matters.
        """
        row = tl.program_id(0)
        cols = tl.arange(0, HN)
        is_q = row < n_q_rows

        q_row = tl.where(is_q, row, 0)
        k_row = tl.where(is_q, 0, row - n_q_rows)

        if Q_GROUPED:
            head = q_row % HEADS
            q_off = (
                (q_row // HEADS) * (HEADS // NPG) + head // NPG
            ) * q_grp_rs + (head % NPG) * HN
        else:
            q_off = q_row * q_in_rs
        xq = tl.load(q_ptr + q_off + cols).to(tl.float32)
        xk = tl.load(k_ptr + k_row * k_in_rs + cols).to(tl.float32)
        x = tl.where(is_q, xq, xk)

        var = tl.sum(x * x, axis=0) / HN
        inv = 1.0 / tl.sqrt(var + eps)
        xn = x * inv

        wq = tl.load(wq_ptr + cols).to(tl.float32)
        wk = tl.load(wk_ptr + cols).to(tl.float32)
        w = tl.where(is_q, wq, wk)
        if ZERO_CENTERED:
            w = w + 1.0
        y = xn * w

        q_store_mask = is_q & (cols < HN)
        k_store_mask = (row >= n_q_rows) & (cols < HN)
        tl.store(qo_ptr + q_row * q_out_rs + cols, y.to(qo_ptr.dtype.element_ty), mask=q_store_mask)
        tl.store(ko_ptr + k_row * k_out_rs + cols, y.to(ko_ptr.dtype.element_ty), mask=k_store_mask)


def fused_qk_rmsnorm(
    query: torch.Tensor,
    key: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    eps: float,
    zero_centered_gamma: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply per-head RMSNorm to ``query`` and ``key`` in a single kernel.

    Args:
        query: ``[.., head_dim]`` (any leading shape); last dim is normalized.
        key: ``[.., head_dim]``; may be a non-contiguous split view — only the
            last dim needs to be contiguous, which holds for the QKV split.
        weight_q / weight_k: ``[head_dim]`` RMSNorm gammas.
        eps: RMSNorm epsilon.
        zero_centered_gamma: if True, apply ``(1 + gamma)`` (matches TE).

    Returns:
        ``(query_normed, key_normed)`` as fresh contiguous tensors with the same
        shapes and dtypes as the inputs.
    """
    hn = query.shape[-1]
    # No-copy 2D views: the last dim is contiguous and the leading strides are
    # exact multiples of the row stride, so reshape returns a view.
    q2 = query.reshape(-1, hn)
    k2 = key.reshape(-1, hn)

    qo = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    ko = torch.empty(key.shape, dtype=key.dtype, device=key.device)
    qo2 = qo.view(-1, hn)
    ko2 = ko.view(-1, hn)

    n_q_rows = q2.shape[0]
    n_rows = n_q_rows + k2.shape[0]

    _fused_qk_rmsnorm_kernel[(n_rows,)](
        q2,
        k2,
        qo2,
        ko2,
        weight_q,
        weight_k,
        n_q_rows,
        q2.stride(0),
        k2.stride(0),
        qo2.stride(0),
        ko2.stride(0),
        float(eps),
        0,
        HN=hn,
        ZERO_CENTERED=zero_centered_gamma,
        Q_GROUPED=False,
        NPG=1,
        HEADS=1,
        num_warps=1,
    )
    return qo, ko


def fused_qk_rmsnorm_grouped(
    grouped_query: torch.Tensor,
    key: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    eps: float,
    zero_centered_gamma: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same norm, but reading the query straight out of the QKV projection.

    Megatron's QKV projection writes ``[sq, b, ng, (np/ng + 2) * hn]`` -- each group's
    q heads sit next to that group's k and v head. Reshaping the q slice to
    ``[sq, b, np, hn]`` therefore **cannot** be a view: merging the group axis with the
    head axis needs the group stride to equal ``(np/ng) * hn``, and it is
    ``(np/ng + 2) * hn`` instead. For Qwen3-30B that is 1280 against 1024, so
    `Tensor.reshape` silently materializes the whole query -- one full-size strided copy
    per layer per step, which showed up in the profile as a 2.8 us
    ``elementwise_kernel<128,4>`` between the QKV GEMM and this norm and took four
    attempts to attribute.

    Since this norm already writes a fresh output, it can absorb the repack for free:
    read with the two strides the grouped layout needs, write the contiguous
    ``[sq, b, np, hn]`` result the rest of attention wants. Values and arithmetic order
    are unchanged, so the result is bit-identical to normalizing the copy.

    Args:
        grouped_query: the q slice of the QKV output, ``[sq, b, ng, (np/ng) * hn]``.
        key: ``[sq, b, ng, hn]``, as for :func:`fused_qk_rmsnorm`.

    Returns:
        ``(query_normed, key_normed)``; the query is ``[sq, b, np, hn]`` contiguous.
    """
    hn = key.shape[-1]
    sq, b, ng = grouped_query.shape[0], grouped_query.shape[1], grouped_query.shape[2]
    npg = grouped_query.shape[3] // hn
    heads = ng * npg

    k2 = key.reshape(-1, hn)
    qo = torch.empty(sq, b, heads, hn, dtype=grouped_query.dtype, device=grouped_query.device)
    ko = torch.empty(key.shape, dtype=key.dtype, device=key.device)
    qo2 = qo.view(-1, hn)
    ko2 = ko.view(-1, hn)

    n_q_rows = sq * b * heads
    n_rows = n_q_rows + k2.shape[0]

    _fused_qk_rmsnorm_kernel[(n_rows,)](
        grouped_query,
        k2,
        qo2,
        ko2,
        weight_q,
        weight_k,
        n_q_rows,
        0,  # unused: grouped q rows have no single stride
        k2.stride(0),
        qo2.stride(0),
        ko2.stride(0),
        float(eps),
        grouped_query.stride(2),
        HN=hn,
        ZERO_CENTERED=zero_centered_gamma,
        Q_GROUPED=True,
        NPG=npg,
        HEADS=heads,
        num_warps=1,
    )
    return qo, ko


def can_use_grouped_qk_norm(
    q_layernorm, k_layernorm, grouped_query: torch.Tensor, key: torch.Tensor
) -> bool:
    """Whether the grouped-read path is safe, i.e. the copy can be skipped entirely.

    Everything :func:`can_use_fused_qk_norm` requires, plus the two assumptions the
    grouped addressing makes about the QKV output: that a token's groups are evenly
    spaced (so a token stride is ``ng * group_stride``), and that the heads inside a
    group are contiguous.
    """
    if not USE_GROUPED_QK_NORM:
        return False
    if grouped_query.ndim != 4 or key.ndim != 4:
        return False
    hn = key.shape[-1]
    if grouped_query.shape[3] % hn != 0:
        return False
    ng = grouped_query.shape[2]
    if grouped_query.stride(3) != 1 or grouped_query.stride(1) != ng * grouped_query.stride(2):
        return False
    # `key` stands in for the query in the shared check: the grouped query's last dim is
    # `(np/ng) * hn` rather than `hn`, and every property that check reads off the query
    # -- head_dim, cuda-ness, last-dim contiguity, token count from `shape[:-2]` -- is
    # identical between the two here, with the query's own layout verified just above.
    return can_use_fused_qk_norm(q_layernorm, k_layernorm, key, key)


def can_use_fused_qk_norm(q_layernorm, k_layernorm, query: torch.Tensor, key: torch.Tensor) -> bool:
    """Whether the fused kernel reproduces this exact q/k RMSNorm contract.

    Requires both norms to be weight-only RMSNorm modules (TE ``RMSNorm``),
    a power-of-two head dim, matching last dims, and CUDA tensors whose reshape
    to ``[-1, head_dim]`` is a view (last dim contiguous). Anything else — L2
    norm, LayerNorm with bias, odd head dims — falls back to the two-call path.
    """
    if not (HAVE_TRITON and USE_FUSED_QK_NORM):
        return False
    if q_layernorm is None or k_layernorm is None:
        return False
    wq = getattr(q_layernorm, "weight", None)
    wk = getattr(k_layernorm, "weight", None)
    if wq is None or wk is None:
        return False
    # RMSNorm has no bias; reject anything that carries one to stay exact.
    if getattr(q_layernorm, "bias", None) is not None or getattr(k_layernorm, "bias", None) is not None:
        return False
    hn = query.shape[-1]
    if hn != key.shape[-1] or hn != wq.numel() or hn != wk.numel():
        return False
    if (hn & (hn - 1)) != 0:  # not a power of two
        return False
    if not (query.is_cuda and key.is_cuda):
        return False
    # Last dim must be contiguous so the [-1, hn] reshape is a no-copy view.
    if query.stride(-1) != 1 or key.stride(-1) != 1:
        return False
    # Decode-only: count tokens (all leading dims except heads and head_dim).
    n_tokens = 1
    for d in query.shape[:-2]:
        n_tokens *= d
    if n_tokens > FUSED_QK_NORM_MAX_TOKENS:
        return False
    return True
