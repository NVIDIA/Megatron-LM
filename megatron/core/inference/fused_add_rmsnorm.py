# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Single-launch fused residual-add + RMSNorm for the decode path.

Every transformer block boundary in the ``inference_optimized`` path runs a
bias-dropout-add (``get_bias_dropout_add``; for Qwen this is a plain
``residual + hidden`` since there is no bias and dropout is disabled at decode)
immediately followed by an RMSNorm (``input_layernorm`` / ``pre_mlp_layernorm``).
In the profile these are two separate kernels per boundary — a
``triton_poi_fused_add_copy`` (~1.5 µs) then a TE ``rmsnorm_fwd_tuned``
(~2.9 µs) — i.e. two launches and two CUDA-graph nodes per boundary, ×2 per
layer ×48 layers. vLLM instead runs one ``triton_..._add_..._rms_norm`` kernel.

This module provides that single kernel: ``new_residual = hidden + residual``
followed by ``rmsnorm(new_residual) * gamma``, returning both the normalized
activation (for the next GEMM) and the updated residual (for the next
bias-dropout-add). The add is done in fp32 and rounded to bf16, matching
PyTorch's bf16 add; the RMSNorm is fp32 mean-of-squares → ``rsqrt(var+eps)`` →
(optionally 1-centered) gamma, matching TE's RMSNorm to bf16 rounding.

Env-gated (``MCORE_FUSED_ADD_NORM``); default off so the untouched two-kernel
path is used unless explicitly enabled.
"""

import os
from typing import Optional, Tuple

import torch

from megatron.core.inference import fusion_diag as _diag

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

USE_FUSED_ADD_NORM: bool = os.environ.get("MCORE_FUSED_ADD_NORM", "0") == "1"

# Same kernel applied one boundary later: the ``mlp_bda`` residual add at the end of
# a layer, fused with the *next* layer's QKV input RMSNorm. Gated separately so the
# two fusions can be A/B'd independently -- this one reaches across the layer
# boundary and into ``linear_qkv``, so it carries more structural risk.
USE_FUSED_ADD_NORM_QKV: bool = os.environ.get("MCORE_FUSED_ADD_NORM_QKV", "0") == "1"

# The one-row-per-CTA kernel is launch/latency bound; it wins in the decode
# regime and should not be used for large prefill token counts.
FUSED_ADD_NORM_MAX_TOKENS: int = int(os.environ.get("MCORE_FUSED_ADD_NORM_MAX_TOKENS", "256"))


if HAVE_TRITON:

    @triton.jit
    def _fused_add_rmsnorm_kernel(
        x_ptr,
        res_ptr,
        w_ptr,
        o_ptr,
        nr_ptr,
        x_rs,
        res_rs,
        o_rs,
        nr_rs,
        eps,
        H: tl.constexpr,
        ZERO_CENTERED: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """One CTA per token row: add the residual, store it, then RMSNorm."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < H

        x = tl.load(x_ptr + row * x_rs + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(res_ptr + row * res_rs + cols, mask=mask, other=0.0).to(tl.float32)
        s = x + r
        tl.store(nr_ptr + row * nr_rs + cols, s.to(nr_ptr.dtype.element_ty), mask=mask)

        var = tl.sum(s * s, axis=0) / H
        inv = 1.0 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        if ZERO_CENTERED:
            w = w + 1.0
        y = s * inv * w
        tl.store(o_ptr + row * o_rs + cols, y.to(o_ptr.dtype.element_ty), mask=mask)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def fused_add_rmsnorm(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    zero_centered_gamma: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse ``residual + hidden`` and RMSNorm into one launch.

    Args:
        hidden: activation to add to the residual; ``[.., H]`` (last dim
            normalized, must be contiguous).
        residual: residual stream tensor, same shape as ``hidden``.
        weight: ``[H]`` RMSNorm gamma.
        eps: RMSNorm epsilon.
        zero_centered_gamma: if True, apply ``(1 + gamma)`` (matches TE).

    Returns:
        ``(normed, new_residual)`` — ``normed`` feeds the next GEMM, and
        ``new_residual = hidden + residual`` is carried to the next add.
    """
    hn = hidden.shape[-1]
    x2 = hidden.reshape(-1, hn)
    r2 = residual.reshape(-1, hn)

    o = torch.empty(hidden.shape, dtype=hidden.dtype, device=hidden.device)
    nr = torch.empty(hidden.shape, dtype=hidden.dtype, device=hidden.device)
    o2 = o.view(-1, hn)
    nr2 = nr.view(-1, hn)

    n_rows = x2.shape[0]
    block = _next_pow2(hn)
    _fused_add_rmsnorm_kernel[(n_rows,)](
        x2,
        r2,
        weight,
        o2,
        nr2,
        x2.stride(0),
        r2.stride(0),
        o2.stride(0),
        nr2.stride(0),
        float(eps),
        H=hn,
        ZERO_CENTERED=zero_centered_gamma,
        BLOCK=block,
        num_warps=8,
    )
    return o, nr


def _tensors_compatible(
    w: torch.Tensor, hidden: torch.Tensor, residual: torch.Tensor, site: str = ""
) -> bool:
    """Shared shape/layout/token-count check for the fused add+RMSNorm kernel.

    This runs twice per layer per decode step, so it keeps the original
    short-circuiting form: the diagnostic breakdown is built only when
    ``MCORE_FUSION_DIAG`` is set, never on the default path.
    """
    if _diag.ENABLED and site:
        _report_compatibility(w, hidden, residual, site)

    hn = hidden.shape[-1]
    if hidden.shape != residual.shape or hn != w.numel():
        return False
    if not (hidden.is_cuda and residual.is_cuda):
        return False
    if hidden.stride(-1) != 1 or residual.stride(-1) != 1:
        return False
    n_tokens = 1
    for d in hidden.shape[:-1]:
        n_tokens *= d
    if n_tokens > FUSED_ADD_NORM_MAX_TOKENS:
        return False
    return True


def _report_compatibility(
    w: torch.Tensor, hidden: torch.Tensor, residual: torch.Tensor, site: str
) -> None:
    """Name the individual compatibility check that rejects this call site.

    Split out because a gate that silently returns False is unfalsifiable from a
    profile: `FUSION-INERT-S17` spent a session on a fusion that was enabled and
    never firing, which this would have answered immediately.
    """
    n_tokens = 1
    for d in hidden.shape[:-1]:
        n_tokens *= d
    _diag.report(
        site,
        lambda: {
            "shapes_match": hidden.shape == residual.shape,
            "gamma_matches_H": hidden.shape[-1] == w.numel(),
            "on_cuda": hidden.is_cuda and residual.is_cuda,
            "last_dim_contiguous": hidden.stride(-1) == 1 and residual.stride(-1) == 1,
            "decode_sized": n_tokens <= FUSED_ADD_NORM_MAX_TOKENS,
            "info:n_tokens": n_tokens,
            "info:max_tokens": FUSED_ADD_NORM_MAX_TOKENS,
            "info:hidden": tuple(hidden.shape),
        },
    )


def can_use_fused_add_rmsnorm(norm_module, hidden: torch.Tensor, residual: torch.Tensor) -> bool:
    """Whether the fused kernel reproduces this exact add+RMSNorm contract.

    Requires a weight-only RMSNorm module (no bias), matching contiguous
    ``[.., H]`` shapes on CUDA, and a decode-sized token count.
    """
    if not (HAVE_TRITON and USE_FUSED_ADD_NORM):
        return False
    if norm_module is None:
        return False
    w = getattr(norm_module, "weight", None)
    if w is None:
        return False
    if getattr(norm_module, "bias", None) is not None:
        return False
    return _tensors_compatible(w, hidden, residual, site="pre_mlp_layernorm")


def can_use_fused_add_rmsnorm_qkv(
    weight: Optional[torch.Tensor], hidden: torch.Tensor, residual: torch.Tensor
) -> bool:
    """Whether the fused kernel can absorb the *next* layer's QKV input RMSNorm.

    Separate gate from :data:`USE_FUSED_ADD_NORM` because this fuses across the
    layer boundary: the norm being absorbed belongs to the following layer's
    ``linear_qkv``, and its weight arrives as a raw tensor
    (``layer_norm_weight``) rather than as a norm module.
    """
    if not (HAVE_TRITON and USE_FUSED_ADD_NORM_QKV):
        return False
    if weight is None:
        return False
    return _tensors_compatible(weight, hidden, residual, site="next_layer_qkv_norm")
