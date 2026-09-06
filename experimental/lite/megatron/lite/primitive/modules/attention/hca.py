# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.fusions.fused_mhc_kernels import fused_h_aggregate, fused_h_post_bda


# Core's mHC helpers are ``@torch.compile``d for the same reason: the mapping
# maths is a long chain of narrow elementwise ops over ``[s, b, (2 + n) * n]``,
# where each eager op is its own launch reading and writing all of HBM. Compiling
# fuses the chain without touching the arithmetic. It is deliberately not
# replaced by ``fused_sinkhorn``: Core regularises with ``softmax(...) + eps`` and
# denominators of ``sum + eps``, while this module uses ``exp(l - max)`` with
# ``clamp(min=eps)`` denominators. The two agree to within the regularisation but
# are not the same function, and swapping them here would change DeepSeek-V4
# numerics under the heading of a fusion change.
@torch.compile
def split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    split_sizes = [hc_mult, hc_mult, hc_mult * hc_mult]
    pre_mix, post_mix, comb_mix = mixes.split(split_sizes, dim=-1)
    base_pre, base_post, base_comb = hc_base.to(dtype=mixes.dtype, device=mixes.device).split(
        split_sizes, dim=-1
    )
    scale = hc_scale.to(dtype=mixes.dtype, device=mixes.device)
    pre = torch.sigmoid(pre_mix * scale[0] + base_pre)
    post = 2 * torch.sigmoid(post_mix * scale[1] + base_post)
    comb_logits = (comb_mix * scale[2] + base_comb).view(*comb_mix.shape[:-1], hc_mult, hc_mult)
    comb = torch.exp(comb_logits - comb_logits.max(dim=-1, keepdim=True).values)
    for _ in range(iters):
        comb = comb / comb.sum(dim=-1, keepdim=True).clamp(min=eps)
        comb = comb / comb.sum(dim=-2, keepdim=True).clamp(min=eps)
    return pre, post, comb


def _aggregate_native(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """CPU form of ``fused_h_aggregate``; see ``_use_fused`` for why it exists."""
    return torch.sum(pre.unsqueeze(-1) * x, dim=2)


def _post_native(
    x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
) -> torch.Tensor:
    """CPU form of ``fused_h_post_bda``; see ``_use_fused`` for why it exists."""
    dtype = x.dtype
    placed = post.to(dtype).unsqueeze(-1) * x.unsqueeze(-2)
    return placed + torch.matmul(comb.to(dtype), residual.to(dtype))


def _use_fused(x: torch.Tensor) -> bool:
    """Core's fused mHC entry points are CUDA-only in practice.

    ``fused_h_post_bda`` and ``fused_h_aggregate`` dispatch to Triton whenever
    Triton is importable and never look at the device, so they raise
    ``ValueError: Pointer argument cannot be accessed from Triton`` on CPU
    tensors. Core never hits that because it only calls them on CUDA; this module
    is also exercised on CPU, so the device decides here instead.
    """
    return x.is_cuda


class HyperConnection(nn.Module):
    def __init__(self, hidden_size: int, hc_mult: int, sinkhorn_iters: int, eps: float):
        super().__init__()
        mix = (2 + hc_mult) * hc_mult
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        self.fn = nn.Parameter(torch.empty(mix, hc_mult * hidden_size, dtype=torch.float32))
        self.base = nn.Parameter(torch.empty(mix, dtype=torch.float32))
        self.scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fn)
        nn.init.zeros_(self.base)
        nn.init.ones_(self.scale)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            x = x.unsqueeze(2).expand(*x.shape[:2], self.hc_mult, x.size(-1))
        shape, dtype = x.shape, x.dtype
        xf = x.flatten(2)
        rms_inv = 1.0 / (xf.norm(dim=-1, keepdim=True) / math.sqrt(xf.shape[-1]) + self.eps)
        mixes = F.linear(xf, self.fn.to(device=x.device, dtype=dtype)) * rms_inv
        pre, post, comb = split_sinkhorn(
            mixes, self.scale, self.base, self.hc_mult, self.sinkhorn_iters, self.eps
        )
        # ``fused_h_aggregate`` is ``(x * h_pre.unsqueeze(-1)).sum(dim=2)``, the
        # same expression written here, so this is a kernel swap and not a change
        # of formula -- unlike the Sinkhorn and compute_h helpers next to it,
        # which differ from Core in their regularisation.
        xs = xf.view(shape)
        y = fused_h_aggregate(xs, pre) if _use_fused(xs) else _aggregate_native(xs, pre)
        return y.to(dtype), post, comb

    @staticmethod
    def post(
        x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
    ) -> torch.Tensor:
        dtype = x.dtype
        # Core defines the mixing term as ``h_res.T @ residual`` while this module
        # carries ``comb`` in the opposite orientation, so the transpose converts
        # between the two conventions rather than being a layout tweak: passing
        # ``comb`` unchanged silently computes a different residual mixing.
        # ``contiguous()`` is required because the native path reshapes with
        # ``view()``; the copy spans ``[s, b, hc_mult, hc_mult]`` and is negligible
        # beside the batched matmul it feeds.
        if not _use_fused(x):
            return _post_native(x, residual, post, comb)
        h_res = comb.to(dtype).transpose(-1, -2).contiguous()
        return fused_h_post_bda(h_res, residual.to(dtype), post.to(dtype), x, None)
