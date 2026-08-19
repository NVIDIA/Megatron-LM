"""TileLang-visible mHC functions with native functional VJPs."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F

from ._recompute import visible_functional_vjp


def _split_sinkhorn(mixes, scale, base, mult, iters, eps):
    sizes = [mult, mult, mult * mult]
    pre_mix, post_mix, comb_mix = mixes.split(sizes, dim=-1)
    base_pre, base_post, base_comb = base.to(
        dtype=mixes.dtype, device=mixes.device
    ).split(sizes, dim=-1)
    scale = scale.to(dtype=mixes.dtype, device=mixes.device)
    pre = torch.sigmoid(pre_mix * scale[0] + base_pre)
    post = 2 * torch.sigmoid(post_mix * scale[1] + base_post)
    logits = (comb_mix * scale[2] + base_comb).view(
        *comb_mix.shape[:-1], mult, mult
    )
    comb = torch.exp(logits - logits.max(dim=-1, keepdim=True).values)
    for _ in range(iters):
        comb = comb / comb.sum(dim=-1, keepdim=True).clamp(min=eps)
        comb = comb / comb.sum(dim=-2, keepdim=True).clamp(min=eps)
    return pre, post, comb


def _pre_graph(x, fn, scale, base, *, mult, iters, eps):
    residual = x
    if residual.ndim == 2:
        residual = residual.unsqueeze(-2).expand(*residual.shape[:-1], mult, residual.shape[-1])
    shape = residual.shape
    flat = residual.flatten(-2)
    rms_inv = 1.0 / (flat.norm(dim=-1, keepdim=True) / math.sqrt(flat.shape[-1]) + eps)
    mixes = F.linear(flat, fn.to(flat.dtype)) * rms_inv
    pre, post, comb = _split_sinkhorn(mixes, scale, base, mult, iters, eps)
    hidden = torch.sum(pre.unsqueeze(-1) * residual, dim=-2)
    return residual, post.unsqueeze(-1), comb, hidden


def _post_graph(x, residual, post, comb):
    post = post.squeeze(-1)
    return post.to(x.dtype).unsqueeze(-1) * x.unsqueeze(-2) + torch.matmul(
        comb.to(x.dtype), residual.to(x.dtype)
    )


def mhc_pre_broadcast(
    visible_op: Callable,
    x: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    mult: int,
    iters: int,
    eps: float,
    norm_eps: float,
):
    def functional(x_, fn_, scale_, base_, norm_weight_):
        residual, post, comb, hidden = _pre_graph(
            x_, fn_, scale_, base_, mult=mult, iters=iters, eps=eps
        )
        hidden = F.rms_norm(hidden, (hidden.shape[-1],), norm_weight_, norm_eps)
        return residual, post, comb, hidden

    return visible_functional_vjp(
        visible_op,
        functional,
        (x, fn, scale, base, norm_weight),
        version_indices=(1, 2, 3, 4),
    )


def mhc_post(visible_op: Callable, x, residual, post, comb):
    return visible_functional_vjp(
        visible_op, _post_graph, (x, residual, post, comb)
    )


def mhc_head(
    visible_op: Callable,
    x,
    fn,
    scale,
    base,
    *,
    eps: float,
):
    def functional(x_, fn_, scale_, base_):
        flat = x_.flatten(-2).float()
        rstd = torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
        mixes = F.linear(flat, fn_.float()) * rstd
        pre = torch.sigmoid(mixes * scale_.float() + base_.float()) + eps
        return torch.sum(pre.unsqueeze(-1) * x_.float(), dim=-2).to(x_.dtype)

    return visible_functional_vjp(
        visible_op, functional, (x, fn, scale, base), version_indices=(1, 2, 3)
    )


__all__ = ["mhc_head", "mhc_post", "mhc_pre_broadcast"]
