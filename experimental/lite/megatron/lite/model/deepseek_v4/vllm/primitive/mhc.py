from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F

from megatron.lite.primitive.modules.attention.hca import HyperConnection, split_sinkhorn

from ._recompute import visible_functional_vjp


def _pre_graph(x, fn, scale, base, *, mult, iters, eps):
    residual = x
    if residual.ndim == 2:
        residual = residual.unsqueeze(-2).expand(*residual.shape[:-1], mult, residual.shape[-1])
    shape = residual.shape
    flat = residual.flatten(-2)
    rms_inv = 1.0 / (flat.norm(dim=-1, keepdim=True) / math.sqrt(flat.shape[-1]) + eps)
    mixes = F.linear(flat, fn.to(flat.dtype)) * rms_inv
    pre, post, comb = split_sinkhorn(mixes, scale, base, mult, iters, eps)
    hidden = torch.sum(pre.unsqueeze(-1) * residual, dim=-2)
    return residual, post.unsqueeze(-1), comb, hidden


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
        visible_op,
        lambda x_, residual_, post_, comb_: HyperConnection.post(
            x_, residual_, post_.squeeze(-1), comb_
        ),
        (x, residual, post, comb),
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
