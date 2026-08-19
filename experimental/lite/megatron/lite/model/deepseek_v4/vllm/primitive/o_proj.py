"""vLLM inverse-RoPE FP8 O-projection with BF16-master VJP."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from ._recompute import visible_functional_vjp


def _inverse_rope(o, positions, cache, nope_dim, rope_dim):
    prefix, rope = o[..., :nope_dim], o[..., nope_dim : nope_dim + rope_dim]
    selected = cache.index_select(0, positions.long()).float()
    cos = selected[..., : rope_dim // 2]
    sin = selected[..., rope_dim // 2 : rope_dim]
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    even, odd = rope[..., 0::2].float(), rope[..., 1::2].float()
    rotated = torch.stack((even * cos + odd * sin, odd * cos - even * sin), dim=-1)
    return torch.cat((prefix.float(), rotated.flatten(-2)), dim=-1)


def o_projection(
    visible_op: Callable,
    o: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    *,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
):
    def functional(o_, wa_, wb_):
        inverse = _inverse_rope(o_, positions, cos_sin_cache, nope_dim, rope_dim)
        grouped = inverse.reshape(inverse.shape[0], n_groups, -1)
        wa = wa_.float().reshape(n_groups, o_lora_rank, -1)
        z = torch.einsum("tgd,grd->tgr", grouped, wa)
        return F.linear(z.flatten(1), wb_.float()).to(o_.dtype)

    return visible_functional_vjp(
        visible_op, functional, (o, wo_a, wo_b), version_indices=(1, 2)
    )


__all__ = ["o_projection"]
