"""Pure grouped-query attention helpers."""

from __future__ import annotations

import torch

from megatron.lite.primitive.utils import ensure_divisible


def split_grouped_qkvg(
    qkv: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lead = qkv.shape[:-1]
    q_heads_per_group = ensure_divisible(num_heads, num_kv_heads)
    group_width = (2 * q_heads_per_group + 2) * head_dim
    grouped = qkv.reshape(*lead, num_kv_heads, group_width)
    query, gate, key, value = grouped.split(
        [
            q_heads_per_group * head_dim,
            q_heads_per_group * head_dim,
            head_dim,
            head_dim,
        ],
        dim=-1,
    )
    return (
        query.reshape(*lead, num_heads, head_dim),
        gate.reshape(*lead, num_heads, head_dim),
        key.reshape(*lead, num_kv_heads, head_dim),
        value.reshape(*lead, num_kv_heads, head_dim),
    )


__all__ = ["split_grouped_qkvg"]
