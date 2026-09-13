# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pure grouped-query attention helpers."""

from __future__ import annotations

import torch

from megatron.lite.primitive.utils import ensure_divisible


def split_grouped_qkvg(
    qkv: torch.Tensor, *, num_heads: int, num_kv_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lead = qkv.shape[:-1]
    q_heads_per_group = ensure_divisible(num_heads, num_kv_heads)
    group_width = (2 * q_heads_per_group + 2) * head_dim
    grouped = qkv.reshape(*lead, num_kv_heads, group_width)
    query, gate, key, value = grouped.split(
        [q_heads_per_group * head_dim, q_heads_per_group * head_dim, head_dim, head_dim], dim=-1
    )
    return (
        query.reshape(*lead, num_heads, head_dim),
        gate.reshape(*lead, num_heads, head_dim),
        key.reshape(*lead, num_kv_heads, head_dim),
        value.reshape(*lead, num_kv_heads, head_dim),
    )


def split_grouped_qkvg_for_tp(
    qkv: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    tp_rank: int,
    tp_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select one replicated KV group and this TP rank's query heads."""
    replicas_per_kv_head = ensure_divisible(tp_size, num_kv_heads)
    q_heads_per_group = ensure_divisible(num_heads, num_kv_heads)
    q_heads_per_rank = ensure_divisible(q_heads_per_group, replicas_per_kv_head)
    group_width = (2 * q_heads_per_group + 2) * head_dim
    kv_group_rank = tp_rank // replicas_per_kv_head
    q_rank_in_group = tp_rank % replicas_per_kv_head
    local_group = qkv.narrow(-1, kv_group_rank * group_width, group_width)
    query, gate, key, value = split_grouped_qkvg(
        local_group,
        num_heads=q_heads_per_group,
        num_kv_heads=1,
        head_dim=head_dim,
    )
    q_start = q_rank_in_group * q_heads_per_rank
    q_end = q_start + q_heads_per_rank
    return query[..., q_start:q_end, :], gate[..., q_start:q_end, :], key, value


__all__ = ["split_grouped_qkvg", "split_grouped_qkvg_for_tp"]
