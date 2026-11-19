# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Row-sharded Sinkhorn adapter for DeepSeek-V4.1 Algorithm 1.

Rows are token identities, columns are complete feature vectors. The supplied group
contains distinct row shards, never replicas. No matrix all-gather is performed.
"""

import math

import torch
import torch.distributed as dist


def balance_row_shards(
    update: torch.Tensor,
    *,
    global_rows: int,
    group: dist.ProcessGroup | None,
    upstream_balance,
    num_normalization_steps: int = 11,
    zero_row_threshold: float = 1e-3,
    eps: float = 1e-20,
    local: bool = False,
) -> torch.Tensor:
    """Balance complete feature rows using global or approximate shard-local statistics.

    Empty local shards participate in global reductions. A locally wide shard is
    valid even though the upstream serial API requires a globally tall matrix.
    The caller validates arguments and the partition at optimizer construction.
    """
    world_size = 1 if local or group is None else dist.get_world_size(group)
    if world_size == 1 and update.shape[0] >= update.shape[1] and update.shape[0] > 0:
        return upstream_balance(
            update,
            eps=eps,
            num_normalization_steps=num_normalization_steps,
            zero_row_threshold=zero_row_threshold,
        )

    values = update.to(dtype=torch.float32, copy=True)
    row_norms = torch.linalg.vector_norm(values, dim=1, keepdim=True)
    norm_sum = row_norms.sum()
    if world_size > 1:
        dist.all_reduce(norm_sum, group=group)
    rows = values.shape[0] if local else global_rows
    mean_norm = norm_sum / max(rows, 1)
    values.masked_fill_(row_norms <= zero_row_threshold * mean_norm, 0.0)
    # Reuse the pre-masking row norms, as upstream does; masked rows remain zero.
    values.div_(row_norms.add_(eps))
    for _ in range((num_normalization_steps - 1) // 2):
        column_squares = values.square().sum(dim=0, keepdim=True)
        if world_size > 1:
            dist.all_reduce(column_squares, group=group)
        values.div_(column_squares.sqrt_().add_(eps))
        values.div_(torch.linalg.vector_norm(values, dim=1, keepdim=True).add_(eps))
    values.mul_(math.sqrt(update.shape[1]))
    return values.to(update.dtype)
