# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Context-parallel batch partitioning helpers."""

from typing import Any, Dict

import torch


def _get_batch_on_this_cp_rank_contiguous(
    batch: Dict[str, Any], cp_group: torch.distributed.ProcessGroup
) -> Dict[str, Any]:
    """Use contiguous CP shards while keeping dense attention-mask queries zigzag."""
    cp_size = torch.distributed.get_world_size(cp_group)
    cp_rank = torch.distributed.get_rank(cp_group)

    sequence_keys = ('tokens', 'labels', 'loss_mask', 'position_ids')
    if cp_size == 1:
        return batch

    for key, val in batch.items():
        if val is None:
            continue
        if key == 'attention_mask':
            seq_dim = 2
            if val.shape[seq_dim] % (2 * cp_size) != 0:
                raise ValueError(
                    "The attention-mask sequence length must be divisible by 2 * CP size, "
                    f"got {val.shape[seq_dim]} and CP size {cp_size}."
                )
            segment_len = val.shape[seq_dim] // (2 * cp_size)
            front = val.narrow(seq_dim, cp_rank * segment_len, segment_len)
            back = val.narrow(seq_dim, (2 * cp_size - cp_rank - 1) * segment_len, segment_len)
            batch[key] = torch.cat((front, back), dim=seq_dim)
            continue

        if key not in sequence_keys:
            continue

        seq_dim = 1
        if val.shape[seq_dim] % (2 * cp_size) != 0:
            raise ValueError(
                "The sequence length must be divisible by 2 * CP size so the contiguous shard "
                "can be redistributed to zigzag attention, "
                f"got {val.shape[seq_dim]} and CP size {cp_size} for {key!r}."
            )
        local_seq_len = val.shape[seq_dim] // cp_size
        batch[key] = val.narrow(seq_dim, cp_rank * local_seq_len, local_seq_len).contiguous()

    return batch
