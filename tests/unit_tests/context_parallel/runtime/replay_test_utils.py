# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared packed-token layout helpers for runtime CP replay tests."""

import torch


def _zigzag_shard(hidden_states, lengths, cp_size, cp_rank):
    """Slice each packed sequence into its rank's paired front/back CP chunks."""
    parts = []
    for sequence in hidden_states.split(lengths):
        chunks = sequence.chunk(2 * cp_size)
        parts.extend((chunks[cp_rank], chunks[2 * cp_size - cp_rank - 1]))
    return torch.cat(parts).contiguous()
