# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""On-device tokenizer compression and official Engram n-gram hashing."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core.utils import get_pg_rank, get_pg_size


def compress_token_ids(input_ids: Tensor, tokenizer_remap: Tensor) -> Tensor:
    """Map nonnegative raw token IDs to canonical IDs entirely on device."""
    safe_ids = input_ids.clamp_min(0)
    compressed = tokenizer_remap[safe_ids]
    return torch.where(input_ids >= 0, compressed, input_ids)


def shift_right_reset_at_eos(token_ids: Tensor, shift: int, eos_token_id: int) -> Tensor:
    """Shift tokens right by ``shift`` positions without crossing EOS document boundaries.

    Faithful to the official Qwen ``_shift_right_ignore_eos``: for every position, the segment
    starts right after the previous EOS in the row; positions whose shifted source would fall
    before their segment start (or before the row start) yield ``eos_token_id`` instead. EOS
    tokens themselves terminate the segment they end, so n-grams never mix two documents. This
    covers plain rows and packed rows alike, because boundaries live in the token stream.
    """
    if shift == 0:
        return token_ids
    batch_size, sequence_length = token_ids.shape
    positions = torch.arange(sequence_length, device=token_ids.device, dtype=torch.int64)
    eos_positions = torch.where(token_ids == eos_token_id, positions, -1)
    previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
    previous_eos = torch.cat(
        [eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim=1
    )
    position_in_segment = positions.unsqueeze(0) - (previous_eos + 1)
    source_positions = positions - shift
    gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
    shifted = token_ids.gather(dim=1, index=gather_positions)
    valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
    return torch.where(valid, shifted, token_ids.new_full((), eos_token_id))


def build_ngram_hashes(
    input_ids: Tensor,
    tokenizer_remap: Tensor | None,
    multipliers: Tensor,
    table_sizes: Tensor,
    max_ngram_order: int,
    num_hash_heads: int,
    boundary_token_id: int,
    reset_at_boundary: bool = False,
) -> Tensor:
    """Compute official multiplicative-XOR multi-head hashes.

    Args:
        input_ids: Raw token IDs with shape ``[batch, sequence]``.
        tokenizer_remap: Raw-to-compressed token map with shape ``[vocab]``, or None to hash
            raw token IDs directly (qwen variant).
        multipliers: Odd int64 multiplier per suffix position.
        table_sizes: Prime modulus for each order/head in order-major layout.
        max_ngram_order: Largest suffix order to hash.
        num_hash_heads: Number of distinct prime tables per order.
        boundary_token_id: Value filling the n-gram window before the start of a document.
        reset_at_boundary: When True, suffix windows reset at every ``boundary_token_id``
            occurrence so n-grams never cross packed document boundaries.

    Returns:
        Hash IDs with shape ``[batch, sequence, (max_ngram_order - 1) * heads]``.
    """
    if input_ids.ndim != 2:
        raise ValueError(
            f"Engram input_ids must have shape [batch, sequence], got {input_ids.shape}."
        )
    tokens = input_ids.to(torch.int64)
    compressed = tokens if tokenizer_remap is None else compress_token_ids(tokens, tokenizer_remap)
    sequence_length = compressed.shape[1]
    if reset_at_boundary:
        suffixes = [
            shift_right_reset_at_eos(compressed, shift, boundary_token_id)
            for shift in range(max_ngram_order)
        ]
    else:
        suffixes = [compressed]
        for shift in range(1, max_ngram_order):
            suffixes.append(
                F.pad(compressed, (shift, 0), value=boundary_token_id)[:, :sequence_length]
            )

    hashes = []
    table_index = 0
    for order in range(2, max_ngram_order + 1):
        mixed = suffixes[0] * multipliers[0]
        for suffix_index in range(1, order):
            mixed = torch.bitwise_xor(mixed, suffixes[suffix_index] * multipliers[suffix_index])
        for _ in range(num_hash_heads):
            hashes.append(torch.remainder(mixed, table_sizes[table_index]))
            table_index += 1
    return torch.stack(hashes, dim=-1)


def slice_hashes_for_sequence_parallel(
    hash_ids: Tensor, local_sequence_length: int, tp_group
) -> Tensor:
    """Select the contiguous SP interval after hashes were computed globally."""
    full_sequence_length = hash_ids.shape[1]
    tp_size = get_pg_size(tp_group)
    if full_sequence_length == local_sequence_length:
        return hash_ids
    expected_sequence_length = local_sequence_length * tp_size
    if full_sequence_length != expected_sequence_length:
        raise ValueError(
            "Engram full hash sequence is incompatible with the local SP hidden state: "
            f"full={full_sequence_length}, local={local_sequence_length}, TP={tp_size}."
        )
    start = get_pg_rank(tp_group) * local_sequence_length
    return hash_ids[:, start : start + local_sequence_length]
