"""Shared DS4 context-parallel compression geometry.

The Lite and vLLM implementations intentionally differ at their numerical
kernel boundaries, but must not independently implement token ownership,
fixed-capacity compressor packing, or rank/sequence row remapping.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils


@dataclass(frozen=True)
class CPCompressionGeometry:
    cu_seqlens_compressed: torch.Tensor
    hidden_compact: torch.Tensor
    compressed_group_ids: torch.Tensor
    seq_to_rank_row: torch.Tensor


def prepare_cp_compression_geometry(
    hidden_local: torch.Tensor,
    boundary_hidden: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    global_start: int,
    cp_size: int,
    ratio: int,
) -> CPCompressionGeometry:
    """Build the single authoritative MCore CP compressor row layout."""
    compressed_lens = torch.div(
        cu_seqlens[1:] - cu_seqlens[:-1], ratio, rounding_mode="floor"
    )
    cu_seqlens_compressed = torch.cat(
        (
            torch.zeros_like(cu_seqlens[:1]),
            torch.cumsum(compressed_lens, dim=0, dtype=torch.int32),
        )
    )
    hidden_compact, compressed_group_ids, seq_to_rank_row = (
        cp_utils.prepare_cp_compressor_input(
            hidden_local,
            boundary_hidden,
            cu_seqlens,
            cu_seqlens_compressed,
            global_start,
            cp_size,
            ratio,
        )
    )
    return CPCompressionGeometry(
        cu_seqlens_compressed=cu_seqlens_compressed,
        hidden_compact=hidden_compact,
        compressed_group_ids=compressed_group_ids,
        seq_to_rank_row=seq_to_rank_row,
    )


def gather_cp_compressed_rows(
    local_rows: torch.Tensor,
    seq_to_rank_row: torch.Tensor,
    *,
    cp_group,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return both physical rank-major and logical sequence-major row views."""
    rank_major = gather_from_sequence_parallel_region(local_rows, group=cp_group)
    sequence_major = torch.index_select(
        rank_major, 0, seq_to_rank_row.clamp_min(0).long()
    )
    return rank_major, sequence_major


__all__ = [
    "CPCompressionGeometry",
    "gather_cp_compressed_rows",
    "prepare_cp_compression_geometry",
]
