"""Context-parallel geometry delegated to MCore CSA utilities."""

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
        cu_seqlens_compressed,
        hidden_compact,
        compressed_group_ids,
        seq_to_rank_row,
    )


def gather_cp_compressed_rows(
    local_rows: torch.Tensor,
    seq_to_rank_row: torch.Tensor,
    *,
    cp_group,
) -> tuple[torch.Tensor, torch.Tensor]:
    rank_major = gather_from_sequence_parallel_region(local_rows, group=cp_group)
    return rank_major, torch.index_select(
        rank_major, 0, seq_to_rank_row.clamp_min(0).long()
    )
