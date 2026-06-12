# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Context/pipeline parallel sequence splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from megatron.lite.primitive.utils import ensure_divisible

if TYPE_CHECKING:
    from megatron.lite.primitive.parallel.state import ParallelState


@dataclass
class PipelineChunkLayout:
    layer_indices: list[int] = field(default_factory=list)
    has_embed: bool = False
    has_head: bool = False


def build_pipeline_chunk_layout(
    num_hidden_layers: int,
    ps: ParallelState,
    vpp: int | None = None,
    vpp_chunk_id: int | None = None,
) -> PipelineChunkLayout:
    """Compute layer_indices, has_embed, has_head for this PP rank / VPP chunk."""
    if vpp_chunk_id is not None:
        assert vpp is not None
        layers_per_chunk = ensure_divisible(num_hidden_layers, ps.pp_size * vpp)
        start = ps.pp_rank * layers_per_chunk + vpp_chunk_id * (ps.pp_size * layers_per_chunk)
        layer_indices = list(range(start, start + layers_per_chunk))
        has_embed = ps.pp_is_first and vpp_chunk_id == 0
        has_head = ps.pp_is_last and vpp_chunk_id == vpp - 1
    elif vpp is not None:
        layers_per_chunk = ensure_divisible(num_hidden_layers, ps.pp_size * vpp)
        layer_indices = []
        for chunk in range(vpp):
            start = ps.pp_rank * layers_per_chunk + chunk * (ps.pp_size * layers_per_chunk)
            layer_indices.extend(range(start, start + layers_per_chunk))
        has_embed = ps.pp_is_first
        has_head = ps.pp_is_last
    else:
        layers_per_stage = ensure_divisible(num_hidden_layers, ps.pp_size)
        start = ps.pp_rank * layers_per_stage
        layer_indices = list(range(start, start + layers_per_stage))
        has_embed = ps.pp_is_first
        has_head = ps.pp_is_last
    return PipelineChunkLayout(layer_indices=layer_indices, has_embed=has_embed, has_head=has_head)


__all__ = ["PipelineChunkLayout", "build_pipeline_chunk_layout"]
