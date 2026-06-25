# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pipeline parallel layer layout — a thin wrapper over Megatron-core's
``PipelineParallelLayerLayout`` (authorized mcore-reuse), which owns the per-stage
split for non-divisible decoder counts plus MTP. Two pp-only modes:

* **auto** (default, only ``pp`` set): balance ``[E, decoder*N, mtp*K, loss]`` across stages.
* **custom** (``ParallelConfig.pp_layout``): an explicit mcore layout string/list,
  e.g. ``"E|t*5|t*6|t,m,L"``.

Not supported (raise, never mis-place): VPP, and standalone MTP (``m`` off the
final/loss stage — mlite's MTP shares the head there; cross-stage MTP is a follow-up).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.lite.primitive.parallel.state import ParallelState


@dataclass
class PipelineChunkLayout:
    layer_indices: list[int] = field(default_factory=list)
    has_embed: bool = False
    has_head: bool = False
    has_mtp: bool = False


def _auto_layout(num_hidden_layers: int, pp_size: int, num_mtp_layers: int):
    """Balance ``[E, decoder*N, mtp*K, loss]`` into even contiguous chunks; the
    embedding/MTP/loss slots make their stages carry fewer decoders (e.g. 6/pp4 ->
    [1,2,2,1]) — Megatron's embedding/loss split accounting."""
    from megatron.core.transformer.pipeline_parallel_layer_layout import (
        PipelineParallelLayerLayout,
    )

    units = ["embedding"] + ["decoder"] * num_hidden_layers + ["mtp"] * max(num_mtp_layers, 0) + ["loss"]
    base, remainder = divmod(len(units), pp_size)
    rows, pos = [], 0
    for size in (base + (1 if s < remainder else 0) for s in range(pp_size)):
        rows.append(units[pos : pos + size])
        pos += size
    return PipelineParallelLayerLayout(rows, pipeline_model_parallel_size=pp_size)


def build_pipeline_chunk_layout(
    num_hidden_layers: int,
    ps: ParallelState,
    vpp: int | None = None,
    vpp_chunk_id: int | None = None,
    *,
    num_mtp_layers: int = 0,
) -> PipelineChunkLayout:
    """``layer_indices`` / ``has_embed`` / ``has_head`` / ``has_mtp`` for this PP rank,
    from ``ps.pp_layout`` (custom) or an auto-balanced layout. ``has_mtp`` follows the
    layout's ``m`` placement, so MTP is built where the layout says, not a fixed rank."""
    if (vpp is not None and vpp > 1) or vpp_chunk_id is not None:
        raise NotImplementedError("VPP / interleaved pipeline layout is not supported (use vpp=1).")

    if ps.pp_size <= 1:  # no pipeline: this stage owns everything
        return PipelineChunkLayout(
            layer_indices=list(range(num_hidden_layers)),
            has_embed=True,
            has_head=True,
            has_mtp=num_mtp_layers > 0,
        )

    from megatron.core.transformer.enums import LayerType
    from megatron.core.transformer.pipeline_parallel_layer_layout import (
        PipelineParallelLayerLayout,
    )

    pp_layout = getattr(ps, "pp_layout", None)
    if pp_layout is not None:
        layout = PipelineParallelLayerLayout(pp_layout, pipeline_model_parallel_size=ps.pp_size)
        if layout.virtual_pipeline_model_parallel_size > 1:
            raise NotImplementedError("VPP pp_layout is not supported (one stage per pp rank).")
    else:
        layout = _auto_layout(num_hidden_layers, ps.pp_size, num_mtp_layers)

    # validate_layer_layout checks legality and returns mtp_standalone=True when `m` is
    # off the final stage — which mlite's head-coupled MTP cannot run.
    if layout.validate_layer_layout(num_hidden_layers, num_mtp_layers or None):
        raise NotImplementedError(
            "Standalone MTP (pp_layout with `m` off the final/loss stage) is not "
            "implemented yet — mlite's MTP shares the output head there. Use the auto "
            "layout (set only `pp`), or place `m` on the same stage as `L`."
        )
    return PipelineChunkLayout(
        layer_indices=layout.get_layer_id_list(LayerType.decoder, vp_stage=0, pp_rank=ps.pp_rank),
        has_embed=ps.pp_is_first,
        has_head=ps.pp_is_last,
        has_mtp=layout.get_num_layers_to_build(LayerType.mtp, vp_stage=0, pp_rank=ps.pp_rank) > 0,
    )


__all__ = ["PipelineChunkLayout", "build_pipeline_chunk_layout"]
