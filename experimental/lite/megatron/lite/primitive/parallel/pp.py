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

from collections.abc import Sequence
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


def _auto_layout(
    num_hidden_layers: int,
    pp_size: int,
    num_mtp_layers: int,
    *,
    rows: list[list[str]] | None = None,
):
    """Balance ``[E, decoder*N, mtp*K, loss]`` into even contiguous chunks; the
    embedding/MTP/loss slots make their stages carry fewer decoders (e.g. 6/pp4 ->
    [1,2,2,1]) — Megatron's embedding/loss split accounting."""
    from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout

    if rows is None:
        units = ["embedding"] + ["decoder"] * num_hidden_layers + ["mtp"] * max(num_mtp_layers, 0) + ["loss"]
        base, remainder = divmod(len(units), pp_size)
        rows, pos = [], 0
        for size in (base + (1 if s < remainder else 0) for s in range(pp_size)):
            rows.append(units[pos : pos + size])
            pos += size
    return PipelineParallelLayerLayout(rows, pipeline_model_parallel_size=pp_size)


def _validate_decoder_layer_groups(
    num_hidden_layers: int,
    decoder_layer_groups: Sequence[Sequence[int]],
) -> list[list[int]]:
    groups = [list(group) for group in decoder_layer_groups]
    if any(not group for group in groups):
        raise ValueError("decoder_layer_groups must not contain empty groups.")
    flattened = [layer_idx for group in groups for layer_idx in group]
    expected = list(range(num_hidden_layers))
    if flattened != expected:
        raise ValueError(
            "decoder_layer_groups must cover decoder layers 0..num_hidden_layers-1 "
            f"exactly once in order; got {flattened[:8]}...{flattened[-8:]} "
            f"for num_hidden_layers={num_hidden_layers}."
        )
    for group in groups:
        if group != list(range(group[0], group[0] + len(group))):
            raise ValueError(f"decoder_layer_groups must be contiguous; got {group}.")
    return groups


def _auto_layout_with_decoder_groups(
    num_hidden_layers: int,
    pp_size: int,
    num_mtp_layers: int,
    decoder_layer_groups: Sequence[Sequence[int]],
):
    """Auto-balance like ``_auto_layout`` while never splitting protected decoder groups."""
    groups = _validate_decoder_layer_groups(num_hidden_layers, decoder_layer_groups)
    lengths = [len(group) for group in groups]
    prefix = [0]
    for length in lengths:
        prefix.append(prefix[-1] + length)

    tail_slots = max(num_mtp_layers, 0) + 1  # MTP slots plus loss.
    total_cells = num_hidden_layers + 1 + tail_slots
    base, remainder = divmod(total_cells, pp_size)
    target_cells = [base + (1 if stage < remainder else 0) for stage in range(pp_size)]

    def overhead(stage: int) -> int:
        return (1 if stage == 0 else 0) + (tail_slots if stage == pp_size - 1 else 0)

    # Linear partition DP: minimize the largest per-stage cell count, then prefer
    # non-empty decoder stages and closeness to Megatron's unprotected target sizes.
    n_groups = len(groups)
    dp: list[dict[int, tuple[int, int, int, int | None]]] = []
    first: dict[int, tuple[int, int, int, int | None]] = {}
    for end in range(n_groups + 1):
        cells = overhead(0) + prefix[end]
        first[end] = (
            cells,
            1 if end == 0 else 0,
            abs(cells - target_cells[0]),
            None,
        )
    dp.append(first)

    for stage in range(1, pp_size):
        current: dict[int, tuple[int, int, int, int | None]] = {}
        for end in range(n_groups + 1):
            best: tuple[int, int, int, int | None] | None = None
            for start, prev in dp[stage - 1].items():
                if start > end:
                    continue
                cells = overhead(stage) + prefix[end] - prefix[start]
                score = (
                    max(prev[0], cells),
                    prev[1] + (1 if end == start else 0),
                    prev[2] + abs(cells - target_cells[stage]),
                    start,
                )
                if best is None or score[:3] < best[:3]:
                    best = score
            assert best is not None
            current[end] = best
        dp.append(current)

    segments: list[tuple[int, int]] = []
    end = n_groups
    for stage in range(pp_size - 1, 0, -1):
        start = dp[stage][end][3]
        assert start is not None
        segments.append((start, end))
        end = start
    segments.append((0, end))
    segments.reverse()

    rows: list[list[str]] = []
    for stage, (start, end) in enumerate(segments):
        row: list[str] = []
        if stage == 0:
            row.append("embedding")
        for group_len in lengths[start:end]:
            row.extend(["decoder"] * group_len)
        if stage == pp_size - 1:
            row.extend(["mtp"] * max(num_mtp_layers, 0))
            row.append("loss")
        rows.append(row)
    return _auto_layout(
        num_hidden_layers,
        pp_size,
        num_mtp_layers,
        rows=rows,
    )


def build_pipeline_chunk_layout(
    num_hidden_layers: int,
    ps: ParallelState,
    vpp: int | None = None,
    vpp_chunk_id: int | None = None,
    *,
    num_mtp_layers: int = 0,
    decoder_layer_groups: Sequence[Sequence[int]] | None = None,
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
    from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout

    pp_layout = getattr(ps, "pp_layout", None)
    if pp_layout is not None:
        layout = PipelineParallelLayerLayout(pp_layout, pipeline_model_parallel_size=ps.pp_size)
        if layout.virtual_pipeline_model_parallel_size > 1:
            raise NotImplementedError("VPP pp_layout is not supported (one stage per pp rank).")
    elif decoder_layer_groups is not None:
        layout = _auto_layout_with_decoder_groups(
            num_hidden_layers, ps.pp_size, num_mtp_layers, decoder_layer_groups
        )
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
