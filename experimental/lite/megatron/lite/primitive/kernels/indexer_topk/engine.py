# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The LiteTopK selection engine (internal).

:class:`LiteTopKEngine` runs the tiles a :class:`~.planner.SegmentPlan` gives to LiteTopK
through one route of an external plugin (ABI v1). Per segment it:

1. starts a plugin call under a private rolling carry key (``begin_call`` drops a stale carry);
2. copies the segment's quantized keys into a block-major key cache; the plugin and the
   reference selector therefore read the same key bytes;
3. seeds the first tile group: with ``seed_bootstrap="reference"`` the selections of the plan's
   vote rows (already selected by the reference selector) are stashed as the HOT carry, with
   ``"identity"`` the plugin starts cold;
4. runs every tile group: one ``prepare_permuted_gather`` plan per group, then one
   ``try_large_exact_once_chunk`` per tile with host-computed key windows, writing the ids
   into the output rows and one status code per row into ``status``. The last tile of a group
   publishes the carry that seeds the next group;
5. drops the rolling carry, also when an error interrupts the segment. No carry outlives a
   call or crosses segments.

A host-side decline (the plugin returns no plan, or refuses a tile) hands the rest of that group
to the caller's fallback, and the next group is seeded again from the rows that precede it.
Nothing here synchronizes with the device: row statuses are read by the caller.

The engine never reads the environment and issues no collective: every rank of a
context-parallel group selects its own rows from keys the model has already gathered.
"""

from __future__ import annotations

import contextlib
from collections import Counter
from collections.abc import Callable, Hashable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from megatron.lite.primitive.kernels.indexer_topk.config import (
    IndexerTopKFormat,
    ResolvedIndexerTopKTuning,
)
from megatron.lite.primitive.kernels.indexer_topk.layout import QueryLayout
from megatron.lite.primitive.kernels.indexer_topk.planner import SegmentPlan, Tile, TileGroup
from megatron.lite.primitive.kernels.indexer_topk.plugins.abi import RouteCapability
from megatron.lite.primitive.kernels.indexer_topk.plugins.cache import (
    KeyCachePool,
    KeyCacheViews,
    pack_key_cache,
)
from megatron.lite.primitive.kernels.indexer_topk.plugins.loader import (
    LoadedLiteTopKPlugin,
    loaded_litetopk_plugins,
)
from megatron.lite.primitive.kernels.indexer_topk.reference import QuantizedKeys, quantize_queries

__all__ = [
    "STATUS_CAPACITY",
    "STATUS_FAILED",
    "STATUS_OK",
    "STATUS_REFINE_OVERFLOW",
    "IndexerOperands",
    "IndexerTopKStats",
    "LiteTopKEngine",
    "release_indexer_topk_workspaces",
]

# The per-row status codes a plugin writes (ABI v1). REFINE_OVERFLOW and CAPACITY invalidate one
# row; FAILED invalidates the whole tile.
STATUS_OK = 0
STATUS_REFINE_OVERFLOW = 1
STATUS_CAPACITY = 2
STATUS_FAILED = 3

# The block-major key caches of every engine in the process (one per device, stream and format).
_KEY_CACHES = KeyCachePool()


@dataclass
class IndexerTopKStats:
    """Host-side counters of the selection calls of one binding.

    Every selected row is counted once in ``litetopk_rows``, ``reference_rows`` or
    ``padding_rows``; ``recomputed_rows`` counts again the rows a status recompute replaced.

    Attributes:
        calls: Selection calls.
        rows: Local query rows of those calls.
        litetopk_rows: Rows of the tiles the plugin selected.
        reference_rows: Rows the reference selector selected: the rows planned for it and the
            tiles the plugin declined on the host.
        bootstrap_rows: The part of ``reference_rows`` in tiles selected by the reference
            selector to vote a first seed.
        padding_rows: Rows outside every segment (all -1).
        tiles: Tiles the plugin selected.
        plans: Tile group plans requested from the plugin.
        carry_stashes: HOT carries stashed from selected rows (first seeds and re-seeds).
        reference_calls: Reference selector invocations that selected rows.
        reference_score_calls: Score kernel calls of the matched reference selector (each with
            one top-k kernel call).
        recomputed_rows: Rows recomputed by the reference selector after the status read.
        recomputed_tiles: Tiles recomputed entirely because the plugin reported a failure.
        reseeded_groups: Tile groups seeded again from selected rows, after a declined group or
            before the second run of a failed group.
        rerun_groups: Failed tile groups run a second time with a seed from repaired rows.
        litetopk_segments: Segments whose plan gives tiles to LiteTopK.
        reference_segments: Segments whose plan gives every row to the reference selector, by
            the plan's reason (for example ``fewer keys than the route minimum``).
        declined_calls: Calls a module declined before selecting, by reason.
        declined_tiles: Tiles the plugin declined on the host, by reason.
        tile_rows: Tiles the plugin selected, by tile length.
        status_rows: Rows whose plugin status was not OK at the status read, by status code.
    """

    calls: int = 0
    rows: int = 0
    litetopk_rows: int = 0
    reference_rows: int = 0
    bootstrap_rows: int = 0
    padding_rows: int = 0
    tiles: int = 0
    plans: int = 0
    carry_stashes: int = 0
    reference_calls: int = 0
    reference_score_calls: int = 0
    recomputed_rows: int = 0
    recomputed_tiles: int = 0
    reseeded_groups: int = 0
    rerun_groups: int = 0
    litetopk_segments: int = 0
    reference_segments: Counter[str] = field(default_factory=Counter)
    declined_calls: Counter[str] = field(default_factory=Counter)
    declined_tiles: Counter[str] = field(default_factory=Counter)
    tile_rows: Counter[int] = field(default_factory=Counter)
    status_rows: Counter[int] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready copy (counter keys as strings, sorted)."""
        result: dict[str, Any] = {}
        for name, value in vars(self).items():
            if isinstance(value, Counter):
                result[name] = {str(key): value[key] for key in sorted(value)}
            else:
                result[name] = value
        return result

    def reset(self) -> None:
        """Set every counter back to zero."""
        for name, value in vars(self).items():
            if isinstance(value, Counter):
                value.clear()
            else:
                setattr(self, name, 0)


@dataclass(frozen=True)
class IndexerOperands:
    """The operands of one selection call, shared by the reference selector and the plugin.

    The keys are quantized once per call; query rows are quantized per scoring call or tile by
    :func:`~.reference.quantize_queries`, which quantizes every row independently. Both selectors
    therefore score byte-identical operands.

    Attributes:
        fmt: Operand format: ``fp8``.
        q: Indexer queries ``[rows, H, D]``.
        weights: Per-head weights ``[rows, H]``, before ``softmax_scale``.
        keys: The quantized keys, covering every key a row of the call sees; None when no
            selector of the call reads them.
        layout: The rows of the call and the keys each one sees.
        topk: Keys selected per row.
        softmax_scale: Positive score scale folded into the weights.
    """

    fmt: IndexerTopKFormat
    q: Tensor
    weights: Tensor
    keys: QuantizedKeys | None
    layout: QueryLayout
    topk: int
    softmax_scale: float

    def queries(
        self, row_start: int, row_end: int, *, kernel_heads: int
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        """Return the score kernel operands of the local rows ``[row_start, row_end)``.

        Args:
            row_start: First local row.
            row_end: One past the last local row.
            kernel_heads: Heads of the score kernel (see :func:`~.reference.quantize_queries`).

        Returns:
            ``(data, scales, weights)`` as :func:`~.reference.quantize_queries` returns them.
        """
        return quantize_queries(
            self.q[row_start:row_end],
            self.weights[row_start:row_end],
            self.fmt,
            softmax_scale=self.softmax_scale,
            kernel_heads=kernel_heads,
        )


@dataclass(frozen=True)
class _SegmentState:
    """What the tiles of one segment share while the plugin selects them."""

    plan: SegmentPlan
    hot_key: Hashable
    views: KeyCacheViews
    key_scales: Tensor


# Called with the tiles the plugin declined on the host and the reason; selects their rows with
# the reference selector (or raises when LiteTopK is required).
TileFallback = Callable[[Sequence[Tile], str], None]


class LiteTopKEngine:
    """Runs planned LiteTopK tiles of one layer through one plugin route.

    Args:
        plugin: The loaded plugin.
        route: The plugin route that serves the layer.
        tuning: The resolved settings of the layer.
        exact: Request exact selection from the plugin (the route must advertise it).
        kernel_heads: Heads the plugin kernels run with.
        layer_key: The identity of the layer: part of the rolling carry keys, so layers and
            segments never share a carry.
    """

    def __init__(
        self,
        plugin: LoadedLiteTopKPlugin,
        route: RouteCapability,
        tuning: ResolvedIndexerTopKTuning,
        *,
        exact: bool,
        kernel_heads: int,
        layer_key: Hashable,
    ) -> None:
        self.plugin = plugin
        self.route = route
        self.tuning = tuning
        self.exact = exact
        self.kernel_heads = kernel_heads
        self.vote_rows = int(plugin.module.carry_vote_rows())
        self._layer_key = layer_key

    def run_segment(
        self,
        operands: IndexerOperands,
        plan: SegmentPlan,
        segment_index: int,
        out: Tensor,
        status: Tensor,
        *,
        fallback: TileFallback,
        stats: IndexerTopKStats,
    ) -> list[tuple[Tile, ...]]:
        """Select the LiteTopK tiles of one planned segment.

        Writes the ids of every dispatched tile (plus the segment's ``index_base``) into its
        rows of ``out`` in the plugin's slot order, and the plugin's status codes into its rows
        of ``status``. The vote rows of the plan must already hold their selections.

        Args:
            operands: The operands of the call.
            plan: The plan of the segment; it has at least one tile group.
            segment_index: The index of the segment in the layout.
            out: int32 ``[layout.rows, topk]`` destination of the call.
            status: int32 ``[layout.rows]`` status destination of the call.
            fallback: Selects the tiles the plugin declines on the host.
            stats: Counters to update.

        Returns:
            The tiles the plugin selected, per tile group of the plan.
        """
        dispatched: list[tuple[Tile, ...]] = []
        with self._segment(operands, plan, segment_index, out.device) as state:
            if plan.seed == "reference":
                self._stash(state, operands, out, plan.vote_rows, stats)
            reseed = False
            for index, group in enumerate(plan.groups):
                if reseed:
                    self._stash(state, operands, out, self._vote_before(plan, group), stats)
                    stats.reseeded_groups += 1
                count, reason = self._run_group(
                    state,
                    operands,
                    group,
                    out,
                    status,
                    publish=index + 1 < len(plan.groups),
                    stats=stats,
                )
                dispatched.append(group.tiles[:count])
                reseed = count < len(group.tiles)
                if reseed:
                    declined = group.tiles[count:]
                    stats.declined_tiles[reason] += len(declined)
                    fallback(declined, reason)
        return dispatched

    def rerun_group(
        self,
        operands: IndexerOperands,
        plan: SegmentPlan,
        segment_index: int,
        group_index: int,
        out: Tensor,
        status: Tensor,
        *,
        stats: IndexerTopKStats,
    ) -> bool:
        """Seed one tile group from the rows that precede it and select it again.

        Used after the group before it failed: that group's carry was voted from invalid rows.
        The rows that precede the group must hold valid selections.

        Args:
            operands: The operands of the call.
            plan: The plan of the segment.
            segment_index: The index of the segment in the layout.
            group_index: The index of the tile group in the plan.
            out: int32 ``[layout.rows, topk]`` destination of the call.
            status: int32 ``[layout.rows]`` status destination of the call.
            stats: Counters to update (the tiles are not counted again).

        Returns:
            Whether the plugin selected every tile of the group; when it declines, the rows of
            the group are left unspecified.
        """
        group = plan.groups[group_index]
        with self._segment(operands, plan, segment_index, out.device) as state:
            self._stash(state, operands, out, self._vote_before(plan, group), stats)
            stats.reseeded_groups += 1
            stats.rerun_groups += 1
            count, _reason = self._run_group(
                state, operands, group, out, status, publish=False, stats=None
            )
        return count == len(group.tiles)

    @contextlib.contextmanager
    def _segment(
        self, operands: IndexerOperands, plan: SegmentPlan, segment_index: int, device: torch.device
    ) -> Iterator[_SegmentState]:
        """Start the plugin call of a segment and pack its keys; drop its carry at the end."""
        module = self.plugin.module
        segment = plan.segment
        keys = operands.keys
        if keys is None or keys.data.shape[0] < segment.key_start + segment.key_count:
            raise ValueError("the quantized keys do not cover the keys of a LiteTopK segment")
        hot_key = (self._layer_key, "rolling", segment_index)
        module.begin_call(device, hot_key, segment.key_count)
        try:
            views = _KEY_CACHES.acquire(
                self.route.fmt, segment.key_count, device, head_dim=operands.q.shape[2]
            )
            first, last = segment.key_start, segment.key_start + segment.key_count
            pack_key_cache(views.cache, keys.data[first:last], keys.scale[first:last])
            yield _SegmentState(
                plan=plan,
                hot_key=hot_key,
                views=views,
                key_scales=views.scales.view(keys.scale.dtype).reshape(segment.key_count),
            )
        finally:
            module.drop_carry(device, hot_key)

    def _vote_before(self, plan: SegmentPlan, group: TileGroup) -> tuple[int, int]:
        """The rows that precede a tile group and vote its seed."""
        first = group.tiles[0].row_start
        return max(plan.segment.row_start, first - self.vote_rows), first

    def _stash(
        self,
        state: _SegmentState,
        operands: IndexerOperands,
        out: Tensor,
        vote: tuple[int, int],
        stats: IndexerTopKStats,
    ) -> None:
        """Publish the selections of the local rows ``vote`` as the segment's HOT carry."""
        segment = state.plan.segment
        start, end = vote
        ids = out[start:end]
        if segment.index_base:
            ids = torch.where(ids >= 0, ids - segment.index_base, ids)
        self.plugin.module.stash_carry(
            state.hot_key,
            ids,
            operands.layout.visible_keys(segment, end - 1),
            min_index=0,
            recent_rows_hint=self.vote_rows,
        )
        stats.carry_stashes += 1

    def _run_group(
        self,
        state: _SegmentState,
        operands: IndexerOperands,
        group: TileGroup,
        out: Tensor,
        status: Tensor,
        *,
        publish: bool,
        stats: IndexerTopKStats | None,
    ) -> tuple[int, str]:
        """Plan one tile group and select its tiles.

        Returns:
            The number of tiles the plugin selected, from the first, and why it declined the
            next one (an empty reason when it selected all).
        """
        module = self.plugin.module
        segment = state.plan.segment
        layout = operands.layout
        views = state.views
        handle = module.prepare_permuted_gather(
            views.cache,
            views.keys,
            views.scales,
            views.block_table,
            sequence_length=segment.key_count,
            query_length=group.tiles[0].rows,
            num_reqs=1,
            common_end=group.common_end,
            window_start=0,
            hot_key=state.hot_key,
        )
        if stats is not None:
            stats.plans += 1
        if handle is None:
            return 0, "plan declined"
        for index, tile in enumerate(group.tiles):
            # Causal key windows [0, visible keys) of the tile rows, computed on the host.
            ends = torch.arange(
                tile.position + 1,
                tile.position + tile.rows + 1,
                dtype=torch.int64,
                device=out.device,
            )
            ends = ends.clamp_(max=segment.key_count).to(torch.int32)
            data, scales, weights = operands.queries(
                tile.row_start, tile.row_end, kernel_heads=self.kernel_heads
            )
            target = out[tile.row_start : tile.row_end]
            dispatched = module.try_large_exact_once_chunk(
                data,
                views.keys,
                state.key_scales,
                weights,
                torch.zeros_like(ends),
                ends,
                target,
                operands.topk,
                permuted_plan=handle,
                num_reqs=1,
                ke_min_hint=group.common_end,
                # Part of the ABI v1 call; no route this version accepts uses a candidate
                # capacity.
                cap=None,
                hot_key=state.hot_key,
                ks_common_hint=0,
                carry_extent_hint=layout.visible_keys(segment, tile.row_end - 1),
                carry_recent_rows_hint=self.vote_rows,
                q_sf=scales,
                carry_io=publish and index + 1 == len(group.tiles),
                exact=self.exact,
                status_out=status[tile.row_start : tile.row_end],
            )
            if not dispatched:
                return index, "tile declined"
            if segment.index_base:
                target.copy_(torch.where(target >= 0, target + segment.index_base, target))
            if stats is not None:
                stats.tiles += 1
                stats.tile_rows[tile.rows] += 1
                stats.litetopk_rows += tile.rows
        return len(group.tiles), ""


def release_indexer_topk_workspaces(device: torch.device | None = None) -> None:
    """Free the memory the indexer top-k selectors keep between calls.

    Releases the pooled block-major key caches and asks every loaded LiteTopK plugin to free
    its scratch memory. Later selections allocate them again. Nothing is loaded by this call.

    Args:
        device: The device to release, or None for every CUDA device.
    """
    _KEY_CACHES.release(device)
    plugins = loaded_litetopk_plugins()
    if not plugins:
        return
    if device is not None:
        devices = [torch.device(device)]
    elif torch.cuda.is_available():
        devices = [torch.device("cuda", index) for index in range(torch.cuda.device_count())]
    else:
        devices = []
    for plugin in plugins:
        for target in devices:
            if target.type == "cuda":
                plugin.module.release(target, release_scratch=True)
