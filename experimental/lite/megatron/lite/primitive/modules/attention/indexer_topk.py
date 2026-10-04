# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Optional per-layer indexer top-k selectors for sparse attention modules.

A sparse attention module that owns an indexer can be bound to an
:class:`IndexerTopKBinding`, which selects the indexer top-k of its inference forward passes (eval
mode with gradients disabled) with the matched-precision reference selector and, where a plan
allows it, an external LiteTopK plugin. :func:`configure_indexer_topk` builds the bindings of a
model from one ``IndexerTopKConfig`` (``megatron.lite.primitive.kernels.indexer_topk``, which also
holds the selectors themselves); without it no module is bound and every module runs its upstream
selector.

A selection call is self-contained: there is no request scope and nothing is carried between
calls. :meth:`IndexerTopKBinding.select`

1. plans every segment of the call's ``QueryLayout`` on the host: the rows the reference
   selector selects and the tiles LiteTopK selects;
2. quantizes the keys once and selects all reference rows of the call in one batched pass;
3. runs the LiteTopK tiles of every segment; the plugin writes one status code per row on the
   device;
4. reads the status once. Rows that report a candidate overflow are recomputed with the
   reference selector, and so is every tile that reports a failure; a failed tile group whose
   seed was voted from a failed tile is first run again with a seed voted from the repaired
   rows, at most twice per call and not after such a second run failed as well;
5. sorts every row (ids ascending, -1 last).

The selectors of a binding score the head counts negotiated when it is built (see
``megatron.lite.primitive.kernels.indexer_topk.heads``): the plugin's kernels may need zero heads
appended to the layer's (``IndexerTopKConfig.head_padding``), and the reference selector then
scores the same padded operands.

Selection issues no collective: under context parallelism every rank selects its local rows
against keys the model has already gathered, so a fallback on one rank cannot desynchronize the
ranks. Bindings select only in eval mode with gradients disabled: they are inactive while autograd
is enabled, and modules consult them only in eval mode, because a training forward can run without
autograd (a reentrant activation recompute runs it under ``torch.no_grad()`` and again with
gradients in the backward pass, and both runs must select the same top-k). Bindings refuse to run
under CUDA graph capture, because tiles are planned on the host.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from megatron.lite.primitive.kernels.indexer_topk.config import (
    IndexerTopKConfig,
    IndexerTopKConfigError,
    IndexerTopKFormat,
    IndexerTopKRuntimeError,
    IndexerTopKTuning,
    ResolvedIndexerTopKTuning,
    normalize_indexer_topk_config,
    resolve_indexer_topk_tuning,
)
from megatron.lite.primitive.kernels.indexer_topk.engine import (
    STATUS_FAILED,
    STATUS_OK,
    IndexerOperands,
    IndexerTopKStats,
    LiteTopKEngine,
    release_indexer_topk_workspaces,
)
from megatron.lite.primitive.kernels.indexer_topk.heads import IndexerHeads, negotiate_indexer_heads
from megatron.lite.primitive.kernels.indexer_topk.layout import IndexerGeometry, QueryLayout
from megatron.lite.primitive.kernels.indexer_topk.order import sort_topk_rows_
from megatron.lite.primitive.kernels.indexer_topk.planner import SegmentPlan, Tile, plan_segment
from megatron.lite.primitive.kernels.indexer_topk.plugins.abi import RouteCapability
from megatron.lite.primitive.kernels.indexer_topk.plugins.loader import (
    LoadedLiteTopKPlugin,
    load_litetopk_plugin,
)
from megatron.lite.primitive.kernels.indexer_topk.reference import (
    QuantizedKeys,
    ReferenceSelector,
    TopKKernel,
    quantize_keys,
    score_kernel_heads,
    topk_kernel,
)

__all__ = [
    "IndexerTopKBinding",
    "IndexerTopKConsumer",
    "IndexerTopKInstallation",
    "configure_indexer_topk",
    "release_indexer_topk_workspaces",
]

logger = logging.getLogger(__name__)

_ROUTE_OF_FORMAT = {"fp8": "fp8_paged", "mxfp4": "fp4_slab"}
# A failed tile group whose seed was voted from a failed tile is run again with a seed voted from
# the repaired rows at most this many times per call; later failed groups, and every failed
# group after such a second run failed as well (its seed was valid, so seeds do not explain the
# failures), go straight to the reference selector.
_MAX_GROUP_RERUNS = 2
RowRanges = Sequence[tuple[int, int]]


class IndexerTopKConsumer(Protocol):
    """A module whose indexer top-k can be selected by an :class:`IndexerTopKBinding`.

    A consumer selects through its binding only in eval mode while the binding is active (see
    :meth:`IndexerTopKBinding.active`) and runs its upstream selector otherwise.
    """

    def indexer_geometry(self) -> IndexerGeometry | None:
        """Return the shape of the module's indexer, or None when it selects no top-k."""
        ...

    def set_indexer_topk(self, binding: IndexerTopKBinding | None) -> None:
        """Install the binding the module selects with, or None for its upstream selector."""
        ...


def _num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _compute_capability(device: torch.device) -> tuple[int, int] | None:
    return torch.cuda.get_device_capability(device) if device.type == "cuda" else None


def _is_capturing(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def _module_device(module: nn.Module) -> torch.device:
    """The CUDA device a module will select on, as far as it is known at build time."""
    for tensor in module.parameters():
        if tensor.device.type == "cuda":
            return tensor.device
        break
    if not torch.cuda.is_available():
        raise IndexerTopKConfigError(
            "indexer top-k bindings select on a CUDA device; none is available"
        )
    return torch.device("cuda", torch.cuda.current_device())


def _device_scope(device: torch.device) -> contextlib.AbstractContextManager:
    """Make ``device`` the current CUDA device (the plugins allocate on the current device)."""
    return torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext()


@dataclass
class _DeviceState:
    """The device-dependent parts of a binding, resolved at the first selection on a device.

    ``reference_heads`` is what the reference selector scores with: the negotiated
    ``heads.reference_heads`` when the plan can give LiteTopK rows, else
    ``heads.baseline_heads`` (no LiteTopK operands to match).
    """

    tuning: ResolvedIndexerTopKTuning
    num_sms: int
    engine: LiteTopKEngine | None
    heads: IndexerHeads
    reference_heads: int | None
    reference: ReferenceSelector | None = None


class IndexerTopKBinding:
    """The indexer top-k selector of one module.

    Built by :func:`configure_indexer_topk`. The binding is not an ``nn.Module``: it holds no
    parameter or buffer and never enters a ``state_dict``. A deep copy of its module shares the
    binding (the loaded plugins are process state).

    Attributes:
        name: The name of the bound module, used in logs and error messages.
        geometry: The indexer shape the binding was negotiated for.
        config: The configuration it was built from.
        heads: The head counts its selectors score with, negotiated by
            :func:`configure_indexer_topk` (see
            :func:`~megatron.lite.primitive.kernels.indexer_topk.heads.negotiate_indexer_heads`)
            and again on every device it selects on; None for a binding built without it.
        stats: The counters of its selection calls.
    """

    def __init__(
        self,
        *,
        name: str,
        geometry: IndexerGeometry,
        config: IndexerTopKConfig,
        fmt: IndexerTopKFormat,
        tuning: IndexerTopKTuning | None,
        kernel: TopKKernel | None,
        plugin: LoadedLiteTopKPlugin | None,
        route: RouteCapability | None,
        heads: IndexerHeads | None = None,
    ) -> None:
        self.name = name
        self.geometry = geometry
        self.config = config
        self.heads = heads
        self.stats = IndexerTopKStats()
        self._fmt = fmt
        self._tuning = tuning
        self._kernel = kernel
        self._plugin = plugin
        self._route = route
        self._required = bool(tuning is not None and tuning.required)
        # An identity-hashed key: layers never share a plugin carry.
        self._layer_key = object()
        self._states: dict[str, _DeviceState] = {}

    def __deepcopy__(self, memo: dict) -> IndexerTopKBinding:
        """Return self: copies of a module share its binding and the plugins behind it."""
        return self

    @property
    def plugin(self) -> LoadedLiteTopKPlugin | None:
        """The loaded LiteTopK plugin, or None for the reference backend."""
        return self._plugin

    def active(self) -> bool:
        """Return whether autograd lets the binding select: only while it is disabled.

        Consumers also select through a binding only in eval mode: a training forward that runs
        without autograd (as a reentrant activation recompute does) keeps its upstream selector.
        """
        return not torch.is_grad_enabled()

    def decline(self, reason: str) -> None:
        """Record that the module runs its upstream selector for a call the binding cannot take.

        Args:
            reason: Why, for example ``"batch>1"``.

        Raises:
            IndexerTopKRuntimeError: If the binding is required (``IndexerTopKTuning.required``).
        """
        self.stats.declined_calls[reason] += 1
        if self._required:
            raise IndexerTopKRuntimeError(
                f"required indexer top-k binding of layer {self.name} declined a call: {reason}"
            )

    def resolved_tuning(self, device: torch.device) -> ResolvedIndexerTopKTuning:
        """Return the selection plan settings the binding uses on ``device``."""
        return self._state(torch.device(device)).tuning

    def select(
        self,
        q: Tensor,
        k: Tensor,
        weights: Tensor,
        *,
        layout: QueryLayout,
        topk: int,
        softmax_scale: float,
    ) -> Tensor | None:
        """Select the indexer top-k of the local query rows of one module call.

        Args:
            q: Indexer queries ``[layout.rows, H, D]`` (after RoPE).
            k: Indexer keys ``[N, D]``; the keys of a segment are
                ``k[key_start : key_start + key_count]``.
            weights: Per-head weights ``[layout.rows, H]``, not yet multiplied by
                ``softmax_scale``.
            layout: The local rows and the keys each one sees.
            topk: Keys selected per row, at most the top-k of the bound geometry.
            softmax_scale: Positive score scale, folded into the weights.

        Returns:
            int32 ``[layout.rows, topk]`` key ids in the layout's id space, -1 for missing keys;
            every row ascending with -1 last unless the tuning keeps the selectors' order. None
            when the binding is inactive (autograd enabled): the caller then runs its upstream
            selector.

        Raises:
            ValueError: If the tensors do not match the layout or the bound geometry.
            IndexerTopKConfigError: If the settings the device needs cannot be resolved.
            IndexerTopKRuntimeError: Under CUDA graph capture, on a device LiteTopK does not
                support, when a launch-time environment key of the plugin changed, or when a
                required LiteTopK selection declined or failed.
        """
        if not self.active():
            return None
        self._check_operands(q, k, weights, layout, topk)
        device = q.device
        if _is_capturing(device):
            raise IndexerTopKRuntimeError(
                "indexer top-k binding cannot run under CUDA graph capture: LiteTopK plans "
                "tiles on the host"
            )
        if self._plugin is not None:
            self._plugin.check_launch_time_env()
        with _device_scope(device):
            return self._select(q, k, weights, layout, topk, softmax_scale)

    def _select(
        self,
        q: Tensor,
        k: Tensor,
        weights: Tensor,
        layout: QueryLayout,
        topk: int,
        softmax_scale: float,
    ) -> Tensor:
        """The body of :meth:`select`, with ``q.device`` as the current device."""
        device = q.device
        state = self._state(device)
        tuning = state.tuning
        stats = self.stats
        stats.calls += 1
        stats.rows += layout.rows
        out = torch.empty((layout.rows, topk), dtype=torch.int32, device=device)
        if not layout.segments:
            stats.padding_rows += layout.rows
            return out.fill_(-1)

        engine = state.engine
        plans = [
            plan_segment(
                segment,
                key_ratio=layout.key_ratio,
                route=None if engine is None else engine.route,
                tuning=tuning,
                topk=topk,
                vote_rows=1 if engine is None else engine.vote_rows,
                max_tile_keys=(
                    None if engine is None else engine.max_tile_keys(segment.key_count, topk)
                ),
            )
            for segment in layout.segments
        ]
        for plan in plans:
            if plan.groups:
                stats.litetopk_segments += 1
            else:
                stats.reference_segments[plan.reason] += 1
        self._fill_padding(layout, out)
        operands = IndexerOperands(
            fmt=self._fmt,
            q=q,
            weights=weights,
            keys=self._quantize_keys(k, layout, plans),
            layout=layout,
            topk=topk,
            softmax_scale=softmax_scale,
        )

        def reference(ranges: RowRanges) -> None:
            self._reference(state, operands, ranges, out)

        planned = [rows for plan in plans for rows in plan.reference_rows]
        reference(planned)
        stats.reference_rows += sum(end - start for start, end in planned)
        stats.bootstrap_rows += sum(
            plan.bootstrap_tile.rows for plan in plans if plan.bootstrap_tile is not None
        )
        if engine is not None and any(plan.groups for plan in plans):
            self._select_tiles(engine, operands, plans, out, reference)
        if tuning.index_order == "ascending":
            sort_topk_rows_(out)
        return out

    def _check_operands(
        self, q: Tensor, k: Tensor, weights: Tensor, layout: QueryLayout, topk: int
    ) -> None:
        geometry = self.geometry
        expected = (layout.rows, geometry.num_heads, geometry.head_dim)
        if (
            tuple(q.shape) != expected
            or k.ndim != 2
            or k.shape[1] != geometry.head_dim
            or tuple(weights.shape) != expected[:2]
            or k.device != q.device
            or weights.device != q.device
        ):
            raise ValueError(
                f"layer {self.name}: expected q {list(expected)}, k [N, {geometry.head_dim}] and "
                f"weights {list(expected[:2])} on one device; got q {tuple(q.shape)} on "
                f"{q.device}, k {tuple(k.shape)} on {k.device}, weights {tuple(weights.shape)} "
                f"on {weights.device}"
            )
        if type(topk) is not int or not 0 < topk <= geometry.topk:
            raise ValueError(
                f"layer {self.name}: topk must be an integer in [1, {geometry.topk}], got {topk!r}"
            )
        if layout.key_ratio != geometry.key_ratio:
            raise ValueError(
                f"layer {self.name}: the layout has key ratio {layout.key_ratio}, the indexer "
                f"{geometry.key_ratio}"
            )
        key_rows = max(
            (segment.key_start + segment.key_count for segment in layout.segments), default=0
        )
        if key_rows > k.shape[0]:
            raise ValueError(
                f"layer {self.name}: the layout's segments need {key_rows} keys; k has "
                f"{k.shape[0]}"
            )

    def _state(self, device: torch.device) -> _DeviceState:
        """Resolve the device-dependent settings once per device."""
        key = str(device)
        state = self._states.get(key)
        if state is not None:
            return state
        plugin, route = self._plugin, self._route
        if plugin is not None:
            capability = _compute_capability(device)
            if capability is None or capability[0] != 10:
                found = device.type if capability is None else f"{capability[0]}.{capability[1]}"
                raise IndexerTopKRuntimeError(
                    f"LiteTopK requires an SM100 (Blackwell) GPU, got compute capability {found}"
                )
        num_sms = _num_sms(device)
        heads = _negotiate_heads(
            self.name, self.geometry, self.config, self._fmt, plugin, route, device=device
        )
        tuning = resolve_indexer_topk_tuning(
            self._tuning,
            fmt=self._fmt,
            route=route,
            geometry=self.geometry,
            precision=self.config.precision,
            num_sms=num_sms,
            heads=heads,
        )
        engine = None
        if plugin is not None and route is not None:
            if tuning.plugin_settings != plugin.settings:
                raise IndexerTopKRuntimeError(
                    f"layer {self.name}: LiteTopK source {plugin.source_id} was loaded with "
                    f"{plugin.settings}, but {device} ({num_sms} SMs) needs "
                    f"{tuning.plugin_settings}; plugin settings are fixed for the process"
                )
            engine = LiteTopKEngine(
                plugin,
                route,
                tuning,
                exact=self.config.precision == "exact",
                kernel_heads=heads.litetopk_heads,
                layer_key=self._layer_key,
            )
        # Without LiteTopK rows there are no plugin operands to match.
        litetopk = engine is not None and tuning.startup_position is not None
        state = _DeviceState(
            tuning=tuning,
            num_sms=num_sms,
            engine=engine,
            heads=heads,
            reference_heads=heads.reference_heads if litetopk else heads.baseline_heads,
        )
        self._states[key] = state
        logger.info(
            "indexer top-k binding of layer %s on %s: backend %s, precision %s, format %s, "
            "route %s, source %s, heads %d (LiteTopK kernels %s, reference kernel %s), "
            "tuning %s",
            self.name,
            device,
            self.config.backend,
            self.config.precision,
            self._fmt,
            None if route is None else route.name,
            None if plugin is None else plugin.source_id,
            self.geometry.num_heads,
            _kernel_heads_text(heads.litetopk_heads, self.geometry.num_heads),
            _kernel_heads_text(state.reference_heads, self.geometry.num_heads),
            tuning.as_dict(),
        )
        return state

    def _reference_selector(self, state: _DeviceState) -> ReferenceSelector:
        if state.reference is None:
            state.reference = ReferenceSelector(
                fmt=self._fmt,
                topk_kernel=self._kernel,
                kernel_heads=state.reference_heads,
                budget_bytes=state.tuning.reference_budget_bytes,
                rows_per_call=state.tuning.reference_rows_per_call,
                num_sms=state.num_sms,
            )
        return state.reference

    def _quantize_keys(
        self, k: Tensor, layout: QueryLayout, plans: list[SegmentPlan]
    ) -> QuantizedKeys | None:
        """Quantize the keys the selectors of the call read (None when none reads them)."""
        needed = 0
        for plan in plans:
            segment = plan.segment
            if plan.groups:
                # The plugin gathers every key of the sequence.
                needed = max(needed, segment.key_start + segment.key_count)
            else:
                needed = max(
                    needed, segment.key_start + layout.visible_keys(segment, segment.row_end - 1)
                )
        return quantize_keys(k, self._fmt, rows=needed) if needed else None

    def _fill_padding(self, layout: QueryLayout, out: Tensor) -> None:
        """Write -1 into the rows outside every segment."""
        cursor = 0
        for segment in layout.segments:
            if cursor < segment.row_start:
                out[cursor : segment.row_start].fill_(-1)
                self.stats.padding_rows += segment.row_start - cursor
            cursor = segment.row_end
        if cursor < layout.rows:
            out[cursor:].fill_(-1)
            self.stats.padding_rows += layout.rows - cursor

    def _reference(
        self, state: _DeviceState, operands: IndexerOperands, ranges: RowRanges, out: Tensor
    ) -> None:
        """Select the local row ranges (all inside segments) with the reference selector."""
        ranges = [(start, end) for start, end in ranges if start < end]
        if not ranges:
            return
        self.stats.reference_calls += 1
        if state.reference_heads > self.geometry.num_heads:
            self.stats.padded_reference_rows += sum(end - start for start, end in ranges)
        self.stats.reference_score_calls += self._reference_selector(state).select(
            operands.q,
            operands.weights,
            operands.keys,
            layout=operands.layout,
            row_ranges=ranges,
            topk=operands.topk,
            softmax_scale=operands.softmax_scale,
            out=out,
        )

    def _select_tiles(
        self,
        engine: LiteTopKEngine,
        operands: IndexerOperands,
        plans: list[SegmentPlan],
        out: Tensor,
        reference: Callable[[RowRanges], None],
    ) -> None:
        """Run the LiteTopK tiles of the call and act on the row statuses."""
        stats = self.stats
        status = torch.zeros((operands.layout.rows,), dtype=torch.int32, device=out.device)

        def fallback(tiles: Sequence[Tile], reason: str) -> None:
            rows = sum(tile.rows for tile in tiles)
            if self._required:
                raise self._required_error(rows, operands, reason)
            reference([(tile.row_start, tile.row_end) for tile in tiles])
            stats.reference_rows += rows

        runs = []
        for index, plan in enumerate(plans):
            if plan.groups:
                dispatched = engine.run_segment(
                    operands, plan, index, out, status, fallback=fallback, stats=stats
                )
                runs.append((index, plan, dispatched))
        if engine.tuning.status_check == "device_assert":
            torch._assert_async(
                (status == STATUS_OK).all(),
                "a LiteTopK tile reported a candidate overflow or a selection failure",
            )
            return
        if self._worst_status(status) != STATUS_OK:
            self._repair(engine, operands, runs, out, status, reference)

    def _worst_status(self, status: Tensor) -> int:
        """Read the largest status code of the call: its one device-to-host synchronization."""
        return int(status.max())

    def _required_error(
        self, rows: int, operands: IndexerOperands, reason: str
    ) -> IndexerTopKRuntimeError:
        keys = max(segment.key_count for segment in operands.layout.segments)
        return IndexerTopKRuntimeError(
            f"required LiteTopK fell back to the reference for {rows} planned rows in layer "
            f"{self.name} (S={keys}): {reason}"
        )

    def _repair(
        self,
        engine: LiteTopKEngine,
        operands: IndexerOperands,
        runs: list[tuple[int, SegmentPlan, list[tuple[Tile, ...]]]],
        out: Tensor,
        status: Tensor,
        reference: Callable[[RowRanges], None],
    ) -> None:
        """Recompute what the plugin reported as invalid (rare: more synchronizations are fine).

        A row with a row-level code (a candidate overflow) is recomputed alone. A tile with a
        failed row is recomputed entirely. A failed group whose seed was voted from a failed
        tile (the last tile of the group before it) is first run again with a seed voted from
        the repaired rows that precede it, at most ``_MAX_GROUP_RERUNS`` times per call; once
        such a second run fails too, its seed was valid, so seeds do not explain the failures
        and no later group runs again (failed tiles go straight to the reference selector).
        With ``status_check="sync_recompute_tail"`` every tile from a failed one to the end of
        its segment is recomputed instead.
        """
        stats = self.stats
        tail = engine.tuning.status_check == "sync_recompute_tail"
        reruns_left = _MAX_GROUP_RERUNS
        codes = status.cpu()
        for code, count in zip(*(values.tolist() for values in codes.unique(return_counts=True))):
            if code != STATUS_OK:
                stats.status_rows[code] += count

        def failed_tiles(tiles: Sequence[Tile]) -> list[Tile]:
            return [
                tile
                for tile in tiles
                if bool((codes[tile.row_start : tile.row_end] == STATUS_FAILED).any())
            ]

        overflow: list[tuple[int, int]] = []
        for segment_index, plan, dispatched in runs:
            seed_failed = False  # the seed of the group was voted from a failed tile
            tail_failed = False
            for group_index, tiles in enumerate(dispatched):
                complete = len(tiles) == len(plan.groups[group_index].tiles)
                failed = failed_tiles(tiles)
                next_seed_failed = complete and bool(failed) and failed[-1] is tiles[-1]
                if failed and self._required:
                    rows = sum(tile.rows for tile in failed)
                    raise self._required_error(rows, operands, "the plugin reported a failure")
                if tail_failed:
                    failed = list(tiles)
                elif failed and tail:
                    failed = list(tiles[tiles.index(failed[0]) :])
                    tail_failed = True
                elif failed and seed_failed and complete and reruns_left > 0:
                    first, last = tiles[0].row_start, tiles[-1].row_end
                    reruns_left -= 1
                    if engine.rerun_group(
                        operands, plan, segment_index, group_index, out, status, stats=stats
                    ):
                        codes[first:last] = status[first:last].cpu()
                        failed = failed_tiles(tiles)
                    else:
                        failed = list(tiles)
                    if failed:
                        # A group with a valid seed failed: stop running groups again.
                        reruns_left = 0
                if failed:
                    reference([(tile.row_start, tile.row_end) for tile in failed])
                    stats.recomputed_tiles += len(failed)
                    stats.recomputed_rows += sum(tile.rows for tile in failed)
                seed_failed = next_seed_failed
                repaired = {tile.row_start for tile in failed}
                for tile in tiles:
                    if tile.row_start not in repaired:
                        rows = torch.nonzero(codes[tile.row_start : tile.row_end]).flatten()
                        overflow.extend(_row_ranges((rows + tile.row_start).tolist()))
        if overflow:
            reference(overflow)
            stats.recomputed_rows += sum(end - start for start, end in overflow)


def _row_ranges(rows: list[int]) -> list[tuple[int, int]]:
    """Merge ascending row indices into ``(start, end)`` ranges of consecutive rows."""
    ranges: list[tuple[int, int]] = []
    for row in rows:
        if ranges and ranges[-1][1] == row:
            ranges[-1] = (ranges[-1][0], row + 1)
        else:
            ranges.append((row, row + 1))
    return ranges


@dataclass(frozen=True)
class IndexerTopKInstallation:
    """The bindings :func:`configure_indexer_topk` installed on a model.

    Attributes:
        config: The normalized configuration.
        bindings: The bindings, one per bound module, in module order.
        plugin_info: The provenance of the loaded LiteTopK plugin (location, hashes, rendered
            environment and its ``plugin_info()``), or None for the reference backend.
    """

    config: IndexerTopKConfig
    bindings: tuple[IndexerTopKBinding, ...]
    plugin_info: Mapping[str, Any] | None

    def stats(self) -> dict[str, IndexerTopKStats]:
        """Return the counters of every binding, keyed by module name."""
        return {binding.name: binding.stats for binding in self.bindings}

    def heads(self) -> dict[str, dict[str, Any]]:
        """Return the negotiated head counts of every binding, keyed by module name.

        For provenance records: the layer's indexer heads, the heads the LiteTopK kernels and
        the reference score kernel run with, and whether they are zero-padded (see
        :class:`~megatron.lite.primitive.kernels.indexer_topk.heads.IndexerHeads`).
        """
        return {binding.name: binding.heads.as_dict() for binding in self.bindings}

    def reset_stats(self) -> None:
        """Reset the counters of every binding."""
        for binding in self.bindings:
            binding.stats.reset()

    def release(self, device: torch.device | None = None) -> None:
        """Free the key caches and plugin scratch memory (see
        :func:`release_indexer_topk_workspaces`)."""
        release_indexer_topk_workspaces(device)


def _consumers(chunks: Sequence[nn.Module]) -> list[tuple[str, Any]]:
    consumers = []
    for index, chunk in enumerate(chunks):
        prefix = f"chunk{index}." if len(chunks) > 1 else ""
        for name, module in chunk.named_modules():
            if callable(getattr(module, "indexer_geometry", None)) and callable(
                getattr(module, "set_indexer_topk", None)
            ):
                consumers.append((prefix + (name or type(module).__name__), module))
    return consumers


def _route(name: str, fmt: IndexerTopKFormat, plugin: LoadedLiteTopKPlugin) -> RouteCapability:
    """Return the plugin route of the layer's operand format, or raise when there is none."""
    route_name = _ROUTE_OF_FORMAT[fmt]
    route = plugin.info.route(route_name)
    if route is None:
        raise IndexerTopKConfigError(
            f"LiteTopK source {plugin.source_id} has no {route_name} route, which {fmt} "
            f"indexers need (layer {name}); use another plugin or backend='reference'"
        )
    return route


def _check_route(
    name: str,
    geometry: IndexerGeometry,
    config: IndexerTopKConfig,
    required: bool,
    plugin: LoadedLiteTopKPlugin,
    route: RouteCapability,
) -> None:
    """Raise when the route cannot select a layer with the configured precision and top-k."""
    if config.precision == "exact" and not route.exact:
        raise IndexerTopKConfigError(
            f"precision='exact' needs a LiteTopK route that advertises exact selection; source "
            f"{plugin.source_id} route {route.name} does not (use an exact-capable plugin, or "
            "precision='fast')"
        )
    if required and not route.supports_topk(geometry.topk):
        raise IndexerTopKConfigError(
            f"LiteTopK route {route.name} (source {plugin.source_id}) does not select top-k "
            f"{geometry.topk} (layer {name}), so a required LiteTopK selection cannot run"
        )


def _negotiate_heads(
    name: str,
    geometry: IndexerGeometry,
    config: IndexerTopKConfig,
    fmt: IndexerTopKFormat,
    plugin: LoadedLiteTopKPlugin | None,
    route: RouteCapability | None,
    *,
    device: torch.device,
) -> IndexerHeads:
    """The head counts the selectors of a layer score with on ``device``."""

    def reference_heads(heads: int) -> int:
        return score_kernel_heads(heads, fmt=fmt, head_dim=geometry.head_dim, device=device)

    return negotiate_indexer_heads(
        geometry,
        name=name,
        precision=config.precision,
        head_padding=config.head_padding,
        route=route,
        source_id=None if plugin is None else plugin.source_id,
        reference_heads=reference_heads,
    )


def _kernel_heads_text(heads: int | None, num_heads: int) -> str:
    if heads is None:
        return "none"
    return f"{heads} (zero-padded)" if heads > num_heads else str(heads)


def configure_indexer_topk(
    chunks: Sequence[nn.Module],
    config: IndexerTopKConfig | Mapping[str, Any] | None,
    *,
    native_format: IndexerTopKFormat,
    tuning: IndexerTopKTuning | None = None,
) -> IndexerTopKInstallation | None:
    """Bind the indexer top-k selectors of a model.

    Every module of ``chunks`` that implements :class:`IndexerTopKConsumer` and reports an
    indexer geometry gets a binding; modules without a geometry (layers that reuse another
    layer's top-k) are unbound. Configuring again replaces the binding of every module; plugin
    loads are cached for the process. Every module is validated before any is changed: its head
    count is negotiated (see
    :func:`~megatron.lite.primitive.kernels.indexer_topk.heads.negotiate_indexer_heads`) with the
    reference score kernel, which is probed for the head counts it supports, and, for backend
    ``litetopk``, with the plugin route, which zero-pads a head count it has no kernels for
    only with ``head_padding``. When the configuration is invalid, a module cannot be served or
    a plugin fails to load, the call raises and every module keeps the binding it had before.

    Args:
        chunks: The model chunks whose modules are bound.
        config: The selection configuration (or its dict form). None or backend ``default``
            unbinds every module: the modules then run their upstream selectors.
        native_format: The operand format of the model's indexers: ``fp8`` (E4M3 rows with
            float32 scales) or ``mxfp4`` (E2M1 values with packed UE8M0 group scales).
        tuning: Expert overrides of the selection plan (harnesses and tests only).

    Returns:
        The installation, or None when nothing is bound.

    Raises:
        IndexerTopKConfigError: If the configuration is invalid, the reference score kernel
            supports no head count for a layer, a plugin route cannot serve a layer (also when
            its head count could be padded but ``head_padding`` is off), or the tuning requires
            LiteTopK where the plan gives it no row.
        IndexerTopKPluginError: If a plugin or the exact-tie package fails to load.
        IndexerTopKRuntimeError: If DeepGEMM, which the reference selector scores with, is
            missing.
    """
    config = normalize_indexer_topk_config(config)
    consumers = _consumers(chunks)
    if config is None or config.backend == "default":
        for _name, module in consumers:
            module.set_indexer_topk(None)
        return None
    if native_format not in _ROUTE_OF_FORMAT:
        raise IndexerTopKConfigError(
            f"native_format must be one of {sorted(_ROUTE_OF_FORMAT)}, got {native_format!r}"
        )
    if tuning is not None and not isinstance(tuning, IndexerTopKTuning):
        raise TypeError(f"expected IndexerTopKTuning or None, got {type(tuning).__name__}")
    required = bool(tuning is not None and tuning.required)
    kernel = topk_kernel(config.exact_topk)
    bound: list[tuple[Any, IndexerTopKBinding | None]] = []
    plugins: dict[str, LoadedLiteTopKPlugin] = {}
    for name, module in consumers:
        geometry = module.indexer_geometry()
        if geometry is None:
            bound.append((module, None))
            continue
        # The device the module selects on.
        device = _module_device(module)
        plugin = route = None
        if config.backend == "litetopk":
            settings = resolve_indexer_topk_tuning(
                tuning,
                fmt=native_format,
                route=None,
                geometry=geometry,
                precision=config.precision,
                num_sms=_num_sms(device),
            ).plugin_settings
            plugin = load_litetopk_plugin(config.litetopk, settings)
            plugins[plugin.owner] = plugin
            route = _route(name, native_format, plugin)
        # Fails here, at build time, when a selector cannot serve the layer's head count. The
        # reference score kernel's answers are cached per device architecture for the
        # selections, which negotiate again on their device.
        heads = _negotiate_heads(
            name, geometry, config, native_format, plugin, route, device=device
        )
        if route is not None:
            _check_route(name, geometry, config, required, plugin, route)
        # Validate the overrides now rather than at the first selection.
        resolve_indexer_topk_tuning(
            tuning,
            fmt=native_format,
            route=route,
            geometry=geometry,
            precision=config.precision,
            num_sms=1 if route is None else _num_sms(device),
            heads=heads,
        )
        binding = IndexerTopKBinding(
            name=name,
            geometry=geometry,
            config=config,
            fmt=native_format,
            tuning=tuning,
            kernel=kernel,
            plugin=plugin,
            route=route,
            heads=heads,
        )
        bound.append((module, binding))
    for module, binding in bound:
        module.set_indexer_topk(binding)
    provenance = next(iter(plugins.values())).provenance() if plugins else None
    return IndexerTopKInstallation(
        config=config,
        bindings=tuple(binding for _module, binding in bound if binding is not None),
        plugin_info=provenance,
    )
