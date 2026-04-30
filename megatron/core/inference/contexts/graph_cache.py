# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""LRU-bounded CUDA graph cache keyed by ``(snapshot_buffer_id,
batch_dimensions)`` (v3 plan §2.5a / commit 11).

Owns capture, lookup, eviction, and metric emission. Three capture
modes:
- ``warmup_only`` (default): all graphs captured at warmup against an
  explicit list of expected ``(slot, batch_dims)`` pairs; a missed
  capture is a hard error rather than a silent stutter.
- ``on_first_use``: mid-run capture allowed, logs a warning.
- ``on_first_use_with_eviction``: mid-run capture + LRU eviction under
  the byte budget.

Memory-budget enforcement: when ``memory_budget_bytes`` is set, the
cache evicts least-recently-used entries to stay under the budget. The
``max_captures`` cap is applied alongside.
"""

from __future__ import annotations

import logging
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Capture-mode enum kept as string constants to align with InferenceConfig's
# existing string-typed knob (commits 2 / 11 both consume the same names).
CAPTURE_MODE_WARMUP_ONLY = "warmup_only"
CAPTURE_MODE_ON_FIRST_USE = "on_first_use"
CAPTURE_MODE_ON_FIRST_USE_WITH_EVICTION = "on_first_use_with_eviction"

VALID_CAPTURE_MODES = frozenset(
    [
        CAPTURE_MODE_WARMUP_ONLY,
        CAPTURE_MODE_ON_FIRST_USE,
        CAPTURE_MODE_ON_FIRST_USE_WITH_EVICTION,
    ]
)


@dataclass
class GraphCacheEntry:
    """One captured-graph record."""

    key: Tuple[Hashable, ...]
    graph_handle: Any
    bytes_held: int


@dataclass
class GraphCacheMetrics:
    """Cumulative counters emitted by ``CUDAGraphCache``."""

    captures: int = 0
    evictions: int = 0
    miss_causes_capture: int = 0
    miss_causes_eager_fallback: int = 0
    cache_size_bytes: int = 0
    capture_mode_history: List[str] = field(default_factory=list)


class CudaGraphCaptureBudgetError(RuntimeError):
    """Raised in ``warmup_only`` mode when a key not seen during warmup is
    looked up at runtime — the architecture cannot proceed because the
    expected graph wasn't captured."""


class CUDAGraphCache:
    """LRU-bounded CUDA graph cache.

    Public API: ``lookup``, ``capture``, ``get_or_capture``, ``invalidate``,
    plus the read-only ``metrics`` and ``size_bytes`` accessors.
    """

    def __init__(
        self,
        capture_mode: str = CAPTURE_MODE_WARMUP_ONLY,
        memory_budget_bytes: Optional[int] = None,
        max_captures: Optional[int] = None,
    ) -> None:
        if capture_mode not in VALID_CAPTURE_MODES:
            raise ValueError(
                f"Unknown CUDA graph capture mode {capture_mode!r}; "
                f"valid: {sorted(VALID_CAPTURE_MODES)}"
            )
        self._mode = capture_mode
        self._memory_budget_bytes = memory_budget_bytes
        self._max_captures = max_captures
        self._entries: "OrderedDict[Tuple[Hashable, ...], GraphCacheEntry]" = OrderedDict()
        self._metrics = GraphCacheMetrics()
        self._metrics.capture_mode_history.append(capture_mode)

    # ------------------------------------------------------------------
    # Read accessors
    # ------------------------------------------------------------------

    @property
    def metrics(self) -> GraphCacheMetrics:
        return self._metrics

    @property
    def capture_mode(self) -> str:
        return self._mode

    @property
    def size_bytes(self) -> int:
        return self._metrics.cache_size_bytes

    def __len__(self) -> int:
        return len(self._entries)

    def keys(self) -> List[Tuple[Hashable, ...]]:
        return list(self._entries.keys())

    # ------------------------------------------------------------------
    # Lookup / capture / eviction
    # ------------------------------------------------------------------

    def lookup(self, key: Tuple[Hashable, ...]) -> Optional[Any]:
        """LRU-bumping lookup. Returns the graph handle or ``None`` on miss."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return entry.graph_handle

    def capture(
        self,
        key: Tuple[Hashable, ...],
        graph_handle: Any,
        bytes_held: int,
    ) -> None:
        """Insert a captured graph under ``key``. Triggers eviction if the
        memory budget or max-captures cap would be exceeded.
        """
        if key in self._entries:
            old = self._entries[key]
            self._metrics.cache_size_bytes -= old.bytes_held
            del self._entries[key]
        self._entries[key] = GraphCacheEntry(
            key=key, graph_handle=graph_handle, bytes_held=bytes_held
        )
        self._metrics.cache_size_bytes += bytes_held
        self._metrics.captures += 1
        self._enforce_limits()

    def get_or_capture(
        self,
        key: Tuple[Hashable, ...],
        capture_fn: Callable[[], Tuple[Any, int]],
    ) -> Any:
        """Look up ``key``; on miss, drive capture according to the mode.

        ``capture_fn`` returns ``(graph_handle, bytes_held)``. Raises
        ``CudaGraphCaptureBudgetError`` in ``warmup_only`` mode on miss.
        Logs a warning in ``on_first_use``; allows eviction in
        ``on_first_use_with_eviction``.
        """
        existing = self.lookup(key)
        if existing is not None:
            return existing
        if self._mode == CAPTURE_MODE_WARMUP_ONLY:
            self._metrics.miss_causes_eager_fallback += 1
            raise CudaGraphCaptureBudgetError(
                f"CUDA graph cache miss for key {key} in warmup_only mode; "
                "expected (slot, batch_dims) pair was not captured at warmup."
            )
        if self._mode == CAPTURE_MODE_ON_FIRST_USE:
            warnings.warn(
                f"CUDA graph cache miss for key {key}; capturing on first use.",
                stacklevel=2,
            )
        # on_first_use modes capture; on_first_use_with_eviction may evict
        # to make room (handled inside capture()).
        graph_handle, bytes_held = capture_fn()
        self._metrics.miss_causes_capture += 1
        self.capture(key, graph_handle, bytes_held)
        return graph_handle

    def invalidate(self, key: Tuple[Hashable, ...]) -> None:
        """Drop ``key`` from the cache; no-op if absent."""
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._metrics.cache_size_bytes -= entry.bytes_held

    def invalidate_slot(self, slot_idx: int) -> int:
        """Drop every entry whose key starts with ``slot_idx``. Used when a
        snapshot slot is repurposed and its captured graphs are stale.
        Returns the number of evicted entries.
        """
        evicted: List[Tuple[Hashable, ...]] = []
        for key in list(self._entries.keys()):
            if len(key) > 0 and key[0] == slot_idx:
                evicted.append(key)
        for key in evicted:
            self.invalidate(key)
        return len(evicted)

    # ------------------------------------------------------------------
    # Internal: limit enforcement
    # ------------------------------------------------------------------

    def _enforce_limits(self) -> None:
        if self._mode == CAPTURE_MODE_WARMUP_ONLY:
            # warmup_only never evicts — every capture is intentional.
            return
        if self._mode == CAPTURE_MODE_ON_FIRST_USE:
            # Bookkeeping only; respect max_captures by evicting LRU when
            # exceeded but leave the budget unenforced (mode is for
            # debugging unanticipated workloads).
            if self._max_captures is not None:
                while len(self._entries) > self._max_captures:
                    self._evict_one()
            return
        # CAPTURE_MODE_ON_FIRST_USE_WITH_EVICTION: enforce both caps.
        if self._max_captures is not None:
            while len(self._entries) > self._max_captures:
                self._evict_one()
        if self._memory_budget_bytes is not None:
            while (
                self._metrics.cache_size_bytes > self._memory_budget_bytes
                and self._entries
            ):
                self._evict_one()

    def _evict_one(self) -> None:
        key, entry = self._entries.popitem(last=False)
        self._metrics.cache_size_bytes -= entry.bytes_held
        self._metrics.evictions += 1
        logger.debug("CUDAGraphCache evicted key=%s bytes=%d", key, entry.bytes_held)
