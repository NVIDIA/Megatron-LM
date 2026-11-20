# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bounded metadata caching for readers with asynchronous block prefetch.

The bound applies to retained cache entries, not bytes or objects already held by
active block reads. Group indexes vary in size; the existing lane/prefetch limits
bound active readers separately. Pinning the consumer's current group prevents
prefetch catch-up from evicting it and issuing another GET for every sample.
"""

import threading
from collections import OrderedDict
from concurrent.futures import Future
from typing import Any, Callable, Hashable, Iterable


class BoundedIndexCache:
    """A thread-safe LRU with pinned consumer entries and single-flight loading.

    A loader runs outside the cache lock. Other threads requesting the same key
    wait for that load instead of sending duplicate metadata requests. Exceptions
    are not cached, so callers may retry a failed load. The number of retained
    entries never exceeds ``max_entries``. Concurrent loads can temporarily hold
    additional objects, as can callers still using an evicted entry.
    """

    def __init__(self, max_entries: int):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("Index cache max_entries must be a positive integer")
        self.max_entries = max_entries
        self._entries: OrderedDict[Hashable, Any] = OrderedDict()
        self._pinned: set[Hashable] = set()
        self._loading: dict[Hashable, Future] = {}
        self._lock = threading.Lock()
        self._generation = 0

    def set_pinned(self, keys: Iterable[Hashable]) -> None:
        """Keep the current consumer's keys resident while prefetch loads other groups."""
        pinned = set(keys)
        if len(pinned) > self.max_entries:
            raise ValueError("Pinned indexes exceed index cache capacity")
        with self._lock:
            self._pinned = pinned
            self._evict()

    def _evict(self) -> None:
        while len(self._entries) > self.max_entries:
            # At most max_entries keys are pinned, so an evictable key exists.
            key = next(key for key in self._entries if key not in self._pinned)
            del self._entries[key]

    def get_or_load(self, key: Hashable, loader: Callable[[], Any]) -> Any:
        """Return a resident index, or load and cache it once across concurrent readers."""
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return self._entries[key]
            generation = self._generation
            pending = self._loading.get(key)
            owner = pending is None
            if owner:
                pending = self._loading[key] = Future()
        if not owner:
            return pending.result()
        try:
            value = loader()
        except BaseException as error:
            with self._lock:
                del self._loading[key]
            pending.set_exception(error)
            raise
        with self._lock:
            del self._loading[key]
            if generation == self._generation:
                self._entries[key] = value
                self._entries.move_to_end(key)
                self._evict()
        pending.set_result(value)
        return value

    def clear(self) -> None:
        """Release cached references without invalidating objects held by active reads."""
        with self._lock:
            self._generation += 1
            self._entries.clear()
            self._pinned.clear()
