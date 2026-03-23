# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Sequence

import numpy as np

if TYPE_CHECKING:
    from megatron.core.inference.config import PrefixCachingCoordinatorPolicy


class HashRankTable:
    """Numpy-backed table mapping block hashes to per-rank assignment timestamps.

    Internally stores a 2D array of shape ``(capacity, n_ranks)`` where each row
    corresponds to a unique hash value and each column to a rank index.  A cell
    value of 0 means the rank has never been assigned that hash; a positive value
    is the monotonically-increasing assignment timestamp.

    The table grows automatically when new hashes are inserted beyond the current
    capacity.

    Args:
        n_ranks: Number of data-parallel ranks (columns in the table).
        initial_capacity: Initial number of hash rows to pre-allocate.
    """

    def __init__(self, n_ranks: int, initial_capacity: int = 256):
        self._n_ranks = n_ranks
        self._hash_to_row: Dict[int, int] = {}
        self._next_row = 0
        self._timestamps = np.zeros((initial_capacity, n_ranks), dtype=np.int64)
        self._assignment_counter = 0

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def record(self, rank_idx: int, hashes: Sequence[int]) -> None:
        """Record that *rank_idx* now owns the given *hashes*.

        A new monotonically-increasing timestamp is generated for the entire
        batch so that recency can be used as a tiebreaker elsewhere.
        """
        self._assignment_counter += 1
        ts = self._assignment_counter
        for h in hashes:
            row = self._get_or_create_row(h)
            self._timestamps[row, rank_idx] = ts

    def set(self, h: int, rank_idx: int, timestamp: int) -> None:
        """Directly set the timestamp for a single (hash, rank) pair.

        Primarily useful in tests; production code should prefer :meth:`record`.
        """
        row = self._get_or_create_row(h)
        self._timestamps[row, rank_idx] = timestamp

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def match_vector(
        self, hashes: Sequence[int], policy: PrefixCachingCoordinatorPolicy | None = None
    ) -> np.ndarray:
        """Return a float64 score vector of shape ``(n_ranks,)`` quantifying
        how well each rank matches the given *hashes*.

        The score semantics depend on *policy*:

        * ``FIRST_PREFIX_BLOCK`` (default): binary – 1.0 if the rank has the
          first block hash cached, 0.0 otherwise.  Only ``hashes[:1]`` is
          checked.
        * ``LONGEST_PREFIX``: reverse-scans hashes to find the deepest block
          present in the table (parent-chained hashes guarantee that a match at
          index *i* implies all earlier blocks are also cached).  Ranks that
          hold that hash score ``(i + 1) / len(hashes)``; all others score 0.
        """
        from megatron.core.inference.config import PrefixCachingCoordinatorPolicy

        if policy is None:
            policy = PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK

        if policy == PrefixCachingCoordinatorPolicy.LONGEST_PREFIX:
            return self._match_vector_longest_prefix(hashes)

        # FIRST_PREFIX_BLOCK: binary check on the first block hash only.
        if not hashes:
            return np.zeros(self._n_ranks, dtype=np.float64)
        row = self._hash_to_row.get(hashes[0])
        if row is not None:
            return (self._timestamps[row] > 0).astype(np.float64)
        return np.zeros(self._n_ranks, dtype=np.float64)

    def _match_vector_longest_prefix(self, hashes: Sequence[int]) -> np.ndarray:
        """Longest-prefix scoring via reverse scan.

        Scans *hashes* from the end and stops at the first (deepest) hash
        found in the table.  Ranks that hold that hash score
        ``(depth + 1) / len(hashes)``; all others score 0.  Because hashes are
        parent-chained, a single deepest-match lookup is sufficient.
        """
        n = len(hashes)
        if n == 0:
            return np.zeros(self._n_ranks, dtype=np.float64)

        for i in range(n - 1, -1, -1):
            row = self._hash_to_row.get(hashes[i])
            if row is None:
                continue
            present = self._timestamps[row] > 0
            if present.any():
                return present.astype(np.float64) * ((i + 1.0) / n)

        return np.zeros(self._n_ranks, dtype=np.float64)

    def get_row(self, h: int) -> np.ndarray | None:
        """Return the timestamp row for hash *h*, or ``None`` if unseen.

        The returned array has shape ``(n_ranks,)``; entries > 0 indicate
        which ranks hold this hash and when it was assigned.
        """
        row = self._hash_to_row.get(h)
        if row is None:
            return None
        return self._timestamps[row]

    def max_timestamps(self, hashes: Sequence[int]) -> np.ndarray:
        """Return the element-wise max timestamp across *hashes* per rank.

        Returns a int64 array of shape ``(n_ranks,)``.
        """
        rows = [self._hash_to_row[h] for h in hashes if h in self._hash_to_row]
        if not rows:
            return np.zeros(self._n_ranks, dtype=np.int64)
        return self._timestamps[rows].max(axis=0)

    def get_timestamp(self, h: int, rank_idx: int) -> int:
        """Return the timestamp for a single (hash, rank) pair, or 0."""
        row = self._hash_to_row.get(h)
        if row is None:
            return 0
        return int(self._timestamps[row, rank_idx])

    def has(self, h: int, rank_idx: int) -> bool:
        """Return whether *rank_idx* has been assigned hash *h*."""
        return self.get_timestamp(h, rank_idx) > 0

    @property
    def assignment_counter(self) -> int:
        """The current assignment counter (number of :meth:`record` calls)."""
        return self._assignment_counter

    @property
    def n_ranks(self) -> int:
        """Number of ranks."""
        return self._n_ranks

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def compact(self) -> None:
        """Remove stale rows and shrink the backing array when under-utilized.

        A row is *stale* when all of its timestamps are zero (no rank holds the
        hash).  After removing stale rows, the backing array is halved
        repeatedly while live rows occupy less than half the capacity.
        """
        # Identify live rows (at least one non-zero timestamp).
        live_mask = (self._timestamps[: self._next_row] != 0).any(axis=1)
        live_indices = np.nonzero(live_mask)[0]
        n_live = len(live_indices)
        removed_rows = n_live < self._next_row

        if removed_rows:
            # Rebuild the table with only live rows, preserving order.
            new_timestamps = self._timestamps[live_indices]  # (n_live, n_ranks)

            # Rebuild hash→row mapping.
            old_to_new = {int(old): new for new, old in enumerate(live_indices)}
            self._hash_to_row = {
                h: old_to_new[row] for h, row in self._hash_to_row.items() if row in old_to_new
            }
            self._next_row = n_live
        else:
            new_timestamps = self._timestamps[:n_live]

        # Shrink: halve capacity while live rows fit in less than half.
        new_capacity = self._timestamps.shape[0]
        while n_live < 0.5 * new_capacity and new_capacity > 256:
            new_capacity //= 2

        if new_capacity != self._timestamps.shape[0] or removed_rows:
            self._timestamps = np.zeros((new_capacity, self._n_ranks), dtype=np.int64)
            self._timestamps[:n_live] = new_timestamps

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_row(self, h: int) -> int:
        row = self._hash_to_row.get(h)
        if row is None:
            row = self._next_row
            self._next_row += 1
            self._ensure_capacity(self._next_row)
            self._hash_to_row[h] = row
        return row

    def add_rank(self) -> int:
        """Append a new rank column and return its index.

        All existing hash timestamps for the new rank are initialised to zero.
        """
        new_idx = self._n_ranks
        self._n_ranks += 1
        # Expand timestamps array by one column.
        new_col = np.zeros((self._timestamps.shape[0], 1), dtype=np.int64)
        self._timestamps = np.concatenate([self._timestamps, new_col], axis=1)
        return new_idx

    def _ensure_capacity(self, needed_rows: int) -> None:
        current = self._timestamps.shape[0]
        if needed_rows <= current:
            return
        new_capacity = current
        while new_capacity < needed_rows:
            new_capacity *= 2
        new_table = np.zeros((new_capacity, self._n_ranks), dtype=np.int64)
        new_table[:current, :] = self._timestamps
        self._timestamps = new_table
