# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Dict, List, Sequence

import numpy as np


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

    def match_vector(self, hashes: Sequence[int]) -> np.ndarray:
        """Return a float64 vector of shape ``(n_ranks,)`` with 1.0 where *any*
        of the given *hashes* are present for that rank, 0.0 otherwise.

        This is fully vectorized: the only Python-level iteration is over the
        hash→row dict lookups; the per-rank reduction is done with numpy.
        """
        row_indices = [self._hash_to_row[h] for h in set(hashes) if h in self._hash_to_row]
        if row_indices:
            rows = self._timestamps[row_indices]  # (num_hashes, n_ranks)
            return (rows > 0).any(axis=0).astype(np.float64)
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
        return self._n_ranks

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
