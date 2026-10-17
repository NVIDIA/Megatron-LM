# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Indexer shapes and the query layouts of indexer top-k selection calls.

A selection call scores the local query rows of one module invocation against a key tensor.
The rows can belong to several sequences (packed inputs) and start anywhere in a sequence (a
context-parallel shard), and every sequence owns a contiguous run of keys. A
:class:`QueryLayout` describes this with host integers only, so a call is planned without
reading device memory:

* a :class:`QuerySegment` is a run of consecutive local rows of one sequence, with the causal
  position of its first row, where the sequence's keys start in the key tensor, how many there
  are, and the offset added to the sequence-local key ids that the call returns;
* the row at causal position ``p`` sees the first ``min(key_count, (p + 1) // key_ratio)`` keys
  of its sequence, where ``key_ratio`` is the number of query tokens per key (1 when every token
  has a key, 4 when keys are compressed four to one);
* local rows outside every segment (padding) select nothing: their output rows are all -1.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from torch import Tensor

__all__ = ["IndexerGeometry", "QueryLayout", "QuerySegment"]

# Output key ids are int32.
_MAX_KEY_ID = 2**31 - 1


def _check_count(owner: str, name: str, value: object, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{owner}.{name} must be an integer >= {minimum}, got {value!r}")


@dataclass(frozen=True, slots=True)
class IndexerGeometry:
    """The shape of the indexer a selector serves.

    Attributes:
        num_heads: Indexer query heads whose weighted scores are summed; the full head count, as
            selection needs the sum over every head.
        head_dim: Indexer query and key head dimension.
        topk: Keys selected per query row.
        key_ratio: Query tokens per key: 1 when every token has a key, 4 for keys compressed four
            to one.
    """

    num_heads: int
    head_dim: int
    topk: int
    key_ratio: int

    def __post_init__(self) -> None:
        for name in ("num_heads", "head_dim", "topk", "key_ratio"):
            _check_count("IndexerGeometry", name, getattr(self, name), 1)


@dataclass(frozen=True, slots=True)
class QuerySegment:
    """Consecutive local query rows of one sequence.

    Attributes:
        row_start: First local row of the segment.
        row_end: One past the last local row of the segment.
        position: Causal position of ``row_start`` within its sequence (in query tokens).
        key_start: Row of the sequence's first key in the key tensor.
        key_count: Keys of the sequence (compressed keys when the key ratio is above 1).
        index_base: Added to the sequence-local key ids the selection returns: 0 for
            sequence-relative ids, ``key_start`` for ids into the key tensor.
    """

    row_start: int
    row_end: int
    position: int
    key_start: int
    key_count: int
    index_base: int

    def __post_init__(self) -> None:
        for name in ("row_start", "position", "key_start", "key_count", "index_base"):
            _check_count("QuerySegment", name, getattr(self, name), 0)
        _check_count("QuerySegment", "row_end", self.row_end, self.row_start + 1)
        if self.index_base + self.key_count - 1 > _MAX_KEY_ID:
            raise ValueError(
                f"QuerySegment ids up to {self.index_base + self.key_count - 1} do not fit int32"
            )

    @property
    def rows(self) -> int:
        """Number of rows in the segment."""
        return self.row_end - self.row_start


@dataclass(frozen=True, slots=True)
class QueryLayout:
    """The local query rows of a selection call and the keys each row sees.

    Attributes:
        rows: Local query rows of the call; rows outside every segment select nothing.
        key_ratio: Query tokens per key. The row at causal position ``p`` of a segment sees
            ``min(key_count, (p + 1) // key_ratio)`` keys.
        segments: Disjoint segments ordered by ``row_start``, within ``[0, rows)``.
    """

    rows: int
    key_ratio: int
    segments: tuple[QuerySegment, ...]

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def full(cls, rows: int, *, keys: int, key_ratio: int = 1) -> QueryLayout:
        """All rows of one sequence from its first token, such as a whole prompt.

        Args:
            rows: Query rows, starting at causal position 0.
            keys: Keys of the sequence, which start at row 0 of the key tensor.
            key_ratio: Query tokens per key.

        Returns:
            A layout with one segment ``(0, rows, 0, 0, keys, 0)``, or none when ``rows`` is 0.
        """
        return cls.contiguous(rows, position=0, keys=keys, key_ratio=key_ratio)

    @classmethod
    def contiguous(cls, rows: int, *, position: int, keys: int, key_ratio: int = 1) -> QueryLayout:
        """Consecutive rows of one sequence, such as a contiguous context-parallel shard.

        Args:
            rows: Query rows.
            position: Causal position of the first row.
            keys: Keys of the whole sequence (not only the visible ones), which start at row 0
                of the key tensor.
            key_ratio: Query tokens per key.

        Returns:
            A layout with one segment ``(0, rows, position, 0, keys, 0)``, or none when ``rows``
            is 0.
        """
        segments = (QuerySegment(0, rows, position, 0, keys, 0),) if rows else ()
        return cls(rows=rows, key_ratio=key_ratio, segments=segments)

    @classmethod
    def packed(
        cls,
        cu_seqlens: Sequence[int],
        *,
        row_start: int,
        rows: int,
        key_ratio: int = 1,
        key_counts: Sequence[int] | None = None,
        absolute_ids: bool,
    ) -> QueryLayout:
        """Local rows of packed sequences, such as one contiguous shard of a packed batch.

        Sequence ``s`` holds the packed query tokens ``[cu_seqlens[s], cu_seqlens[s + 1])``; its
        ``key_counts[s]`` keys follow those of the earlier sequences in the key tensor. The call
        holds the packed tokens ``[row_start, row_start + rows)``; local rows past the last
        sequence are padding.

        Args:
            cu_seqlens: Cumulative query token counts of the packed sequences as host integers
                (for example ``cu_seqlens.tolist()``), starting at 0 and non-decreasing.
            row_start: Packed token of local row 0.
            rows: Local query rows.
            key_ratio: Query tokens per key.
            key_counts: Keys of every sequence; defaults to ``length // key_ratio``.
            absolute_ids: Return ids into the key tensor (``index_base = key_start``) instead of
                sequence-relative ids (``index_base = 0``).

        Returns:
            One segment per sequence that intersects the local rows.

        Raises:
            TypeError: If ``cu_seqlens`` or ``key_counts`` is a tensor.
            ValueError: If the offsets or counts are inconsistent.
        """
        if isinstance(cu_seqlens, Tensor) or isinstance(key_counts, Tensor):
            raise TypeError(
                "QueryLayout.packed takes host integers; pass cu_seqlens.tolist() so that the "
                "device synchronization is explicit at the call site"
            )
        offsets = list(cu_seqlens)
        if (
            not offsets
            or any(type(offset) is not int for offset in offsets)
            or offsets[0] != 0
            or any(later < earlier for earlier, later in zip(offsets, offsets[1:]))
        ):
            raise ValueError(
                f"cu_seqlens must be non-decreasing integers starting at 0, got {offsets}"
            )
        _check_count("QueryLayout.packed", "row_start", row_start, 0)
        _check_count("QueryLayout.packed", "rows", rows, 0)
        _check_count("QueryLayout.packed", "key_ratio", key_ratio, 1)
        lengths = [later - earlier for earlier, later in zip(offsets, offsets[1:])]
        if key_counts is None:
            counts = [length // key_ratio for length in lengths]
        else:
            counts = list(key_counts)
            if len(counts) != len(lengths) or any(
                type(count) is not int or count < 0 for count in counts
            ):
                raise ValueError(
                    f"key_counts must hold {len(lengths)} non-negative integers, got {counts}"
                )
        row_end = row_start + rows
        segments = []
        key_start = 0
        for sequence, count in enumerate(counts):
            first = max(offsets[sequence], row_start)
            last = min(offsets[sequence + 1], row_end)
            if first < last:
                segments.append(
                    QuerySegment(
                        row_start=first - row_start,
                        row_end=last - row_start,
                        position=first - offsets[sequence],
                        key_start=key_start,
                        key_count=count,
                        index_base=key_start if absolute_ids else 0,
                    )
                )
            key_start += count
        return cls(rows=rows, key_ratio=key_ratio, segments=tuple(segments))

    def visible_keys(self, segment: QuerySegment, row: int) -> int:
        """Return how many keys of its sequence the local ``row`` of ``segment`` sees.

        A host computation; nothing is read from the device.

        Args:
            segment: A segment of this layout.
            row: A local row of ``segment``.

        Returns:
            ``min(key_count, (position + 1) // key_ratio)`` for the row's causal position.

        Raises:
            ValueError: If ``row`` is outside ``segment``.
        """
        if not segment.row_start <= row < segment.row_end:
            raise ValueError(
                f"row {row} is outside the segment rows [{segment.row_start}, {segment.row_end})"
            )
        position = segment.position + row - segment.row_start
        return min(segment.key_count, (position + 1) // self.key_ratio)

    def validate(self) -> None:
        """Check that the segments are ordered, disjoint and within the rows.

        Raises:
            ValueError: If the rows, key ratio or segments are inconsistent.
        """
        _check_count("QueryLayout", "rows", self.rows, 0)
        _check_count("QueryLayout", "key_ratio", self.key_ratio, 1)
        if not isinstance(self.segments, tuple) or not all(
            isinstance(segment, QuerySegment) for segment in self.segments
        ):
            raise ValueError("QueryLayout.segments must be a tuple of QuerySegment")
        previous_end = 0
        for segment in self.segments:
            if segment.row_start < previous_end or segment.row_end > self.rows:
                raise ValueError(
                    f"QueryLayout segments must be ordered, disjoint and within [0, {self.rows}); "
                    f"got {self.segments}"
                )
            previous_end = segment.row_end
