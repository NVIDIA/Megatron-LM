# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tile planning of indexer top-k selection calls (internal).

:func:`plan_segment` splits one :class:`~.layout.QuerySegment` into the rows the
matched-precision reference selector selects and the query tiles the LiteTopK plugin selects,
using host integers only. A tile row is eligible when its causal position is at least
``startup_position`` (below it the reference selector is faster; no row is eligible when it is
None) and it sees the route's HOT prefix (the keys scored for the seed of every tile);
visibility grows with the position, so the eligible rows of a segment are a suffix of it.

Tile grids, by route:

* ``fp8_paged`` (any tile of a multiple of four rows up to the admitted length): tiles of
  ``tile_rows`` rows from the first eligible row and a shorter last tile. When the eligible rows
  are not a multiple of four, the remainder goes to the reference selector at their start.
* ``fp4_slab`` (qualified tile lengths only): the segment is covered from its end by tiles of
  ``tile_rows`` rows and, when its rows are not a multiple of ``tile_rows``, a last tile of the
  longest qualified length below ``tile_rows``. The leading remainder and the tiles that start
  before the first eligible row go to the reference selector.

Tiles of lengths the route does not admit go to the reference selector, and so do tiles whose
rows see more keys than ``max_tile_keys`` (the candidate capacity of a slab route).

The first tile group of a segment gets its HOT seed from the reference selections of the rows
that precede it (``seed_bootstrap="reference"``): the last ``vote_rows`` rows before the first
tile when that many precede it and they see the HOT prefix (a seed needs that many keys to
vote among), otherwise the first tile itself, which the reference selector then selects. With
``seed_bootstrap="identity"`` the plugin starts from its cold-start seed.
Later groups use the seed the previous group carries. Consecutive tiles of equal length form
groups of at most ``group_tiles`` tiles that share one plugin plan.

A segment whose LiteTopK tiles cover fewer than ``min_litetopk_pairs`` query-key pairs (the
per-segment crossover) goes to the reference selector entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from megatron.lite.primitive.kernels.indexer_topk.layout import QuerySegment

if TYPE_CHECKING:
    from megatron.lite.primitive.kernels.indexer_topk.config import ResolvedIndexerTopKTuning
    from megatron.lite.primitive.kernels.indexer_topk.plugins.abi import RouteCapability

__all__ = ["SegmentPlan", "Tile", "TileGroup", "plan_segment"]


@dataclass(frozen=True, slots=True)
class Tile:
    """Consecutive query rows the LiteTopK plugin selects in one call.

    Attributes:
        row_start: First local row.
        row_end: One past the last local row.
        position: Causal position of ``row_start`` within its sequence.
    """

    row_start: int
    row_end: int
    position: int

    @property
    def rows(self) -> int:
        """Number of rows in the tile."""
        return self.row_end - self.row_start


@dataclass(frozen=True, slots=True)
class TileGroup:
    """Consecutive tiles of equal length that share one plugin plan.

    Attributes:
        tiles: The tiles, in row order.
        common_end: Keys visible to the first row of the group, the key window the shared plan
            gathers (every later row sees at least these keys).
    """

    tiles: tuple[Tile, ...]
    common_end: int


@dataclass(frozen=True, slots=True)
class SegmentPlan:
    """How the rows of one segment are selected.

    Attributes:
        segment: The planned segment.
        reference_rows: Ordered, disjoint local row ranges the reference selector selects: every
            row of the segment outside the LiteTopK tiles, including a bootstrap tile.
        groups: LiteTopK tile groups in row order; empty when LiteTopK selects no row.
        seed: How the first group gets its HOT seed: ``reference`` (the votes of ``vote_rows``)
            or ``identity``; None without groups.
        vote_rows: Local rows ``(start, end)`` whose reference selections vote the first seed.
        vote_extent: Keys visible to the last vote row, the key extent of the voted seed.
        bootstrap_tile: The segment's first tile when the reference selector selects it for the
            vote, because the rows that precede it are too few or see too few keys.
        litetopk_pairs: Query-key pairs the LiteTopK tiles cover (visible keys summed over their
            rows).
        reason: Why LiteTopK selects no row of the segment, or None.
    """

    segment: QuerySegment
    reference_rows: tuple[tuple[int, int], ...]
    groups: tuple[TileGroup, ...] = ()
    seed: Literal["reference", "identity"] | None = None
    vote_rows: tuple[int, int] | None = None
    vote_extent: int = 0
    bootstrap_tile: Tile | None = None
    litetopk_pairs: int = 0
    reason: str | None = None

    @property
    def tiles(self) -> tuple[Tile, ...]:
        """The LiteTopK tiles of all groups, in row order."""
        return tuple(tile for group in self.groups for tile in group.tiles)

    @property
    def litetopk_rows(self) -> int:
        """Rows the LiteTopK tiles cover."""
        return sum(tile.rows for group in self.groups for tile in group.tiles)


def _visible(segment: QuerySegment, key_ratio: int, row: int) -> int:
    return min(segment.key_count, (segment.position + row - segment.row_start + 1) // key_ratio)


def _prefix_pairs(tokens: int, keys: int, key_ratio: int) -> int:
    """Sum of ``min(keys, x // key_ratio)`` over ``x`` in ``[0, tokens)``."""
    capped = min(tokens, keys * key_ratio)
    quotient, remainder = divmod(capped, key_ratio)
    return (
        key_ratio * quotient * (quotient - 1) // 2 + remainder * quotient + keys * (tokens - capped)
    )


def _pairs(segment: QuerySegment, key_ratio: int, row_start: int, row_end: int) -> int:
    """Visible keys summed over the local rows ``[row_start, row_end)`` of ``segment``."""
    first = segment.position + row_start - segment.row_start + 1
    last = segment.position + row_end - segment.row_start + 1
    return _prefix_pairs(last, segment.key_count, key_ratio) - _prefix_pairs(
        first, segment.key_count, key_ratio
    )


def _paged_grid(first: int, end: int, tile_rows: int) -> list[tuple[int, int]]:
    """Tiles of ``tile_rows`` rows over ``[first, end)`` after a remainder of up to 3 rows."""
    start = first + (end - first) % 4
    return [(row, min(row + tile_rows, end)) for row in range(start, end, tile_rows)]


def _slab_grid(
    start: int, end: int, tile_rows: int, tail_rows: int | None
) -> list[tuple[int, int]]:
    """Tiles covering ``[start, end)`` from its end: ``tile_rows`` rows, then a ``tail_rows`` tile.

    When the rows are a multiple of ``tile_rows``, or no tail length fits, only full tiles
    (aligned to the end) are used; the leading remainder is not covered.
    """
    rows = end - start
    if rows % tile_rows == 0 or tail_rows is None or rows < tail_rows:
        first = rows % tile_rows
        return [(start + row, start + row + tile_rows) for row in range(first, rows, tile_rows)]
    tail_start = rows - tail_rows
    tiles = [
        (start + row, start + row + tile_rows)
        for row in range(tail_start % tile_rows, tail_start, tile_rows)
    ]
    tiles.append((start + tail_start, end))
    return tiles


def _group(
    segment: QuerySegment, key_ratio: int, tiles: list[Tile], group_tiles: int
) -> tuple[TileGroup, ...]:
    """Group adjacent tiles of equal length, at most ``group_tiles`` per group."""
    groups = []
    members: list[Tile] = []
    for tile in tiles:
        if members and (
            len(members) == group_tiles
            or tile.rows != members[0].rows
            or tile.row_start != members[-1].row_end
        ):
            groups.append(members)
            members = []
        members.append(tile)
    if members:
        groups.append(members)
    return tuple(
        TileGroup(tuple(members), _visible(segment, key_ratio, members[0].row_start))
        for members in groups
    )


def _complement(start: int, end: int, tiles: list[Tile]) -> tuple[tuple[int, int], ...]:
    """The row ranges of ``[start, end)`` outside the ordered ``tiles``."""
    ranges = []
    cursor = start
    for tile in tiles:
        if cursor < tile.row_start:
            ranges.append((cursor, tile.row_start))
        cursor = tile.row_end
    if cursor < end:
        ranges.append((cursor, end))
    return tuple(ranges)


def plan_segment(
    segment: QuerySegment,
    *,
    key_ratio: int,
    route: RouteCapability | None,
    tuning: ResolvedIndexerTopKTuning,
    topk: int,
    vote_rows: int,
    max_tile_keys: int | None = None,
) -> SegmentPlan:
    """Plan which rows of a segment the reference selector and the LiteTopK plugin select.

    A host computation; see the module docstring for the rules.

    Args:
        segment: The segment to plan.
        key_ratio: Query tokens per key of the layout.
        route: The plugin route of the layer, or None when only the reference selector runs.
        tuning: The resolved settings of the layer.
        topk: Keys selected per row.
        vote_rows: Rows whose selections vote a HOT seed (the plugin's ``carry_vote_rows()``).
        max_tile_keys: The most keys a LiteTopK tile row may see, or None for no limit. A
            route with a fixed candidate capacity holds at most one candidate per visible key,
            so tiles within the capacity cannot overflow it.

    Returns:
        The segment's plan. Without LiteTopK tiles every row goes to the reference selector and
        ``reason`` says why.

    Raises:
        ValueError: If ``key_ratio`` or ``vote_rows`` is not a positive integer.
    """
    if type(key_ratio) is not int or key_ratio < 1:
        raise ValueError(f"key_ratio must be a positive integer, got {key_ratio!r}")
    if type(vote_rows) is not int or vote_rows < 1:
        raise ValueError(f"vote_rows must be a positive integer, got {vote_rows!r}")

    def reference_only(reason: str) -> SegmentPlan:
        return SegmentPlan(
            segment=segment, reference_rows=((segment.row_start, segment.row_end),), reason=reason
        )

    if route is None:
        return reference_only("no LiteTopK route")
    if not route.supports_topk(topk):
        return reference_only("top-k not supported by the route")
    if segment.key_count < route.min_keys:
        return reference_only("fewer keys than the route minimum")
    if segment.key_count > route.max_keys:
        return reference_only("more keys than the route maximum")
    if tuning.startup_position is None:
        return reference_only("no LiteTopK start position for the kernel heads")
    first_position = max(
        tuning.startup_position, route.hot_prefix * key_ratio - 1, segment.position
    )
    first_row = segment.row_start + first_position - segment.position
    if segment.key_count < route.hot_prefix or first_row >= segment.row_end:
        return reference_only("no row after the startup position sees the HOT prefix")

    if route.name == "fp8_paged":
        grid = _paged_grid(first_row, segment.row_end, tuning.tile_rows)
    else:
        tail_rows = max(
            (rows for rows in route.qualified_query_lengths if rows < tuning.tile_rows),
            default=None,
        )
        grid = [
            tile
            for tile in _slab_grid(segment.row_start, segment.row_end, tuning.tile_rows, tail_rows)
            if tile[0] >= first_row
        ]
    tiles = [
        Tile(start, end, segment.position + start - segment.row_start)
        for start, end in grid
        if route.admits_query_length(end - start)
    ]
    if not tiles:
        return reference_only("no tile of an admitted length")
    if max_tile_keys is not None:
        tiles = [
            tile
            for tile in tiles
            if _visible(segment, key_ratio, tile.row_end - 1) <= max_tile_keys
        ]
        if not tiles:
            return reference_only("every tile sees more keys than the candidate capacity")

    bootstrap_tile = None
    vote = None
    if tuning.seed_bootstrap == "reference":
        first = tiles[0].row_start
        if (
            first - segment.row_start >= vote_rows
            and _visible(segment, key_ratio, first - 1) >= route.hot_prefix
        ):
            vote = (first - vote_rows, first)
        else:
            bootstrap_tile = tiles.pop(0)
            vote = (
                bootstrap_tile.row_end - min(vote_rows, bootstrap_tile.rows),
                bootstrap_tile.row_end,
            )
        if not tiles:
            return reference_only("the only tile bootstraps the seed")
    pairs = sum(_pairs(segment, key_ratio, tile.row_start, tile.row_end) for tile in tiles)
    if pairs < tuning.min_litetopk_pairs:
        return reference_only("below the LiteTopK crossover")
    return SegmentPlan(
        segment=segment,
        reference_rows=_complement(segment.row_start, segment.row_end, tiles),
        groups=_group(segment, key_ratio, tiles, tuning.group_tiles),
        seed=tuning.seed_bootstrap,
        vote_rows=vote,
        vote_extent=0 if vote is None else _visible(segment, key_ratio, vote[1] - 1),
        bootstrap_tile=bootstrap_tile,
        litetopk_pairs=pairs,
    )
