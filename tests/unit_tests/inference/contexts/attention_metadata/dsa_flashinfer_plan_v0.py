# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Verbatim copy of the FlashInfer DSA step planner as first written (before its readability
refactor into ``dsa_flashinfer_plan``), kept only as the golden reference of
``test_dsa_flashinfer_plan.test_build_step_plan_matches_v0``. Do not modify; delete together
with that test once the refactored planner has settled.
"""

from typing import Dict, Optional, Sequence

import numpy as np

FI_PAGE = 64
TILE_ROWS = 128
SAMPLES, PAGES = 3, 2


def union_group(heads: int) -> int:
    """Tokens per union-core group: the most whose query heads fit FA4's 128-row tile."""
    return 1 << ((TILE_ROWS // heads).bit_length() - 1)


def plan_sizes(
    padded_rows: int,
    padded_requests: int,
    decode_only: bool,
    padded_prefill_requests: Optional[int] = None,
) -> Dict[str, int]:
    """Static plan sizes for a padded batch: decode-path rows, prefill tiles and requests.

    Only requests that take the tile path can leave a partial tile, so the tiles beyond the
    rows' own are bounded by ``padded_prefill_requests`` (the batch's padded prefill request
    count; ``padded_requests`` when not given). Leading decode requests take the decode path.
    """
    if decode_only:
        return {"decode_rows": padded_rows, "tiles": 0, "prefill_requests": 0, "compact_rows": 0}
    prefill = padded_requests if padded_prefill_requests is None else padded_prefill_requests
    return {
        "decode_rows": min(padded_rows, padded_requests),
        "tiles": -(-padded_rows // TILE_ROWS) + prefill,
        "prefill_requests": prefill,
        # The per-row sample kernels run on the live prefill rows, at most the padded rows,
        # rather than on every tiled row.
        "compact_rows": padded_rows,
    }


def build_step_plan(
    query_lengths: Sequence[int],
    kv_lengths: Sequence[int],
    *,
    padded_rows: int,
    padded_requests: int,
    decode_only: bool,
    heads: int,
    padded_prefill_requests: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """The plan arrays (int32, CPU) of a step whose request ``i`` adds ``query_lengths[i]`` rows,
    in order, the last at key ``kv_lengths[i] - 1``; sizes depend only on the padded batch.

    The leading single-token requests whose rows are among the first ``decode_rows`` rows take
    the decode path; every other request's rows go into prefill tiles.
    """
    sizes = plan_sizes(padded_rows, padded_requests, decode_only, padded_prefill_requests)
    group = union_group(heads)
    q_len = np.asarray(query_lengths, dtype=np.int64)
    kv_len = np.asarray(kv_lengths, dtype=np.int64)
    first_row = np.cumsum(q_len) - q_len
    # The decode path takes the leading single-token requests (the dynamic context orders
    # decode requests first), so the prefill rows stay consecutive for the union core.
    multi = np.flatnonzero(q_len > 1)
    leading = np.arange(len(q_len)) < (multi[0] if len(multi) else len(q_len))
    on_decode = leading & (first_row < sizes["decode_rows"])
    decode_requests = np.zeros(sizes["decode_rows"], np.int32)
    decode_lengths = np.zeros(sizes["decode_rows"], np.int32)
    decode_ids = np.flatnonzero(on_decode)
    decode_requests[first_row[decode_ids]] = decode_ids
    decode_lengths[first_row[decode_ids]] = kv_len[decode_ids]
    if decode_only:
        assert on_decode.sum() == len(q_len), "A decode-only step holds single-token requests."
    tiles, prefill_requests = sizes["tiles"], sizes["prefill_requests"]
    prefill = np.flatnonzero(~on_decode)
    assert len(prefill) <= prefill_requests
    request_tiles = (q_len[prefill] + TILE_ROWS - 1) // TILE_ROWS
    assert request_tiles.sum() <= tiles
    tile_ordinal = np.repeat(np.arange(len(prefill)), request_tiles)
    tile_start = TILE_ROWS * (
        np.arange(len(tile_ordinal))
        - np.repeat(np.cumsum(request_tiles) - request_tiles, request_tiles)
    )
    request = prefill[tile_ordinal]
    count = np.minimum(TILE_ROWS, q_len[request] - tile_start)
    lane = np.arange(TILE_ROWS)
    live = lane < count[:, None]
    row_ends = np.zeros((tiles, TILE_ROWS), np.int32)
    query_rows = np.full((tiles, TILE_ROWS), -1, np.int32)
    real = len(tile_ordinal)
    row_ends[:real] = np.where(
        live, (kv_len - q_len)[request][:, None] + tile_start[:, None] + lane + 1, 0
    )
    query_rows[:real] = np.where(live, first_row[request][:, None] + tile_start[:, None] + lane, -1)
    tile_requests = np.zeros(tiles, np.int32)
    tile_requests[:real] = request
    tile_ordinals = np.zeros(tiles, np.int32)
    tile_ordinals[:real] = tile_ordinal
    request_rows = np.zeros(prefill_requests, np.int32)
    request_rows[: len(prefill)] = prefill
    sample_ends = np.zeros(prefill_requests, np.int32)
    sample_ends[: len(prefill)] = kv_len[prefill] // (PAGES * FI_PAGE) * SAMPLES
    # Groups of `group` tiled rows: each is a union-core batch over its live rows, which are
    # consecutive step rows; an empty group is an empty batch at the previous one's end.
    group_rows = query_rows.reshape(-1, group)
    group_live = (group_rows >= 0).sum(1)
    live_groups = group_live > 0
    ends = np.where(live_groups, group_rows[:, 0] + group_live, 0)
    first = int(group_rows[live_groups, 0].min()) if live_groups.any() else 0
    # An empty group starts (and ends) where the last live group before it ended.
    carried = np.maximum(np.maximum.accumulate(ends), first)
    previous = np.concatenate(([first], carried[:-1]))
    batch_starts = np.empty(len(ends) + 1, np.int32)
    batch_starts[:-1] = np.where(live_groups, group_rows[:, 0], previous)
    batch_starts[-1] = carried[-1] if len(carried) else first
    # The live tiled rows in order, for the per-row sample kernels; padding tiled rows map to
    # the one past the last compact row, which is empty.
    compact = sizes["compact_rows"]
    live_flat = np.flatnonzero(query_rows.ravel() >= 0)
    assert len(live_flat) <= compact
    compact_rows = np.zeros(compact, np.int32)
    compact_rows[: len(live_flat)] = live_flat
    compact_ends = np.zeros(compact, np.int32)
    compact_ends[: len(live_flat)] = row_ends.ravel()[live_flat]
    compact_ordinals = np.zeros(compact, np.int32)
    compact_ordinals[: len(live_flat)] = tile_ordinals[live_flat // TILE_ROWS]
    tiled_compact = np.full(tiles * TILE_ROWS, compact, np.int32)
    tiled_compact[live_flat] = np.arange(len(live_flat))
    return {
        "decode_requests": decode_requests,
        "decode_lengths": decode_lengths,
        "tile_requests": tile_requests,
        "tile_ordinals": tile_ordinals,
        "row_ends": row_ends.ravel(),
        "query_rows": query_rows.ravel(),
        "request_rows": request_rows,
        "sample_ends": sample_ends,
        "batch_starts": batch_starts,
        "compact_rows": compact_rows,
        "compact_ends": compact_ends,
        "compact_ordinals": compact_ordinals,
        "tiled_compact": tiled_compact,
    }
