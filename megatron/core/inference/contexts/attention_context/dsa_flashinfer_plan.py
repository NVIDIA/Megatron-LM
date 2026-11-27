# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Host-side step plan of DSA-over-GQA inference through FlashInfer's DSA kernels.

The FlashInfer DSA backend (``dsa_gqa_flashinfer``) splits a dynamic-batching step into two paths:

- **decode rows**: the leading single-token requests. Each such request has one row, which
  scores all of its keys, selects its top-k and attends to them;
- **prefill tiles**: every other request's rows, cut into ``TILE_ROWS``-row tiles of one request
  each. A tile's rows select their keys through a sampled threshold, and attend in groups of
  ``union_group(heads)`` consecutive rows (the "union core").

Every kernel launch has a shape fixed by the *padded* batch, never by the live requests, so the
plan can be built once per step on the CPU and replayed inside a CUDA graph. This module builds
that plan; it imports no FlashInfer code and runs on any machine.

How a step is planned (``build_step_plan``), on a worked example used throughout this module:
four requests with query lengths ``[1, 1, 300, 1]`` and KV lengths ``[50, 9000, 1300, 77]``,
padded to 400 rows and 8 requests, with 16 query heads (union groups of 8 rows):

1. ``_StepRequests``: request ``i`` owns step rows ``first_row[i] ..`` (``[0, 1, 2, 302]``) and
   its rows end at keys ``kv_len - q_len + 1 ..`` (request 2: keys 1001 .. 1300).
2. ``_plan_decode_rows``: requests 0 and 1 are leading single-token requests, so they take
   decode rows 0 and 1. Request 3 is single-token too but follows a prefill, so it is tiled.
3. ``_tile_prefill_requests``: request 2 needs 3 tiles (300 rows), request 3 one tile, so
   tiles ``[2, 2, 2, 3]`` with starting rows ``[0, 128, 256, 0]`` within their request.
4. ``_tiled_row_layout``: each tile's 128 lanes get a step row and a causal key count; lanes past
   the request's rows are padding (row -1, keys 0).
5. ``_union_core_batch_starts``: every 8 lanes form one union-core batch; the batches' first
   step rows are FA4's ``cu_seqlens_q`` (an empty batch starts where the previous one ended).
6. ``_compact_live_rows``: the per-row threshold-sample kernels run on the live tiled rows only
   (301 here), so their row count is bounded by the step's tokens, not by its tiles.
7. ``_threshold_sample_ends``: each prefill request samples ``SAMPLES`` keys per run of
   ``PAGES`` pages of its keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, TypedDict

import numpy as np
import torch

# FlashInfer's indexer page: the indexer-key cache is read as pages of this many keys.
FI_PAGE = 64
# Rows of one prefill query tile (FA4's query tile; every tile holds rows of one request).
TILE_ROWS = 128
# Threshold samples: SAMPLES keys at random offsets into each run of PAGES pages of a request's
# keys.
SAMPLES, PAGES = 3, 2
# gvr_2's top-k launches one grid row per input row, and a grid's y dimension holds at most 65535.
MAX_TOPK_ROWS = 65535


def ceil_div(numerator: int, denominator: int) -> int:
    """``ceil(numerator / denominator)`` for integers (also elementwise on integer arrays)."""
    return -(-numerator // denominator)


def union_group(heads: int) -> int:
    """Tokens per union-core group: the most whose query heads fit FA4's 128-row tile.

    The largest power of two ``g`` with ``g * heads <= TILE_ROWS``: 8 for 16 heads, 1 for 128.
    It always divides ``TILE_ROWS``, so a group never spans two tiles.
    """
    return 1 << ((TILE_ROWS // heads).bit_length() - 1)


class PlanSizes(TypedDict):
    """Static sizes of a step plan; they depend only on the padded batch."""

    decode_rows: int  # rows [0, decode_rows) may take the decode path
    tiles: int  # prefill query tiles (0: no prefill path)
    prefill_requests: int  # requests the prefill path can hold
    compact_rows: int  # live prefill rows the per-row sample kernels can hold


class StepPlanArrays(TypedDict):
    """The CPU arrays (int32) of one step's plan; shapes follow ``PlanSizes``.

    Padding entries are 0 unless noted. "Tiled rows" are the ``tiles * TILE_ROWS`` lanes of the
    prefill tiles, tile-major.
    """

    decode_requests: np.ndarray  # [decode_rows] request (block-table row) of each decode row
    decode_lengths: np.ndarray  # [decode_rows] its key count; 0: not a decode row
    tile_requests: np.ndarray  # [tiles] request of each prefill tile
    tile_ordinals: np.ndarray  # [tiles] prefill-request ordinal of each tile
    row_ends: np.ndarray  # [tiles * TILE_ROWS] causal key count of each tiled row (0: padding)
    query_rows: np.ndarray  # [tiles * TILE_ROWS] step row of each tiled row (-1: padding)
    request_rows: np.ndarray  # [prefill_requests] request of each prefill request ordinal
    sample_ends: np.ndarray  # [prefill_requests] threshold samples of each prefill request
    batch_starts: np.ndarray  # [tiles * TILE_ROWS / group + 1] first step row of each group
    compact_rows: np.ndarray  # [compact_rows] tiled index of each live tiled row
    compact_ends: np.ndarray  # [compact_rows] causal key count of each compact row
    compact_ordinals: np.ndarray  # [compact_rows] prefill-request ordinal of each compact row
    tiled_compact: np.ndarray  # [tiles * TILE_ROWS] compact index of each tiled row (pad: last)


@dataclass
class StepPlan:
    """One step's plan as device tensors (``StepPlanArrays`` on the GPU) and its static sizes."""

    decode_rows: int  # rows [0, decode_rows) may take the decode path
    decode_requests: torch.Tensor  # [decode_rows] block-table row of each decode-path row
    decode_lengths: torch.Tensor  # [decode_rows] its keys, 0 for rows not on the decode path
    num_tiles: int  # prefill query tiles (0: no prefill path)
    tile_requests: torch.Tensor  # [tiles] block-table row of each tile
    tile_ordinals: torch.Tensor  # [tiles] the tile's prefill request ordinal
    row_ends: torch.Tensor  # [tiles * 128] causal key count of each tiled row (0: none)
    query_rows: torch.Tensor  # [tiles * 128] the step row of each tiled row (-1: none)
    request_rows: torch.Tensor  # [prefill requests] block-table row of each prefill request
    sample_ends: torch.Tensor  # [prefill requests] sampled keys of each prefill request
    batch_starts: torch.Tensor  # [tiles * 128 / group + 1] first step row of each group
    compact_rows: torch.Tensor  # [padded rows] tiled index of each live prefill row (pad: 0)
    compact_ends: torch.Tensor  # [padded rows] causal key count of each compact row (pad: 0)
    compact_ordinals: torch.Tensor  # [padded rows] prefill request ordinal of each compact row
    tiled_compact: torch.Tensor  # [tiles * 128] compact index of each tiled row (pad: padded rows)
    max_context: int  # bound on every row length (the graph's bound under CUDA graphs)
    page_table: torch.Tensor  # [requests, pages] FlashInfer's 64-key pages (fill_page_table)


def plan_sizes(
    padded_rows: int,
    padded_requests: int,
    decode_only: bool,
    padded_prefill_requests: Optional[int] = None,
) -> PlanSizes:
    """Static plan sizes for a padded batch: decode-path rows, prefill tiles and requests.

    Only requests that take the tile path can leave a partial tile, so the tiles beyond the
    rows' own are bounded by ``padded_prefill_requests`` (the batch's padded prefill request
    count; ``padded_requests`` when not given). Leading decode requests take the decode path.

    Example: 400 rows and 8 requests, not decode-only, give 8 decode rows, ``ceil(400 / 128) +
    8 = 12`` tiles, 8 prefill requests and 400 compact rows.
    """
    if decode_only:
        return {"decode_rows": padded_rows, "tiles": 0, "prefill_requests": 0, "compact_rows": 0}
    prefill = padded_requests if padded_prefill_requests is None else padded_prefill_requests
    return {
        "decode_rows": min(padded_rows, padded_requests),
        "tiles": ceil_div(padded_rows, TILE_ROWS) + prefill,
        "prefill_requests": prefill,
        # The per-row sample kernels run on the live prefill rows, at most the padded rows,
        # rather than on every tiled row.
        "compact_rows": padded_rows,
    }


def check_plan_rows(max_tokens: int) -> None:
    """Reject a token budget whose per-row top-k exceeds one gvr_2 launch.

    The sample and decode top-k run on at most ``max_tokens`` real rows plus one empty row,
    however many requests the batch admits (the request-tiled layout is used only by the
    threshold select, whose grid has no such limit).
    """
    rows = max_tokens + 1
    if rows > MAX_TOPK_ROWS:
        raise ValueError(
            f"The FlashInfer DSA backend selects top-k over up to {rows} rows at "
            f"max_tokens={max_tokens}, above its limit of {MAX_TOPK_ROWS}. Lower max_tokens, "
            f"or use dsa_backend='triton'."
        )


def _segment_starts(counts: np.ndarray) -> np.ndarray:
    """Start index of each segment when segments of ``counts`` items are laid end to end.

    ``[2, 3, 1] -> [0, 2, 5]``.
    """
    return np.cumsum(counts) - counts


def _index_within_segment(counts: np.ndarray) -> np.ndarray:
    """For segments of ``counts`` items laid end to end, each item's index in its segment.

    ``[2, 3] -> [0, 1, 0, 1, 2]``.
    """
    return np.arange(counts.sum()) - np.repeat(_segment_starts(counts), counts)


@dataclass
class _StepRequests:
    """A step's requests: request ``i`` adds rows ``first_row[i] .. first_row[i] + q_len[i] - 1``
    and its last row is at key ``kv_len[i] - 1``."""

    q_len: np.ndarray  # int64 [requests] rows (query tokens) each request adds this step
    kv_len: np.ndarray  # int64 [requests] keys each request has after this step
    first_row: np.ndarray  # int64 [requests] step row of each request's first query token

    @classmethod
    def from_lengths(cls, query_lengths: Sequence[int], kv_lengths: Sequence[int]):
        """The requests of the given per-request query and KV lengths, in step-row order."""
        q_len = np.asarray(query_lengths, dtype=np.int64)
        kv_len = np.asarray(kv_lengths, dtype=np.int64)
        return cls(q_len=q_len, kv_len=kv_len, first_row=_segment_starts(q_len))


def _plan_decode_rows(requests: _StepRequests, decode_rows: int, decode_only: bool):
    """Which requests take the decode path, and the decode rows' arrays.

    The decode path takes the *leading* single-token requests (the dynamic context orders
    decode requests first) whose row is among the first ``decode_rows`` rows. A single-token
    request after the first multi-token one is tiled instead, so the prefill rows stay
    consecutive for the union core. Example: query lengths ``[1, 1, 300, 1]`` put requests 0
    and 1 on decode rows 0 and 1, with their KV lengths as the rows' key counts.

    Returns:
        (on_decode bool [requests], decode_requests int32 [decode_rows],
        decode_lengths int32 [decode_rows]); unused decode rows have length 0.
    """
    q_len, first_row = requests.q_len, requests.first_row
    multi_token = np.flatnonzero(q_len > 1)
    num_leading = multi_token[0] if len(multi_token) else len(q_len)
    on_decode = (np.arange(len(q_len)) < num_leading) & (first_row < decode_rows)
    decode_requests = np.zeros(decode_rows, np.int32)
    decode_lengths = np.zeros(decode_rows, np.int32)
    decode_ids = np.flatnonzero(on_decode)
    decode_requests[first_row[decode_ids]] = decode_ids
    decode_lengths[first_row[decode_ids]] = requests.kv_len[decode_ids]
    if decode_only and on_decode.sum() != len(q_len):
        raise ValueError("A decode-only step holds single-token requests only.")
    return on_decode, decode_requests, decode_lengths


@dataclass
class _PrefillTiles:
    """The live prefill tiles of a step (unpadded: one entry per tile in use)."""

    requests: np.ndarray  # [prefill requests] request of each prefill request ordinal
    tile_ordinal: np.ndarray  # [live tiles] prefill-request ordinal of each tile
    tile_request: np.ndarray  # [live tiles] request of each tile
    tile_start: np.ndarray  # [live tiles] index, in its request's rows, of each tile's first row


def _tile_prefill_requests(
    requests: _StepRequests, on_decode: np.ndarray, sizes: PlanSizes
) -> _PrefillTiles:
    """Cut every request not on the decode path into ``TILE_ROWS``-row tiles, in order.

    Example: prefill requests 2 (300 rows) and 3 (1 row) give tiles of requests
    ``[2, 2, 2, 3]``, prefill ordinals ``[0, 0, 0, 1]`` and starts ``[0, 128, 256, 0]``.
    """
    prefill = np.flatnonzero(~on_decode)
    if len(prefill) > sizes["prefill_requests"]:
        raise ValueError(
            f"{len(prefill)} requests take the prefill path, more than the plan's "
            f"{sizes['prefill_requests']}."
        )
    tiles_per_request = ceil_div(requests.q_len[prefill], TILE_ROWS)
    if tiles_per_request.sum() > sizes["tiles"]:
        raise ValueError(
            f"The prefill requests need {tiles_per_request.sum()} tiles, more than the plan's "
            f"{sizes['tiles']}."
        )
    tile_ordinal = np.repeat(np.arange(len(prefill)), tiles_per_request)
    return _PrefillTiles(
        requests=prefill,
        tile_ordinal=tile_ordinal,
        tile_request=prefill[tile_ordinal],
        tile_start=TILE_ROWS * _index_within_segment(tiles_per_request),
    )


def _tiled_row_layout(requests: _StepRequests, tiles: _PrefillTiles, num_tiles: int):
    """Each tiled row's causal key count (``row_ends``) and step row (``query_rows``).

    Lane ``j`` of a tile starting at row ``s`` of request ``r`` is the request's row ``s + j``,
    i.e. step row ``first_row[r] + s + j``, whose causal prefix ends at key
    ``kv_len[r] - q_len[r] + s + j + 1``. Lanes past the request's rows, and every lane of the
    padding tiles beyond the live ones, are padding: key count 0, step row -1.

    Returns:
        (row_ends, query_rows), int32 ``[num_tiles, TILE_ROWS]``.
    """
    request, start = tiles.tile_request, tiles.tile_start
    rows_in_tile = np.minimum(TILE_ROWS, requests.q_len[request] - start)
    lane = np.arange(TILE_ROWS)
    live = lane < rows_in_tile[:, None]
    # Request row (in the request's own rows) of each lane of each live tile.
    request_row = start[:, None] + lane
    first_key = (requests.kv_len - requests.q_len)[request][:, None]
    row_ends = np.zeros((num_tiles, TILE_ROWS), np.int32)
    query_rows = np.full((num_tiles, TILE_ROWS), -1, np.int32)
    live_tiles = len(request)
    row_ends[:live_tiles] = np.where(live, first_key + request_row + 1, 0)
    query_rows[:live_tiles] = np.where(live, requests.first_row[request][:, None] + request_row, -1)
    return row_ends, query_rows


def _union_core_batch_starts(query_rows: np.ndarray, group: int) -> np.ndarray:
    """FA4 ``cu_seqlens_q`` of the union core: the first step row of each ``group``-row batch.

    Every ``group`` consecutive tiled rows form one union-core batch over its live rows, which
    are consecutive step rows (live lanes are a prefix of each tile, and ``group`` divides
    ``TILE_ROWS``). ``cu_seqlens_q`` must not decrease, so a batch with no live rows is an empty
    batch placed where the last live batch before it ended (or at the first live row, for
    leading empty batches). The final entry is where the last batch ends.

    Example (group 8): tiles of request 2 cover step rows 2 .. 301, so the first batches start
    at rows 2, 10, 18, ...; the padding lanes of its last tile give empty batches at 302.

    Returns:
        int32 ``[tiles * TILE_ROWS / group + 1]``.
    """
    group_rows = query_rows.reshape(-1, group)
    group_live = (group_rows >= 0).sum(1)
    is_live = group_live > 0
    ends = np.where(is_live, group_rows[:, 0] + group_live, 0)
    first = int(group_rows[is_live, 0].min()) if is_live.any() else 0
    # Where the last live group at or before each group ended (at least the first live row).
    end_so_far = np.maximum(np.maximum.accumulate(ends), first)
    end_before = np.concatenate(([first], end_so_far[:-1]))
    batch_starts = np.empty(len(ends) + 1, np.int32)
    batch_starts[:-1] = np.where(is_live, group_rows[:, 0], end_before)
    batch_starts[-1] = end_so_far[-1] if len(end_so_far) else first
    return batch_starts


def _compact_live_rows(
    row_ends: np.ndarray, query_rows: np.ndarray, tile_ordinals: np.ndarray, compact: int
):
    """The live tiled rows in order, for the per-row threshold-sample kernels.

    The sample kernels work per row, so running them on every tiled row would scale with the
    tiles (up to ``padded_requests`` partial tiles); the live rows are at most the step's padded
    rows. ``tiled_compact`` maps each tiled row back to its compact row; padding tiled rows map
    to ``compact``, one past the last compact row, which the kernels keep empty.

    Returns:
        (compact_rows, compact_ends, compact_ordinals) int32 ``[compact]``, and
        tiled_compact int32 ``[tiles * TILE_ROWS]``.
    """
    live_flat = np.flatnonzero(query_rows >= 0)
    if len(live_flat) > compact:
        raise ValueError(f"{len(live_flat)} live prefill rows exceed the plan's {compact}.")
    num_live = len(live_flat)
    compact_rows = np.zeros(compact, np.int32)
    compact_rows[:num_live] = live_flat
    compact_ends = np.zeros(compact, np.int32)
    compact_ends[:num_live] = row_ends[live_flat]
    compact_ordinals = np.zeros(compact, np.int32)
    compact_ordinals[:num_live] = tile_ordinals[live_flat // TILE_ROWS]
    tiled_compact = np.full(len(query_rows), compact, np.int32)
    tiled_compact[live_flat] = np.arange(num_live)
    return compact_rows, compact_ends, compact_ordinals, tiled_compact


def _threshold_sample_ends(kv_len: np.ndarray) -> np.ndarray:
    """Threshold samples of requests with ``kv_len`` keys: ``SAMPLES`` per whole run of
    ``PAGES`` 64-key pages (example: 1300 keys -> 1300 // 128 * 3 = 30)."""
    return kv_len // (PAGES * FI_PAGE) * SAMPLES


def _padded(values: np.ndarray, size: int) -> np.ndarray:
    """``values`` as int32 in the leading entries of a zero array of ``size``."""
    out = np.zeros(size, np.int32)
    out[: len(values)] = values
    return out


def build_step_plan(
    query_lengths: Sequence[int],
    kv_lengths: Sequence[int],
    *,
    padded_rows: int,
    padded_requests: int,
    decode_only: bool,
    heads: int,
    padded_prefill_requests: Optional[int] = None,
) -> StepPlanArrays:
    """The plan arrays (int32, CPU) of a step whose request ``i`` adds ``query_lengths[i]`` rows,
    in order, the last at key ``kv_lengths[i] - 1``; sizes depend only on the padded batch.

    The leading single-token requests whose rows are among the first ``decode_rows`` rows take
    the decode path; every other request's rows go into prefill tiles. See the module docstring
    for a worked example.
    """
    sizes = plan_sizes(padded_rows, padded_requests, decode_only, padded_prefill_requests)
    requests = _StepRequests.from_lengths(query_lengths, kv_lengths)
    on_decode, decode_requests, decode_lengths = _plan_decode_rows(
        requests, sizes["decode_rows"], decode_only
    )
    tiles = _tile_prefill_requests(requests, on_decode, sizes)
    row_ends, query_rows = _tiled_row_layout(requests, tiles, sizes["tiles"])
    tile_ordinals = _padded(tiles.tile_ordinal, sizes["tiles"])
    row_ends, query_rows = row_ends.ravel(), query_rows.ravel()
    compact_rows, compact_ends, compact_ordinals, tiled_compact = _compact_live_rows(
        row_ends, query_rows, tile_ordinals, sizes["compact_rows"]
    )
    return {
        "decode_requests": decode_requests,
        "decode_lengths": decode_lengths,
        "tile_requests": _padded(tiles.tile_request, sizes["tiles"]),
        "tile_ordinals": tile_ordinals,
        "row_ends": row_ends,
        "query_rows": query_rows,
        "request_rows": _padded(tiles.requests, sizes["prefill_requests"]),
        "sample_ends": _padded(
            _threshold_sample_ends(requests.kv_len[tiles.requests]), sizes["prefill_requests"]
        ),
        "batch_starts": _union_core_batch_starts(query_rows, union_group(heads)),
        "compact_rows": compact_rows,
        "compact_ends": compact_ends,
        "compact_ordinals": compact_ordinals,
        "tiled_compact": tiled_compact,
    }


def page_table_columns(max_kv_blocks: int, page_size: int, max_context: int, index_topk: int):
    """Columns of the FlashInfer page table: the most of

    - the 64-key pages of a request's blocks (``max_kv_blocks * page_size / 64``);
    - the pages covering the decode score width (``max(max_context, index_topk)`` rounded up
      to 4), which ``indexer_scores`` reads;
    - twice the 128-key tiles of ``max_context``, which ``indexer_threshold_select`` reads.

    Entries past a row's length are never read.
    """
    score_width = ceil_div(max(max_context, index_topk), 4) * 4
    key_tiles = ceil_div(max_context, TILE_ROWS)
    return max(
        max_kv_blocks * (page_size // FI_PAGE), ceil_div(score_width, FI_PAGE), 2 * key_tiles
    )


def fill_page_table(block_table: torch.Tensor, page_size: int, out: torch.Tensor) -> None:
    """Write FlashInfer's 64-key pages of [requests, blocks] page_size-key blocks into the leading
    columns of ``out`` (block b holds pages b * per .. b * per + per - 1), once per step.

    Example: page_size 256 (4 pages per block) and block table row ``[2, 0]`` give pages
    ``[8, 9, 10, 11, 0, 1, 2, 3]``.
    """
    per = page_size // FI_PAGE
    requests, blocks = block_table.shape
    pages = out[:requests, : blocks * per].view(requests, blocks, per)
    # One kernel: arange(per) + per * block.
    offsets = torch.arange(per, device=out.device, dtype=out.dtype)
    torch.add(offsets, block_table[:, :, None], alpha=per, out=pages)
