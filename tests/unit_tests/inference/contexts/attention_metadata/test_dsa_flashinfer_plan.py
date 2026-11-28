# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Host-side step plan of the FlashInfer DSA backend (``dsa_flashinfer_plan``).

Three kinds of checks:

- worked examples whose arrays are spelled out;
- a slow per-request, per-row reference planner (``_reference_plan``) written from the plan's
  definition, against which the vectorized planner is compared on random batches;
- exact equality with the planner as first written (``dsa_flashinfer_plan_v0``), which pins the
  readability refactor to bit-identical output.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from megatron.core.inference.contexts.attention_context import dsa_flashinfer_plan as plan_lib
from megatron.core.inference.contexts.attention_context.dsa_flashinfer_plan import (
    FI_PAGE,
    PAGES,
    SAMPLES,
    TILE_ROWS,
    build_step_plan,
    check_plan_rows,
    fill_page_table,
    plan_sizes,
    union_group,
)

NUM_HEADS = 16
ARRAY_NAMES = (
    "decode_requests",
    "decode_lengths",
    "tile_requests",
    "tile_ordinals",
    "row_ends",
    "query_rows",
    "request_rows",
    "sample_ends",
    "batch_starts",
    "compact_rows",
    "compact_ends",
    "compact_ordinals",
    "tiled_compact",
)


def _load_v0():
    path = Path(__file__).with_name("dsa_flashinfer_plan_v0.py")
    spec = importlib.util.spec_from_file_location("dsa_flashinfer_plan_v0", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flashinfer_step_plan():
    """Leading single-token requests take the decode path; the rest fill 128-row tiles whose
    8-row groups are consecutive, gap-free union-core batches; sizes follow the padded batch."""
    # Two decode requests, a 300-token chunk, a single-token request after it (prefill path),
    # padded to 400 rows and 8 requests.
    plan = build_step_plan(
        [1, 1, 300, 1],
        [50, 9000, 1300, 77],
        padded_rows=400,
        padded_requests=8,
        decode_only=False,
        heads=NUM_HEADS,
    )
    sizes = plan_sizes(400, 8, False)
    assert sizes == {"decode_rows": 8, "tiles": 4 + 8, "prefill_requests": 8, "compact_rows": 400}
    assert plan["decode_lengths"].tolist() == [50, 9000] + [0] * 6
    assert plan["decode_requests"][:2].tolist() == [0, 1]
    assert len(plan["query_rows"]) == 12 * 128 and len(plan["batch_starts"]) == 12 * 16 + 1
    rows = plan["query_rows"][plan["query_rows"] >= 0]
    assert rows.tolist() == list(range(2, 303))
    # Request 2's rows end at keys 1001 .. 1300; request 3's single row at key 77.
    ends = plan["row_ends"][plan["query_rows"] >= 0]
    assert ends.tolist() == list(range(1001, 1301)) + [77]
    assert plan["tile_requests"][:4].tolist() == [2, 2, 2, 3]
    starts = plan["batch_starts"]
    assert (np.diff(starts) >= 0).all() and starts[0] == 2 and starts[-1] == 303
    assert plan["sample_ends"][:2].tolist() == [1300 // 128 * 3, 77 // 128 * 3]
    # The sample kernels' compact rows: the 301 live tiled rows in order, then padding that
    # every padding tiled row maps past.
    live = np.flatnonzero(plan["query_rows"] >= 0)
    assert plan["compact_rows"][:301].tolist() == live.tolist()
    assert plan["compact_ends"][:301].tolist() == ends.tolist()
    assert plan["compact_ordinals"][:301].tolist() == [0] * 300 + [1]
    assert (plan["compact_ends"][301:] == 0).all() and len(plan["compact_rows"]) == 400
    assert plan["tiled_compact"][live].tolist() == list(range(301))
    assert (np.delete(plan["tiled_compact"], live) == 400).all()
    decode_plan = build_step_plan(
        [1] * 3, [10, 20, 30], padded_rows=4, padded_requests=4, decode_only=True, heads=NUM_HEADS
    )
    assert decode_plan["decode_lengths"].tolist() == [10, 20, 30, 0]
    assert decode_plan["query_rows"].size == 0
    assert decode_plan["compact_rows"].size == 0


def test_flashinfer_step_plan_speculative_rows():
    """With speculative decoding a decode request adds up to 1 + k rows; each row takes the
    decode path with its own causal key count, and a longer request goes to the tiles."""
    plan = build_step_plan(
        [3, 2, 300, 3],
        [50, 9000, 1300, 77],
        padded_rows=400,
        padded_requests=8,
        decode_only=False,
        heads=NUM_HEADS,
        decode_tokens=3,
    )
    assert plan_sizes(400, 8, False, decode_tokens=3)["decode_rows"] == 24
    assert plan["decode_requests"][:5].tolist() == [0, 0, 0, 1, 1]
    assert plan["decode_lengths"].tolist() == [48, 49, 50, 8999, 9000] + [0] * 19
    # The 300-row chunk and the 3-row request after it fill tiles from step row 5.
    rows = plan["query_rows"][plan["query_rows"] >= 0]
    assert rows.tolist() == list(range(5, 308))
    ends = plan["row_ends"][plan["query_rows"] >= 0]
    assert ends.tolist() == list(range(1001, 1301)) + [75, 76, 77]
    decode_plan = build_step_plan(
        [3, 3],
        [10, 20],
        padded_rows=9,
        padded_requests=3,
        decode_only=True,
        heads=NUM_HEADS,
        decode_tokens=3,
    )
    assert decode_plan["decode_requests"][:6].tolist() == [0, 0, 0, 1, 1, 1]
    assert decode_plan["decode_lengths"].tolist() == [8, 9, 10, 18, 19, 20, 0, 0, 0]
    with pytest.raises(ValueError, match="at most 3 rows"):
        build_step_plan(
            [4],
            [10],
            padded_rows=9,
            padded_requests=3,
            decode_only=True,
            heads=NUM_HEADS,
            decode_tokens=3,
        )


def test_flashinfer_compact_rows_are_bounded_by_tokens():
    """Many small prefills tile far more rows than they hold; the per-row sample kernels run on
    at most the padded rows however many requests the batch admits."""
    lengths = [2] * 1024
    plan = build_step_plan(
        lengths,
        [100] * 1024,
        padded_rows=2048,
        padded_requests=1024,
        decode_only=False,
        heads=NUM_HEADS,
    )
    assert len(plan["query_rows"]) == (16 + 1024) * 128 > 65535
    assert len(plan["compact_rows"]) == 2048
    assert (plan["compact_ends"] > 0).sum() == 2048


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_fill_page_table():
    """The page table splits each KV block into FlashInfer's 64-key pages."""
    device = torch.device("cuda", torch.cuda.current_device())
    table = torch.tensor([[2, 0], [5, 1]], device=device, dtype=torch.int32)
    pages = torch.full((3, 10), -7, device=device, dtype=torch.int32)
    fill_page_table(table, 256, pages)
    assert pages[:2, :8].tolist() == [[8, 9, 10, 11, 0, 1, 2, 3], [20, 21, 22, 23, 4, 5, 6, 7]]
    assert (pages[:, 8:] == -7).all() and (pages[2] == -7).all()


@pytest.mark.parametrize(
    "max_tokens, ok", [(8192, True), (32768, True), (65534, True), (65535, False)]
)
def test_check_plan_rows(max_tokens, ok):
    """The top-k rows are bounded by the token budget alone, whatever the request budget."""
    if ok:
        check_plan_rows(max_tokens)
    else:
        with pytest.raises(ValueError, match="Lower max_tokens"):
            check_plan_rows(max_tokens)


def test_plan_rejects_more_prefill_requests_than_planned():
    """A step with more prefill requests than the padded prefill count is a caller error."""
    with pytest.raises(ValueError, match="take the prefill path"):
        build_step_plan(
            [5, 5, 5],
            [5, 5, 5],
            padded_rows=16,
            padded_requests=4,
            decode_only=False,
            heads=NUM_HEADS,
            padded_prefill_requests=2,
        )


def _reference_plan(
    query_lengths,
    kv_lengths,
    *,
    padded_rows,
    padded_requests,
    decode_only,
    heads,
    padded_prefill_requests=None,
    decode_tokens=1,
):
    """The plan built one request and one row at a time, from its definition."""
    sizes = plan_sizes(
        padded_rows, padded_requests, decode_only, padded_prefill_requests, decode_tokens
    )
    group = union_group(heads)
    plan = {
        "decode_requests": [0] * sizes["decode_rows"],
        "decode_lengths": [0] * sizes["decode_rows"],
        "tile_requests": [0] * sizes["tiles"],
        "tile_ordinals": [0] * sizes["tiles"],
        "row_ends": [0] * (sizes["tiles"] * TILE_ROWS),
        "query_rows": [-1] * (sizes["tiles"] * TILE_ROWS),
        "request_rows": [0] * sizes["prefill_requests"],
        "sample_ends": [0] * sizes["prefill_requests"],
    }
    # 1. Decode rows: leading requests of at most `decode_tokens` rows whose rows are all among
    # the decode rows; row j of a request adding q rows sees its first kv_len - q + j + 1 keys.
    prefill, first_row, leading = [], 0, True
    for request, (q_len, kv_len) in enumerate(zip(query_lengths, kv_lengths)):
        leading = leading and q_len <= decode_tokens
        if leading and first_row + q_len <= sizes["decode_rows"]:
            for j in range(q_len):
                plan["decode_requests"][first_row + j] = request
                plan["decode_lengths"][first_row + j] = kv_len - q_len + j + 1
        else:
            prefill.append((request, first_row, q_len, kv_len))
        first_row += q_len
    # 2. Prefill tiles: TILE_ROWS rows of one request each, in request order.
    tile = 0
    for ordinal, (request, first_row, q_len, kv_len) in enumerate(prefill):
        plan["request_rows"][ordinal] = request
        plan["sample_ends"][ordinal] = kv_len // (PAGES * FI_PAGE) * SAMPLES
        for start in range(0, q_len, TILE_ROWS):
            plan["tile_requests"][tile] = request
            plan["tile_ordinals"][tile] = ordinal
            for lane in range(min(TILE_ROWS, q_len - start)):
                row = start + lane  # the request's own row
                plan["query_rows"][tile * TILE_ROWS + lane] = first_row + row
                plan["row_ends"][tile * TILE_ROWS + lane] = kv_len - q_len + row + 1
            tile += 1
    # 3. Union-core batches: every `group` tiled rows; an empty batch sits where the last live
    # batch before it ended (or at the first live row).
    live_rows = [row for row in plan["query_rows"] if row >= 0]
    end = live_rows[0] if live_rows else 0
    batch_starts = []
    for g in range(0, len(plan["query_rows"]), group):
        rows = [row for row in plan["query_rows"][g : g + group] if row >= 0]
        if rows:
            batch_starts.append(rows[0])
            end = rows[0] + len(rows)
        else:
            batch_starts.append(end)
    batch_starts.append(end)
    plan["batch_starts"] = batch_starts
    # 4. Compact rows: the live tiled rows in order; padding tiled rows map past the last.
    compact = sizes["compact_rows"]
    plan["compact_rows"] = [0] * compact
    plan["compact_ends"] = [0] * compact
    plan["compact_ordinals"] = [0] * compact
    plan["tiled_compact"] = [compact] * len(plan["query_rows"])
    index = 0
    for tiled, row in enumerate(plan["query_rows"]):
        if row >= 0:
            plan["compact_rows"][index] = tiled
            plan["compact_ends"][index] = plan["row_ends"][tiled]
            plan["compact_ordinals"][index] = plan["tile_ordinals"][tiled // TILE_ROWS]
            plan["tiled_compact"][tiled] = index
            index += 1
    return {name: np.asarray(values, dtype=np.int32) for name, values in plan.items()}


def _random_batch(rng, max_decode_tokens=1):
    """Random step arguments: decode-only or mixed batches, with lengths around tile edges.

    With ``max_decode_tokens > 1`` some batches are speculative-decoding steps, whose decode
    requests add up to ``decode_tokens`` rows each."""
    heads = int(rng.choice([8, 16, 32, 64, 128]))
    decode_tokens = int(rng.integers(1, max_decode_tokens + 1))
    # Only pass `decode_tokens` when it is not the default, so the batches also suit planners
    # without it.
    extra = {"decode_tokens": decode_tokens} if decode_tokens > 1 else {}
    if rng.random() < 0.25:
        n = int(rng.integers(0, 65))
        q_len = [int(q) for q in rng.integers(1, decode_tokens + 1, n)]
        kv_len = [q + int(k) for q, k in zip(q_len, rng.integers(0, 20000, n))]
        padded_requests = max(n + int(rng.integers(0, 8)), 1)
        return (
            q_len,
            kv_len,
            dict(
                padded_rows=padded_requests * decode_tokens,
                padded_requests=padded_requests,
                decode_only=True,
                heads=heads,
                **extra,
            ),
        )
    num_decode = int(rng.integers(0, 33))
    num_prefill = int(rng.integers(0, 17))
    edge = [1, 2, TILE_ROWS - 1, TILE_ROWS, TILE_ROWS + 1, 2 * TILE_ROWS]
    prefill_q = [
        int(rng.choice(edge)) if rng.random() < 0.5 else int(rng.integers(1, 700))
        for _ in range(num_prefill)
    ]
    q_len = [int(q) for q in rng.integers(1, decode_tokens + 1, num_decode)] + prefill_q
    kv_len = [q + int(rng.integers(0, 20000)) for q in q_len]
    tokens = sum(q_len)
    padded_rows = max(tokens + int(rng.integers(0, 300)), 1)
    padded_requests = len(q_len) + int(rng.integers(0, 16))
    # Requests past the decode rows (or after the first one longer than `decode_tokens`) take
    # the prefill path.
    decode_rows = min(padded_rows, padded_requests * decode_tokens)
    q = np.asarray(q_len, dtype=np.int64)
    first = np.cumsum(q) - q
    leading = np.cumprod(q <= decode_tokens).astype(bool)
    prefill_count = int((~(leading & (first + q <= decode_rows))).sum())
    padded_prefill = None
    if rng.random() < 0.5:
        padded_prefill = prefill_count + int(rng.integers(0, 4))
    return (
        q_len,
        kv_len,
        dict(
            padded_rows=padded_rows,
            padded_requests=max(padded_requests, 1),
            decode_only=False,
            heads=heads,
            padded_prefill_requests=padded_prefill,
            **extra,
        ),
    )


def _edge_batches():
    """Explicit edge cases: empty, all-decode with padding, one exact tile, a full budget."""
    yield [], [], dict(padded_rows=8, padded_requests=8, decode_only=False, heads=16)
    yield [], [], dict(padded_rows=8, padded_requests=8, decode_only=True, heads=16)
    yield [1] * 5, [7, 8, 9, 10, 11], dict(
        padded_rows=8, padded_requests=8, decode_only=True, heads=32
    )
    yield [TILE_ROWS], [TILE_ROWS], dict(
        padded_rows=TILE_ROWS, padded_requests=1, decode_only=False, heads=16
    )
    yield [1, TILE_ROWS, 1], [5, 300, 9], dict(
        padded_rows=TILE_ROWS + 2, padded_requests=3, decode_only=False, heads=128
    )
    yield [2] * 4096, [100] * 4096, dict(
        padded_rows=8192, padded_requests=4096, decode_only=False, heads=16
    )
    yield [1] * 200 + [8192 - 200], [50] * 200 + [9000], dict(
        padded_rows=8192, padded_requests=256, decode_only=False, heads=8
    )


def _assert_plans_equal(actual, expected, context):
    assert list(actual) == list(ARRAY_NAMES), context
    for name in ARRAY_NAMES:
        assert actual[name].dtype == expected[name].dtype == np.int32, (name, context)
        assert np.array_equal(actual[name], expected[name]), (name, context)


@pytest.mark.parametrize("seed", range(4))
def test_build_step_plan_matches_reference(seed):
    """The vectorized planner equals the per-row reference on random batches, speculative
    decoding steps (decode requests of up to 1 + k rows) included."""
    rng = np.random.default_rng(1000 + seed)
    for _ in range(250):
        q_len, kv_len, kwargs = _random_batch(rng, max_decode_tokens=4)
        _assert_plans_equal(
            build_step_plan(q_len, kv_len, **kwargs),
            _reference_plan(q_len, kv_len, **kwargs),
            (q_len, kv_len, kwargs),
        )
    for q_len, kv_len, kwargs in _edge_batches():
        _assert_plans_equal(
            build_step_plan(q_len, kv_len, **kwargs),
            _reference_plan(q_len, kv_len, **kwargs),
            kwargs,
        )


def test_build_step_plan_matches_v0():
    """Bit-identical to the planner as first written, on 20k random batches and edge cases."""
    v0 = _load_v0()
    rng = np.random.default_rng(20261120)
    cases = [_random_batch(rng) for _ in range(20000)] + list(_edge_batches())
    for q_len, kv_len, kwargs in cases:
        assert plan_sizes(
            kwargs["padded_rows"],
            kwargs["padded_requests"],
            kwargs["decode_only"],
            kwargs.get("padded_prefill_requests"),
        ) == v0.plan_sizes(
            kwargs["padded_rows"],
            kwargs["padded_requests"],
            kwargs["decode_only"],
            kwargs.get("padded_prefill_requests"),
        )
        _assert_plans_equal(
            build_step_plan(q_len, kv_len, **kwargs),
            v0.build_step_plan(q_len, kv_len, **kwargs),
            (q_len, kv_len, kwargs),
        )


def test_union_group():
    """Union-core groups hold the most tokens whose heads fit a 128-row FA4 tile."""
    assert [union_group(h) for h in (8, 16, 24, 32, 64, 96, 128)] == [16, 8, 4, 4, 2, 1, 1]


def test_page_table_columns():
    """The page table is wide enough for the blocks, the score width and the key tiles."""
    assert plan_lib.page_table_columns(36, 256, 9216, 512) == max(36 * 4, 9216 // 64, 2 * 72)
    assert plan_lib.page_table_columns(1, 64, 100, 2048) == 2048 // 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_metadata_holds_the_step_plan_on_the_gpu():
    """``FlashInferDSAMetadata`` stages the host plan in its static buffers, copies it to the GPU
    and derives the page table; ``plan()`` views exactly the step's arrays."""
    from megatron.core.inference.contexts.attention_context.dsa_flashinfer_metadata import (
        FlashInferDSAMetadata,
    )

    metadata = FlashInferDSAMetadata(
        max_requests=8,
        max_tokens=512,
        max_kv_blocks=40,
        block_size_tokens=256,
        max_sequence_length=9216,
        query_heads=NUM_HEADS,
        index_topk=512,
    )
    query_lengths, kv_lengths = [1, 1, 300, 1], [50, 9000, 1300, 77]
    metadata.update(
        query_lengths, kv_lengths, padded_tokens=400, padded_requests=8, decode_only=False
    )
    block_table = torch.arange(8 * 40, dtype=torch.int32, device="cuda").view(8, 40)
    metadata.transfer(block_table, non_blocking=False)
    plan = metadata.plan()
    expected = build_step_plan(
        query_lengths,
        kv_lengths,
        padded_rows=400,
        padded_requests=8,
        decode_only=False,
        heads=NUM_HEADS,
    )
    for name, values in expected.items():
        assert getattr(plan, name).tolist() == values.tolist(), name
    sizes = plan_sizes(400, 8, False)
    assert (plan.decode_rows, plan.num_tiles) == (sizes["decode_rows"], sizes["tiles"])
    assert plan.max_context == 9216
    pages = torch.zeros_like(metadata.page_table)
    fill_page_table(block_table, 256, pages)
    assert torch.equal(plan.page_table, pages[:8])
    metadata.reset()
    assert metadata.plan().num_tiles == 0 and metadata.plan().row_ends.numel() == 0
