# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts of the indexer top-k tile planner and its tuning policy seam.

The expected plans are those of the previous LiteTopK integration, an earlier out-of-tree
integration into a Megatron-LM fork (its FP8 wave tiles are restated below as an oracle), and
the per-rank context-parallel plans derived from them.
"""

from __future__ import annotations

import dataclasses
from collections import Counter

import pytest

from megatron.lite.primitive.kernels.indexer_topk import (
    IndexerGeometry,
    IndexerTopKConfigError,
    IndexerTopKTuning,
    LiteTopKPluginSettings,
    QueryLayout,
    plan_score_rows,
    resolve_indexer_topk_tuning,
)
from megatron.lite.primitive.kernels.indexer_topk.planner import Tile, plan_segment
from megatron.lite.primitive.kernels.indexer_topk.plugins.abi import RouteCapability

pytestmark = pytest.mark.mlite

B200_SMS = 148
VOTE_ROWS = 1536
FP8_ROUTE = RouteCapability(
    name="fp8_paged",
    fmt="fp8",
    heads=frozenset({32}),
    head_dims=frozenset({128}),
    topk=frozenset({2048}),
    max_topk=2048,
    qualified_query_lengths=frozenset({1016, 1024, 2040, 2048, 4088, 4096}),
    admitted_max_query_len=1776,
    min_keys=196608,
    max_keys=1048576,
    hot_prefix=12288,
    exact=True,
    tie_policies=frozenset({"logical-id", "storage"}),
    score_policies=frozenset({"folded", "native-fp32"}),
)
FP8_GEOMETRY = IndexerGeometry(num_heads=32, head_dim=128, topk=2048)


def _fp8_tuning(**overrides):
    return resolve_indexer_topk_tuning(
        IndexerTopKTuning(**overrides),
        fmt="fp8",
        route=FP8_ROUTE,
        geometry=FP8_GEOMETRY,
        precision="exact",
        num_sms=B200_SMS,
    )


def _plan(layout, route, tuning, topk):
    return [
        plan_segment(segment, route=route, tuning=tuning, topk=topk, vote_rows=VOTE_ROWS)
        for segment in layout.segments
    ]


def _wave_tile_slices(sequence_length: int) -> list[tuple[int, int]]:
    """The FP8 wave tiles of the previous integration: Q1776 tiles from query 188416."""
    return [
        (first, min(first + 1776, sequence_length))
        for first in range(188416, sequence_length, 1776)
    ]


def _score_chunk_rows(keys: int, *, num_sms: int) -> int:
    """The FP8 reference score chunk rule of the previous integration (2 GiB, BLOCK_Q 4)."""
    maximum = max(4, ((2**31 // (4 * keys)) // 4) * 4)
    wave_rows = 4 * num_sms
    return maximum // wave_rows * wave_rows if maximum >= wave_rows else maximum


@pytest.mark.parametrize(
    ("sequence", "tiles", "tail", "groups"),
    [
        (262144, 42, 912, 7),
        (524288, 190, 208, 25),
        (786432, 337, 1280, 43),
        (1048576, 485, 576, 62),
    ],
)
def test_fp8_wave_tables(sequence, tiles, tail, groups):
    (plan,) = _plan(QueryLayout.full(sequence, keys=sequence), FP8_ROUTE, _fp8_tuning(), 2048)
    assert [(tile.row_start, tile.row_end) for tile in plan.tiles] == _wave_tile_slices(sequence)
    assert len(plan.tiles) == tiles and plan.tiles[-1].rows == tail
    assert all(tile.position == tile.row_start for tile in plan.tiles)
    assert len(plan.groups) == groups
    full = tiles - 1  # groups of up to eight equal tiles; the shorter last tile stands alone
    assert [len(group.tiles) for group in plan.groups] == [8] * (full // 8) + (
        [full % 8] if full % 8 else []
    ) + [1]
    assert all(group.common_end == group.tiles[0].row_start + 1 for group in plan.groups)
    assert plan.reference_rows == ((0, 188416),)
    assert plan.seed == "reference" and plan.bootstrap_tile is None
    assert plan.vote_rows == (188416 - VOTE_ROWS, 188416) and plan.vote_extent == 188416
    assert plan.litetopk_rows == sequence - 188416
    assert plan.litetopk_pairs == sum(range(188417, sequence + 1))
    if sequence == 262144:
        assert 21 * len(plan.tiles) == 882  # 21 full indexer layers per forward


def _global(plan, rank_start):
    return [(start + rank_start, end + rank_start) for start, end in plan.reference_rows]


def test_cp8_segment_plans():
    fp8 = _fp8_tuning()

    # GLM-style FP8, CP8 at 256K (32768 rows per rank).
    plans = []
    for rank in range(8):
        layout = QueryLayout.contiguous(32768, position=rank * 32768, keys=262144)
        (plan,) = _plan(layout, FP8_ROUTE, fp8, 2048)
        plans.append(plan)
    assert all(not plan.tiles and plan.reason for plan in plans[:5])
    assert _global(plans[5], 5 * 32768) == [(163840, 188416)]
    assert plans[5].vote_rows == (24576 - VOTE_ROWS, 24576) and plans[5].bootstrap_tile is None
    assert Counter(tile.rows for tile in plans[5].tiles) == {1776: 4, 1088: 1}
    for rank in (6, 7):
        assert plans[rank].bootstrap_tile == Tile(0, 1776, rank * 32768)
        assert plans[rank].reference_rows == ((0, 1776),)
        assert plans[rank].vote_rows == (1776 - VOTE_ROWS, 1776)
        assert Counter(tile.rows for tile in plans[rank].tiles) == {1776: 17, 800: 1}

    # GLM-style FP8, CP8 at 1M (131072 rows per rank).
    plans = []
    for rank in range(8):
        layout = QueryLayout.contiguous(131072, position=rank * 131072, keys=1048576)
        (plan,) = _plan(layout, FP8_ROUTE, fp8, 2048)
        plans.append(plan)
    assert not plans[0].tiles and plans[0].reference_rows == ((0, 131072),)
    assert _global(plans[1], 131072) == [(131072, 188416)]
    assert Counter(tile.rows for tile in plans[1].tiles) == {1776: 41, 912: 1}
    for plan in plans[2:]:
        assert plan.bootstrap_tile is not None and plan.reference_rows == ((0, 1776),)
        assert Counter(tile.rows for tile in plan.tiles) == {1776: 72, 1424: 1}


def test_bootstrap_choice():
    fp8 = _fp8_tuning(startup_position=0, tile_rows=512)
    layout = QueryLayout.contiguous(8192, position=200000, keys=262144)
    # No preceding rows: the first tile is selected by the reference selector and votes.
    (plan,) = _plan(layout, FP8_ROUTE, fp8, 2048)
    assert plan.bootstrap_tile == Tile(0, 512, 200000)
    assert plan.vote_rows == (0, 512) and plan.vote_extent == 200512
    assert plan.reference_rows == ((0, 512),) and plan.tiles[0].row_start == 512

    # Enough preceding reference rows: they vote, every tile runs LiteTopK.
    fp8 = _fp8_tuning(startup_position=202000, tile_rows=512)
    (plan,) = _plan(layout, FP8_ROUTE, fp8, 2048)
    assert plan.bootstrap_tile is None and plan.reference_rows == ((0, 2000),)
    assert plan.vote_rows == (2000 - VOTE_ROWS, 2000) and plan.vote_extent == 202000

    # Fewer preceding rows than the vote: bootstrap tile after them.
    fp8 = _fp8_tuning(startup_position=201000, tile_rows=512)
    (plan,) = _plan(layout, FP8_ROUTE, fp8, 2048)
    assert plan.bootstrap_tile == Tile(1000, 1512, 201000)
    assert plan.reference_rows == ((0, 1512),) and plan.vote_rows == (1000, 1512)

    # A remainder of rows that is not a multiple of four goes to the reference selector first.
    fp8 = _fp8_tuning(startup_position=201001, tile_rows=512)
    (plan,) = _plan(layout, FP8_ROUTE, fp8, 2048)
    assert plan.bootstrap_tile.row_start == 1004 and plan.reference_rows[0] == (0, 1516)
    assert all(tile.rows % 4 == 0 for tile in plan.tiles)

    # Preceding rows that do not see the HOT prefix cannot vote a seed: the first tile does.
    early = QueryLayout.full(12287 + 4096, keys=262144)
    (plan,) = _plan(early, FP8_ROUTE, _fp8_tuning(startup_position=0, tile_rows=512), 2048)
    assert plan.bootstrap_tile == Tile(12287, 12799, 12287)  # row 12286 sees 12287 keys
    assert plan.vote_rows == (12287, 12799) and plan.vote_extent == 12799
    assert plan.reference_rows == ((0, 12799),) and plan.tiles[0].row_start == 12799
    (plan,) = _plan(early, FP8_ROUTE, _fp8_tuning(startup_position=12288, tile_rows=512), 2048)
    assert plan.bootstrap_tile is None and plan.vote_rows == (12291 - VOTE_ROWS, 12291)
    assert plan.vote_extent == 12291 and plan.tiles[0].row_start == 12291

    # Identity seeds need no reference rows; the only tile of a segment cannot bootstrap itself.
    (plan,) = _plan(layout, FP8_ROUTE, _fp8_tuning(seed_bootstrap="identity"), 2048)
    assert plan.seed == "identity" and plan.vote_rows is None and plan.bootstrap_tile is None
    short = QueryLayout.contiguous(1776, position=200000, keys=262144)
    (plan,) = _plan(short, FP8_ROUTE, _fp8_tuning(), 2048)
    assert not plan.tiles and plan.reason == "the only tile bootstraps the seed"


def test_crossover_threshold():
    layout = QueryLayout.contiguous(32768, position=5 * 32768, keys=262144)
    (plan,) = _plan(layout, FP8_ROUTE, _fp8_tuning(), 2048)
    pairs = plan.litetopk_pairs
    assert pairs == sum(range(188417, 196609))
    (plan,) = _plan(layout, FP8_ROUTE, _fp8_tuning(min_litetopk_pairs=pairs), 2048)
    assert plan.tiles
    (plan,) = _plan(layout, FP8_ROUTE, _fp8_tuning(min_litetopk_pairs=pairs + 1), 2048)
    assert not plan.tiles and plan.reason == "below the LiteTopK crossover"
    assert plan.reference_rows == ((0, 32768),)
    # A reference bootstrap tile does not count as LiteTopK work.
    layout = QueryLayout.contiguous(32768, position=6 * 32768, keys=262144)
    (plan,) = _plan(layout, FP8_ROUTE, _fp8_tuning(), 2048)
    assert plan.litetopk_pairs == sum(range(6 * 32768 + 1777, 7 * 32768 + 1))
    # The pairs are the visible keys summed over the tile rows, for any shard start and with the
    # key count as the cap.
    for position, keys in ((188416, 196608), (190001, 196608), (250002, 1 << 20)):
        layout = QueryLayout.contiguous(12286, position=position, keys=keys)
        (plan,) = _plan(layout, FP8_ROUTE, _fp8_tuning(startup_position=0), 2048)
        (segment,) = layout.segments
        assert plan.tiles and plan.litetopk_pairs == sum(
            layout.visible_keys(segment, row)
            for tile in plan.tiles
            for row in range(tile.row_start, tile.row_end)
        )


def test_segments_without_litetopk():
    fp8 = _fp8_tuning()
    full = QueryLayout.full(262144, keys=262144)
    cases = {
        "no LiteTopK route": dict(route=None),
        "top-k not supported by the route": dict(topk=512),
        "fewer keys than the route minimum": dict(layout=QueryLayout.full(131072, keys=131072)),
        "more keys than the route maximum": dict(
            layout=QueryLayout.full(262144, keys=(1 << 20) + 4)
        ),
        "no row after the startup position sees the HOT prefix": dict(
            layout=QueryLayout.contiguous(1000, position=0, keys=262144)
        ),
        "no tile of an admitted length": dict(
            route=dataclasses.replace(FP8_ROUTE, admitted_max_query_len=0)
        ),
    }
    for reason, case in cases.items():
        layout = case.get("layout", full)
        (plan,) = [
            plan_segment(
                segment,
                route=case.get("route", FP8_ROUTE),
                tuning=fp8,
                topk=case.get("topk", 2048),
                vote_rows=VOTE_ROWS,
            )
            for segment in layout.segments
        ]
        assert plan.reason == reason and not plan.groups
        assert plan.reference_rows == ((0, layout.rows),)
    # Packed sequences are planned independently; short ones stay with the reference selector.
    packed = QueryLayout.packed([0, 1000, 263144], row_start=0, rows=263144, absolute_ids=True)
    short, long = _plan(packed, FP8_ROUTE, fp8, 2048)
    assert short.reason and not short.tiles
    assert long.reference_rows == ((1000, 189416),) and len(long.tiles) == 42


def test_plan_score_rows_matches_a():
    budget = 2 << 30
    assert plan_score_rows(1048576, kernel_heads=32, num_sms=B200_SMS, budget_bytes=budget) == 512
    assert plan_score_rows(786432, kernel_heads=32, num_sms=B200_SMS, budget_bytes=budget) == 592
    assert plan_score_rows(188416, kernel_heads=32, num_sms=B200_SMS, budget_bytes=budget) == 2368
    assert plan_score_rows(262144, kernel_heads=32, num_sms=B200_SMS, budget_bytes=budget) == 1776
    for keys in list(range(1, 5000, 37)) + list(range(100000, 1 << 20, 9973)) + [1 << 20]:
        assert plan_score_rows(
            keys, kernel_heads=32, num_sms=B200_SMS, budget_bytes=budget
        ) == _score_chunk_rows(keys, num_sms=B200_SMS)
    # 64 heads: two rows per SM wave; below one wave, multiples of four and at least four.
    assert plan_score_rows(65536, kernel_heads=64, num_sms=B200_SMS, budget_bytes=1 << 30) == 3848
    assert plan_score_rows(1 << 20, kernel_heads=64, num_sms=B200_SMS, budget_bytes=1 << 30) == 256
    assert plan_score_rows(1 << 20, kernel_heads=32, num_sms=B200_SMS, budget_bytes=16) == 4
    with pytest.raises(ValueError, match="keys"):
        plan_score_rows(0, kernel_heads=32, num_sms=B200_SMS, budget_bytes=budget)


def test_seam_defaults():
    fp8 = _fp8_tuning()
    assert (fp8.tile_rows, fp8.startup_position, fp8.group_tiles, fp8.seed_bootstrap) == (
        1776,
        188416,
        8,
        "reference",
    )
    assert (fp8.required, fp8.min_litetopk_pairs, fp8.reference_rows_per_call) == (False, 0, None)
    assert fp8.reference_budget_bytes == 2 << 30
    assert (fp8.index_order, fp8.status_check) == ("ascending", "sync_recompute")
    # The FP8 plugin settings the paged route was measured fastest with (tiered 12K seed; the
    # pool size does not change speed, 13 pages leave ample headroom), for both precisions.
    assert fp8.plugin_settings == LiteTopKPluginSettings(
        tie_policy="logical-id",
        score_policy="native-fp32",
        paged_pool_pages_per_row=13,
        fp8_row_tiles=2,
        fp8_paged_admit_max_query_len=1776,
        tiered_seed_12k=True,
        coldstart_identity=True,
    )
    fast = resolve_indexer_topk_tuning(
        None, fmt="fp8", route=FP8_ROUTE, geometry=FP8_GEOMETRY, precision="fast", num_sms=132
    )
    assert fast.tile_rows == 3 * 132 * 4 and fast.plugin_settings.tie_policy is None
    assert fast.plugin_settings.fp8_paged_admit_max_query_len == 1584
    assert fast.plugin_settings == dataclasses.replace(
        fp8.plugin_settings, tie_policy=None, score_policy=None, fp8_paged_admit_max_query_len=1584
    )
    sixteen = resolve_indexer_topk_tuning(
        None,
        fmt="fp8",
        route=None,
        geometry=IndexerGeometry(16, 128, 2048),
        precision="fast",
        num_sms=B200_SMS,
    )
    assert sixteen.tile_rows == 3 * B200_SMS * 8 and sixteen.startup_position == 0

    # Overrides replace single values; plugin settings merge field by field.
    tuned = _fp8_tuning(
        tile_rows=2048,
        group_tiles=4,
        required=True,
        reference_rows_per_call=4096,
        plugin_settings=LiteTopKPluginSettings(paged_pool_pages_per_row=13, tiered_seed_12k=True),
    )
    assert (tuned.tile_rows, tuned.group_tiles, tuned.required) == (2048, 4, True)
    assert tuned.reference_rows_per_call == 4096
    assert tuned.plugin_settings == dataclasses.replace(
        fp8.plugin_settings,
        paged_pool_pages_per_row=13,
        tiered_seed_12k=True,
        fp8_paged_admit_max_query_len=2048,
    )
    assert tuned.as_dict()["plugin_settings"]["paged_pool_pages_per_row"] == 13


def test_tuning_validation():
    for field, value, match in (
        ("tile_rows", 1778, "multiple of 4"),
        ("tile_rows", 0, "tile_rows"),
        ("seed_bootstrap", "zeros", "seed_bootstrap"),
        ("reference_budget_bytes", 0, "reference_budget_bytes"),
        ("required", 1, "required"),
        ("plugin_settings", {}, "plugin_settings"),
    ):
        with pytest.raises(IndexerTopKConfigError, match=match):
            IndexerTopKTuning(**{field: value})
    with pytest.raises(IndexerTopKConfigError, match="tie_policy='logical-id'"):
        _fp8_tuning(plugin_settings=LiteTopKPluginSettings(tie_policy="logical-id-desc"))
    with pytest.raises(IndexerTopKConfigError, match="fmt must be one of"):
        resolve_indexer_topk_tuning(
            None, fmt="bf16", route=None, geometry=FP8_GEOMETRY, precision="fast", num_sms=148
        )
    with pytest.raises(IndexerTopKConfigError, match="serves bf16 operands"):
        resolve_indexer_topk_tuning(
            None,
            fmt="fp8",
            route=dataclasses.replace(FP8_ROUTE, fmt="bf16"),
            geometry=FP8_GEOMETRY,
            precision="fast",
            num_sms=B200_SMS,
        )
    with pytest.raises(IndexerTopKConfigError, match="precision"):
        resolve_indexer_topk_tuning(
            None, fmt="fp8", route=None, geometry=FP8_GEOMETRY, precision="auto", num_sms=148
        )
    with pytest.raises(IndexerTopKConfigError, match="concrete values"):
        dataclasses.replace(_fp8_tuning(), tile_rows=None)
