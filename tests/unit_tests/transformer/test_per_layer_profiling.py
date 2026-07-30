# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the pure-logic (CPU, no CUDA) parts of per-layer profiling.

Covers timing aggregation (``_agg``, ``_LayerTiming``, ``summarize_timing``), the
backward boundary pairing (``PerLayerProfiler._on_bwd_boundary``), the allocator-
trace analysis (``_allocated_curve``, ``_addr_lifetimes``, ``_reserved_overhead``,
``summarize_trace_step``, ``summarize_trace``, ``build_trace_windows``), and the
report builder / renderer (``build_per_layer_report``, ``_render_per_layer_report``,
``log_per_layer_resource_usage``).

The CUDA-dependent paths (deferred events, ``_record_memory_history`` collection,
hook attach/detach, flush) are exercised by GPU tests; ``_on_bwd_boundary`` is
tested with sentinel string "events" so its pairing logic runs on CPU.
"""

import pytest

from megatron.core.transformer.per_layer_profiling import (
    PerLayerProfiler,
    _addr_lifetimes,
    _agg,
    _allocated_curve,
    _LayerTiming,
    _MemTraceGlobalAccumulator,
    _mib,
    _reserved_overhead,
    _TraceWindow,
    build_per_layer_report,
    build_trace_windows,
    log_per_layer_resource_usage,
    merge_across_ranks,
    summarize_timing,
    summarize_trace,
    summarize_trace_step,
)


# ---------------------------------------------------------------------------
# _agg
# ---------------------------------------------------------------------------
class TestAgg:
    def test_empty_returns_zeros(self):
        assert _agg([]) == (0.0, 0.0)

    def test_mean_and_max(self):
        mean, mx = _agg([1.0, 2.0, 3.0, 4.0])
        assert mean == pytest.approx(2.5)
        assert mx == 4.0


# ---------------------------------------------------------------------------
# _LayerTiming
# ---------------------------------------------------------------------------
class TestLayerTiming:
    def test_record_appends_by_phase(self):
        s = _LayerTiming(layer_idx=0)
        s.record("fwd", 1.0)
        s.record("fwd", 3.0)
        s.record("bwd", 5.0)
        assert s.time_ms["fwd"] == [1.0, 3.0]
        assert s.time_ms["bwd"] == [5.0]
        # num_samples counts forward passes only.
        assert s.num_samples == 2

    def test_reset_clears_everything(self):
        s = _LayerTiming(layer_idx=0)
        s.record("fwd", 1.0)
        s.record("bwd", 2.0)
        s.reset()
        assert s.time_ms == {"fwd": [], "bwd": []}
        assert s.num_samples == 0

    def test_moe_tag_preserved(self):
        s = _LayerTiming(layer_idx=7, is_moe_layer=True)
        assert s.is_moe_layer is True
        assert s.layer_idx == 7


# ---------------------------------------------------------------------------
# summarize_timing
# ---------------------------------------------------------------------------
def _make_profiler_with_samples() -> PerLayerProfiler:
    """Build a profiler and inject timing directly, bypassing all CUDA."""
    prof = PerLayerProfiler(enabled=False)
    s0 = _LayerTiming(layer_idx=0, is_moe_layer=False)
    s0.record("fwd", 2.0)
    s0.record("fwd", 4.0)  # fwd mean 3.0 / max 4.0
    s0.record("bwd", 6.0)
    s0.record("bwd", 8.0)  # bwd mean 7.0 / max 8.0
    s1 = _LayerTiming(layer_idx=1, is_moe_layer=True)
    s1.record("fwd", 10.0)
    s1.record("bwd", 20.0)
    prof._stats = {0: s0, 1: s1}
    prof._global_peak_bytes = [70_000, 80_000]  # mean 75000 / max 80000
    prof._global_reserved_peak_bytes = [100_000, 120_000]  # mean 110000 / max 120000
    return prof


class TestSummarizeTiming:
    def test_global_peaks(self):
        out = summarize_timing(_make_profiler_with_samples())
        assert out["global_peak_bytes"] == {"mean": pytest.approx(75_000.0), "max": 80_000.0}
        assert out["global_reserved_peak_bytes"]["max"] == 120_000.0

    def test_layer_timing(self):
        l0 = summarize_timing(_make_profiler_with_samples())["layers"][0]
        assert l0["is_moe"] is False
        assert l0["num_samples"] == 2
        assert l0["fwd_time_ms"] == {"mean": pytest.approx(3.0), "max": 4.0}
        assert l0["bwd_num_samples"] == 2
        assert l0["bwd_time_ms"] == {"mean": pytest.approx(7.0), "max": 8.0}

    def test_moe_flag_and_offset(self):
        out = summarize_timing(_make_profiler_with_samples(), layer_offset=16)
        assert set(out["layers"].keys()) == {16, 17}
        assert out["layers"][17]["is_moe"] is True

    def test_empty_profiler(self):
        out = summarize_timing(PerLayerProfiler(enabled=False))
        assert out["layers"] == {}
        assert out["global_peak_bytes"] == {"mean": 0.0, "max": 0.0}


# ---------------------------------------------------------------------------
# PerLayerProfiler._on_bwd_boundary  (pure pairing; sentinel string "events")
# ---------------------------------------------------------------------------
def _empty_profiler_with_layers(n: int) -> PerLayerProfiler:
    prof = PerLayerProfiler(enabled=False)
    prof._stats = {i: _LayerTiming(layer_idx=i) for i in range(n)}
    return prof


class TestOnBwdBoundary:
    def test_pairs_adjacent_and_covers_first_layer(self):
        prof = _empty_profiler_with_layers(3)
        for lid in (2, 1, 0, -1):  # decreasing, then block-input (-1) closes layer 0
            prof._on_bwd_boundary(lid, f"evt{lid}")
        assert len(prof._stats[2]._pending["bwd"]) == 1
        assert len(prof._stats[1]._pending["bwd"]) == 1
        assert len(prof._stats[0]._pending["bwd"]) == 1

    def test_interval_orders_start_before_end(self):
        prof = _empty_profiler_with_layers(2)
        for lid in (1, 0, -1):
            prof._on_bwd_boundary(lid, f"evt{lid}")
        assert prof._stats[1]._pending["bwd"][0] == ("evt1", "evt0")

    def test_no_cross_microbatch_pairing(self):
        prof = _empty_profiler_with_layers(2)
        for _ in range(2):
            for lid in (1, 0, -1):
                prof._on_bwd_boundary(lid, "e")
        assert len(prof._stats[1]._pending["bwd"]) == 2
        assert len(prof._stats[0]._pending["bwd"]) == 2


# ---------------------------------------------------------------------------
# _mib
# ---------------------------------------------------------------------------
class TestMib:
    def test_one_mib(self):
        assert _mib(1024 * 1024) == pytest.approx(1.0)

    def test_zero(self):
        assert _mib(0) == 0.0


# ---------------------------------------------------------------------------
# Allocator-trace analysis (pure; synthetic event dicts)
# ---------------------------------------------------------------------------
class TestAllocatedCurve:
    def test_anchor_and_alloc_free(self):
        events = [
            {"action": "alloc", "addr": 1, "size": 100},
            {"action": "alloc", "addr": 2, "size": 30},
            {"action": "free_requested", "addr": 2, "size": 30},
        ]
        assert _allocated_curve(events, base=1000) == [1100, 1130, 1100]

    def test_free_completed_is_ignored(self):
        events = [
            {"action": "alloc", "addr": 1, "size": 100},
            {"action": "free_completed", "addr": 1, "size": 100},
        ]
        assert _allocated_curve(events, base=0) == [100, 100]

    def test_missing_size_defensive(self):
        events = [{"action": "alloc", "addr": 1}, {"action": "segment_alloc"}]
        assert _allocated_curve(events, base=5) == [5, 5]


class TestAddrLifetimes:
    def test_matches_alloc_to_free(self):
        events = [
            {"action": "alloc", "addr": 1, "size": 10},
            {"action": "alloc", "addr": 2, "size": 20},
            {"action": "free_requested", "addr": 1, "size": 10},
        ]
        assert _addr_lifetimes(events) == {0: 2, 1: None}

    def test_address_reuse_starts_fresh(self):
        events = [
            {"action": "alloc", "addr": 1, "size": 10},
            {"action": "free_requested", "addr": 1, "size": 10},
            {"action": "alloc", "addr": 1, "size": 10},
        ]
        assert _addr_lifetimes(events) == {0: 1, 2: None}


class TestReservedOverhead:
    def test_none_segments(self):
        assert _reserved_overhead(None) == {
            "reserved_bytes": None,
            "allocated_bytes": None,
            "reserved_overhead_bytes": None,
        }

    def test_overhead_is_reserved_minus_active(self):
        out = _reserved_overhead(
            [
                {
                    "total_size": 2048,
                    "blocks": [
                        {"state": "active_allocated", "size": 1000},
                        {"state": "inactive", "size": 1048},
                    ],
                }
            ]
        )
        assert out == {
            "reserved_bytes": 2048,
            "allocated_bytes": 1000,
            "reserved_overhead_bytes": 1048,
        }


class TestSummarizeTraceStep:
    _EVENTS = [
        {"action": "alloc", "addr": 1, "size": 100, "stream": 0},  # 0 fwd retained
        {"action": "alloc", "addr": 2, "size": 30, "stream": 0},  # 1 fwd transient
        {"action": "free_requested", "addr": 2, "size": 30, "stream": 0},  # 2
        {"action": "alloc", "addr": 3, "size": 200, "stream": 9},  # 3 fwd comm/transient
        {"action": "free_requested", "addr": 3, "size": 200, "stream": 9},  # 4
        {"action": "alloc", "addr": 4, "size": 500, "stream": 0},  # 5 bwd transient
        {"action": "free_requested", "addr": 1, "size": 100, "stream": 0},  # 6 free retained
        {"action": "free_requested", "addr": 4, "size": 500, "stream": 0},  # 7
    ]
    _WINDOWS = [_TraceWindow(0, "fwd", 0, 4), _TraceWindow(0, "bwd", 5, 7)]

    def _run(self):
        return summarize_trace_step(
            self._EVENTS, self._WINDOWS, base_allocated=1000, comm_stream_ids={9}
        )

    def test_single_window_mean_equals_max(self):
        # One micro-batch -> one window per (layer, phase) -> mean == max.
        fwd = self._run()["layers"][0]["fwd"]
        assert fwd["peak_bytes"] == {"mean": 1300.0, "max": 1300.0}  # curve max over [0,4]
        assert fwd["retained_bytes"] == {"mean": 100.0, "max": 100.0}  # a1, freed later at idx 6
        assert fwd["transient_bytes"] == {"mean": 230.0, "max": 230.0}  # a2 + a3, freed in-window
        assert fwd["largest_alloc_bytes"] == {"mean": 200.0, "max": 200.0}
        assert fwd["comm_bytes"] == {"mean": 200.0, "max": 200.0}  # a3 on stream 9

    def test_backward_window(self):
        bwd = self._run()["layers"][0]["bwd"]
        assert bwd["peak_bytes"]["max"] == 1600
        assert bwd["transient_bytes"]["max"] == 500
        assert bwd["retained_bytes"]["max"] == 0

    def test_comm_none_reports_zero(self):
        out = summarize_trace_step(self._EVENTS, self._WINDOWS, base_allocated=1000)
        assert out["layers"][0]["fwd"]["comm_bytes"] == {"mean": 0.0, "max": 0.0}

    def test_window_bounds_clamped(self):
        out = summarize_trace_step(
            self._EVENTS, [_TraceWindow(0, "fwd", 0, 999)], base_allocated=1000
        )
        assert out["layers"][0]["fwd"]["peak_bytes"]["max"] == 1600

    def test_multiple_microbatch_windows_are_aggregated_not_overwritten(self):
        # Regression test: two micro-batches for the SAME (layer, phase) must
        # both contribute (mean/max over both), not have the second silently
        # overwrite the first (the bug this shape fixes -- under 1F1B the last
        # micro-batch is typically in cooldown with a LOWER peak, so overwriting
        # would systematically under-report the true peak).
        events = [
            {"action": "alloc", "addr": 1, "size": 100},  # 0: microbatch A window
            {"action": "free_requested", "addr": 1, "size": 100},  # 1
            {"action": "alloc", "addr": 2, "size": 40},  # 2: microbatch B window (lower peak)
            {"action": "free_requested", "addr": 2, "size": 40},  # 3
        ]
        windows = [_TraceWindow(0, "fwd", 0, 1), _TraceWindow(0, "fwd", 2, 3)]
        out = summarize_trace_step(events, windows, base_allocated=1000)
        cell = out["layers"][0]["fwd"]["peak_bytes"]
        # microbatch A peak = 1100, microbatch B peak = 1040.
        assert cell["max"] == 1100.0  # the higher of the two, not just the last
        assert cell["mean"] == pytest.approx(1070.0)  # both counted, not just B


class TestSummarizeTrace:
    _STEPS = [
        {
            "layers": {
                0: {
                    "fwd": {
                        "peak_bytes": {"mean": 100, "max": 120},
                        "retained_bytes": {"mean": 40, "max": 40},
                    }
                }
            },
            "reserved_overhead_bytes": 10,
        },
        {
            "layers": {
                0: {
                    "fwd": {
                        "peak_bytes": {"mean": 300, "max": 340},
                        "retained_bytes": {"mean": 60, "max": 60},
                    }
                }
            },
            "reserved_overhead_bytes": 30,
        },
    ]

    def test_mean_of_means_and_max_of_maxes_over_steps(self):
        out = summarize_trace(self._STEPS)
        cell = out["layers"][0]["fwd"]["peak_bytes"]
        assert cell == {"mean": pytest.approx(200.0), "max": 340.0}  # mean(100,300) / max(120,340)
        assert out["layers"][0]["fwd"]["retained_bytes"]["mean"] == pytest.approx(50.0)

    def test_reserved_overhead_aggregated(self):
        out = summarize_trace(self._STEPS)
        assert out["reserved_overhead_bytes"] == {"mean": pytest.approx(20.0), "max": 30.0}

    def test_layer_offset_relabels(self):
        assert set(summarize_trace(self._STEPS, layer_offset=16)["layers"].keys()) == {16}

    def test_skips_non_cell_values(self):
        out = summarize_trace([{"layers": {0: {"fwd": {"bogus": "not-a-cell"}}}}])
        assert out["layers"] == {}

    def test_empty(self):
        assert summarize_trace([]) == {"layers": {}}


class TestBuildTraceWindows:
    def test_forward_fifo(self):
        fwd = [(0, "pre", 1), (0, "post", 4), (0, "pre", 20), (0, "post", 25)]
        assert build_trace_windows(fwd, []) == [
            _TraceWindow(0, "fwd", 1, 4),
            _TraceWindow(0, "fwd", 20, 25),
        ]

    def test_unmatched_post_ignored(self):
        assert build_trace_windows([(0, "post", 3)], []) == []

    def test_backward_decreasing_and_block_marker(self):
        bwd = [(2, 100), (1, 110), (0, 120), (-1, 130)]
        wins = build_trace_windows([], bwd)
        assert wins == [
            _TraceWindow(2, "bwd", 100, 110),
            _TraceWindow(1, "bwd", 110, 120),
            _TraceWindow(0, "bwd", 120, 130),
        ]
        assert all(w.layer_idx >= 0 for w in wins)

    def test_backward_no_cross_pass(self):
        assert build_trace_windows([], [(1, 10), (0, 20), (1, 30), (0, 40)]) == [
            _TraceWindow(1, "bwd", 10, 20),
            _TraceWindow(1, "bwd", 30, 40),
        ]


# ---------------------------------------------------------------------------
# build_per_layer_report  (trace vs timing-only)
# ---------------------------------------------------------------------------
_TIMING = {
    "global_peak_bytes": {"mean": 500, "max": 600},
    "global_reserved_peak_bytes": {"mean": 700, "max": 800},
    "layers": {
        0: {
            "is_moe": False,
            "num_samples": 2,
            "fwd_time_ms": {"mean": 1.0, "max": 2.0},
            "bwd_num_samples": 2,
            "bwd_time_ms": {"mean": 3.0, "max": 4.0},
        }
    },
}
_MEM = {
    "layers": {
        0: {
            "fwd": {
                "peak_bytes": {"mean": 510, "max": 610},
                "retained_bytes": {"mean": 40, "max": 45},
                "transient_bytes": {"mean": 5, "max": 6},
                "largest_alloc_bytes": {"mean": 100, "max": 110},
                "comm_bytes": {"mean": 0, "max": 0},
            },
            "bwd": {
                "peak_bytes": {"mean": 720, "max": 820},
                "retained_bytes": {"mean": 0, "max": 0},
                "transient_bytes": {"mean": 500, "max": 520},
                "largest_alloc_bytes": {"mean": 300, "max": 310},
                "comm_bytes": {"mean": 0, "max": 0},
            },
        }
    },
    "reserved_overhead_bytes": {"mean": 10, "max": 12},
}


class TestBuildPerLayerReport:
    def test_trace_uses_trace_memory_but_timing_time(self):
        r = build_per_layer_report(_TIMING, _MEM)
        assert r["source"] == "trace"
        fwd = r["layers"][0]["fwd"]
        assert fwd["time_ms"] == {"mean": 1.0, "max": 2.0}  # always from timing
        assert fwd["peak"] == {"mean": 510, "max": 610}  # from trace
        assert fwd["transient"] == {"mean": 5, "max": 6}
        assert r["reserved_overhead_bytes"] == {"mean": 10, "max": 12}
        assert r["global_peak_bytes"] == {"mean": 500, "max": 600}

    def test_timing_only_has_no_per_layer_memory(self):
        r = build_per_layer_report(_TIMING, None)
        assert r["source"] == "timing-only"
        fwd = r["layers"][0]["fwd"]
        assert fwd["time_ms"] == {"mean": 1.0, "max": 2.0}
        assert fwd["peak"] is None and fwd["retained"] is None and fwd["transient"] is None
        assert r["global_peak_bytes"] == {"mean": 500, "max": 600}

    def test_empty_mem_is_timing_only(self):
        assert build_per_layer_report(_TIMING, {"layers": {}})["source"] == "timing-only"


class TestLogPerLayerResourceUsage:
    def test_returns_report_and_prints_on_log_rank(self, capsys):
        report = log_per_layer_resource_usage(_make_profiler_with_samples(), is_log_rank=True)
        assert report["source"] == "timing-only"  # profiler's mem_trace disabled
        assert set(report["layers"].keys()) == {0, 1}
        out = capsys.readouterr().out
        assert "per-layer report" in out
        assert "TIMING-ONLY" in out

    def test_no_print_off_log_rank(self, capsys):
        report = log_per_layer_resource_usage(_make_profiler_with_samples(), is_log_rank=False)
        assert report["source"] == "timing-only"
        assert capsys.readouterr().out == ""

    def test_offset_reflected(self):
        report = log_per_layer_resource_usage(
            _make_profiler_with_samples(), layer_offset=16, is_log_rank=False
        )
        assert set(report["layers"].keys()) == {16, 17}


# ---------------------------------------------------------------------------
# _MemTraceGlobalAccumulator  (fold interval reports into a per-rank global)
# ---------------------------------------------------------------------------
def _report(gidx, is_moe, fwd_peak, bwd_peak=None):
    layers = {gidx: {"is_moe": is_moe, "fwd": {"peak": fwd_peak}, "bwd": {}}}
    if bwd_peak is not None:
        layers[gidx]["bwd"] = {"peak": bwd_peak}
    return {"layers": layers}


class TestMemTraceGlobalAccumulator:
    def test_folds_mean_of_means_and_max_of_maxes(self):
        acc = _MemTraceGlobalAccumulator()
        acc.add(_report(0, False, {"mean": 100, "max": 100}))
        acc.add(_report(0, False, {"mean": 300, "max": 400}))
        cell = acc.result()["layers"][0]["fwd"]["peak"]
        assert cell["mean"] == pytest.approx(200.0)
        assert cell["max"] == 400.0

    def test_skips_none_cells(self):
        acc = _MemTraceGlobalAccumulator()
        acc.add(
            {
                "layers": {
                    0: {"is_moe": False, "fwd": {"peak": None, "time_ms": {"mean": 1, "max": 2}}}
                }
            }
        )
        fwd = acc.result()["layers"][0]["fwd"]
        assert "peak" not in fwd  # None never folded
        assert fwd["time_ms"]["max"] == 2

    def test_is_moe_carried(self):
        acc = _MemTraceGlobalAccumulator()
        acc.add(_report(3, True, {"mean": 1, "max": 1}))
        assert acc.result()["layers"][3]["is_moe"] is True

    def test_empty(self):
        assert _MemTraceGlobalAccumulator().result() == {"layers": {}}


# ---------------------------------------------------------------------------
# merge_across_ranks  (end-of-run cross-rank reduction)
# ---------------------------------------------------------------------------
def _rank_global(rank, gidx, is_moe, fwd_peak):
    return {"rank": rank, "layers": {gidx: {"is_moe": is_moe, "fwd": {"peak": fwd_peak}}}}


class TestMergeAcrossRanks:
    def test_pp_union_disjoint_layers(self):
        merged = merge_across_ranks(
            [
                _rank_global(0, 0, False, {"mean": 100, "max": 100}),
                _rank_global(1, 8, True, {"mean": 200, "max": 200}),
            ]
        )
        assert set(merged.keys()) == {0, 8}
        assert merged[8]["is_moe"] is True
        assert merged[0]["fwd"]["peak"]["max"] == 100

    def test_max_mean_spread_argmax_over_ranks(self):
        merged = merge_across_ranks(
            [
                _rank_global(0, 0, False, {"mean": 90, "max": 100}),
                _rank_global(1, 0, False, {"mean": 120, "max": 130}),
                _rank_global(2, 0, False, {"mean": 60, "max": 70}),
            ]
        )
        cell = merged[0]["fwd"]["peak"]
        assert cell["max"] == 130
        assert cell["argmax_rank"] == 1
        assert cell["mean"] == pytest.approx(90.0)
        assert cell["spread"] == pytest.approx(40.0)
        assert cell["count"] == 3

    def test_never_sums(self):
        merged = merge_across_ranks(
            [
                _rank_global(0, 0, False, {"mean": 100, "max": 100}),
                _rank_global(1, 0, False, {"mean": 100, "max": 100}),
            ]
        )
        assert merged[0]["fwd"]["peak"]["max"] == 100  # not 200

    def test_empty(self):
        assert merge_across_ranks([]) == {}
