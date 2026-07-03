# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the pure-logic (CPU, no CUDA) parts of per-layer profiling.

Covers aggregation and summarization only: ``_agg``, ``PerLayerProfileStats``
record/reset, ``summarize_stats`` (including the PP local->global layer-index
offset), ``_mib``, and ``log_per_layer_resource_usage`` return / rank gating.

The CUDA-dependent paths (deferred events, memory_stats snapshots, hook
attach/detach, flush) are intentionally not covered here; they are exercised by
the functional tests on GPU. Everything below runs on CPU without a CUDA device.
"""

import pytest

from megatron.core.transformer.per_layer_profiling import (
    PerLayerProfiler,
    PerLayerProfileStats,
    _agg,
    _mib,
    log_per_layer_resource_usage,
    summarize_stats,
)


# ---------------------------------------------------------------------------
# _agg
# ---------------------------------------------------------------------------
class TestAgg:
    def test_empty_returns_zeros(self):
        assert _agg([]) == (0.0, 0.0)

    def test_single_value(self):
        assert _agg([5.0]) == (5.0, 5.0)

    def test_mean_and_max(self):
        mean, mx = _agg([1.0, 2.0, 3.0, 4.0])
        assert mean == pytest.approx(2.5)
        assert mx == 4.0

    def test_negative_values(self):
        # reserved/allocated deltas can be negative; _agg must not clamp.
        mean, mx = _agg([-4.0, -2.0])
        assert mean == pytest.approx(-3.0)
        assert mx == -2.0


# ---------------------------------------------------------------------------
# PerLayerProfileStats.record_fwd / reset
# ---------------------------------------------------------------------------
class TestPerLayerProfileStats:
    def _sample(self, s: PerLayerProfileStats, n: int = 1):
        for i in range(n):
            s.record_fwd(
                time_ms=float(i + 1),
                mem_delta_bytes=(i + 1) * 100,
                reserved_delta_bytes=(i + 1) * 1000,
                peak_after_bytes=(i + 1) * 10_000,
                peak_rise_bytes=(i + 1) * 5,
            )

    def test_record_appends_all_columns(self):
        s = PerLayerProfileStats(layer_idx=0)
        self._sample(s, n=3)
        assert s.num_samples == 3
        assert s.fwd_time_ms == [1.0, 2.0, 3.0]
        assert s.fwd_mem_allocated_delta_bytes == [100, 200, 300]
        assert s.fwd_mem_reserved_delta_bytes == [1000, 2000, 3000]
        assert s.fwd_mem_peak_after_bytes == [10_000, 20_000, 30_000]
        assert s.fwd_mem_peak_rise_bytes == [5, 10, 15]

    def test_reset_clears_everything(self):
        s = PerLayerProfileStats(layer_idx=0)
        self._sample(s, n=2)
        s.reset()
        assert s.num_samples == 0
        assert s.fwd_time_ms == []
        assert s.fwd_mem_allocated_delta_bytes == []
        assert s.fwd_mem_reserved_delta_bytes == []
        assert s.fwd_mem_peak_after_bytes == []
        assert s.fwd_mem_peak_rise_bytes == []

    def test_moe_tag_preserved(self):
        s = PerLayerProfileStats(layer_idx=7, is_moe_layer=True)
        assert s.is_moe_layer is True
        assert s.layer_idx == 7


# ---------------------------------------------------------------------------
# summarize_stats  (build a profiler by hand, no CUDA)
# ---------------------------------------------------------------------------
def _make_profiler_with_samples() -> PerLayerProfiler:
    """Construct a profiler and inject stats directly, bypassing all CUDA.

    PerLayerProfiler(enabled=False) does no CUDA work in __init__ beyond
    torch.cuda.is_available() (safe on CPU). We populate _stats and the global
    peak lists directly to test the pure summarization path.
    """
    prof = PerLayerProfiler(enabled=False)

    s0 = PerLayerProfileStats(layer_idx=0, is_moe_layer=False)
    s0.record_fwd(2.0, 100, 1000, 50_000, 10)
    s0.record_fwd(4.0, 300, 3000, 70_000, 30)  # time mean 3.0 / max 4.0

    s1 = PerLayerProfileStats(layer_idx=1, is_moe_layer=True)
    s1.record_fwd(10.0, 500, 5000, 90_000, 40)

    prof._stats = {0: s0, 1: s1}
    prof._global_peak_bytes = [70_000, 80_000]  # mean 75000 / max 80000
    prof._global_reserved_peak_bytes = [100_000, 120_000]  # mean 110000 / max 120000
    return prof


class TestSummarizeStats:
    def test_global_peaks(self):
        out = summarize_stats(_make_profiler_with_samples())
        assert out["global_peak_bytes"]["mean"] == pytest.approx(75_000.0)
        assert out["global_peak_bytes"]["max"] == 80_000.0
        assert out["global_reserved_peak_bytes"]["mean"] == pytest.approx(110_000.0)
        assert out["global_reserved_peak_bytes"]["max"] == 120_000.0

    def test_layer_count_and_keys(self):
        out = summarize_stats(_make_profiler_with_samples())
        assert set(out["layers"].keys()) == {0, 1}

    def test_layer_aggregation(self):
        out = summarize_stats(_make_profiler_with_samples())
        l0 = out["layers"][0]
        assert l0["is_moe"] is False
        assert l0["num_samples"] == 2
        assert l0["fwd_time_ms"]["mean"] == pytest.approx(3.0)
        assert l0["fwd_time_ms"]["max"] == 4.0
        assert l0["mem_delta_bytes"]["mean"] == pytest.approx(200.0)
        assert l0["mem_delta_bytes"]["max"] == 300.0
        assert l0["mem_reserved_delta_bytes"]["mean"] == pytest.approx(2000.0)
        assert l0["mem_peak_after_bytes"]["max"] == 70_000.0
        assert l0["mem_peak_rise_bytes"]["max"] == 30.0

    def test_moe_flag(self):
        out = summarize_stats(_make_profiler_with_samples())
        assert out["layers"][1]["is_moe"] is True

    def test_layer_offset_maps_to_global_index(self):
        # PP: local indices 0,1 with offset 16 -> global 16,17.
        out = summarize_stats(_make_profiler_with_samples(), layer_offset=16)
        assert set(out["layers"].keys()) == {16, 17}
        # content still follows the original local layer
        assert out["layers"][16]["num_samples"] == 2
        assert out["layers"][17]["is_moe"] is True

    def test_empty_profiler(self):
        prof = PerLayerProfiler(enabled=False)
        out = summarize_stats(prof)
        assert out["layers"] == {}
        assert out["global_peak_bytes"] == {"mean": 0.0, "max": 0.0}
        assert out["global_reserved_peak_bytes"] == {"mean": 0.0, "max": 0.0}


# ---------------------------------------------------------------------------
# _mib
# ---------------------------------------------------------------------------
class TestMib:
    def test_one_mib(self):
        assert _mib(1024 * 1024) == pytest.approx(1.0)

    def test_zero(self):
        assert _mib(0) == 0.0


# ---------------------------------------------------------------------------
# log_per_layer_resource_usage  (return value + rank gating)
# ---------------------------------------------------------------------------
class TestLogPerLayerResourceUsage:
    def test_returns_summary_on_log_rank(self, capsys):
        prof = _make_profiler_with_samples()
        summary = log_per_layer_resource_usage(prof, is_log_rank=True)
        assert summary is not None
        assert set(summary["layers"].keys()) == {0, 1}
        # a table was printed
        out = capsys.readouterr().out
        assert "per-layer resource usage" in out

    def test_returns_summary_but_no_print_off_log_rank(self, capsys):
        prof = _make_profiler_with_samples()
        summary = log_per_layer_resource_usage(prof, is_log_rank=False)
        # summary still returned (useful for TB/W&B), but nothing printed
        assert summary is not None
        assert set(summary["layers"].keys()) == {0, 1}
        assert capsys.readouterr().out == ""

    def test_offset_reflected_in_output(self):
        prof = _make_profiler_with_samples()
        summary = log_per_layer_resource_usage(prof, layer_offset=16, is_log_rank=False)
        assert set(summary["layers"].keys()) == {16, 17}
