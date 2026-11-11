# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Step-time histograms by step class: classification, bucketing, the EP-consensus flag, the
engine hook, Prometheus rendering, and the steady-state client's window diff."""

import os
import sys
from types import SimpleNamespace

import pytest

from megatron.core.inference.batch_dimensions_utils import (
    CUDAGraphBatchDimensionBuilder,
    InferenceBatchDimensions,
)
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.engines.step_time_metrics import (
    STEP_TIME_BUCKETS_MS,
    StepMixCounter,
    StepTimeHistogram,
    classify_step,
    classify_step_mix,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    DecodeOnly,
)
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.metrics import (
    render_prometheus,
)

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "inference-bench", "server")
)
import steady_client  # noqa: E402


@pytest.mark.parametrize(
    "local_prefill, ep_any_prefill, expected",
    [
        (True, True, "prefill"),
        (True, False, "prefill"),
        (True, None, "prefill"),
        (False, True, "decode_interfered"),
        (False, False, "decode_clean"),
        (False, None, "decode"),
    ],
)
def test_classify_step(local_prefill, ep_any_prefill, expected):
    assert classify_step(local_prefill, ep_any_prefill) == expected


def test_histogram_buckets_are_cumulative_with_inf():
    hist = StepTimeHistogram()
    for ms in (0.1, 0.25, 0.3, 3.0, 2000.0):
        hist.observe("decode_clean", ms)
    snap = hist.snapshot()["decode_clean"]
    bounds = STEP_TIME_BUCKETS_MS
    assert len(snap["buckets"]) == len(bounds) + 1
    # le=0.25 holds 0.1 and 0.25 (bounds are inclusive); le=0.5 adds 0.3; le=4 adds 3.0.
    assert snap["buckets"][bounds.index(0.25)] == 2
    assert snap["buckets"][bounds.index(0.5)] == 3
    assert snap["buckets"][bounds.index(2.0)] == 3
    assert snap["buckets"][bounds.index(4.0)] == 4
    assert snap["buckets"][-2] == 4  # le=1024 excludes 2000 ms
    assert snap["buckets"][-1] == snap["count"] == 5  # +Inf
    assert snap["sum"] == pytest.approx(2003.65)


def _row_bound_context(any_non_decode):
    """Just enough of a DynamicInferenceContext for _match_graph_with_ep_moe_row_bound."""
    return SimpleNamespace(
        cuda_graph_batch_dimensions_list=[],
        is_hybrid_model=True,
        expert_model_parallel_group=None,
        _ep_zmq_communicator=None,
        _moe_sp_size=1,
        _eager_padded_batch_dimensions=lambda: InferenceBatchDimensions(
            token_count=64, prefill_req_count=0, decode_req_count=4
        ),
        step_ep_any_prefill=None,
    )


@pytest.mark.parametrize("any_non_decode", [True, False])
def test_ep_consensus_sets_step_flag(monkeypatch, any_non_decode):
    monkeypatch.setattr(
        InferenceBatchDimensions,
        "sync_ep_moe_row_bound_state",
        staticmethod(lambda *a, **k: (8, any_non_decode, True, 256)),
    )
    monkeypatch.setattr(
        CUDAGraphBatchDimensionBuilder, "match_graph_config", staticmethod(lambda *a, **k: None)
    )
    context = _row_bound_context(any_non_decode)
    dims = InferenceBatchDimensions(token_count=4, prefill_req_count=0, decode_req_count=4)
    DynamicInferenceContext._match_graph_with_ep_moe_row_bound(context, dims)
    assert context.step_ep_any_prefill is any_non_decode


@pytest.mark.parametrize(
    "decode_only, ep_any_prefill, expected",
    [
        (DecodeOnly(consumed=True, launched=True), True, "decode_interfered"),
        (DecodeOnly(consumed=True, launched=True), False, "decode_clean"),
        (DecodeOnly(consumed=True, launched=False), True, "prefill"),
        (DecodeOnly(consumed=None, launched=True), None, "decode"),
        (DecodeOnly(consumed=True, launched=None), True, None),  # nothing launched: skipped
    ],
)
def test_engine_observes_launched_forward(decode_only, ep_any_prefill, expected):
    engine = SimpleNamespace(
        context=SimpleNamespace(step_ep_any_prefill=ep_any_prefill),
        _step_time_histogram=StepTimeHistogram(),
    )
    DynamicInferenceEngine._observe_step_time(engine, decode_only, 0.003)
    snap = engine._step_time_histogram.snapshot()
    if expected is None:
        assert snap == {}
    else:
        assert list(snap) == [expected]
        assert snap[expected]["count"] == 1
        assert snap[expected]["sum"] == pytest.approx(3.0)


def _engine_metrics(step_times):
    hist = StepTimeHistogram()
    for step_class, ms in step_times:
        hist.observe(step_class, ms)
    return {"generation_tokens_total": 0, "step_time_ms": hist.snapshot()}


def test_render_and_window_diff_round_trip():
    before = {
        "e0": _engine_metrics([("decode_clean", 1.5)]),
        "e1": _engine_metrics([("decode_clean", 1.5), ("prefill", 40.0)]),
    }
    after = {
        "e0": _engine_metrics([("decode_clean", 1.5), ("decode_clean", 1.7), ("prefill", 70.0)]),
        "e1": _engine_metrics(
            [("decode_clean", 1.5), ("prefill", 40.0), ("decode_interfered", 9.0)]
        ),
    }
    text0, text1 = render_prometheus(before), render_prometheus(after)
    assert "# TYPE mcore:step_time_ms histogram" in text1
    assert 'mcore:step_time_ms_bucket{engine="e0",step="decode_clean",le="2"} 2' in text1
    assert 'mcore:step_time_ms_count{step="decode_clean"} 3' in text1  # summed over engines

    window = steady_client.diff_histograms(
        steady_client.parse_histograms(text0), steady_client.parse_histograms(text1)
    )
    assert window["e0"]["decode_clean"]["count"] == 1
    assert window["e0"]["decode_clean"]["mean_ms"] == pytest.approx(1.7)
    assert window["e0"]["decode_clean"]["buckets"]["2"] == 1
    assert window["e0"]["prefill"] == {
        "count": 1,
        "mean_ms": pytest.approx(70.0),
        "buckets": {**{f"{le:g}": 0 for le in STEP_TIME_BUCKETS_MS}, "+Inf": 0, "128": 1},
    }
    assert "prefill" not in window["e1"]  # no new prefill on e1 in the window
    assert window["all"]["decode_interfered"]["count"] == 1
    assert window["all"]["decode_interfered"]["buckets"]["16"] == 1
    assert sum(window["all"]["decode_clean"]["buckets"].values()) == 1


@pytest.mark.parametrize(
    "local_prefill, decode_requests, ep_any_prefill, expected",
    [
        (True, 0, True, "prefill_only"),
        (True, 3, None, "mixed"),
        (False, 5, True, "decode_interfered"),
        (False, 5, False, "decode_clean"),
        (False, 5, None, "decode"),
    ],
)
def test_classify_step_mix(local_prefill, decode_requests, ep_any_prefill, expected):
    assert classify_step_mix(local_prefill, decode_requests, ep_any_prefill) == expected


def test_step_mix_counter_times_gaps_and_skips_idle():
    counter = StepMixCounter()
    counter.observe("decode_clean", 8, now=10.000)  # starts the clock, untimed
    counter.observe("decode_clean", 8, now=10.010)
    counter.observe("mixed", 600, now=10.060)
    counter.observe("decode_clean", 8, now=20.0)  # after an idle gap: counted, not timed
    counter.idle()
    counter.observe("prefill_only", 900, now=20.5)  # clock restarted: untimed
    snap = counter.snapshot()
    assert snap["decode_clean"] == {"steps": 3, "tokens": 24, "wall_ms": pytest.approx(10.0)}
    assert snap["mixed"] == {"steps": 1, "tokens": 600, "wall_ms": pytest.approx(50.0)}
    assert snap["prefill_only"] == {"steps": 1, "tokens": 900, "wall_ms": 0.0}


def _mix_metrics(counts):
    counter = StepMixCounter()
    t = 0.0
    counter.observe("decode_clean", 0, now=t)
    for step_class, tokens, ms in counts:
        t += ms / 1e3
        counter.observe(step_class, tokens, now=t)
    return {"generation_tokens_total": 0, "step_mix": counter.snapshot()}


def test_step_mix_render_and_window_diff_by_pool():
    before = {
        "prefill/e0": _mix_metrics([("prefill_only", 4000, 100.0)]),
        "decode/e1": _mix_metrics([("decode_clean", 64, 10.0)]),
    }
    after = {
        "prefill/e0": _mix_metrics([("prefill_only", 4000, 100.0), ("prefill_only", 2000, 60.0)]),
        "decode/e1": _mix_metrics(
            [("decode_clean", 64, 10.0), ("decode_clean", 64, 12.0), ("decode_clean", 64, 14.0)]
        ),
    }
    text0, text1 = render_prometheus(before), render_prometheus(after)
    assert 'mcore:step_mix_steps_total{engine="decode/e1",step="decode_clean"} 4' in text1
    assert 'mcore:step_mix_tokens_total{step="prefill_only"} 6000' in text1

    window = steady_client.diff_step_mix(
        steady_client.parse_step_mix(text0), steady_client.parse_step_mix(text1)
    )
    assert window["prefill"]["prefill_only"]["steps"] == 1
    assert window["prefill"]["prefill_only"]["mean_ms"] == pytest.approx(60.0)
    assert window["decode"]["decode_clean"]["steps"] == 2
    assert window["decode"]["decode_clean"]["mean_ms"] == pytest.approx(13.0)
    assert window["decode"]["decode_clean"]["step_share"] == 1.0
    assert window["all"]["decode_clean"]["steps"] == 2
