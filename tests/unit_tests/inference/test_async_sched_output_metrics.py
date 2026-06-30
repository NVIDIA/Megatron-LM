# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
from argparse import Namespace
from types import SimpleNamespace

from examples.inference.offline_inference import _capture_engine_stats
from examples.inference.utils import dump_inference_results_to_json
from tests.functional_tests.python_test_utils.test_inference_regular_pipeline import (
    _NON_REQUEST_TOP_LEVEL_KEYS,
)


def test_dump_inference_results_to_json_writes_async_sched_counters(tmp_path):
    """Ensure async scheduling counters are emitted as top-level JSON metadata."""
    output_path = tmp_path / "results.json"
    args = Namespace(
        output_path=str(output_path),
        output_every_n_results=1,
        output_request_events=False,
        record_throughput=True,
    )
    request = SimpleNamespace(
        request_id=7,
        prompt="prompt",
        generated_text="generated",
        generated_tokens=[1, 2],
        latency=None,
        ttft=None,
        sampling_params=SimpleNamespace(return_log_probs=False),
    )

    dump_inference_results_to_json(
        args=args,
        results=[request],
        throughputs=[12.5],
        peak_mem_stats={"mem-max-allocated-bytes": 1024},
        step_count=3,
        lifetime_prefill_token_count=4,
        async_sched_step_count=5,
        async_sched_compaction_step_count=6,
    )

    output = json.loads(output_path.read_text())
    assert output["async_sched_step_count"] == 5
    assert output["async_sched_compaction_step_count"] == 6
    assert output["7"]["step_count"] == 3


def test_inference_comparator_ignores_async_sched_counters():
    """Ensure async scheduling counters are treated as metadata, not request IDs."""
    assert "async_sched_step_count" in _NON_REQUEST_TOP_LEVEL_KEYS
    assert "async_sched_compaction_step_count" in _NON_REQUEST_TOP_LEVEL_KEYS


def test_capture_engine_stats_includes_async_sched_counters():
    """Ensure offline reporting captures async scheduling counters from the engine context."""
    context = SimpleNamespace(
        step_count=1,
        lifetime_prefill_token_count=2,
        async_sched_step_count=3,
        async_sched_compaction_step_count=4,
    )
    llm = SimpleNamespace(engine=SimpleNamespace(context=context, capture_stats={"graphs": 5}))

    assert _capture_engine_stats(llm) == {
        "step_count": 1,
        "lifetime_prefill_token_count": 2,
        "async_sched_step_count": 3,
        "async_sched_compaction_step_count": 4,
        "capture_stats": {"graphs": 5},
    }
