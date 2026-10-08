# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
from argparse import Namespace
from types import SimpleNamespace
from unittest import mock

import pytest

from examples.inference import utils as inference_utils
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


@pytest.mark.parametrize(
    ("do_broadcast", "distributed_initialized", "world_size"),
    [(False, True, 2), (True, False, 2), (True, True, 1)],
)
def test_get_curr_time_avoids_cuda_when_rank_sync_is_unnecessary(
    monkeypatch, do_broadcast, distributed_initialized, world_size
):
    """Ensure local timing never synchronizes the CUDA compute stream."""
    monkeypatch.setattr(inference_utils.time, "time_ns", lambda: 123_000_000_000)
    monkeypatch.setattr(
        inference_utils.torch.distributed, "is_initialized", lambda: distributed_initialized
    )
    monkeypatch.setattr(inference_utils.torch.distributed, "get_world_size", lambda: world_size)
    cuda_long_tensor = mock.Mock()
    broadcast = mock.Mock()
    monkeypatch.setattr(inference_utils.torch.cuda, "LongTensor", cuda_long_tensor)
    monkeypatch.setattr(inference_utils.torch.distributed, "broadcast", broadcast)

    assert inference_utils.get_curr_time(do_broadcast=do_broadcast) == 123.0
    cuda_long_tensor.assert_not_called()
    broadcast.assert_not_called()


def test_get_curr_time_broadcasts_for_multi_rank_sync(monkeypatch):
    """Ensure explicit multi-rank timing still broadcasts a rank-zero timestamp."""
    timestamp = mock.Mock()
    timestamp.item.return_value = 123_000_000_000
    monkeypatch.setattr(inference_utils.time, "time_ns", lambda: 123_000_000_000)
    monkeypatch.setattr(inference_utils.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(inference_utils.torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(inference_utils.torch.cuda, "LongTensor", lambda _value: timestamp)
    broadcast = mock.Mock()
    monkeypatch.setattr(inference_utils.torch.distributed, "broadcast", broadcast)

    assert inference_utils.get_curr_time() == 123.0
    broadcast.assert_called_once_with(timestamp, src=0)
