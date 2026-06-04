# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for per-request metric derivation in DynamicInferenceEngine.

These tests exercise `_build_request_metrics` directly against synthetic event
timelines. The method only reads `context.block_size_tokens`,
`context.step_count`, and `inference_step_offset` off `self`, so a
`SimpleNamespace` stand-in is sufficient.
"""

from types import SimpleNamespace
from typing import List, Optional

import pytest
import torch

from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.inference_request import (
    DynamicInferenceEvent,
    DynamicInferenceEventType,
    DynamicInferenceRequest,
)
from megatron.core.inference.sampling_params import SamplingParams


def _fake_engine(
    block_size_tokens: int = 16, step_count: int = 100, inference_step_offset: int = 0
) -> SimpleNamespace:
    return SimpleNamespace(
        context=SimpleNamespace(block_size_tokens=block_size_tokens, step_count=step_count),
        inference_step_offset=inference_step_offset,
    )


def _evt(type_: DynamicInferenceEventType, ts: float, token_id: int = 0) -> DynamicInferenceEvent:
    # GENERATED_TOKEN requires a payload with token_id; all other types must be None.
    payload = {"token_id": token_id} if type_ is DynamicInferenceEventType.GENERATED_TOKEN else None
    return DynamicInferenceEvent(type=type_, timestamp=ts, payload=payload)


def _make_request(
    prompt_len: int = 4,
    generated_tokens: Optional[List[int]] = None,
    events: Optional[List[DynamicInferenceEvent]] = None,
    ttft: Optional[float] = None,
    tpot: Optional[List[float]] = None,
    prefix_blocks_matched: int = 0,
    max_tokens: int = 10,
) -> DynamicInferenceRequest:
    req = DynamicInferenceRequest(
        request_id=42,
        prompt_tokens=torch.arange(prompt_len, dtype=torch.long),
        sampling_params=SamplingParams(num_tokens_to_generate=max_tokens, termination_id=0),
    )
    if generated_tokens is not None:
        req.generated_tokens = list(generated_tokens)
    if events:
        req.events = list(events)
        for evt in events:
            if evt.type is DynamicInferenceEventType.ADD_ENGINE:
                req.event_add_engine = evt
                break
    if ttft is not None:
        req.ttft = ttft
    if tpot is not None:
        req.tpot = list(tpot)
    req.prefix_blocks_matched = prefix_blocks_matched
    return req


def test_build_request_metrics_happy_path():
    """Full ADD_ENGINE -> ADD_CONTEXT -> 3xGENERATED_TOKEN -> FINISH timeline.
    Verifies that every derived timing field is computed from the event timestamps."""
    req = _make_request(
        prompt_len=4,
        generated_tokens=[10, 11, 12],
        ttft=2.0,
        tpot=[0.1, 0.2, 0.3],
        max_tokens=10,
        events=[
            _evt(DynamicInferenceEventType.ADD_ENGINE, 0.0),
            _evt(DynamicInferenceEventType.ADD_CONTEXT, 1.0),
            _evt(DynamicInferenceEventType.GENERATED_TOKEN, 2.0),
            _evt(DynamicInferenceEventType.GENERATED_TOKEN, 3.0),
            _evt(DynamicInferenceEventType.GENERATED_TOKEN, 4.0),
            _evt(DynamicInferenceEventType.FINISH, 5.0),
        ],
    )
    m = DynamicInferenceEngine._build_request_metrics(_fake_engine(), req)

    assert m['request_id'] == 42
    assert m['prompt_len'] == 4
    assert m['output_len'] == 3
    assert m['max_tokens'] == 10
    assert m['hit_max_tokens'] == 0
    assert m['num_forward_passes'] == 3
    assert m['num_pauses'] == 0
    assert m['num_evicts'] == 0
    assert m['time_paused_s'] == 0.0
    assert m['time_evicted_s'] == 0.0
    assert m['arrival_ts'] == 0.0
    assert m['add_context_ts'] == 1.0
    assert m['first_token_ts'] == 2.0
    assert m['finish_ts'] == 5.0
    assert m['ttft_s'] == 2.0
    assert m['queue_wait_s'] == 1.0
    assert m['prefill_time_s'] == 1.0
    assert m['decode_time_s'] == 3.0
    assert m['total_wall_s'] == 5.0
    assert m['mean_tpot_s'] == pytest.approx(0.2)
    assert m['num_tpot_samples'] == 3


def test_build_request_metrics_pause_resume_accumulates_dormant_time():
    """A PAUSE followed by a non-PAUSE event closes the dormant interval into time_paused_s."""
    req = _make_request(
        generated_tokens=[10, 11],
        events=[
            _evt(DynamicInferenceEventType.ADD_ENGINE, 0.0),
            _evt(DynamicInferenceEventType.ADD_CONTEXT, 1.0),
            _evt(DynamicInferenceEventType.GENERATED_TOKEN, 2.0),
            _evt(DynamicInferenceEventType.PAUSE, 3.0),
            _evt(DynamicInferenceEventType.GENERATED_TOKEN, 5.0),
            _evt(DynamicInferenceEventType.FINISH, 6.0),
        ],
    )
    m = DynamicInferenceEngine._build_request_metrics(_fake_engine(), req)
    assert m['num_pauses'] == 1
    assert m['time_paused_s'] == 2.0


def test_build_request_metrics_evict_resume_accumulates_dormant_time():
    """EVICT followed by a non-EVICT event closes the dormant interval into time_evicted_s."""
    req = _make_request(
        generated_tokens=[10],
        events=[
            _evt(DynamicInferenceEventType.ADD_ENGINE, 0.0),
            _evt(DynamicInferenceEventType.ADD_CONTEXT, 1.0),
            _evt(DynamicInferenceEventType.GENERATED_TOKEN, 2.0),
            _evt(DynamicInferenceEventType.EVICT, 3.0),
            _evt(DynamicInferenceEventType.ADD_CONTEXT, 6.0),
            _evt(DynamicInferenceEventType.GENERATED_TOKEN, 7.0),
            _evt(DynamicInferenceEventType.FINISH, 8.0),
        ],
    )
    m = DynamicInferenceEngine._build_request_metrics(_fake_engine(), req)
    assert m['num_evicts'] == 1
    assert m['time_evicted_s'] == 3.0


def test_build_request_metrics_pause_open_at_finish_is_closed_by_finish():
    """If the request FINISHes while still paused, FINISH must close the interval —
    otherwise the dormant time silently disappears for the trailing pause."""
    req = _make_request(
        generated_tokens=[10],
        events=[
            _evt(DynamicInferenceEventType.ADD_ENGINE, 0.0),
            _evt(DynamicInferenceEventType.ADD_CONTEXT, 1.0),
            _evt(DynamicInferenceEventType.GENERATED_TOKEN, 2.0),
            _evt(DynamicInferenceEventType.PAUSE, 3.0),
            _evt(DynamicInferenceEventType.FINISH, 5.0),
        ],
    )
    m = DynamicInferenceEngine._build_request_metrics(_fake_engine(), req)
    assert m['num_pauses'] == 1
    assert m['time_paused_s'] == 2.0


def test_build_request_metrics_prefix_cache_math():
    """prefix_tokens_reused = blocks_matched * block_size_tokens;
    prompt_tokens_recomputed = prompt_len - prefix_tokens_reused."""
    req = _make_request(prompt_len=64, prefix_blocks_matched=2)
    m = DynamicInferenceEngine._build_request_metrics(_fake_engine(block_size_tokens=16), req)
    assert m['prefix_blocks_matched'] == 2
    assert m['prefix_tokens_reused'] == 32
    assert m['prompt_tokens_recomputed'] == 32


def test_build_request_metrics_omits_timing_when_no_token_generated():
    """Requests that failed before generating must not emit None for missing timing
    fields — the fields must be absent from the dict so wandb histograms skip them."""
    req = _make_request(events=[_evt(DynamicInferenceEventType.ADD_ENGINE, 0.0)])
    m = DynamicInferenceEngine._build_request_metrics(_fake_engine(), req)
    for absent_key in (
        'ttft_s',
        'queue_wait_s',
        'prefill_time_s',
        'decode_time_s',
        'total_wall_s',
        'add_context_ts',
        'first_token_ts',
        'finish_ts',
    ):
        assert absent_key not in m, f"{absent_key} should be omitted, got {m[absent_key]!r}"


def test_build_request_metrics_omits_tpot_when_empty():
    """Empty request.tpot should skip mean_tpot_s and num_tpot_samples entirely."""
    req = _make_request()
    m = DynamicInferenceEngine._build_request_metrics(_fake_engine(), req)
    assert 'mean_tpot_s' not in m
    assert 'num_tpot_samples' not in m


def test_build_request_metrics_hit_max_tokens_flag():
    fake = _fake_engine()
    req_hit = _make_request(generated_tokens=[1, 2, 3, 4, 5], max_tokens=5)
    req_miss = _make_request(generated_tokens=[1, 2], max_tokens=5)
    assert DynamicInferenceEngine._build_request_metrics(fake, req_hit)['hit_max_tokens'] == 1
    assert DynamicInferenceEngine._build_request_metrics(fake, req_miss)['hit_max_tokens'] == 0


def test_build_request_metrics_inference_step_uses_offset():
    """inference_step = inference_step_offset + context.step_count."""
    req = _make_request()
    m = DynamicInferenceEngine._build_request_metrics(
        _fake_engine(step_count=200, inference_step_offset=1000), req
    )
    assert m['inference_step'] == 1200
