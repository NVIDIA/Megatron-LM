# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from examples.mimo.training import encoder_prefetch
from examples.mimo.training import step as mimo_step
from examples.mimo.training.encoder_prefetch import (
    PREFETCHED_FEATURES_KEY,
    PROJECTION_TIMER_KEY,
    EncoderPrefetchLoader,
    prefetch_frozen_features,
    validate_encoder_prefetch_args,
)
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules

ENCODER = "clip_encoder"


def _args(**overrides):
    values = {
        "mimo_encoder_prefetch": True,
        "mimo_encoder_prefetch_depth": 2,
        "freeze_vit": True,
        "freeze_projection": False,
        "encoder_tp": 1,
        "rerun_mode": "disabled",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("freeze_vit", False, "freeze-vit"),
        ("freeze_projection", True, "trainable projection"),
        ("encoder_tp", 2, "TP=1"),
        ("encoder_cp", 2, "CP=1"),
        ("encoder_pp", 2, "PP=1"),
        ("encoder_ep", 2, "EP=1"),
        ("mimo_encoder_prefetch_depth", 0, "positive"),
        ("rerun_mode", "validate_results", "rerun"),
    ],
)
def test_prefetch_validation(field, value, match):
    with pytest.raises(ValueError, match=match):
        validate_encoder_prefetch_args(_args(**{field: value}))


class _LinearEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)

    def forward(self, *, x):
        return self.linear(x)


def test_prefetch_skips_backbone_autograd_but_keeps_projection_gradients():
    torch.manual_seed(123)
    submodule = VisionModalitySubmodules(
        encoders={"radio": _LinearEncoder()}, input_projections=[nn.Linear(4, 4, bias=False)]
    )
    submodule.encoders.requires_grad_(False)
    inputs = {"radio": {"x": torch.randn(2, 3, 4)}}

    expected = submodule(encoder_inputs=inputs)
    features = prefetch_frozen_features(submodule, inputs)
    actual = submodule(hidden_states=features)

    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert all(parameter.grad is None for parameter in submodule.encoders.parameters())
    assert all(parameter.grad is not None for parameter in submodule.input_projections.parameters())
    with pytest.raises(ValueError, match="mutually exclusive"):
        submodule(encoder_inputs=inputs, hidden_states=features)


class _FakeEvent:
    def __init__(self):
        self.recorded_on = None
        self.record_calls = 0
        self.synchronized = False

    def record(self, stream=None):
        self.recorded_on = stream
        self.record_calls += 1

    def synchronize(self):
        self.synchronized = True

    def query(self):
        return self.record_calls > 0

    def elapsed_time(self, _end_event):
        return 1.25


class _FakeStream:
    def __init__(self):
        self.waited_events = []
        self.synchronize_calls = 0

    def wait_event(self, event):
        self.waited_events.append(event)

    def synchronize(self):
        self.synchronize_calls += 1


@pytest.fixture
def fake_cuda(monkeypatch):
    current = _FakeStream()
    producer = _FakeStream()
    events = []

    def make_event(**_kwargs):
        event = _FakeEvent()
        events.append(event)
        return event

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current)
    monkeypatch.setattr(torch.cuda, "Event", make_event)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(encoder_prefetch, "move_batch_to_cuda", lambda value: value)
    return SimpleNamespace(current=current, producer=producer, events=events)


class _Source:
    def __init__(self, count):
        self.count = count
        self.position = 0
        self.condition = threading.Condition()

    def __iter__(self):
        return self

    def __next__(self):
        with self.condition:
            if self.position == self.count:
                raise StopIteration
            sequence = self.position
            self.position += 1
            self.condition.notify_all()
        return {
            "input_ids": torch.tensor(sequence),
            "modality_inputs": {ENCODER: {"radio": {"x": torch.tensor(sequence)}}},
        }


def _wait_until(predicate):
    deadline = time.monotonic() + 2
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.005)


@pytest.mark.parametrize("depth", (1, 2, 4))
def test_depth_is_completed_batches_and_refills_after_pop(fake_cuda, depth):
    source = _Source(8)
    produced = []

    def produce(inputs):
        produced.append(int(inputs["radio"]["x"].item()))
        return torch.tensor(produced[-1])

    loader = EncoderPrefetchLoader(
        source=source,
        encoder_name=ENCODER,
        feature_producer=produce,
        depth=depth,
        stream=fake_cuda.producer,
    )
    loader.start()
    _wait_until(lambda: len(loader._ready) == depth)

    first = next(loader)
    _wait_until(lambda: len(loader._ready) == depth)

    assert first[PREFETCHED_FEATURES_KEY][ENCODER].item() == 0
    assert PROJECTION_TIMER_KEY not in first
    assert len(fake_cuda.producer.waited_events) == 1
    assert fake_cuda.current.waited_events == []
    loader.close()


def test_pending_encode_can_be_claimed_while_cpu_read_ahead_blocks(fake_cuda, caplog):
    caplog.set_level("INFO", logger=f"{encoder_prefetch.__name__}.debug")
    read_ahead_started = threading.Event()
    release_read_ahead = threading.Event()

    class _BlockingSource(_Source):
        def __next__(self):
            if self.position == 1:
                read_ahead_started.set()
                release_read_ahead.wait(timeout=2)
            return super().__next__()

    loader = EncoderPrefetchLoader(
        source=_BlockingSource(2),
        encoder_name=ENCODER,
        feature_producer=lambda inputs: torch.tensor(inputs["radio"]["x"].item()),
        depth=1,
        stream=fake_cuda.producer,
        debug=True,
    )
    loader.start()
    assert read_ahead_started.wait(timeout=1)

    try:
        assert len(loader._ready) == 0
        completion_event = loader._pending[1]
        batch = next(loader)
        assert batch[PREFETCHED_FEATURES_KEY][ENCODER].item() == 0
        assert fake_cuda.current.waited_events[-1] is completion_event
        assert loader._pending is None
    finally:
        release_read_ahead.set()
        loader.close()

    assert "consumer-wait batch=0 encoder_wait_ms=1.250" in caplog.text
    assert "claimed_pending=1" in caplog.text


def test_cpu_read_ahead_overlaps_encode_without_enqueuing_another_batch(fake_cuda):
    class _ObservedSource(_Source):
        def __next__(self):
            if self.position == 1:
                assert not fake_cuda.events[-1].synchronized
            return super().__next__()

    source = _ObservedSource(2)
    produced = []

    def produce(inputs):
        produced.append(inputs["radio"]["x"].item())
        return torch.tensor(produced[-1])

    loader = EncoderPrefetchLoader(
        source=source,
        encoder_name=ENCODER,
        feature_producer=produce,
        depth=1,
        stream=fake_cuda.producer,
    )
    loader.start()
    _wait_until(lambda: len(loader._ready) == 1)

    assert source.position == 2
    assert produced == [0]

    for expected in (0, 1):
        batch = next(loader)
        assert batch[PREFETCHED_FEATURES_KEY][ENCODER].item() == expected
    with pytest.raises(StopIteration):
        next(loader)
    loader.close()


def test_prefetch_keeps_input_ids_on_cpu_path(fake_cuda, monkeypatch):
    input_ids = torch.tensor([[511, 1]])
    encoder_inputs = {"radio": {"x": torch.tensor(0)}}
    moved = []

    def record_move(value):
        moved.append(value)
        return value

    monkeypatch.setattr(encoder_prefetch, "move_batch_to_cuda", record_move)
    loader = EncoderPrefetchLoader(
        source=[{"input_ids": input_ids, "modality_inputs": {ENCODER: encoder_inputs}}],
        encoder_name=ENCODER,
        feature_producer=lambda _inputs: torch.tensor(0),
        depth=1,
        stream=fake_cuda.producer,
    )
    loader.start()
    _wait_until(lambda: len(loader._ready) == 1)

    batch = next(loader)
    loader.close()

    assert batch["input_ids"] is input_ids
    assert len(moved) == 1
    assert moved[0] is encoder_inputs


def test_debug_logs_prefetch_timing_and_queue_state(fake_cuda, caplog):
    module_logger_level = encoder_prefetch.logger.level
    caplog.set_level("INFO", logger=f"{encoder_prefetch.__name__}.debug")
    loader = EncoderPrefetchLoader(
        source=_Source(2),
        encoder_name=ENCODER,
        feature_producer=lambda inputs: torch.tensor(inputs["radio"]["x"].item()),
        depth=1,
        stream=fake_cuda.producer,
        debug=True,
    )
    loader.start()
    _wait_until(lambda: len(loader._ready) == 1)

    first = next(loader)
    with first.pop(PROJECTION_TIMER_KEY):
        pass
    _wait_until(lambda: len(loader._ready) == 1)
    second = next(loader)
    with second.pop(PROJECTION_TIMER_KEY):
        pass
    with pytest.raises(StopIteration):
        next(loader)
    loader.close()

    assert encoder_prefetch.logger.level == module_logger_level
    assert "encoder-prefetch-debug consumer batch=0 ready_at_request=1/1" in caplog.text
    assert "encoder-prefetch-debug producer batch=1" in caplog.text
    assert "encoder-prefetch-debug projection batch=0 projection_ms=1.250" in caplog.text


def test_producer_failure_is_terminal_and_preserves_ready_fifo(fake_cuda):
    class _FailingSource(_Source):
        def __next__(self):
            if self.position == 2:
                raise ValueError("boom")
            return super().__next__()

    loader = EncoderPrefetchLoader(
        source=_FailingSource(8),
        encoder_name=ENCODER,
        feature_producer=lambda inputs: torch.tensor(inputs["radio"]["x"].item()),
        depth=2,
        stream=fake_cuda.producer,
    )
    loader.start()
    _wait_until(lambda: len(loader._ready) == 2)

    for expected in (0, 1):
        batch = next(loader)
        assert batch[PREFETCHED_FEATURES_KEY][ENCODER].item() == expected
    with pytest.raises(RuntimeError, match="producer failed") as exc_info:
        next(loader)
    assert isinstance(exc_info.value.__cause__, ValueError)
    loader.close()


def test_close_does_not_raise_when_worker_is_stuck(fake_cuda, caplog):
    entered = threading.Event()
    release = threading.Event()

    def produce(_inputs):
        entered.set()
        release.wait(timeout=2)
        return torch.tensor(1)

    loader = EncoderPrefetchLoader(
        source=_Source(1),
        encoder_name=ENCODER,
        feature_producer=produce,
        depth=1,
        stream=fake_cuda.producer,
        worker_join_timeout_s=0.01,
    )
    loader.start()
    assert entered.wait(timeout=1)

    loader.close()

    assert "worker did not stop" in caplog.text
    assert fake_cuda.producer.synchronize_calls == 0
    release.set()
    assert loader._worker is not None
    loader._worker.join(timeout=1)
    assert not loader._worker.is_alive()


def test_forward_step_projects_prefetched_features_inside_debug_timer(monkeypatch):
    events = []

    class _Lease:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *_args):
            events.append("exit")

    class _Model:
        role = SimpleNamespace(modality_module_names=(ENCODER,))

        def _forward_encoders(self, input_ids, modality_inputs, input_tensors):
            assert modality_inputs is None
            events.append("project")
            return input_tensors

    features = {ENCODER: torch.ones(2, 4)}
    batch = {
        "input_ids": torch.tensor([[511]]),
        PREFETCHED_FEATURES_KEY: features,
        PROJECTION_TIMER_KEY: _Lease(),
    }
    monkeypatch.setattr(
        mimo_step,
        "move_batch_to_cuda",
        lambda _value: pytest.fail("prefetched batches are already CUDA-resident"),
    )

    output, _ = mimo_step.mimo_forward_step(iter([batch]), _Model())

    assert output is features
    assert events == ["enter", "project", "exit"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_cuda_pending_handoff_waits_for_encode():
    torch.cuda.set_device(0)
    read_ahead_started = threading.Event()
    release_read_ahead = threading.Event()
    backbone = nn.Linear(4, 4, bias=False, device="cuda")

    class _BlockingSource:
        def __init__(self):
            self.position = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.position == 1:
                read_ahead_started.set()
                release_read_ahead.wait(timeout=2)
                raise StopIteration
            self.position += 1
            return {
                "input_ids": torch.tensor([[511]]),
                "modality_inputs": {ENCODER: {"radio": {"x": torch.ones(32, 4)}}},
            }

    def produce(inputs):
        with torch.no_grad():
            return backbone(inputs["radio"]["x"])

    loader = EncoderPrefetchLoader(
        source=_BlockingSource(), encoder_name=ENCODER, feature_producer=produce, depth=1
    )
    loader.start()
    assert read_ahead_started.wait(timeout=1)

    try:
        assert len(loader._ready) == 0
        batch = next(loader)
        expected = backbone(torch.ones(32, 4, device="cuda"))
        torch.testing.assert_close(batch[PREFETCHED_FEATURES_KEY][ENCODER], expected)
    finally:
        release_read_ahead.set()
        loader.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_cuda_handoff_and_projection_gradients():
    torch.cuda.set_device(0)
    backbone = nn.Linear(4, 4, bias=False, device="cuda")
    projection = nn.Linear(4, 1, bias=False, device="cuda")
    backbone.requires_grad_(False)
    source = [
        {
            "input_ids": torch.tensor([[sequence]]),
            "modality_inputs": {ENCODER: {"radio": {"x": torch.full((32, 4), float(sequence))}}},
        }
        for sequence in range(8)
    ]

    def produce(inputs):
        with torch.no_grad():
            return backbone(inputs["radio"]["x"])

    loader = EncoderPrefetchLoader(
        source=source, encoder_name=ENCODER, feature_producer=produce, depth=2
    )
    loader.start()
    losses = []
    for sequence in range(8):
        batch = next(loader)
        assert not batch["input_ids"].is_cuda
        features = batch[PREFETCHED_FEATURES_KEY][ENCODER]
        reference = backbone(torch.full((32, 4), float(sequence), device="cuda"))
        torch.testing.assert_close(features, reference)
        losses.append(projection(features).sum())
    torch.stack(losses).sum().backward()
    loader.close()

    assert projection.weight.grad is not None
    assert torch.isfinite(projection.weight.grad).all()
    assert all(parameter.grad is None for parameter in backbone.parameters())
