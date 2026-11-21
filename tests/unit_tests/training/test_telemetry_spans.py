# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-only tests for the training telemetry span lifecycle."""

import signal
from types import SimpleNamespace

import pytest
from opentelemetry import trace

from megatron.training import telemetry_spans


class FakeSpan(trace.NonRecordingSpan):
    """Minimal span that records lifecycle operations."""

    def __init__(self, span_id: int = 1) -> None:
        self.attributes: dict[str, object] = {}
        self.end_calls: list[int | None] = []
        self.context = trace.SpanContext(trace_id=span_id, span_id=span_id, is_remote=False)
        super().__init__(self.context)

    def set_attribute(self, name, value):
        self.attributes[name] = value

    def get_span_context(self):
        return self.context

    def end(self, end_time=None):
        self.end_calls.append(end_time)


class FakeTracer:
    """Tracer that records created spans."""

    def __init__(self):
        self.calls = []

    def start_span(self, name, **kwargs):
        span = FakeSpan(span_id=len(self.calls) + 1)
        self.calls.append((name, kwargs, span))
        return span


class FakeTelemetry:
    """Exporting telemetry handle with a fake tracer."""

    def __init__(self):
        self.is_exporting = True
        self.tracer = FakeTracer()
        self.shutdown_calls = 0

    def shutdown(self):
        self.shutdown_calls += 1


@pytest.fixture(autouse=True)
def reset_telemetry_state(monkeypatch):
    """Keep the module-global lifecycle state isolated between tests."""
    monkeypatch.setattr(telemetry_spans, "_otel_startup_span", None)
    monkeypatch.setattr(telemetry_spans, "_otel_startup_span_ctx", None)
    monkeypatch.setattr(telemetry_spans, "_otel_ctx_module", None)
    monkeypatch.setattr(telemetry_spans, "_otel_startup_ctx_token", None)
    monkeypatch.setattr(telemetry_spans, "_otel_interval_span", None)
    monkeypatch.setattr(telemetry_spans, "_otel_interval_ctx_token", None)
    monkeypatch.setattr(telemetry_spans, "_otel_trace_interval_step", 0)
    monkeypatch.setattr(telemetry_spans, "_otel_shutdown_done", False)
    monkeypatch.setattr(telemetry_spans, "_otel_exit_hooks_installed", False)


def test_startup_span_opens_and_closes_once(monkeypatch):
    telemetry = FakeTelemetry()
    monkeypatch.setattr(telemetry_spans, "get_telemetry", lambda: telemetry)
    monkeypatch.setattr(telemetry_spans, "_otel_sg_enabled", lambda group: group == "job")
    monkeypatch.setattr(
        telemetry_spans,
        "_otel_safe_set_attrs",
        lambda span, attributes: span.attributes.update(attributes),
    )

    telemetry_spans._start_otel_job_spans(
        model_type="encoder_or_decoder", program_start=10.0, startup_timestamps={}
    )
    telemetry_spans._end_otel_startup_span()
    telemetry_spans._end_otel_startup_span()

    assert [name for name, _, _ in telemetry.tracer.calls] == ["megatron.startup"]
    startup_span = telemetry.tracer.calls[0][2]
    assert startup_span.attributes == {
        "is_goodput_span": True,
        "megatron.model_type": "encoder_or_decoder",
    }
    assert startup_span.end_calls == [None]


def test_interval_reroots_at_save_boundaries(monkeypatch):
    telemetry = FakeTelemetry()
    startup_context = FakeSpan(span_id=99).get_span_context()
    previous_span = trace.get_current_span()
    monkeypatch.setattr(telemetry_spans, "get_telemetry", lambda: telemetry)
    monkeypatch.setattr(
        telemetry_spans,
        "get_run_config",
        lambda: SimpleNamespace(checkpoint=SimpleNamespace(save_interval=2)),
    )
    monkeypatch.setattr(telemetry_spans, "_otel_sg_enabled", lambda group: group == "job")
    monkeypatch.setattr(telemetry_spans, "_otel_startup_span_ctx", startup_context)

    telemetry_spans._start_otel_train_span()
    try:
        for step in range(5):
            telemetry_spans._maybe_reroot_otel_interval()

            assert len(telemetry.tracer.calls) == step // 2 + 1
            current = telemetry.tracer.calls[-1][2]
            assert telemetry_spans._otel_interval_span is current
            assert trace.get_current_span() is current
            assert current.end_calls == []
            assert all(span.end_calls == [None] for _, _, span in telemetry.tracer.calls[:-1])

        assert [name for name, _, _ in telemetry.tracer.calls] == ["megatron.train"] * 3
        for index, (_, kwargs, span) in enumerate(telemetry.tracer.calls):
            expected_context = (
                startup_context if index == 0 else telemetry.tracer.calls[index - 1][2].context
            )
            assert len(kwargs["links"]) == 1
            assert kwargs["links"][0].context is expected_context
            assert trace.get_current_span(kwargs["context"]) is trace.INVALID_SPAN
            assert span.attributes == {"is_goodput_span": True}
    finally:
        telemetry_spans._end_otel_train_span()

    assert all(span.end_calls == [None] for _, _, span in telemetry.tracer.calls)
    assert telemetry_spans._otel_interval_span is None
    assert telemetry_spans._otel_interval_ctx_token is None
    assert trace.get_current_span() is previous_span


def test_atexit_hook_is_installed_once_and_closes_spans(monkeypatch):
    telemetry = FakeTelemetry()
    registered = []
    signal_handlers = []
    startup_span = FakeSpan()
    interval_span = FakeSpan()
    monkeypatch.setattr(telemetry_spans, "get_telemetry", lambda: telemetry)
    monkeypatch.setattr(telemetry_spans.atexit, "register", registered.append)
    monkeypatch.setattr(telemetry_spans.signal, "getsignal", lambda signum: signal.SIG_IGN)
    monkeypatch.setattr(
        telemetry_spans.signal,
        "signal",
        lambda signum, handler: signal_handlers.append((signum, handler)),
    )
    monkeypatch.setattr(telemetry_spans, "_otel_startup_span", startup_span)
    monkeypatch.setattr(telemetry_spans, "_otel_interval_span", interval_span)

    telemetry_spans._install_otel_exit_hooks(graceful_drain=False)
    telemetry_spans._install_otel_exit_hooks(graceful_drain=False)
    registered[0]()  # Simulate interpreter teardown after an unhandled exception.
    registered[0]()

    assert registered == [telemetry_spans._end_otel_job_spans]
    assert len(signal_handlers) == 1
    assert startup_span.end_calls == [None]
    assert interval_span.end_calls == [None]
    assert telemetry.shutdown_calls == 1


def test_hard_sigterm_closes_spans_only_once(monkeypatch):
    telemetry = FakeTelemetry()
    signal_handlers = []
    startup_span = FakeSpan()
    interval_span = FakeSpan()
    monkeypatch.setattr(telemetry_spans, "get_telemetry", lambda: telemetry)
    monkeypatch.setattr(telemetry_spans.atexit, "register", lambda callback: None)
    monkeypatch.setattr(telemetry_spans.signal, "getsignal", lambda signum: signal.SIG_IGN)
    monkeypatch.setattr(
        telemetry_spans.signal,
        "signal",
        lambda signum, handler: signal_handlers.append((signum, handler)),
    )
    monkeypatch.setattr(telemetry_spans, "_otel_startup_span", startup_span)
    monkeypatch.setattr(telemetry_spans, "_otel_interval_span", interval_span)

    telemetry_spans._install_otel_exit_hooks(graceful_drain=False)
    sigterm_handler = signal_handlers[0][1]
    sigterm_handler(signal.SIGTERM, None)
    sigterm_handler(signal.SIGTERM, None)

    assert startup_span.end_calls == [None]
    assert interval_span.end_calls == [None]
    assert telemetry_spans._otel_startup_span is None
    assert telemetry_spans._otel_interval_span is None
    assert telemetry.shutdown_calls == 1


def test_graceful_sigterm_flushes_without_closing(monkeypatch):
    telemetry = FakeTelemetry()
    signal_handlers = []
    flush_calls = []
    startup_span = FakeSpan()
    interval_span = FakeSpan()
    monkeypatch.setattr(telemetry_spans, "get_telemetry", lambda: telemetry)
    monkeypatch.setattr(telemetry_spans.atexit, "register", lambda callback: None)
    monkeypatch.setattr(telemetry_spans.signal, "getsignal", lambda signum: signal.SIG_IGN)
    monkeypatch.setattr(
        telemetry_spans.signal,
        "signal",
        lambda signum, handler: signal_handlers.append((signum, handler)),
    )
    monkeypatch.setattr(
        trace,
        "get_tracer_provider",
        lambda: SimpleNamespace(force_flush=lambda: flush_calls.append("flushed")),
    )
    monkeypatch.setattr(telemetry_spans, "_otel_startup_span", startup_span)
    monkeypatch.setattr(telemetry_spans, "_otel_interval_span", interval_span)

    telemetry_spans._install_otel_exit_hooks(graceful_drain=True)
    sigterm_handler = signal_handlers[0][1]
    sigterm_handler(signal.SIGTERM, None)
    sigterm_handler(signal.SIGTERM, None)

    assert flush_calls == ["flushed"]
    assert startup_span.end_calls == []
    assert interval_span.end_calls == []
    assert telemetry_spans._otel_startup_span is startup_span
    assert telemetry_spans._otel_interval_span is interval_span
    assert telemetry.shutdown_calls == 0
