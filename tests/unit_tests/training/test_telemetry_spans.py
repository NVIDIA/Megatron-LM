# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-only tests for the training telemetry span lifecycle."""

import signal
from types import SimpleNamespace

import pytest

from megatron.training import telemetry_spans


class FakeSpan:
    """Minimal span that records lifecycle operations."""

    def __init__(self):
        self.attributes = {}
        self.end_calls = []
        self.context = object()

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
        span = FakeSpan()
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
    reroot_steps = []
    monkeypatch.setattr(telemetry_spans, "get_args", lambda: SimpleNamespace(save_interval=2))
    monkeypatch.setattr(telemetry_spans, "_otel_sg_enabled", lambda group: group == "job")
    monkeypatch.setattr(
        telemetry_spans,
        "_reroot_otel_interval",
        lambda: reroot_steps.append(telemetry_spans._otel_trace_interval_step),
    )

    telemetry_spans._start_otel_train_span()
    for _ in range(5):
        telemetry_spans._maybe_reroot_otel_interval()

    assert reroot_steps == [0, 2, 4]


def test_atexit_hook_is_installed_once_and_closes_spans(monkeypatch):
    telemetry = FakeTelemetry()
    registered = []
    signal_handlers = []
    startup_span = FakeSpan()
    interval_span = FakeSpan()
    monkeypatch.setattr(telemetry_spans, "get_telemetry", lambda: telemetry)
    monkeypatch.setattr(
        telemetry_spans, "get_args", lambda: SimpleNamespace(exit_signal_handler=False)
    )
    monkeypatch.setattr(telemetry_spans.atexit, "register", registered.append)
    monkeypatch.setattr(telemetry_spans.signal, "getsignal", lambda signum: signal.SIG_IGN)
    monkeypatch.setattr(
        telemetry_spans.signal,
        "signal",
        lambda signum, handler: signal_handlers.append((signum, handler)),
    )
    monkeypatch.setattr(telemetry_spans, "_otel_startup_span", startup_span)
    monkeypatch.setattr(telemetry_spans, "_otel_interval_span", interval_span)

    telemetry_spans._install_otel_exit_hooks()
    telemetry_spans._install_otel_exit_hooks()
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
    close_calls = []
    monkeypatch.setattr(telemetry_spans, "get_telemetry", lambda: telemetry)
    monkeypatch.setattr(
        telemetry_spans, "get_args", lambda: SimpleNamespace(exit_signal_handler=False)
    )
    monkeypatch.setattr(telemetry_spans.atexit, "register", lambda callback: None)
    monkeypatch.setattr(telemetry_spans.signal, "getsignal", lambda signum: signal.SIG_IGN)
    monkeypatch.setattr(
        telemetry_spans.signal,
        "signal",
        lambda signum, handler: signal_handlers.append((signum, handler)),
    )
    monkeypatch.setattr(
        telemetry_spans, "_end_otel_job_spans", lambda: close_calls.append("closed")
    )

    telemetry_spans._install_otel_exit_hooks()
    sigterm_handler = signal_handlers[0][1]
    sigterm_handler(signal.SIGTERM, None)
    sigterm_handler(signal.SIGTERM, None)

    assert close_calls == ["closed"]


def test_graceful_sigterm_flushes_without_closing(monkeypatch):
    telemetry = FakeTelemetry()
    signal_handlers = []
    flush_calls = []
    close_calls = []
    monkeypatch.setattr(telemetry_spans, "get_telemetry", lambda: telemetry)
    monkeypatch.setattr(
        telemetry_spans, "get_args", lambda: SimpleNamespace(exit_signal_handler=True)
    )
    monkeypatch.setattr(telemetry_spans.atexit, "register", lambda callback: None)
    monkeypatch.setattr(telemetry_spans.signal, "getsignal", lambda signum: signal.SIG_IGN)
    monkeypatch.setattr(
        telemetry_spans.signal,
        "signal",
        lambda signum, handler: signal_handlers.append((signum, handler)),
    )
    monkeypatch.setattr(telemetry_spans, "_force_flush_otel", lambda: flush_calls.append("flushed"))
    monkeypatch.setattr(
        telemetry_spans, "_end_otel_job_spans", lambda: close_calls.append("closed")
    )

    telemetry_spans._install_otel_exit_hooks()
    signal_handlers[0][1](signal.SIGTERM, None)

    assert flush_calls == ["flushed"]
    assert close_calls == []
