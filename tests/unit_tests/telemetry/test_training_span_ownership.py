# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Production span helpers preserve trainer timing and exception behavior."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from opentelemetry import context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode


@pytest.fixture
def span_runtime(monkeypatch, telemetry_module):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry_module, "is_enabled", lambda group: True)
    try:
        yield telemetry_module, provider.get_tracer("training-span-ownership"), exporter
    finally:
        provider.shutdown()


@pytest.mark.parametrize("raises", [False, True])
def test_report_preserves_parent_extent_and_exception_status(span_runtime, raises):
    telemetry, tracer, exporter = span_runtime
    failure = RuntimeError("report failed")
    with tracer.start_as_current_span("parent"):
        try:
            with telemetry.training_report_span(SimpleNamespace(tracer=tracer)):
                with tracer.start_as_current_span("report-work"):
                    pass
                if raises:
                    raise failure
        except RuntimeError as exc:
            assert raises and exc is failure
        with tracer.start_as_current_span("after-report"):
            pass
    spans = {span.name: span for span in exporter.get_finished_spans()}
    report = spans[telemetry.SPAN_TRAINING_ITERATION_REPORT]
    assert report.parent.span_id == spans["parent"].context.span_id
    assert spans["report-work"].parent.span_id == report.context.span_id
    assert spans["after-report"].parent.span_id == spans["parent"].context.span_id
    assert report.status.status_code == StatusCode.UNSET
    assert not report.events


def test_report_preserves_detach_failure_policy(monkeypatch, telemetry_module):
    events = []
    span = SimpleNamespace(end=lambda: events.append("end"))
    tracer = SimpleNamespace(start_span=lambda name: span)
    monkeypatch.setattr(telemetry_module, "is_enabled", lambda group: True)
    monkeypatch.setattr(context, "attach", lambda value: "token")
    failure = RuntimeError("detach failed")

    def detach(token):
        assert token == "token"
        raise failure

    monkeypatch.setattr(context, "detach", detach)
    with pytest.raises(RuntimeError) as caught:
        with telemetry_module.training_report_span(SimpleNamespace(tracer=tracer)):
            events.append("body")
    assert caught.value is failure
    assert events == ["body"]


def test_step_capture_and_lazy_attribute_factory_timing(monkeypatch, telemetry_module):
    telemetry = telemetry_module
    enabled = [True]
    calls = []
    monkeypatch.setattr(telemetry, "is_enabled", lambda group: enabled[0])
    contexts = []

    def span_cm(name, **kwargs):
        calls.append((name, kwargs))
        result = object()
        contexts.append(result)
        return result

    monkeypatch.setattr(telemetry, "span_cm", span_cm)
    handle = SimpleNamespace(tracer="captured")
    spans = telemetry.TrainingStepSpans(handle)
    handle.tracer = "later"

    def num_microbatches():
        calls.append("num-microbatches")
        return 7

    result = spans.forward_backward(num_microbatches)
    assert result is contexts[0]
    assert calls == [
        "num-microbatches",
        (telemetry.SPAN_TRAINING_FORWARD_BACKWARD, {"tracer": "captured", "num_microbatches": 7}),
    ]
    spans.optimizer()
    assert calls[-1] == (telemetry.SPAN_TRAINING_OPTIMIZER_STEP, {"tracer": "captured"})
    enabled[0] = False
    with spans.forward_backward(lambda: pytest.fail("disabled attribute read")) as span:
        assert span is None
    with spans.optimizer() as span:
        assert span is None
    assert len(calls) == 3


def test_disabled_helpers_do_not_inspect_handles_or_microbatches(monkeypatch, telemetry_module):
    monkeypatch.setattr(telemetry_module, "is_enabled", lambda group: False)
    spans = telemetry_module.TrainingStepSpans(None)
    with spans.forward_backward(lambda: pytest.fail("disabled attribute read")) as span:
        assert span is None
    with spans.optimizer() as span:
        assert span is None
    with telemetry_module.training_report_span(None) as span:
        assert span is None


@pytest.mark.parametrize("method", ["forward_backward", "optimizer"])
def test_step_spans_keep_lens_exception_recording(span_runtime, method):
    telemetry, tracer, exporter = span_runtime
    spans = telemetry.TrainingStepSpans(SimpleNamespace(tracer=tracer))
    cm = spans.forward_backward(lambda: 3) if method == "forward_backward" else spans.optimizer()
    failure = RuntimeError("step failed")
    with pytest.raises(RuntimeError) as caught:
        with cm:
            raise failure
    assert caught.value is failure
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert [event.name for event in span.events] == ["exception"]


def test_training_report_and_step_have_no_raw_tracer_operations():
    path = Path(__file__).resolve().parents[3] / "megatron/training/training.py"
    tree = ast.parse(path.read_text())
    for function in tree.body:
        if isinstance(function, ast.FunctionDef) and function.name in {"train", "train_step"}:
            assert not any(
                isinstance(node, ast.Attribute)
                and node.attr in {"tracer", "start_span", "span_cm", "attach", "detach"}
                for node in ast.walk(function)
            )
