# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for OTel context propagation helpers."""

import pytest

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry import propagate
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.baggage.propagation import W3CBaggagePropagator

from nemo.lens.propagation import extract_context, inject_context


@pytest.fixture(autouse=True)
def setup_w3c_propagator():
    """Install W3C TraceContext + Baggage propagator for all tests."""
    propagate.set_global_textmap(
        CompositePropagator([TraceContextTextMapPropagator(), W3CBaggagePropagator()])
    )
    yield
    # Reset to default
    propagate.set_global_textmap(TraceContextTextMapPropagator())


@pytest.fixture
def tracer():
    """A real tracer backed by an in-memory exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


class TestInjectContext:
    def test_inject_adds_traceparent_key(self, tracer):
        t, _ = tracer
        with t.start_as_current_span("root"):
            carrier = {}
            inject_context(carrier)
        assert 'traceparent' in carrier

    def test_inject_traceparent_format(self, tracer):
        t, _ = tracer
        with t.start_as_current_span("root"):
            carrier = {}
            inject_context(carrier)
        # W3C format: 00-<32 hex>-<16 hex>-<2 hex>
        tp = carrier['traceparent']
        parts = tp.split('-')
        assert len(parts) == 4
        assert parts[0] == '00'
        assert len(parts[1]) == 32  # trace_id
        assert len(parts[2]) == 16  # span_id
        assert len(parts[3]) == 2   # flags

    def test_inject_with_no_active_span(self):
        carrier = {}
        inject_context(carrier)
        # No active span: carrier may be empty or minimal — should not raise
        # (The W3C propagator does not inject if there's no active span)


class TestExtractContext:
    def test_extract_from_valid_carrier(self, tracer):
        t, _ = tracer
        with t.start_as_current_span("root") as root_span:
            carrier = {}
            inject_context(carrier)
            original_trace_id = root_span.get_span_context().trace_id

        ctx = extract_context(carrier)
        assert ctx is not None

    def test_roundtrip_preserves_trace_id(self, tracer):
        t, _ = tracer
        with t.start_as_current_span("root") as root_span:
            carrier = {}
            inject_context(carrier)
            original_trace_id = root_span.get_span_context().trace_id

        ctx = extract_context(carrier)
        span_ctx = trace.get_current_span(ctx).get_span_context()
        assert span_ctx.trace_id == original_trace_id

    def test_roundtrip_preserves_span_id(self, tracer):
        t, _ = tracer
        with t.start_as_current_span("root") as root_span:
            carrier = {}
            inject_context(carrier)
            original_span_id = root_span.get_span_context().span_id

        ctx = extract_context(carrier)
        span_ctx = trace.get_current_span(ctx).get_span_context()
        assert span_ctx.span_id == original_span_id

    def test_extract_from_empty_carrier_returns_context(self):
        ctx = extract_context({})
        # Should not raise; returns an empty context
        assert ctx is not None

    def test_extract_from_invalid_traceparent(self):
        carrier = {'traceparent': 'invalid-value'}
        # Should not raise; returns empty/invalid context
        ctx = extract_context(carrier)
        assert ctx is not None

    def test_child_span_linked_to_extracted_context(self, tracer):
        t, exporter = tracer
        with t.start_as_current_span("parent") as parent:
            carrier = {}
            inject_context(carrier)
            parent_trace_id = parent.get_span_context().trace_id
            parent_span_id = parent.get_span_context().span_id

        # Extract in a "new process"
        ctx = extract_context(carrier)
        with t.start_as_current_span("child", context=ctx) as child:
            child_ctx = child.get_span_context()

        assert child_ctx.trace_id == parent_trace_id
        spans = exporter.get_finished_spans()
        child_span = next(s for s in spans if s.name == "child")
        assert child_span.parent.span_id == parent_span_id
