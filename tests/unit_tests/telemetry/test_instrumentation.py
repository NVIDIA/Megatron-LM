# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Integration-style tests: verify spans emitted at correct training boundaries.

These tests use InMemorySpanExporter + SimpleSpanProcessor to capture spans
without a real OTLP collector, and call lightly-mocked versions of the
training/checkpoint/evaluation boundaries.
"""

import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from nemo.lens.config import NemoLensConfig
from nemo.lens.handle import TelemetryHandle, setup_telemetry
from nemo.lens.helpers import span_cm


@pytest.fixture
def real_tracer():
    """Provide a real tracer + exporter for span capture tests."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    return tracer, exporter


class TestSpanNames:
    def test_pretrain_span_name(self, real_tracer):
        tracer, exporter = real_tracer
        with span_cm("megatron.pretrain", tracer=tracer, model_type="GPT"):
            pass
        spans = exporter.get_finished_spans()
        assert any(s.name == "megatron.pretrain" for s in spans)

    def test_train_span_name(self, real_tracer):
        tracer, exporter = real_tracer
        with span_cm("megatron.train", tracer=tracer, train_iters=100):
            pass
        spans = exporter.get_finished_spans()
        assert any(s.name == "megatron.train" for s in spans)

    def test_save_checkpoint_span_name(self, real_tracer):
        tracer, exporter = real_tracer
        with span_cm("megatron.save_checkpoint", tracer=tracer, **{"megatron.iteration": 100}):
            pass
        spans = exporter.get_finished_spans()
        assert any(s.name == "megatron.save_checkpoint" for s in spans)

    def test_evaluate_span_name(self, real_tracer):
        tracer, exporter = real_tracer
        with span_cm("megatron.evaluate", tracer=tracer, eval_iters=10):
            pass
        spans = exporter.get_finished_spans()
        assert any(s.name == "megatron.evaluate" for s in spans)

    def test_train_step_span_name(self, real_tracer):
        tracer, exporter = real_tracer
        with span_cm("megatron.train_step", tracer=tracer, **{"megatron.iteration": 5}):
            pass
        spans = exporter.get_finished_spans()
        assert any(s.name == "megatron.train_step" for s in spans)

    def test_inference_request_span_name(self, real_tracer):
        tracer, exporter = real_tracer
        with span_cm("megatron.inference.request", tracer=tracer):
            pass
        spans = exporter.get_finished_spans()
        assert any(s.name == "megatron.inference.request" for s in spans)


class TestSpanAttributes:
    def test_pretrain_span_has_model_type(self, real_tracer):
        tracer, exporter = real_tracer
        with span_cm("megatron.pretrain", tracer=tracer, model_type="GPT"):
            pass
        span = exporter.get_finished_spans()[0]
        assert span.attributes.get('model_type') == 'GPT'

    def test_train_step_span_has_iteration(self, real_tracer):
        tracer, exporter = real_tracer
        with span_cm("megatron.train_step", tracer=tracer, **{"megatron.iteration": 42}):
            pass
        span = exporter.get_finished_spans()[0]
        assert span.attributes.get('megatron.iteration') == 42

    def test_save_checkpoint_span_has_iteration(self, real_tracer):
        tracer, exporter = real_tracer
        with span_cm("megatron.save_checkpoint", tracer=tracer, **{"megatron.iteration": 500}):
            pass
        span = exporter.get_finished_spans()[0]
        assert span.attributes.get('megatron.iteration') == 500

    def test_evaluate_span_has_eval_iters(self, real_tracer):
        tracer, exporter = real_tracer
        with span_cm("megatron.evaluate", tracer=tracer, eval_iters=20):
            pass
        span = exporter.get_finished_spans()[0]
        assert span.attributes.get('eval_iters') == 20


class TestParentChildRelationships:
    def test_train_step_inside_train(self, real_tracer):
        tracer, exporter = real_tracer
        with span_cm("megatron.train", tracer=tracer):
            with span_cm("megatron.train_step", tracer=tracer, **{"megatron.iteration": 1}):
                pass

        spans = exporter.get_finished_spans()
        train_span = next(s for s in spans if s.name == "megatron.train")
        step_span = next(s for s in spans if s.name == "megatron.train_step")

        # train_step's parent should be train
        assert step_span.parent is not None
        assert step_span.parent.span_id == train_span.context.span_id

    def test_pretrain_is_root(self, real_tracer):
        tracer, exporter = real_tracer
        with span_cm("megatron.pretrain", tracer=tracer):
            with span_cm("megatron.train", tracer=tracer):
                pass

        spans = exporter.get_finished_spans()
        pretrain_span = next(s for s in spans if s.name == "megatron.pretrain")
        train_span = next(s for s in spans if s.name == "megatron.train")

        assert pretrain_span.parent is None
        assert train_span.parent.span_id == pretrain_span.context.span_id

    def test_full_hierarchy(self, real_tracer):
        tracer, exporter = real_tracer
        with span_cm("megatron.pretrain", tracer=tracer, model_type="GPT"):
            with span_cm("megatron.train", tracer=tracer, train_iters=3):
                for i in range(3):
                    with span_cm("megatron.train_step", tracer=tracer, **{"megatron.iteration": i}):
                        pass
            with span_cm("megatron.evaluate", tracer=tracer, eval_iters=5):
                pass

        spans = exporter.get_finished_spans()
        span_names = [s.name for s in spans]
        assert "megatron.pretrain" in span_names
        assert "megatron.train" in span_names
        assert span_names.count("megatron.train_step") == 3
        assert "megatron.evaluate" in span_names


class TestDisabledTelemetry:
    def test_disabled_telemetry_no_spans(self):
        """When telemetry is disabled, InMemorySpanExporter should see 0 spans."""
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        # Simulate disabled: use a no-op tracer (not this provider's tracer)
        from opentelemetry.trace import NoOpTracerProvider
        noop_tracer = NoOpTracerProvider().get_tracer("test")

        with span_cm("megatron.pretrain", tracer=noop_tracer):
            with span_cm("megatron.train", tracer=noop_tracer):
                pass

        # The in-memory exporter for the real provider should have 0 spans
        assert len(exporter.get_finished_spans()) == 0

    def test_setup_telemetry_disabled_returns_handle(self):
        cfg = NemoLensConfig(enabled=False)
        handle = setup_telemetry(cfg, rank=0, world_size=1)
        assert isinstance(handle, TelemetryHandle)

    def test_setup_telemetry_disabled_noop_tracing(self):
        cfg = NemoLensConfig(enabled=False)
        handle = setup_telemetry(cfg, rank=0, world_size=1)
        with handle.tracer.start_as_current_span("test") as span:
            pass
