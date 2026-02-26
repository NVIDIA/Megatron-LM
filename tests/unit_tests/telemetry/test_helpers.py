# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for telemetry helpers: span_cm, safe_set_span_attributes, redact_value."""

import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry import trace

from nemo.lens.helpers import (
    DEFAULT_REDACT_KEYS,
    managed_span,
    redact_value,
    safe_set_span_attributes,
    span_cm,
    trace_fn,
)


@pytest.fixture
def in_memory_tracer():
    """Return a tracer backed by an InMemorySpanExporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    return tracer, exporter


class TestRedactValue:
    def test_non_redacted_key_returns_value(self):
        assert redact_value('loss', 'hello', DEFAULT_REDACT_KEYS) == 'hello'

    def test_redacted_key_returns_placeholder(self):
        assert redact_value('prompt', 'my secret', DEFAULT_REDACT_KEYS) == '[REDACTED]'

    def test_all_default_redact_keys(self):
        for key in DEFAULT_REDACT_KEYS:
            result = redact_value(key, 'sensitive', DEFAULT_REDACT_KEYS)
            assert result == '[REDACTED]', f"Expected redaction for key {key!r}"

    def test_custom_redact_keys(self):
        custom = frozenset({'custom_key'})
        assert redact_value('custom_key', 'value', custom) == '[REDACTED]'
        assert redact_value('other_key', 'value', custom) == 'value'

    def test_empty_redact_keys(self):
        assert redact_value('prompt', 'value', frozenset()) == 'value'


class TestSafeSetSpanAttributes:
    def test_sets_string_attribute(self, in_memory_tracer):
        tracer, exporter = in_memory_tracer
        with tracer.start_as_current_span("test") as span:
            safe_set_span_attributes(span, {'model_type': 'gpt'})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get('model_type') == 'gpt'

    def test_sets_int_attribute(self, in_memory_tracer):
        tracer, exporter = in_memory_tracer
        with tracer.start_as_current_span("test") as span:
            safe_set_span_attributes(span, {'iteration': 42})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get('iteration') == 42

    def test_sets_float_attribute(self, in_memory_tracer):
        tracer, exporter = in_memory_tracer
        with tracer.start_as_current_span("test") as span:
            safe_set_span_attributes(span, {'loss': 1.5})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get('loss') == 1.5

    def test_sets_bool_attribute(self, in_memory_tracer):
        tracer, exporter = in_memory_tracer
        with tracer.start_as_current_span("test") as span:
            safe_set_span_attributes(span, {'success': True})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get('success') is True

    def test_skips_tensor_value(self, in_memory_tracer):
        tracer, exporter = in_memory_tracer

        class FakeTensor:
            pass

        with tracer.start_as_current_span("test") as span:
            safe_set_span_attributes(span, {'tensor_val': FakeTensor(), 'loss': 0.5})
        spans = exporter.get_finished_spans()
        # tensor_val should be skipped; loss should be set
        assert 'tensor_val' not in spans[0].attributes
        assert spans[0].attributes.get('loss') == 0.5

    def test_skips_dict_value(self, in_memory_tracer):
        tracer, exporter = in_memory_tracer
        with tracer.start_as_current_span("test") as span:
            safe_set_span_attributes(span, {'nested': {'a': 1}, 'loss': 1.0})
        spans = exporter.get_finished_spans()
        assert 'nested' not in spans[0].attributes
        assert spans[0].attributes.get('loss') == 1.0

    def test_redacts_sensitive_string(self, in_memory_tracer):
        tracer, exporter = in_memory_tracer
        with tracer.start_as_current_span("test") as span:
            safe_set_span_attributes(span, {'prompt': 'my secret prompt'})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get('prompt') == '[REDACTED]'

    def test_no_error_on_non_recording_span(self):
        # NonRecordingSpan (no-op) should not raise
        noop_span = trace.NonRecordingSpan(trace.INVALID_SPAN_CONTEXT)
        safe_set_span_attributes(noop_span, {'key': 'value'})  # Should not raise

    def test_sets_list_of_scalars(self, in_memory_tracer):
        tracer, exporter = in_memory_tracer
        with tracer.start_as_current_span("test") as span:
            safe_set_span_attributes(span, {'ranks': [0, 1, 2]})
        spans = exporter.get_finished_spans()
        assert list(spans[0].attributes.get('ranks')) == [0, 1, 2]

    def test_skips_mixed_type_list(self, in_memory_tracer):
        tracer, exporter = in_memory_tracer
        with tracer.start_as_current_span("test") as span:
            safe_set_span_attributes(span, {'mixed': [1, 'str', None]})
        spans = exporter.get_finished_spans()
        assert 'mixed' not in spans[0].attributes

    def test_skips_none_values(self, in_memory_tracer):
        tracer, exporter = in_memory_tracer
        with tracer.start_as_current_span("test") as span:
            safe_set_span_attributes(span, {'loss': 0.5, 'grad_norm': None, 'iter': 10})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get('loss') == 0.5
        assert spans[0].attributes.get('iter') == 10
        assert 'grad_norm' not in spans[0].attributes


class TestSpanCm:
    def test_span_cm_creates_span(self, in_memory_tracer):
        tracer, exporter = in_memory_tracer
        with span_cm("test.op", tracer=tracer):
            pass
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "test.op"

    def test_span_cm_sets_attributes(self, in_memory_tracer):
        tracer, exporter = in_memory_tracer
        with span_cm("test.op", tracer=tracer, iteration=10, loss=0.5):
            pass
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get('iteration') == 10
        assert spans[0].attributes.get('loss') == 0.5

    def test_span_cm_yields_span(self, in_memory_tracer):
        tracer, exporter = in_memory_tracer
        with span_cm("test.op", tracer=tracer) as span:
            span.set_attribute('extra', 'value')
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get('extra') == 'value'

    def test_span_cm_with_noop_tracer_no_exception(self):
        from opentelemetry.trace import NoOpTracerProvider
        noop_tracer = NoOpTracerProvider().get_tracer("test")
        # Should not raise
        with span_cm("test.op", tracer=noop_tracer, foo='bar'):
            pass

    def test_span_cm_with_default_tracer_no_exception(self):
        # No tracer argument — uses global tracer
        with span_cm("test.op"):
            pass

    def test_span_cm_propagates_exception(self, in_memory_tracer):
        tracer, exporter = in_memory_tracer
        with pytest.raises(ValueError):
            with span_cm("test.op", tracer=tracer):
                raise ValueError("test error")
        # Span should be finished (exception recorded)
        spans = exporter.get_finished_spans()
        assert len(spans) == 1

    def test_span_cm_nested(self, in_memory_tracer):
        tracer, exporter = in_memory_tracer
        with span_cm("outer", tracer=tracer):
            with span_cm("inner", tracer=tracer):
                pass
        spans = exporter.get_finished_spans()
        assert len(spans) == 2
        names = {s.name for s in spans}
        assert names == {"outer", "inner"}


class TestManagedSpan:
    def setup_method(self):
        from nemo.lens.state import set_enabled_span_groups
        set_enabled_span_groups(frozenset())

    def teardown_method(self):
        from nemo.lens.state import set_enabled_span_groups
        set_enabled_span_groups(frozenset())

    def test_yields_none_when_group_disabled(self, in_memory_tracer):
        tracer, exporter = in_memory_tracer
        with managed_span('step', 'megatron.train_step', tracer=tracer) as span:
            assert span is None
        # No spans created
        assert len(exporter.get_finished_spans()) == 0

    def test_creates_span_when_group_enabled(self, in_memory_tracer):
        from nemo.lens.state import set_enabled_span_groups
        set_enabled_span_groups(frozenset({'step'}))
        tracer, exporter = in_memory_tracer
        with managed_span('step', 'megatron.train_step', tracer=tracer) as span:
            assert span is not None
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == 'megatron.train_step'

    def test_sets_attributes_when_enabled(self, in_memory_tracer):
        from nemo.lens.state import set_enabled_span_groups
        set_enabled_span_groups(frozenset({'step'}))
        tracer, exporter = in_memory_tracer
        with managed_span('step', 'megatron.train_step', tracer=tracer, iteration=42):
            pass
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get('iteration') == 42

    def test_span_ends_on_exception(self, in_memory_tracer):
        from nemo.lens.state import set_enabled_span_groups
        from opentelemetry.trace import StatusCode
        set_enabled_span_groups(frozenset({'step'}))
        tracer, exporter = in_memory_tracer
        with pytest.raises(RuntimeError):
            with managed_span('step', 'megatron.train_step', tracer=tracer):
                raise RuntimeError("boom")
        # Span must be finished and have ERROR status
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code == StatusCode.ERROR

    def test_no_span_created_when_disabled_even_on_exception(self, in_memory_tracer):
        tracer, exporter = in_memory_tracer
        with pytest.raises(ValueError):
            with managed_span('step', 'megatron.train_step', tracer=tracer):
                raise ValueError("no span group enabled")
        assert len(exporter.get_finished_spans()) == 0


class TestTraceFn:
    def setup_method(self):
        from nemo.lens.state import set_enabled_span_groups
        set_enabled_span_groups(frozenset())

    def teardown_method(self):
        from nemo.lens.state import set_enabled_span_groups
        set_enabled_span_groups(frozenset())

    def test_no_span_when_group_disabled(self, in_memory_tracer):
        tracer, exporter = in_memory_tracer

        # Use the global tracer (no-op) to verify no span created in real exporter.
        @trace_fn('job', 'megatron.train')
        def my_func():
            return 42

        result = my_func()
        assert result == 42
        assert len(exporter.get_finished_spans()) == 0

    def test_span_created_when_group_enabled(self, in_memory_tracer):
        from nemo.lens.state import set_enabled_span_groups
        tracer, exporter = in_memory_tracer

        set_enabled_span_groups(frozenset({'job'}))

        @trace_fn('job', 'megatron.pretrain', tracer=tracer)
        def my_func():
            return 99

        result = my_func()
        assert result == 99
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == 'megatron.pretrain'

    def test_preserves_function_name(self):
        @trace_fn('job', 'megatron.train')
        def my_training_func():
            """Docstring."""

        assert my_training_func.__name__ == 'my_training_func'
        assert my_training_func.__doc__ == 'Docstring.'

    def test_propagates_exception(self, in_memory_tracer):
        from nemo.lens.state import set_enabled_span_groups
        tracer, exporter = in_memory_tracer

        set_enabled_span_groups(frozenset({'job'}))

        @trace_fn('job', 'megatron.train', tracer=tracer)
        def failing_func():
            raise RuntimeError("training failed")

        with pytest.raises(RuntimeError):
            failing_func()

    def test_group_disabled_bypasses_otel_entirely(self):
        # With group disabled the wrapped function should run and return normally
        # regardless of whether OTel is set up.
        call_log = []

        @trace_fn('nonexistent_group', 'megatron.whatever')
        def side_effect_func():
            call_log.append('called')
            return 'result'

        assert side_effect_func() == 'result'
        assert call_log == ['called']
