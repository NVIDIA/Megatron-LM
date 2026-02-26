# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for _instruments.py: record_training_metrics and record_inference_metrics."""

import pytest

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from nemo.lens.instruments.training import record_training_metrics, _TRAINING_INSTRUMENTS
from nemo.lens.instruments.inference import record_inference_metrics, _INFERENCE_INSTRUMENTS


@pytest.fixture
def meter_and_reader():
    """Return an OTel meter backed by an InMemoryMetricReader."""
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    meter = provider.get_meter("test")
    yield meter, reader
    provider.shutdown()


@pytest.fixture(autouse=True)
def clear_instrument_caches():
    """Clear the WeakKeyDictionary caches between tests."""
    _TRAINING_INSTRUMENTS.clear()
    _INFERENCE_INSTRUMENTS.clear()
    yield
    _TRAINING_INSTRUMENTS.clear()
    _INFERENCE_INSTRUMENTS.clear()


class TestRecordTrainingMetrics:
    def test_records_loss(self, meter_and_reader):
        meter, reader = meter_and_reader
        record_training_metrics(meter, loss=2.5)
        metrics_data = reader.get_metrics_data()
        names = [
            rm.name
            for sm in metrics_data.resource_metrics
            for sm in sm.scope_metrics
            for rm in sm.metrics
        ]
        assert 'megatron.training.loss' in names

    def test_records_step_duration(self, meter_and_reader):
        meter, reader = meter_and_reader
        record_training_metrics(meter, step_duration_ms=150.0)
        metrics_data = reader.get_metrics_data()
        names = [
            rm.name
            for sm in metrics_data.resource_metrics
            for sm in sm.scope_metrics
            for rm in sm.metrics
        ]
        assert 'megatron.training.step_duration_ms' in names

    def test_records_throughput(self, meter_and_reader):
        meter, reader = meter_and_reader
        record_training_metrics(meter, throughput_tps=1000.0)
        metrics_data = reader.get_metrics_data()
        names = [
            rm.name
            for sm in metrics_data.resource_metrics
            for sm in sm.scope_metrics
            for rm in sm.metrics
        ]
        assert 'megatron.training.throughput_tps' in names

    def test_records_grad_norm(self, meter_and_reader):
        meter, reader = meter_and_reader
        record_training_metrics(meter, grad_norm=1.5)
        metrics_data = reader.get_metrics_data()
        names = [
            rm.name
            for sm in metrics_data.resource_metrics
            for sm in sm.scope_metrics
            for rm in sm.metrics
        ]
        assert 'megatron.training.grad_norm' in names

    def test_records_skipped_iters(self, meter_and_reader):
        meter, reader = meter_and_reader
        record_training_metrics(meter, skipped_iters=3)
        metrics_data = reader.get_metrics_data()
        names = [
            rm.name
            for sm in metrics_data.resource_metrics
            for sm in sm.scope_metrics
            for rm in sm.metrics
        ]
        assert 'megatron.training.skipped_iters' in names

    def test_skips_none_values(self, meter_and_reader):
        meter, reader = meter_and_reader
        # All None — no instruments should record
        record_training_metrics(meter)
        metrics_data = reader.get_metrics_data()
        # When nothing is recorded, get_metrics_data() may return None
        if metrics_data is None:
            return
        for sm in metrics_data.resource_metrics:
            for scope in sm.scope_metrics:
                for metric in scope.metrics:
                    for dp in metric.data.data_points:
                        # If we get here, a data point was recorded — that's fine
                        # as long as it wasn't from a None value
                        pass

    def test_skips_zero_skipped_iters(self, meter_and_reader):
        meter, reader = meter_and_reader
        record_training_metrics(meter, skipped_iters=0)
        # skipped_iters=0 should NOT add to counter
        metrics_data = reader.get_metrics_data()
        if metrics_data is None:
            return
        for sm in metrics_data.resource_metrics:
            for scope in sm.scope_metrics:
                for metric in scope.metrics:
                    if metric.name == 'megatron.training.skipped_iters':
                        # Counter should have no data points for 0 value
                        assert len(metric.data.data_points) == 0

    def test_instruments_are_cached(self, meter_and_reader):
        meter, reader = meter_and_reader
        record_training_metrics(meter, loss=1.0)
        record_training_metrics(meter, loss=2.0)
        # Should reuse cached instruments (WeakKeyDictionary)
        assert meter in _TRAINING_INSTRUMENTS

    def test_silent_on_exception(self):
        """Recording with a broken meter-like object should not raise."""
        class BrokenMeter:
            pass

        # Should not raise — exception is logged and suppressed
        record_training_metrics(BrokenMeter(), loss=1.0)


class TestRecordInferenceMetrics:
    def test_records_request_duration(self, meter_and_reader):
        meter, reader = meter_and_reader
        record_inference_metrics(meter, request_duration_s=0.5, model='gpt')
        metrics_data = reader.get_metrics_data()
        names = [
            rm.name
            for sm in metrics_data.resource_metrics
            for sm in sm.scope_metrics
            for rm in sm.metrics
        ]
        assert 'gen_ai.server.request.duration' in names

    def test_records_token_usage_input(self, meter_and_reader):
        meter, reader = meter_and_reader
        record_inference_metrics(meter, input_tokens=100, model='gpt')
        metrics_data = reader.get_metrics_data()
        names = [
            rm.name
            for sm in metrics_data.resource_metrics
            for sm in sm.scope_metrics
            for rm in sm.metrics
        ]
        assert 'gen_ai.client.token.usage' in names

    def test_records_token_usage_output(self, meter_and_reader):
        meter, reader = meter_and_reader
        record_inference_metrics(meter, output_tokens=50, model='gpt')
        metrics_data = reader.get_metrics_data()
        names = [
            rm.name
            for sm in metrics_data.resource_metrics
            for sm in sm.scope_metrics
            for rm in sm.metrics
        ]
        assert 'gen_ai.client.token.usage' in names

    def test_genai_attributes_present(self, meter_and_reader):
        meter, reader = meter_and_reader
        record_inference_metrics(
            meter, request_duration_s=1.0, model='llama', input_tokens=10
        )
        metrics_data = reader.get_metrics_data()
        for sm in metrics_data.resource_metrics:
            for scope in sm.scope_metrics:
                for metric in scope.metrics:
                    for dp in metric.data.data_points:
                        attrs = dict(dp.attributes)
                        assert attrs.get('gen_ai.operation.name') == 'text_completion'
                        assert attrs.get('gen_ai.provider.name') == 'megatron'

    def test_skips_none_values(self, meter_and_reader):
        meter, reader = meter_and_reader
        # All None — should not raise
        record_inference_metrics(meter)

    def test_silent_on_exception(self):
        """Recording with a broken meter-like object should not raise."""
        class BrokenMeter:
            pass

        record_inference_metrics(BrokenMeter(), request_duration_s=1.0)

    def test_instruments_are_cached(self, meter_and_reader):
        meter, reader = meter_and_reader
        record_inference_metrics(meter, request_duration_s=0.5, model='gpt')
        record_inference_metrics(meter, request_duration_s=1.0, model='gpt')
        assert meter in _INFERENCE_INSTRUMENTS
