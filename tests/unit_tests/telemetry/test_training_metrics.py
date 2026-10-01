# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for ``megatron.core.telemetry.training_metrics``.

The tests drive ``record_training_metrics`` with a fake meter rather than a real
OTel ``MeterProvider``, so they need neither ``opentelemetry`` nor ``nemo-lens``
and they can assert exactly which instrument received which value.
"""

import gc

import pytest

from megatron.core.telemetry import training_metrics
from megatron.core.telemetry.training_metrics import record_training_metrics

# Argument name -> (instrument kind, exported metric name).
INSTRUMENTS = {
    "step_duration_ms": ("histogram", training_metrics.MEGATRON_TRAINING_STEP_DURATION_MS),
    "loss": ("gauge", training_metrics.MEGATRON_TRAINING_LOSS),
    "throughput_tflops": ("gauge", training_metrics.MEGATRON_TRAINING_THROUGHPUT_TFLOPS),
    "grad_norm": ("gauge", training_metrics.MEGATRON_TRAINING_GRAD_NORM),
    "skipped_iters": ("counter", training_metrics.MEGATRON_TRAINING_SKIPPED_ITERS),
    "learning_rate": ("gauge", training_metrics.MEGATRON_TRAINING_LEARNING_RATE),
    "tokens_per_sec": ("gauge", training_metrics.MEGATRON_TRAINING_TOKENS_PER_SEC),
    "memory_allocated_gb": ("gauge", training_metrics.MEGATRON_TRAINING_MEMORY_ALLOCATED_GB),
}


class FakeInstrument:
    """Accepts all three OTel write calls and remembers what it was given."""

    def __init__(self, kind, name):
        self.kind = kind
        self.name = name
        self.values = []

    def record(self, value):
        self.values.append(value)

    def set(self, value):
        self.values.append(value)

    def add(self, value):
        self.values.append(value)


class FakeMeter:
    """Stands in for an OTel ``Meter``; must be weak-referenceable."""

    def __init__(self):
        self.instruments = {}
        self.create_calls = 0

    def _create(self, kind, name):
        self.create_calls += 1
        instrument = FakeInstrument(kind, name)
        self.instruments[name] = instrument
        return instrument

    def create_histogram(self, name, unit=None, description=None):
        return self._create("histogram", name)

    def create_gauge(self, name, unit=None, description=None):
        return self._create("gauge", name)

    def create_counter(self, name, unit=None, description=None):
        return self._create("counter", name)


class BrokenMeter:
    """Every instrument factory raises, as an SDK misconfiguration would."""

    def create_histogram(self, *args, **kwargs):
        raise RuntimeError("meter is broken")

    create_gauge = create_histogram
    create_counter = create_histogram


@pytest.fixture(autouse=True)
def clear_instrument_cache():
    """The instrument cache is module-global; keep tests independent."""
    training_metrics._TRAINING_INSTRUMENTS.clear()
    yield
    training_metrics._TRAINING_INSTRUMENTS.clear()


@pytest.fixture(autouse=True)
def otel_available(monkeypatch):
    """Exercise the recording path even when ``opentelemetry`` is absent."""
    if training_metrics.metrics is None:
        monkeypatch.setattr(training_metrics, "metrics", object())


@pytest.fixture
def meter():
    return FakeMeter()


class TestMetricNames:
    """These strings are the queryable metric names; changing one is breaking."""

    @pytest.mark.parametrize("kind_and_name", INSTRUMENTS.values(), ids=list(INSTRUMENTS))
    def test_name_is_namespaced(self, kind_and_name):
        _, name = kind_and_name
        assert name.startswith("megatron.training.")

    def test_names_are_unique(self):
        names = [name for _, name in INSTRUMENTS.values()]
        assert len(set(names)) == len(names)


class TestInstrumentCreation:
    def test_creates_every_instrument_with_the_right_kind(self, meter):
        record_training_metrics(meter, loss=1.0)

        for kind, name in INSTRUMENTS.values():
            assert name in meter.instruments, f"{name} was never created"
            assert meter.instruments[name].kind == kind

    def test_creates_exactly_the_expected_instruments(self, meter):
        record_training_metrics(meter, loss=1.0)

        assert set(meter.instruments) == {name for _, name in INSTRUMENTS.values()}

    def test_instruments_are_created_once_per_meter(self, meter):
        record_training_metrics(meter, loss=1.0)
        creates_after_first_call = meter.create_calls

        record_training_metrics(meter, loss=2.0)
        record_training_metrics(meter, loss=3.0)

        assert meter.create_calls == creates_after_first_call

    def test_each_meter_gets_its_own_instruments(self):
        first, second = FakeMeter(), FakeMeter()
        record_training_metrics(first, loss=1.0)
        record_training_metrics(second, loss=2.0)

        loss_name = training_metrics.MEGATRON_TRAINING_LOSS
        assert first.instruments[loss_name] is not second.instruments[loss_name]
        assert first.instruments[loss_name].values == [1.0]
        assert second.instruments[loss_name].values == [2.0]

    def test_cache_does_not_keep_the_meter_alive(self):
        """The cache is weak-keyed so re-init does not leak meters.

        The meter is built here rather than taken from the fixture, which would
        hold the only reference that stops it being collected.
        """
        meter = FakeMeter()
        record_training_metrics(meter, loss=1.0)
        assert len(training_metrics._TRAINING_INSTRUMENTS) == 1

        del meter
        gc.collect()
        assert len(training_metrics._TRAINING_INSTRUMENTS) == 0


class TestRecording:
    def test_records_every_metric(self, meter):
        record_training_metrics(
            meter,
            step_duration_ms=123.5,
            loss=2.75,
            throughput_tflops=410.0,
            grad_norm=0.9,
            skipped_iters=2,
            learning_rate=1e-4,
            tokens_per_sec=50000.0,
            memory_allocated_gb=64.25,
        )

        expected = {
            training_metrics.MEGATRON_TRAINING_STEP_DURATION_MS: 123.5,
            training_metrics.MEGATRON_TRAINING_LOSS: 2.75,
            training_metrics.MEGATRON_TRAINING_THROUGHPUT_TFLOPS: 410.0,
            training_metrics.MEGATRON_TRAINING_GRAD_NORM: 0.9,
            training_metrics.MEGATRON_TRAINING_SKIPPED_ITERS: 2,
            training_metrics.MEGATRON_TRAINING_LEARNING_RATE: 1e-4,
            training_metrics.MEGATRON_TRAINING_TOKENS_PER_SEC: 50000.0,
            training_metrics.MEGATRON_TRAINING_MEMORY_ALLOCATED_GB: 64.25,
        }
        for name, value in expected.items():
            assert meter.instruments[name].values == [value], name

    def test_records_nothing_when_all_values_are_none(self, meter):
        record_training_metrics(meter)

        assert all(not instrument.values for instrument in meter.instruments.values())

    @pytest.mark.parametrize("argument", list(INSTRUMENTS))
    def test_records_one_metric_in_isolation(self, meter, argument):
        record_training_metrics(meter, **{argument: 1})

        _, recorded_name = INSTRUMENTS[argument]
        for name, instrument in meter.instruments.items():
            assert instrument.values == ([1] if name == recorded_name else []), name

    def test_accumulates_across_calls(self, meter):
        record_training_metrics(meter, loss=1.0)
        record_training_metrics(meter, loss=0.5)

        assert meter.instruments[training_metrics.MEGATRON_TRAINING_LOSS].values == [1.0, 0.5]

    def test_grad_norm_is_coerced_to_float(self, meter):
        """Callers pass a torch scalar; the SDK only accepts a plain float."""

        class ScalarTensor:
            def __float__(self):
                return 1.5

        record_training_metrics(meter, grad_norm=ScalarTensor())

        recorded = meter.instruments[training_metrics.MEGATRON_TRAINING_GRAD_NORM].values
        assert recorded == [1.5]
        assert type(recorded[0]) is float

    @pytest.mark.parametrize("value,expected", [(0, []), (1, [1]), (5, [5])])
    def test_skipped_iters_only_counts_when_positive(self, meter, value, expected):
        """Adding zero to a counter is a pointless export."""
        record_training_metrics(meter, skipped_iters=value)

        assert (
            meter.instruments[training_metrics.MEGATRON_TRAINING_SKIPPED_ITERS].values == expected
        )

    @pytest.mark.parametrize("argument", ["loss", "grad_norm", "learning_rate"])
    def test_zero_is_recorded_for_non_counter_metrics(self, meter, argument):
        """Zero loss is a real observation, unlike zero skipped iterations."""
        record_training_metrics(meter, **{argument: 0.0})

        _, name = INSTRUMENTS[argument]
        assert meter.instruments[name].values == [0.0]


class TestFailureHandling:
    def test_instrument_creation_failure_is_swallowed(self, caplog):
        """Telemetry must never take down the training loop."""
        record_training_metrics(BrokenMeter(), loss=1.0)

        assert "Failed to create training metric instruments" in caplog.text

    def test_a_broken_meter_is_not_cached(self):
        record_training_metrics(BrokenMeter(), loss=1.0)

        assert len(training_metrics._TRAINING_INSTRUMENTS) == 0

    def test_no_op_without_opentelemetry(self, meter, monkeypatch):
        monkeypatch.setattr(training_metrics, "metrics", None)

        record_training_metrics(meter, loss=1.0)

        assert meter.create_calls == 0
        assert len(training_metrics._TRAINING_INSTRUMENTS) == 0
