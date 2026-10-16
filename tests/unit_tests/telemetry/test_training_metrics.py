# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for ``megatron.core.telemetry.training_metrics``.

The tests drive ``record_processed_tokens`` with a fake meter rather than a real
OTel ``MeterProvider``, so they need neither ``opentelemetry`` nor ``nemo-lens``
and they can assert exactly which instrument received which value.
"""

import ast
import gc
import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_PATH = REPO_ROOT / "megatron/training/training.py"
TRAINING_METRICS_PATH = REPO_ROOT / "megatron/core/telemetry/training_metrics.py"
METRICS_DOC_PATH = REPO_ROOT / "docs/user-guide/observability/metrics.md"

_SPEC = importlib.util.spec_from_file_location(
    "_test_training_metrics_module", TRAINING_METRICS_PATH
)
training_metrics = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = training_metrics
_SPEC.loader.exec_module(training_metrics)
record_processed_tokens = training_metrics.record_processed_tokens


def _load_training_token_boundary(*, telemetry_handle, num_microbatches):
    """Load the exact production batch-size and commit-boundary functions."""
    tree = ast.parse(TRAINING_PATH.read_text())
    function_names = {
        "_get_global_batch_size_for_iteration",
        "_record_processed_tokens_for_committed_iteration",
    }
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    namespace = {
        "_otel_training_metrics": training_metrics,
        "get_batch_parallel_size": lambda args: (
            args.data_parallel_size * args.gtp_weight_remat_size
        ),
        "get_num_microbatches": lambda: num_microbatches,
        "get_telemetry": lambda: telemetry_handle,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), TRAINING_PATH, "exec"), namespace)
    return namespace["_record_processed_tokens_for_committed_iteration"]


# Argument name -> (instrument kind, exported metric name).
INSTRUMENTS = {"token_count": ("counter", training_metrics.NV_DL_TRAINING_TOKENS_PROCESSED)}


class FakeInstrument:
    """Accepts all three OTel write calls and remembers what it was given."""

    def __init__(self, kind, name, unit=None):
        self.kind = kind
        self.name = name
        self.unit = unit
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

    def _create(self, kind, name, unit=None):
        self.create_calls += 1
        instrument = FakeInstrument(kind, name, unit)
        self.instruments[name] = instrument
        return instrument

    def create_histogram(self, name, unit=None, description=None):
        return self._create("histogram", name, unit)

    def create_gauge(self, name, unit=None, description=None):
        return self._create("gauge", name, unit)

    def create_counter(self, name, unit=None, description=None):
        return self._create("counter", name, unit)


class BrokenMeter:
    """Every instrument factory raises, as an SDK misconfiguration would."""

    def create_histogram(self, *args, **kwargs):
        raise RuntimeError("meter is broken")

    create_gauge = create_histogram
    create_counter = create_histogram


class BrokenCounterMeter(FakeMeter):
    def create_counter(self, name, unit=None, description=None):
        instrument = super().create_counter(name, unit, description)

        def fail(value):
            raise RuntimeError("counter is broken")

        instrument.add = fail
        return instrument


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
        assert name == "nv.dl.training.tokens.processed"

    def test_names_are_unique(self):
        names = [name for _, name in INSTRUMENTS.values()]
        assert len(set(names)) == len(names)


class TestInstrumentCreation:
    def test_creates_every_instrument_with_the_right_kind(self, meter):
        record_processed_tokens(meter, token_count=1)

        for kind, name in INSTRUMENTS.values():
            assert name in meter.instruments, f"{name} was never created"
            assert meter.instruments[name].kind == kind

    def test_creates_exactly_the_expected_instruments(self, meter):
        record_processed_tokens(meter, token_count=1)

        assert set(meter.instruments) == {name for _, name in INSTRUMENTS.values()}
        assert meter.instruments[training_metrics.NV_DL_TRAINING_TOKENS_PROCESSED].unit == "{token}"

    def test_instruments_are_created_once_per_meter(self, meter):
        record_processed_tokens(meter, token_count=1)
        creates_after_first_call = meter.create_calls

        record_processed_tokens(meter, token_count=2)
        record_processed_tokens(meter, token_count=3)

        assert meter.create_calls == creates_after_first_call

    def test_each_meter_gets_its_own_instruments(self):
        first, second = FakeMeter(), FakeMeter()
        record_processed_tokens(first, token_count=1)
        record_processed_tokens(second, token_count=2)

        token_name = training_metrics.NV_DL_TRAINING_TOKENS_PROCESSED
        assert first.instruments[token_name] is not second.instruments[token_name]
        assert first.instruments[token_name].values == [1]
        assert second.instruments[token_name].values == [2]

    def test_cache_does_not_keep_the_meter_alive(self):
        """The cache is weak-keyed so re-init does not leak meters.

        The meter is built here rather than taken from the fixture, which would
        hold the only reference that stops it being collected.
        """
        meter = FakeMeter()
        record_processed_tokens(meter, token_count=1)
        assert len(training_metrics._TRAINING_INSTRUMENTS) == 1

        del meter
        gc.collect()
        assert len(training_metrics._TRAINING_INSTRUMENTS) == 0


class TestRecording:
    def test_records_every_metric(self, meter):
        record_processed_tokens(meter, token_count=50000)

        expected = {training_metrics.NV_DL_TRAINING_TOKENS_PROCESSED: 50000}
        for name, value in expected.items():
            assert meter.instruments[name].values == [value], name

    def test_records_nothing_when_all_values_are_none(self, meter):
        record_processed_tokens(meter)

        assert all(not instrument.values for instrument in meter.instruments.values())

    @pytest.mark.parametrize("argument", list(INSTRUMENTS))
    def test_records_one_metric_in_isolation(self, meter, argument):
        record_processed_tokens(meter, **{argument: 1})

        _, recorded_name = INSTRUMENTS[argument]
        for name, instrument in meter.instruments.items():
            assert instrument.values == ([1] if name == recorded_name else []), name

    def test_accumulates_across_calls(self, meter):
        record_processed_tokens(meter, token_count=1)
        record_processed_tokens(meter, token_count=5)

        assert meter.instruments[training_metrics.NV_DL_TRAINING_TOKENS_PROCESSED].values == [1, 5]


class TestFailureHandling:
    def test_instrument_creation_failure_is_swallowed(self, caplog):
        """Telemetry must never take down the training loop."""
        record_processed_tokens(BrokenMeter(), token_count=1)

        assert "Failed to create training metric instruments" in caplog.text

    def test_a_broken_meter_is_not_cached(self):
        record_processed_tokens(BrokenMeter(), token_count=1)

        assert len(training_metrics._TRAINING_INSTRUMENTS) == 0

    def test_no_op_without_opentelemetry(self, meter, monkeypatch):
        monkeypatch.setattr(training_metrics, "metrics", None)

        record_processed_tokens(meter, token_count=1)

        assert meter.create_calls == 0
        assert len(training_metrics._TRAINING_INSTRUMENTS) == 0

    def test_counter_write_failure_is_swallowed(self, caplog):
        record_processed_tokens(BrokenCounterMeter(), token_count=1)

        assert "Failed to record processed-token metric" in caplog.text


@pytest.mark.parametrize(
    "global_batch_size,sequence_length,packed,expected",
    [(64, 2048, None, 131072), (64, 2048, 12345.0, 12345), (64, 2048, 0.0, 0)],
)
def test_processed_token_increment(global_batch_size, sequence_length, packed, expected):
    assert (
        training_metrics.processed_token_increment(global_batch_size, sequence_length, packed)
        == expected
    )


@pytest.mark.parametrize("packed", [float("nan"), float("inf"), 12.5, -1.0, True, "12"])
def test_invalid_packed_token_increment_is_skipped(packed, caplog):
    assert training_metrics.processed_token_increment(64, 2048, packed) is None
    assert "Skipping invalid packed processed-token count" in caplog.text


@pytest.mark.parametrize(
    "micro_batch_size,data_parallel_size,gtp_size,num_microbatches,sequence_length,packed,expected",
    [
        pytest.param(2, 4, 1, 3, 10, None, 240, id="unpacked"),
        pytest.param(2, 4, 1, 3, 10, 123.0, 123, id="packed"),
        pytest.param(2, 4, 1, 5, 10, None, 400, id="ramping-batch"),
        pytest.param(2, 4, 8, 3, 10, None, 1920, id="gtp-remat"),
    ],
)
def test_processed_token_boundary_uses_canonical_batch_calculation(
    micro_batch_size,
    data_parallel_size,
    gtp_size,
    num_microbatches,
    sequence_length,
    packed,
    expected,
):
    meter = FakeMeter()
    handle = types.SimpleNamespace(is_exporting=True, meter=meter)
    args = types.SimpleNamespace(
        micro_batch_size=micro_batch_size,
        data_parallel_size=data_parallel_size,
        gtp_weight_remat_size=gtp_size,
        seq_length=sequence_length,
        skip_train=False,
    )
    commit_iteration = _load_training_token_boundary(
        telemetry_handle=handle, num_microbatches=num_microbatches
    )

    commit_iteration(args, packed)

    assert meter.instruments[training_metrics.NV_DL_TRAINING_TOKENS_PROCESSED].values == [expected]


def test_committed_model_work_records_once():
    meter = FakeMeter()
    handle = types.SimpleNamespace(is_exporting=True, meter=meter)
    args = types.SimpleNamespace(
        micro_batch_size=2,
        data_parallel_size=4,
        gtp_weight_remat_size=1,
        seq_length=2048,
        skip_train=False,
    )
    commit_iteration = _load_training_token_boundary(telemetry_handle=handle, num_microbatches=8)

    commit_iteration(args)

    assert meter.instruments[training_metrics.NV_DL_TRAINING_TOKENS_PROCESSED].values == [131072]


def test_inference_only_pass_does_not_reach_the_metric_recorder(monkeypatch):
    calls = []
    monkeypatch.setattr(
        training_metrics,
        "record_processed_tokens_for_iteration",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    args = types.SimpleNamespace(skip_train=True)
    commit_iteration = _load_training_token_boundary(
        telemetry_handle=types.SimpleNamespace(is_exporting=True, meter=FakeMeter()),
        num_microbatches=8,
    )

    commit_iteration(args, 12345.0)

    assert calls == []


@pytest.mark.parametrize(
    "handle",
    [None, types.SimpleNamespace(is_exporting=False, meter=FakeMeter())],
    ids=["missing", "non-exporting"],
)
def test_processed_tokens_require_an_exporting_telemetry_handle(handle):
    training_metrics.record_processed_tokens_for_iteration(
        handle, global_batch_size=64, sequence_length=2048
    )

    if handle is not None:
        assert handle.meter.instruments == {}


def test_processed_tokens_are_replicated_on_every_exporting_rank():
    handles = [
        types.SimpleNamespace(is_exporting=True, meter=FakeMeter()),
        types.SimpleNamespace(is_exporting=True, meter=FakeMeter()),
    ]

    for handle in handles:
        training_metrics.record_processed_tokens_for_iteration(
            handle, global_batch_size=24, sequence_length=1024, packed_token_count=12345.0
        )

    assert [
        handle.meter.instruments[training_metrics.NV_DL_TRAINING_TOKENS_PROCESSED].values
        for handle in handles
    ] == [[12345], [12345]]


def test_processed_token_call_remains_at_the_committed_iteration_boundary():
    tree = ast.parse(TRAINING_PATH.read_text())
    train = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    commit_call = next(
        node
        for node in ast.walk(train)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_record_processed_tokens_for_committed_iteration"
    )

    loop = next(node for node in train.body if isinstance(node, ast.While))
    dummy_guard = next(
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.If) and "args.iterations_to_skip" in ast.unparse(node.test)
    )
    dummy_continue = next(node for node in ast.walk(dummy_guard) if isinstance(node, ast.Continue))
    exit_guard = next(
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "should_exit"
        and any(isinstance(child, ast.Break) for child in ast.walk(node))
    )
    exit_break = next(node for node in ast.walk(exit_guard) if isinstance(node, ast.Break))
    train_step_call = next(
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "train_step"
    )
    workload_exception_call = next(
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_maybe_raise_workload_exception"
    )
    iteration_increment = next(
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "completed_iterations"
    )
    packed_stats_call = next(
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "consume_seqlen_stats_in_iteration"
    )

    assert dummy_continue.lineno < commit_call.lineno
    assert train_step_call.lineno < workload_exception_call.lineno < commit_call.lineno
    assert train_step_call.lineno < exit_break.lineno < commit_call.lineno
    assert iteration_increment.lineno < packed_stats_call.lineno < commit_call.lineno
    assert ast.unparse(commit_call) == (
        "_record_processed_tokens_for_committed_iteration(args, total_real_tokens_in_batch)"
    )
    assert "skipped_iter" not in ast.unparse(commit_call)
    assert all(
        commit_call not in list(ast.walk(node))
        for node in ast.walk(loop)
        if isinstance(node, ast.If) and "skipped_iter" in ast.unparse(node.test)
    )


def test_processed_token_recorder_never_materializes_device_values():
    recorder_source = TRAINING_METRICS_PATH.read_text()
    for forbidden in (".item()", ".tolist()", "torch.cuda.synchronize", "torch.distributed"):
        assert forbidden not in recorder_source
    assert "float(grad_norm)" not in recorder_source


def test_processed_token_docs_describe_counter_rank_policy():
    metrics_doc = METRICS_DOC_PATH.read_text()
    assert "nv.dl.training.tokens.processed" in metrics_doc
    assert "each exporting rank" in metrics_doc.lower()
    assert "select one rank series" in metrics_doc.lower()
