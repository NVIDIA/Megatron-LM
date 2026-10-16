# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Training report events reuse host values at the existing logging boundary."""

import ast
import sys
import types
from pathlib import Path

import pytest

from .test_telemetry import (
    _block_lens_import,
    _import_fresh,
    _install_fake_lens,
    _module_function_ast,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_PATH = REPO_ROOT / "megatron/training/training.py"
METRICS_DOC_PATH = REPO_ROOT / "docs/user-guide/observability/metrics.md"


class _EventSpan:
    def __init__(self, recording=True, fail=False, inspect_fail=False):
        self.recording = recording
        self.fail = fail
        self.inspect_fail = inspect_fail
        self.events = []

    def is_recording(self):
        if self.inspect_fail:
            raise RuntimeError("span inspection failed")
        return self.recording

    def add_event(self, name, attributes=None):
        if self.fail:
            raise RuntimeError("event write failed")
        self.events.append((name, attributes))


def _install_current_span(monkeypatch, span):
    opentelemetry = types.ModuleType("opentelemetry")
    trace = types.ModuleType("opentelemetry.trace")
    trace.get_current_span = lambda: span
    opentelemetry.trace = trace
    monkeypatch.setitem(sys.modules, "opentelemetry", opentelemetry)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", trace)


def test_add_span_event_prefers_explicit_recording_span(monkeypatch):
    _install_fake_lens(monkeypatch)
    telemetry = _import_fresh(monkeypatch)
    explicit = _EventSpan()
    current = _EventSpan()
    _install_current_span(monkeypatch, current)

    telemetry.add_span_event("training.objective", {"value": 1.5}, span=explicit)

    assert explicit.events == [("training.objective", {"value": 1.5})]
    assert current.events == []


def test_add_span_event_falls_back_from_nonrecording_explicit_span(monkeypatch):
    _install_fake_lens(monkeypatch)
    telemetry = _import_fresh(monkeypatch)
    explicit = _EventSpan(recording=False)
    current = _EventSpan()
    _install_current_span(monkeypatch, current)

    telemetry.add_span_event("training.numerics", span=explicit)

    assert explicit.events == []
    assert current.events == [("training.numerics", None)]


def test_add_span_event_uses_current_recording_span(monkeypatch):
    _install_fake_lens(monkeypatch)
    telemetry = _import_fresh(monkeypatch)
    current = _EventSpan()
    _install_current_span(monkeypatch, current)

    telemetry.add_span_event("training.objective", {"step": 7})

    assert current.events == [("training.objective", {"step": 7})]


def test_add_span_event_ignores_missing_recording_span(monkeypatch):
    _install_fake_lens(monkeypatch)
    telemetry = _import_fresh(monkeypatch)
    current = _EventSpan(recording=False)
    _install_current_span(monkeypatch, current)

    assert telemetry.add_span_event("training.objective") is None
    assert current.events == []


def test_add_span_event_is_a_direct_noop_without_lens(monkeypatch):
    _block_lens_import(monkeypatch)
    telemetry = _import_fresh(monkeypatch)

    class ExplodingSpan:
        def is_recording(self):
            raise AssertionError("unavailable telemetry must not inspect spans")

    assert telemetry.add_span_event("training.objective", span=ExplodingSpan()) is None


def test_add_span_event_contains_write_failure_and_logs_debug(monkeypatch, caplog):
    _install_fake_lens(monkeypatch)
    telemetry = _import_fresh(monkeypatch)
    span = _EventSpan(fail=True)
    caplog.set_level("DEBUG", logger=telemetry.__name__)

    assert telemetry.add_span_event("training.objective", span=span) is None

    assert "Could not add span event 'training.objective'" in caplog.text


def test_add_span_event_contains_explicit_span_inspection_failure(monkeypatch, caplog):
    _install_fake_lens(monkeypatch)
    telemetry = _import_fresh(monkeypatch)
    explicit = _EventSpan(inspect_fail=True)
    current = _EventSpan()
    _install_current_span(monkeypatch, current)
    caplog.set_level("DEBUG", logger=telemetry.__name__)

    telemetry.add_span_event("training.objective", {"value": 1.5}, span=explicit)

    assert explicit.events == []
    assert current.events == [("training.objective", {"value": 1.5})]
    assert "Could not inspect span for event 'training.objective'" in caplog.text


def test_add_span_event_contains_current_span_lookup_failure(monkeypatch, caplog):
    _install_fake_lens(monkeypatch)
    telemetry = _import_fresh(monkeypatch)

    def fail_lookup():
        raise RuntimeError("current span lookup failed")

    opentelemetry = types.ModuleType("opentelemetry")
    trace = types.ModuleType("opentelemetry.trace")
    trace.get_current_span = fail_lookup
    opentelemetry.trace = trace
    monkeypatch.setitem(sys.modules, "opentelemetry", opentelemetry)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", trace)
    caplog.set_level("DEBUG", logger=telemetry.__name__)

    assert telemetry.add_span_event("training.numerics") is None
    assert "Could not locate recording span for event 'training.numerics'" in caplog.text


def test_add_span_event_contains_current_span_inspection_failure(monkeypatch, caplog):
    _install_fake_lens(monkeypatch)
    telemetry = _import_fresh(monkeypatch)
    current = _EventSpan(inspect_fail=True)
    _install_current_span(monkeypatch, current)
    caplog.set_level("DEBUG", logger=telemetry.__name__)

    assert telemetry.add_span_event("training.numerics") is None
    assert current.events == []
    assert "Could not locate recording span for event 'training.numerics'" in caplog.text


def test_training_report_events_have_v01_shapes(monkeypatch):
    _install_fake_lens(monkeypatch)
    telemetry = _import_fresh(monkeypatch)
    span = _EventSpan()

    assert not hasattr(telemetry, "TRAINING_GRAD_NORM")
    assert "grad_norm" not in telemetry.add_training_numerics_event.__annotations__

    telemetry.add_training_objective_event(17, "lm loss", 1.25, span=span)
    telemetry.add_training_objective_event(17, "aux loss", 0.5, weight=0.01, span=span)
    telemetry.add_training_numerics_event(
        17,
        learning_rate=1e-4,
        global_batch_size=128,
        consumed_samples=4096,
        grad_zeros=3,
        params_norm=100.0,
        skipped_iterations=1,
        nan_iterations=0,
        loss_scale=None,
        span=span,
    )

    first_name, first = span.events[0]
    assert first_name == telemetry.EVENT_TRAINING_OBJECTIVE
    assert first == {
        telemetry.TRAINING_STEP: 17,
        telemetry.TRAINING_OBJECTIVE_NAME: "lm loss",
        telemetry.TRAINING_OBJECTIVE_VALUE: 1.25,
        telemetry.MEASUREMENT_DOMAIN: telemetry.MEASUREMENT_DOMAIN_DP_MEAN,
    }
    _, weighted = span.events[1]
    assert weighted[telemetry.TRAINING_OBJECTIVE_WEIGHT] == 0.01

    numerics_name, numerics = span.events[2]
    assert numerics_name == telemetry.EVENT_TRAINING_NUMERICS
    assert numerics[telemetry.TRAINING_STEP] == 17
    assert telemetry.MEASUREMENT_DOMAIN not in numerics
    assert telemetry.TRAINING_LOSS_SCALE not in numerics
    assert "nv.dl.training.grad_norm" not in numerics
    assert numerics[telemetry.TRAINING_GRAD_ZEROS] == 3


def test_training_report_boundary_exports_valid_scalar_events(monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    _install_fake_lens(monkeypatch)
    telemetry = _import_fresh(monkeypatch)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)
    report = {
        "learning_rate": 1e-4,
        "global_batch_size": 128,
        "consumed_samples": 4096,
        "grad_zeros": 3,
        "params_norm": 100.0,
        "skipped_iterations": 1,
        "nan_iterations": 0,
        "loss_scale": 65536.0,
    }

    try:
        with tracer.start_as_current_span(telemetry.SPAN_TRAINING_LOG):
            telemetry.add_training_report_events(17, {"lm loss": 1.25, "aux loss": 0.5}, **report)

        provider.force_flush()
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        events = spans[0].events
        assert [event.name for event in events] == [
            telemetry.EVENT_TRAINING_OBJECTIVE,
            telemetry.EVENT_TRAINING_OBJECTIVE,
            telemetry.EVENT_TRAINING_NUMERICS,
        ]
        assert [event.attributes[telemetry.TRAINING_OBJECTIVE_NAME] for event in events[:2]] == [
            "lm loss",
            "aux loss",
        ]
        assert all(
            isinstance(value, (bool, int, float, str))
            for event in events
            for value in event.attributes.values()
        )
        numerics = events[-1].attributes
        assert numerics[telemetry.TRAINING_STEP] == 17
        assert telemetry.MEASUREMENT_DOMAIN not in numerics
        assert "nv.dl.training.grad_norm" not in numerics
    finally:
        provider.shutdown()


@pytest.mark.parametrize(
    "fp16,configured_scale,current_scale,expected_scale",
    [
        (True, None, 32768.0, 32768.0),
        (True, 128.0, 128.0, 128.0),
        (False, 128.0, 128.0, 128.0),
        (False, None, 1.0, None),
    ],
    ids=["fp16-dynamic", "fp16-constant", "bf16-constant", "bf16-unscaled"],
)
@pytest.mark.parametrize("span_name", ["nv.mlm.train.log", "nv.dl.training.iter_block"])
def test_training_log_exports_current_loss_scale(
    monkeypatch, fp16, configured_scale, current_scale, expected_scale, span_name
):
    """Execute the trainer's real reporting call, not a duplicated scale predicate."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    _install_fake_lens(monkeypatch)
    telemetry = _import_fresh(monkeypatch)
    training_log = _module_function_ast(TRAINING_PATH, "training_log")
    calls = [
        node
        for node in ast.walk(training_log)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "_otel"
        and any(keyword.arg == "loss_scale" for keyword in node.keywords)
    ]
    assert len(calls) == 1
    call = calls[0]
    # The reporting statement must use existing host values, with no tensor
    # construction, optimizer reads, item() calls, or distributed operations.
    assert all(
        node is call or (isinstance(node.func, ast.Name) and node.func.id in {"int", "bool"})
        for node in ast.walk(call)
        if isinstance(node, ast.Call)
    )

    class HostScale(float):
        def item(self):
            raise AssertionError("loss scale must already be materialized")

        def __float__(self):
            raise AssertionError("reporting must reuse the current host scale")

    namespace = {
        "_otel": telemetry,
        "args": types.SimpleNamespace(
            fp16=fp16,
            bf16=not fp16,
            loss_scale=configured_scale,
            initial_loss_scale=65536.0,
            consumed_train_samples=4096,
        ),
        "iteration": 17,
        "_otel_objective_snapshot": {"lm loss": 1.25},
        "learning_rate": 1e-4,
        "batch_size": 128,
        "num_zeros_in_grad": 3,
        "params_norm": 100.0,
        "_otel_skipped_iters_snapshot": 1,
        "_otel_nan_iters_snapshot": 0,
        "loss_scale": HostScale(current_scale),
    }
    statement = ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=call)], type_ignores=[]))
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        with provider.get_tracer(__name__).start_as_current_span(span_name):
            exec(compile(statement, str(TRAINING_PATH), "exec"), namespace)
        provider.force_flush()
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == span_name
        numerics = [
            event for event in spans[0].events if event.name == telemetry.EVENT_TRAINING_NUMERICS
        ]
        assert len(numerics) == 1
        attributes = numerics[0].attributes
        assert attributes[telemetry.TRAINING_STEP] == 17
        assert telemetry.MEASUREMENT_DOMAIN not in attributes
        if expected_scale is None:
            assert telemetry.TRAINING_LOSS_SCALE not in attributes
        else:
            assert attributes[telemetry.TRAINING_LOSS_SCALE] == expected_scale
    finally:
        provider.shutdown()


def test_training_telemetry_reuses_existing_host_scalars_without_syncing():
    """Keep telemetry out of device materialization and synchronization paths."""
    tree = ast.parse(TRAINING_PATH.read_text())
    training_log = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "training_log"
    )
    item_calls = {
        ast.unparse(node)
        for node in ast.walk(training_log)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "item"
    }

    # These are the two pre-existing ordinary logging materializations: rejected-loss
    # validation and the average that telemetry now reuses.
    assert item_calls == {"loss_dict[key].float().sum().item()", "total_loss_dict[key].item()"}
    training_source = ast.unparse(training_log)
    assert "_otel_loss_snapshot" not in training_source
    assert "_otel_objective_snapshot[key] = avg" in training_source
    assert ".tolist()" not in training_source
    assert "torch.cuda.synchronize" not in training_source
    assert "torch.distributed.all_" not in training_source


def test_training_report_call_remains_inside_the_ordinary_log_occurrence():
    tree = ast.parse(TRAINING_PATH.read_text())
    training_log = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "training_log"
    )
    report_call = next(
        node
        for node in ast.walk(training_log)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_training_report_events"
    )
    log_guard = next(
        node
        for node in ast.walk(training_log)
        if isinstance(node, ast.If)
        and report_call in list(ast.walk(node))
        and "cfg.logger.log_interval" in ast.unparse(node.test)
    )
    reporter_guard = next(
        node
        for node in ast.walk(log_guard)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "_otel_is_reporter"
        and report_call in list(ast.walk(node))
    )

    assert "is_first_iteration" in ast.unparse(log_guard.test)
    assert reporter_guard is not None
    assert "is_global_last_rank" not in {keyword.arg for keyword in report_call.keywords}
    assert "is_log_occurrence" not in {keyword.arg for keyword in report_call.keywords}
    assert "grad_norm" not in {keyword.arg for keyword in report_call.keywords}


EXPECTED_REPORT_NAMES = {
    'MEASUREMENT_DOMAIN': 'nv.dl.measurement.domain',
    'TRAINING_OBJECTIVE_NAME': 'nv.dl.training.objective.name',
    'TRAINING_OBJECTIVE_VALUE': 'nv.dl.training.objective.value',
    'TRAINING_OBJECTIVE_WEIGHT': 'nv.dl.training.objective.weight',
    'TRAINING_LEARNING_RATE': 'nv.dl.training.learning_rate',
    'TRAINING_GLOBAL_BATCH_SIZE': 'nv.dl.training.global_batch_size',
    'TRAINING_CONSUMED_SAMPLES': 'nv.dl.training.consumed_samples',
    'TRAINING_GRAD_ZEROS': 'nv.dl.training.grad_zeros',
    'TRAINING_PARAMS_NORM': 'nv.dl.training.params_norm',
    'TRAINING_SKIPPED_ITERATIONS': 'nv.dl.training.skipped_iterations',
    'TRAINING_NAN_ITERATIONS': 'nv.dl.training.nan_iterations',
    'TRAINING_LOSS_SCALE': 'nv.dl.training.loss_scale',
    'EVENT_TRAINING_OBJECTIVE': 'nv.dl.training.objective',
    'EVENT_TRAINING_NUMERICS': 'nv.dl.training.numerics',
}


def test_report_names_are_canonical(telemetry_module):
    assert {
        name: getattr(telemetry_module, name) for name in EXPECTED_REPORT_NAMES
    } == EXPECTED_REPORT_NAMES


def test_reporting_docs_describe_events_without_retired_emissions():
    metrics_doc = METRICS_DOC_PATH.read_text()
    active_docs = "\n".join(
        (REPO_ROOT / path).read_text()
        for path in (
            "docs/user-guide/observability/metrics.md",
            "docs/user-guide/observability/extending.md",
            "docs/user-guide/observability/index.md",
            "docs/user-guide/observability/span-groups.md",
            "megatron/core/telemetry/README.md",
        )
    )
    for retired_name in (
        "megatron.training.step_duration_ms",
        "megatron.training.loss",
        "megatron.training.throughput_tflops",
        "megatron.training.tokens_per_sec",
        "megatron.training.grad_norm",
        "megatron.training.skipped_iters",
        "megatron.training.learning_rate",
        "megatron.training.memory_allocated_gb",
    ):
        assert retired_name not in active_docs
    assert "nv.dl.training.objective" in metrics_doc
    assert "nv.dl.training.numerics" in metrics_doc
