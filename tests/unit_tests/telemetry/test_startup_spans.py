# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for explicit-timestamp training startup spans."""

import ast
import builtins
import importlib
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from nemo.lens import span_utilities
from opentelemetry.context import Context

REPO_ROOT = Path(__file__).resolve().parents[3]
TELEMETRY_PATH = REPO_ROOT / "megatron/core/telemetry/telemetry.py"
TRAINING_PATH = REPO_ROOT / "megatron/training/training.py"
GLOBAL_VARS_PATH = REPO_ROOT / "megatron/training/global_vars.py"

EXPECTED_STARTUP_SPANS = {
    "SPAN_TRAINING_STARTUP": "nv.dl.training.startup",
    "SPAN_TRAINING_STARTUP_PYTHON": "nv.dl.training.startup.python",
    "SPAN_TRAINING_STARTUP_IMPORTS": "nv.dl.training.startup.imports",
    "SPAN_TRAINING_STARTUP_ARG_PARSE": "nv.dl.training.startup.arg_parse",
    "SPAN_TRAINING_STARTUP_IN_JOB_SETUP": "nv.dl.training.startup.in_job_setup",
    "SPAN_TRAINING_STARTUP_INITIALIZE_MEGATRON": "nv.dl.training.startup.initialize_megatron",
    "SPAN_TRAINING_STARTUP_JIT_FUSION_OPTIONS": "nv.dl.training.startup.jit_fusion_options",
    "SPAN_TRAINING_STARTUP_MODEL_INIT": "nv.dl.training.startup.model_init",
    "SPAN_TRAINING_STARTUP_DATALOADER": "nv.dl.training.startup.dataloader",
    "SPAN_TRAINING_STARTUP_WEIGHT_HASH_CHECK": "nv.dl.training.startup.weight_hash_check",
    "SPAN_GPU_SNIFF_STARTUP": "nv.dl.resiliency.gpu_sniff.startup",
}


def _function(path, name):
    tree = ast.parse(path.read_text(), filename=str(path))
    return next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _module_string_constants(path):
    constants = {}
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                constants[target.id] = node.value.value
    return constants


def _assigned_dict(function, name):
    assignment = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    )
    return {
        key.value: ast.unparse(value)
        for key, value in zip(assignment.value.keys, assignment.value.values)
    }


@pytest.fixture
def telemetry(monkeypatch, telemetry_module):
    monkeypatch.setattr(telemetry_module, "is_enabled", lambda group: group == telemetry_module.JOB)
    monkeypatch.setattr(telemetry_module, "_ACTIVE_TRAINER_HANDLE", None)
    monkeypatch.setattr(telemetry_module, "_PYTHON_STARTUP_RECORDED", False)
    return telemetry_module


@pytest.fixture
def span_exporter():
    sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
    sdk_export = pytest.importorskip("opentelemetry.sdk.trace.export")
    try:
        in_memory_export = importlib.import_module("opentelemetry.sdk.trace.export")
        exporter_type = in_memory_export.InMemorySpanExporter
    except AttributeError:
        in_memory_export = importlib.import_module(
            "opentelemetry.sdk.trace.export.in_memory_span_exporter"
        )
        exporter_type = in_memory_export.InMemorySpanExporter

    exporter = exporter_type()
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(sdk_export.SimpleSpanProcessor(exporter))
    try:
        yield provider.get_tracer("megatron-startup-test"), exporter
    finally:
        provider.shutdown()


def test_startup_constants_match_canonical_names():
    constants = _module_string_constants(TELEMETRY_PATH)

    assert {name: constants[name] for name in EXPECTED_STARTUP_SPANS} == EXPECTED_STARTUP_SPANS
    assert constants["MODEL_CONFIG_MODEL_TYPE"] == "nv.mcore.model.config.model_type"


@pytest.mark.parametrize(
    ("process_start", "program_start", "main_entry", "expected"),
    [
        (90.0, 95.0, 100.0, 90.0),
        (101.0, 95.0, 100.0, None),
        (96.0, 95.0, 100.0, None),
        (math.nan, 95.0, 100.0, None),
        (-1.0, 95.0, 100.0, None),
        (101.0, math.inf, 100.0, None),
        (90.0, 95.0, math.nan, None),
    ],
)
def test_process_start_selection(
    monkeypatch, telemetry, process_start, program_start, main_entry, expected
):
    monkeypatch.setattr(span_utilities, "linux_process_create_time", lambda: process_start)

    assert telemetry.select_process_start_time(program_start, main_entry) == expected


def test_process_start_selection_does_not_fabricate_os_time(monkeypatch, telemetry):
    def unreadable_proc():
        raise RuntimeError("unreadable /proc")

    monkeypatch.setattr(span_utilities, "linux_process_create_time", unreadable_proc)

    assert telemetry.select_process_start_time(95.0, 100.0) is None


def test_installed_lens_missing_startup_helper_fails_visibly(monkeypatch, telemetry):
    real_import = builtins.__import__

    def missing_helper(name, *args, **kwargs):
        if name == "nemo.lens.span_utilities":
            raise ImportError("installed Lens has no startup helpers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_helper)

    with pytest.raises(ImportError, match="installed Lens has no startup helpers"):
        telemetry.select_process_start_time(95.0, 100.0)


def test_startup_root_and_explicit_children_have_deterministic_topology(
    monkeypatch, telemetry, span_exporter
):
    tracer, exporter = span_exporter
    monkeypatch.setattr(span_utilities, "linux_process_create_time", lambda: 100.0)
    monkeypatch.setattr(telemetry.time, "time", lambda: 104.0)

    with tracer.start_as_current_span("ambient") as ambient:
        startup, startup_context, root_start, root_open = telemetry.start_startup_span(
            tracer, "encoder_or_decoder", 100.5, 102.0
        )
        telemetry.emit_startup_span(
            tracer,
            telemetry.SPAN_TRAINING_STARTUP_IMPORTS,
            100.0,
            101.0,
            context=startup_context,
            root_start=root_start,
            root_open_time=root_open,
        )
        telemetry.emit_startup_span(
            tracer,
            telemetry.SPAN_TRAINING_STARTUP_ARG_PARSE,
            101.0,
            102.0,
            context=startup_context,
            root_start=root_start,
            root_open_time=root_open,
        )
        startup.end(end_time=105_000_000_000)

    spans = {span.name: span for span in exporter.get_finished_spans()}
    startup_span = spans[telemetry.SPAN_TRAINING_STARTUP]
    ambient_span = spans["ambient"]
    assert startup_span.parent is None
    assert startup_span.context.trace_id != ambient_span.context.trace_id
    assert startup_span.start_time == 100_000_000_000
    assert startup_span.attributes == {"nv.mcore.model.config.model_type": "encoder_or_decoder"}
    for name in (
        telemetry.SPAN_TRAINING_STARTUP_IMPORTS,
        telemetry.SPAN_TRAINING_STARTUP_ARG_PARSE,
    ):
        child = spans[name]
        assert child.parent.span_id == startup_span.context.span_id
        assert child.context.trace_id == startup_span.context.trace_id


def test_normal_time_root_fallback_suppresses_historical_children(
    monkeypatch, telemetry, span_exporter
):
    tracer, exporter = span_exporter
    monkeypatch.setattr(span_utilities, "linux_process_create_time", lambda: math.nan)
    monkeypatch.setattr(telemetry.time, "time", lambda: 104.0)

    startup, startup_context, root_start, root_open = telemetry.start_startup_span(
        tracer, "model", math.nan, 102.0
    )
    assert root_start == root_open == 104.0
    assert (
        telemetry.emit_startup_span(
            tracer,
            telemetry.SPAN_TRAINING_STARTUP_IMPORTS,
            100.0,
            101.0,
            context=startup_context,
            root_start=root_start,
            root_open_time=root_open,
        )
        is None
    )
    startup.end()

    assert [span.name for span in exporter.get_finished_spans()] == [
        telemetry.SPAN_TRAINING_STARTUP
    ]


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (None, 102.0),
        (101.0, None),
        (math.nan, 102.0),
        (101.0, math.inf),
        (103.0, 102.0),
        (99.0, 101.0),
        (101.0, 111.0),
    ],
)
def test_invalid_or_uncontained_intervals_skip_only_the_affected_child(telemetry, start, end):
    class FailTracer:
        def start_span(self, *args, **kwargs):
            raise AssertionError("invalid intervals must not touch the tracer")

    result = telemetry.emit_startup_span(
        FailTracer(),
        telemetry.SPAN_TRAINING_STARTUP_IMPORTS,
        start,
        end,
        context=Context(),
        root_start=100.0,
        root_open_time=110.0,
    )

    assert result is None


def test_equal_timestamp_interval_emits_zero_duration_span(monkeypatch, telemetry):
    calls = []
    emitted = object()

    def emit_span(tracer, name, start, end, **kwargs):
        calls.append((tracer, name, start, end, kwargs))
        return emitted

    monkeypatch.setattr(span_utilities, "emit_span", emit_span)
    tracer = object()
    parent_context = Context()

    result = telemetry.emit_startup_span(
        tracer,
        telemetry.SPAN_TRAINING_STARTUP_IMPORTS,
        105.0,
        105.0,
        context=parent_context,
        root_start=100.0,
        root_open_time=110.0,
    )

    assert result is emitted
    assert calls == [
        (tracer, telemetry.SPAN_TRAINING_STARTUP_IMPORTS, 105.0, 105.0, {"context": parent_context})
    ]


def test_disabled_job_group_creates_no_startup_spans(monkeypatch, telemetry):
    monkeypatch.setattr(telemetry, "is_enabled", lambda group: False)
    monkeypatch.setattr(
        span_utilities, "linux_process_create_time", lambda: pytest.fail("disabled OS lookup")
    )

    class FailTracer:
        def start_span(self, *args, **kwargs):
            raise AssertionError("disabled startup must not touch the tracer")

    assert telemetry.start_startup_span(FailTracer(), "model", 1.0, 2.0) is None
    assert (
        telemetry.emit_startup_span(
            FailTracer(),
            telemetry.SPAN_TRAINING_STARTUP_IMPORTS,
            1.0,
            2.0,
            context=Context(),
            root_start=1.0,
            root_open_time=2.0,
        )
        is None
    )


def test_training_wires_startup_children_to_constants_and_explicit_parent_context():
    source = TRAINING_PATH.read_text()
    owner = next(
        node
        for node in ast.parse(TELEMETRY_PATH.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "_TrainerLifecycle"
    )
    emit_helper = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "emit_startup_phase"
    )
    calls = [node for node in ast.walk(emit_helper) if isinstance(node, ast.Call)]
    emit_call = next(
        call
        for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == "emit_startup_span"
    )
    keywords = {keyword.arg: keyword.value for keyword in emit_call.keywords}

    assert ast.unparse(keywords["context"]) == "self._startup_parent_context"
    assert "_startup_span_context" not in ast.unparse(emit_call)
    for constant in EXPECTED_STARTUP_SPANS:
        if constant in {"SPAN_TRAINING_STARTUP", "SPAN_TRAINING_STARTUP_PYTHON"}:
            continue
        assert f"_otel.{constant}" in source
    assert "_otel.SPAN_TRAINING_STARTUP_DATALOADER" in source
    assert "_otel.managed_span(_otel.JOB, _otel.SPAN_TRAINING_STARTUP_DATALOADER)" in source
    assert "_otel.SPAN_GPU_SNIFF_STARTUP" in source
    assert "megatron.startup.inprocess_setup" not in source


def _load_training_startup_boundary(telemetry, tracer):
    """Execute the real facade and the first pretrain emission statements."""
    telemetry._ACTIVE_TRAINER_HANDLE = telemetry._ManagedTrainerTelemetry(
        SimpleNamespace(tracer=tracer, shutdown=lambda: None)
    )
    namespace = {"_otel": telemetry}
    pretrain = _function(TRAINING_PATH, "pretrain")
    start = next(
        i
        for i, node in enumerate(pretrain.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "_otel.start_training_startup"
    )
    # Root, imports, and argument parsing: use the production statements, not
    # a test copy of their argument selection or timestamp arithmetic.
    boundary = compile(
        ast.Module(body=pretrain.body[start : start + 3], type_ignores=[]),
        str(TRAINING_PATH),
        "exec",
    )

    def emit(program_start, main_entry):
        namespace.update(
            model_type="model",
            program_start=program_start,
            main_entry=main_entry,
            pretrain_entry=115.0,
        )
        exec(boundary, namespace)
        telemetry.end_training_startup()

    return emit


def test_production_startup_splits_python_and_imports(monkeypatch, telemetry, span_exporter):
    tracer, exporter = span_exporter
    calls = []

    def process_start():
        calls.append(True)
        return 100.0

    monkeypatch.setattr(span_utilities, "linux_process_create_time", process_start)
    monkeypatch.setattr(telemetry.time, "time", lambda: 120.0)
    emit = _load_training_startup_boundary(telemetry, tracer)
    with tracer.start_as_current_span("ambient"):
        emit(102.0, 110.0)
    spans = {span.name: span for span in exporter.get_finished_spans()}
    root = spans[telemetry.SPAN_TRAINING_STARTUP]
    python = spans[telemetry.SPAN_TRAINING_STARTUP_PYTHON]
    imports = spans[telemetry.SPAN_TRAINING_STARTUP_IMPORTS]
    assert len(calls) == 1
    assert root.parent is None
    assert root.context.trace_id != spans["ambient"].context.trace_id
    assert (python.start_time, python.end_time) == (100_000_000_000, 102_000_000_000)
    assert (imports.start_time, imports.end_time) == (102_000_000_000, 110_000_000_000)
    for child in (python, imports):
        assert child.parent.span_id == root.context.span_id
        assert child.context.trace_id == root.context.trace_id
    assert spans[telemetry.SPAN_TRAINING_STARTUP_ARG_PARSE].start_time == imports.end_time

    # An in-process re-entry is not another interpreter startup. Preserve the
    # existing startup root/import behavior, but do not duplicate the new child.
    emit(102.0, 110.0)
    assert (
        sum(
            span.name == telemetry.SPAN_TRAINING_STARTUP_PYTHON
            for span in exporter.get_finished_spans()
        )
        == 1
    )


@pytest.mark.parametrize(
    ("process_start", "program_start", "main_entry", "expected_root", "expected_imports"),
    [
        (None, 102.0, 110.0, 102.0, True),
        (math.nan, 102.0, 110.0, 102.0, True),
        (math.inf, 102.0, 110.0, 102.0, True),
        (-1.0, 102.0, 110.0, 102.0, True),
        (105.0, 102.0, 110.0, 102.0, True),
        (100.0, None, 110.0, 100.0, False),
        (100.0, math.nan, 110.0, 100.0, False),
        (100.0, 112.0, 110.0, 100.0, False),
        (100.0, 102.0, math.inf, 120.0, False),
        (100.0, 102.0, 130.0, 100.0, False),
    ],
)
def test_production_startup_missing_boundaries_do_not_invent_python_span(
    monkeypatch,
    telemetry,
    span_exporter,
    process_start,
    program_start,
    main_entry,
    expected_root,
    expected_imports,
):
    tracer, exporter = span_exporter

    def os_start():
        if process_start is None:
            raise OSError("/proc unavailable")
        return process_start

    monkeypatch.setattr(span_utilities, "linux_process_create_time", os_start)
    monkeypatch.setattr(telemetry.time, "time", lambda: 120.0)
    monkeypatch.setattr("opentelemetry.sdk.trace.time_ns", lambda: 120_000_000_000)
    _load_training_startup_boundary(telemetry, tracer)(program_start, main_entry)
    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert spans[telemetry.SPAN_TRAINING_STARTUP].start_time == int(expected_root * 1e9)
    assert telemetry.SPAN_TRAINING_STARTUP_PYTHON not in spans
    assert (telemetry.SPAN_TRAINING_STARTUP_IMPORTS in spans) == expected_imports
    if expected_imports:
        assert spans[telemetry.SPAN_TRAINING_STARTUP_IMPORTS].start_time == int(program_start * 1e9)


@pytest.mark.parametrize(
    "entrypoint",
    ["pretrain_hybrid.py", "pretrain_gpt.py", "pretrain_vlm.py", "examples/mimo/pretrain_mimo.py"],
)
def test_entrypoints_preserve_the_local_import_boundary(entrypoint):
    tree = ast.parse((REPO_ROOT / entrypoint).read_text())
    capture = next(
        i
        for i, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_PROGRAM_START_TIME"
            for target in node.targets
        )
    )
    # Only the minimal time import and any required future import precede the
    # early timestamp.
    imports = [
        node for node in tree.body[:capture] if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert all(
        (isinstance(node, ast.Import) and all(alias.name == "time" for alias in node.names))
        or (isinstance(node, ast.ImportFrom) and node.module == "__future__")
        for node in imports
    )
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "set_startup_timestamps"
    ]
    assert any(
        {"program_start": "_PROGRAM_START_TIME", "main_entry": "_MAIN_ENTRY_TIME"}.items()
        <= {kw.arg: ast.unparse(kw.value) for kw in call.keywords}.items()
        for call in calls
    )


def test_telemetry_setup_interval_wraps_the_full_setup_call():
    runtime_services = _function(GLOBAL_VARS_PATH, "initialize_runtime_services")
    wrapper = _function(GLOBAL_VARS_PATH, "_set_telemetry_with_timestamps")
    runtime_services_source = ast.unparse(runtime_services)
    wrapper_source = ast.unparse(wrapper)

    assert (
        "_set_telemetry_with_timestamps(args, include_training=training)" in runtime_services_source
    )
    assert "setup_start = time.time()" in wrapper_source
    assert (
        "try:\n        _set_telemetry(args, include_training=include_training)\n    finally:"
        in wrapper_source
    )
    assert "_GLOBAL_TELEMETRY_SETUP_TIMESTAMPS = (setup_start, time.time())" in wrapper_source
    assert "start_span" not in wrapper_source
    assert "add_event" not in wrapper_source


def test_existing_startup_timer_output_names_and_values_are_preserved():
    pretrain = _function(TRAINING_PATH, "pretrain")
    source = ast.unparse(pretrain)

    assert _assigned_dict(pretrain, "startup_timers") == {
        "startup-program-entry-spread": "program_start - program_start_global",
        "startup-library-setup": "main_entry - program_start",
        "startup-program-setup": "pretrain_entry - main_entry",
        "startup-in-process-setup": "timestamp_after_inprocess_setup - pretrain_entry",
        "startup-in-job-setup": ("timestamp_after_in_job_setup - timestamp_after_inprocess_setup"),
        "startup-initialize-megatron": (
            "timestamp_after_initialize_megatron - timestamp_after_in_job_setup"
        ),
        "startup-set-jit-fusion-options": (
            "timestamp_after_set_jit_fusion_options - timestamp_after_initialize_megatron"
        ),
        "all-reduce-start-timestamps-tensor": (
            "megatron_init_end - timestamp_after_set_jit_fusion_options"
        ),
        "startup-megatron-init-local": "megatron_init_end - pretrain_entry",
        "startup-megatron-init-global": "megatron_init_end - program_start_global",
    }
    assert _assigned_dict(pretrain, "startup_timestamps") == {
        "before library-setup": "program_start",
        "after library-setup": "main_entry",
        "before megatron-init": "pretrain_entry",
    }
    assert "timers.log(list(startup_timers.keys()), barrier=True)" in source
    assert "print_rank_0(f'[{name}] datetime: {ts_str}')" in source
