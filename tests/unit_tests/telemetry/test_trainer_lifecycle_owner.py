# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Trainer lifecycle ownership and process hook behavior."""

import ast
import atexit
import os
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from nemo.lens import span_utilities
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture
def owner_runtime(monkeypatch, telemetry_module):
    telemetry = telemetry_module
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    events = []
    tracer = provider.get_tracer("trainer-owner")

    def shutdown():
        events.append(("provider", [span.name for span in exporter.get_finished_spans()]))

    handle = telemetry._ManagedTrainerTelemetry(
        SimpleNamespace(tracer=tracer, shutdown=shutdown, is_exporting=True)
    )
    handle._publication.callback(lambda: events.append(("carrier",)))
    monkeypatch.setattr(telemetry, "_ACTIVE_TRAINER_HANDLE", handle)
    monkeypatch.setattr(telemetry, "_PYTHON_STARTUP_RECORDED", False)
    monkeypatch.setattr(telemetry, "_TRAINER_EXIT_HOOKS", telemetry._TrainerExitHooks())
    monkeypatch.setattr(telemetry, "is_enabled", lambda group: True)
    monkeypatch.setattr(span_utilities, "linux_process_create_time", lambda: 100.0)
    monkeypatch.setattr(telemetry.time, "time", lambda: 120.0)
    try:
        yield telemetry, handle, tracer, exporter, events
    finally:
        handle.shutdown()
        provider.shutdown()


def test_facade_topology_reset_and_teardown_order(owner_runtime):
    telemetry, handle, tracer, exporter, events = owner_runtime
    with tracer.start_as_current_span("ambient"):
        telemetry.start_training_startup("model", 102.0, 110.0)
        telemetry.emit_training_startup_phase("phase", 110.0, 115.0)
        telemetry.end_training_startup()
        telemetry.prepare_training_loop()
        telemetry.start_training_loop_pass(3)
        telemetry.start_training_loop_pass(4)
        telemetry.end_training_loop_pass()
        telemetry.prepare_training_loop()
        telemetry.start_training_loop_pass(5)
        telemetry.shutdown_training(handle)
        telemetry.shutdown_training(handle)
    spans = list(exporter.get_finished_spans())
    startup = next(span for span in spans if span.name == telemetry.SPAN_TRAINING_STARTUP)
    loops = [span for span in spans if span.name == telemetry.SPAN_TRAINING_ITER_BLOCK]
    assert all(span.parent is None for span in [startup, *loops])
    assert len({span.context.trace_id for span in [startup, *loops]}) == 4
    assert [span.links[0].context.span_id for span in loops] == [
        startup.context.span_id,
        loops[0].context.span_id,
        startup.context.span_id,
    ]
    assert events[0][0] == "provider"
    assert events[0][1].count(telemetry.SPAN_TRAINING_ITER_BLOCK) == 3
    assert telemetry.SPAN_TRAINING_STARTUP in events[0][1]
    assert events[1:] == [("carrier",)]
    assert telemetry._ACTIVE_TRAINER_HANDLE is None


def test_shutdown_closes_unfinished_startup_and_preserves_failure(owner_runtime):
    telemetry, handle, _, exporter, events = owner_runtime
    failure = RuntimeError("provider failed")

    def fail():
        events.append(("provider",))
        assert telemetry.SPAN_TRAINING_STARTUP in [
            span.name for span in exporter.get_finished_spans()
        ]
        raise failure

    handle._handle.shutdown = fail
    telemetry.start_training_startup("model", 102.0, 110.0)
    with pytest.raises(RuntimeError) as caught:
        telemetry.shutdown_training(handle)
    assert caught.value is failure
    telemetry.shutdown_training(handle)
    assert events == [("provider",), ("carrier",)]
    assert telemetry._ACTIVE_TRAINER_HANDLE is None


def test_python_startup_is_process_lifetime_even_if_first_child_was_suppressed(owner_runtime):
    telemetry, handle, tracer, exporter, _ = owner_runtime
    telemetry.start_training_startup("model", None, 110.0)
    telemetry.end_training_startup()
    assert telemetry._PYTHON_STARTUP_RECORDED
    other = telemetry._TrainerLifecycle(SimpleNamespace(tracer=tracer))
    other.start_startup("model", 102.0, 110.0)
    other.end_startup()
    assert not any(
        span.name == telemetry.SPAN_TRAINING_STARTUP_PYTHON
        for span in exporter.get_finished_spans()
    )


def test_disabled_handle_shutdown_still_delegates(monkeypatch, telemetry_module):
    calls = []
    monkeypatch.setattr(telemetry_module, "_ACTIVE_TRAINER_HANDLE", None)
    telemetry_module.shutdown_training(
        SimpleNamespace(is_exporting=False, shutdown=lambda: calls.append("shutdown"))
    )
    telemetry_module.shutdown_training(None)
    assert calls == ["shutdown"]


def test_real_disabled_handle_shutdown_is_idempotent_with_ambient_providers(
    monkeypatch, telemetry_module
):
    from nemo.lens.handle import TelemetryHandle
    from opentelemetry import metrics

    calls = []
    provider = SimpleNamespace(
        force_flush=lambda **kwargs: calls.append(("flush", kwargs)),
        shutdown=lambda: calls.append(("shutdown",)),
    )
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(metrics, "get_meter_provider", lambda: provider)
    monkeypatch.setattr(telemetry_module, "_ACTIVE_TRAINER_HANDLE", None)
    handle = TelemetryHandle(tracer=None, meter=None, is_exporting=False)
    telemetry_module.shutdown_training(handle)
    telemetry_module.shutdown_training(handle)
    assert calls == [
        ("flush", {"timeout_millis": 5000}),
        ("shutdown",),
        ("flush", {"timeout_millis": 5000}),
        ("shutdown",),
    ]


def test_disabled_lifecycle_still_runs_terminal_application_callback(monkeypatch, telemetry_module):
    telemetry = telemetry_module
    monkeypatch.setattr(telemetry, "_ACTIVE_TRAINER_HANDLE", None)
    monkeypatch.setattr(telemetry, "is_enabled", lambda group: False)

    @contextmanager
    def managed_span(group, name, tracer=None, **attributes):
        assert group == telemetry.CKPT
        yield None

    monkeypatch.setattr(telemetry, "managed_span", managed_span)
    monkeypatch.setattr(atexit, "register", lambda *args: pytest.fail("disabled hook"))
    telemetry.start_training_startup("model", None, None)
    telemetry.emit_training_startup_phase("phase", None, None)
    telemetry.end_training_startup()
    telemetry.prepare_training_loop()
    telemetry.start_training_loop_pass(1)
    telemetry.end_training_loop_pass()
    telemetry.install_training_exit_hooks(get_graceful_drain=lambda: pytest.fail("disabled policy"))
    calls = []
    result = telemetry.finalize_training_exit(
        lambda **kwargs: calls.append(kwargs) or "result", terminate=True
    )
    assert result == "result"
    assert calls == [{"blocking": True, "terminate": True}]


def test_terminal_facade_closes_pass_before_application_drain(monkeypatch, owner_runtime):
    telemetry, _, tracer, exporter, _ = owner_runtime

    def managed_span(group, name, tracer=None, **attributes):
        assert group == telemetry.CKPT
        return owner_runtime[2].start_as_current_span(name, attributes=attributes)

    monkeypatch.setattr(telemetry, "managed_span", managed_span)
    telemetry.start_training_loop_pass(9)
    calls = []

    def drain(**kwargs):
        calls.append(kwargs)
        assert telemetry.SPAN_TRAINING_ITER_BLOCK in [
            span.name for span in exporter.get_finished_spans()
        ]
        return 7

    assert telemetry.finalize_training_exit(drain, terminate=True) == 7
    assert calls == [{"blocking": True, "terminate": True}]
    spans = {span.name: span for span in exporter.get_finished_spans()}
    terminal = spans[telemetry.SPAN_CHECKPOINT_EXIT_FINALIZE]
    assert terminal.parent is None
    assert terminal.context.trace_id != spans[telemetry.SPAN_TRAINING_ITER_BLOCK].context.trace_id
    assert spans[telemetry.SPAN_CHECKPOINT_SAVE_FINALIZE].parent.span_id == terminal.context.span_id


@pytest.mark.parametrize("graceful", [False, True])
@pytest.mark.parametrize("previous", ["callable", "default", "ignore"])
def test_signal_hooks_preserve_flush_close_chaining_and_reentry(
    monkeypatch, owner_runtime, graceful, previous
):
    telemetry, handle, _, _, events = owner_runtime
    registered = []
    dispatched = []
    previous_handler = {
        "callable": lambda *args: dispatched.append(("previous", *args)),
        "default": signal.SIG_DFL,
        "ignore": signal.SIG_IGN,
    }[previous]
    monkeypatch.setattr(atexit, "register", lambda callback: registered.append(callback))
    monkeypatch.setattr(signal, "getsignal", lambda signum: previous_handler)
    monkeypatch.setattr(signal, "signal", lambda *args: dispatched.append(("install", *args)))
    monkeypatch.setattr(os, "kill", lambda *args: dispatched.append(("kill", *args)))
    monkeypatch.setattr(
        trace,
        "get_tracer_provider",
        lambda: SimpleNamespace(force_flush=lambda: events.append(("flush",))),
    )
    telemetry.install_training_exit_hooks(get_graceful_drain=lambda: graceful)
    telemetry.install_training_exit_hooks(get_graceful_drain=lambda: pytest.fail("repeated policy"))
    assert len(registered) == 1
    assert len(dispatched) == 1
    handler = dispatched[0][2]
    handler(signal.SIGTERM, None)
    handler(signal.SIGTERM, None)
    assert [event[0] for event in events] == (["flush"] if graceful else ["provider", "carrier"])
    assert not handle._closed if graceful else handle._closed
    assert sum(event[0] == "previous" for event in dispatched) == (
        2 if previous == "callable" else 0
    )
    assert sum(event[0] == "kill" for event in dispatched) == (2 if previous == "default" else 0)
    registered[0]()
    assert handle._closed


@pytest.mark.parametrize("failure", [ValueError, OSError])
def test_signal_install_failure_keeps_atexit_fallback(monkeypatch, owner_runtime, failure):
    telemetry, handle, _, _, _ = owner_runtime
    registered = []
    monkeypatch.setattr(atexit, "register", lambda callback: registered.append(callback))
    monkeypatch.setattr(signal, "getsignal", lambda signum: signal.SIG_IGN)

    def fail(*args):
        raise failure("unsupported")

    monkeypatch.setattr(signal, "signal", fail)
    telemetry.install_training_exit_hooks(get_graceful_drain=lambda: False)
    telemetry.install_training_exit_hooks(get_graceful_drain=lambda: False)
    assert len(registered) == 1
    registered[0]()
    assert handle._closed


def test_hook_policy_failure_keeps_hard_shutdown_and_uses_current_handle(
    monkeypatch, owner_runtime
):
    telemetry, handle, _, _, _ = owner_runtime
    registered = []
    monkeypatch.setattr(atexit, "register", lambda callback: registered.append(callback))
    monkeypatch.setattr(signal, "getsignal", lambda signum: signal.SIG_IGN)
    monkeypatch.setattr(signal, "signal", lambda *args: None)

    def policy():
        raise RuntimeError("policy unavailable")

    telemetry.install_training_exit_hooks(get_graceful_drain=policy)
    calls = []
    monkeypatch.setattr(
        telemetry, "_ACTIVE_TRAINER_HANDLE", SimpleNamespace(shutdown=lambda: calls.append("new"))
    )
    telemetry._TRAINER_EXIT_HOOKS.handle_sigterm(signal.SIGTERM, None)
    assert calls == ["new"]
    assert not handle._closed


def test_training_has_no_raw_otel_provider_context_or_hook_implementation():
    path = Path(__file__).resolve().parents[3] / "megatron/training/training.py"
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(("opentelemetry", "nemo.lens"))
        if isinstance(node, ast.Import):
            assert not any(
                alias.name.startswith(("opentelemetry", "nemo.lens")) for alias in node.names
            )
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"tracer", "get_tracer_provider", "force_flush"}


def test_missing_lens_lifecycle_is_noop_in_isolated_process():
    path = Path(__file__).resolve().parents[3] / "megatron/core/telemetry/telemetry.py"
    training_path = Path(__file__).resolve().parents[3] / "megatron/training/training.py"
    script = """
import ast
import builtins
import importlib.util
import sys
sys.dont_write_bytecode = True
original_import = builtins.__import__
def import_without_lens(name, *args, **kwargs):
    if name == "nemo" or name.startswith("nemo."):
        raise ModuleNotFoundError("Lens unavailable", name="nemo")
    return original_import(name, *args, **kwargs)
builtins.__import__ = import_without_lens
spec = importlib.util.spec_from_file_location("absent_lens_shim", sys.argv[1])
telemetry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(telemetry)
assert not telemetry._AVAILABLE
assert telemetry.start_root_span(telemetry.JOB, "root", object()) is None
assert telemetry.end_root_span(None, None) is None
telemetry.start_training_startup("model", None, None)
telemetry.emit_training_startup_phase("phase", None, None)
telemetry.end_training_startup()
telemetry.prepare_training_loop()
telemetry.start_training_loop_pass(1)
telemetry.end_training_loop_pass()
def forbidden_policy():
    raise AssertionError("missing Lens must not read signal policy")
telemetry.install_training_exit_hooks(get_graceful_drain=forbidden_policy)
assert not telemetry._TRAINER_EXIT_HOOKS._installed
calls = []
assert telemetry.finalize_training_exit(
    lambda **kwargs: calls.append(kwargs) or 7, terminate=True
) == 7
assert calls == [{"blocking": True, "terminate": True}]
training_tree = ast.parse(open(sys.argv[2]).read(), filename=sys.argv[2])
train = next(
    node for node in training_tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "train"
)
parents = {
    child: parent for parent in ast.walk(train) for child in ast.iter_child_nodes(parent)
}
intermediate = next(
    node for node in ast.walk(train)
    if isinstance(node, ast.With)
    and "SPAN_CHECKPOINT_SAVE_FINALIZE" in ast.unparse(node.items[0].context_expr)
    and "terminate=False" in ast.unparse(node)
)
parent = parents[intermediate]
statements = next(
    value for _field, value in ast.iter_fields(parent)
    if isinstance(value, list) and intermediate in value
)
index = statements.index(intermediate)
drain_calls = []
exec(
    compile(
        ast.Module(body=statements[index - 1:index + 1], type_ignores=[]),
        sys.argv[2],
        "exec",
    ),
    {
        "_otel": telemetry,
        "maybe_finalize_async_save": lambda **kwargs: drain_calls.append(kwargs),
    },
)
assert drain_calls == [{"blocking": True, "terminate": False}]
telemetry.shutdown_training(None)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), str(training_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_lifecycle_documentation_preserves_parent_edges():
    directory = Path(__file__).resolve().parents[3] / "docs/user-guide/observability"
    text = (directory / "trace-lifecycle.md").read_text()
    edges = {
        tuple(cell.strip().strip("`") for cell in line.strip("|").split("|"))
        for line in text.splitlines()
        if line.startswith("| `nv.")
    }
    assert {
        ("nv.dl.training.startup", "nv.dl.training.startup.model_init"),
        ("nv.dl.training.startup.model_init", "nv.dl.training.checkpoint.load"),
        ("nv.dl.training.checkpoint.load", "nv.dl.training.checkpoint.load.io_read"),
        ("nv.dl.training.startup", "nv.dl.training.startup.dataloader"),
        ("nv.dl.training.iter_block", "nv.dl.training.evaluate"),
        ("nv.dl.training.evaluate", "nv.dl.training.evaluate.step"),
    } <= edges
    assert "`nv.mlm.checkpoint.exit_finalize` is a separate fresh root" in text
    for document in directory.glob("*.md"):
        assert "megatron.startup" not in document.read_text()
        assert "megatron.checkpoint." not in document.read_text()
