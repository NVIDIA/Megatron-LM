# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for ``megatron.core.telemetry.telemetry``."""

import ast
import builtins
import importlib.util
import re
import subprocess
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

MODULE = "megatron.core.telemetry.telemetry"
PACKAGE = "megatron.core.telemetry"
REPO_ROOT = Path(__file__).resolve().parents[3]
TELEMETRY_PATH = REPO_ROOT / "megatron/core/telemetry/telemetry.py"
GLOBAL_VARS_PATH = REPO_ROOT / "megatron/training/global_vars.py"
TRAINING_PATH = REPO_ROOT / "megatron/training/training.py"

EXPECTED_GROUPS = frozenset(
    [
        "megatron.job",
        "megatron.train",
        "megatron.ckpt",
        "megatron.eval",
        "megatron.inference",
        "megatron.detail",
    ]
)
EXPECTED_PRESETS = {
    "default": frozenset(["megatron.job", "megatron.ckpt", "megatron.eval", "megatron.inference"]),
    "per_step": frozenset(
        ["megatron.job", "megatron.train", "megatron.ckpt", "megatron.eval", "megatron.inference"]
    ),
    "profiling": EXPECTED_GROUPS,
}


def _drop_telemetry_module(monkeypatch):
    sys.modules.pop(MODULE, None)
    package = sys.modules.get(PACKAGE)
    if package is not None:
        monkeypatch.delattr(package, "telemetry", raising=False)


def _import_fresh(monkeypatch):
    _drop_telemetry_module(monkeypatch)
    spec = importlib.util.spec_from_file_location(MODULE, TELEMETRY_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, MODULE, module)
    spec.loader.exec_module(module)
    return module


def _block_lens_import(monkeypatch):
    for name in list(sys.modules):
        if name == "nemo.lens" or name.startswith("nemo.lens."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "nemo.lens" or name.startswith("nemo.lens."):
            raise ModuleNotFoundError(
                "nemo-lens intentionally hidden for this test", name="nemo.lens"
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)


def _install_fake_lens(monkeypatch, enabled_groups=None):
    calls = []
    enabled = set(enabled_groups or [])

    class FakeSpanRegistry:
        @classmethod
        def register(cls, namespace, groups, presets=None):
            calls.append((namespace, frozenset(groups), dict(presets or {})))

    class FakeSpan:
        def __init__(self):
            self.attributes = {}

        def is_recording(self):
            return True

        def set_attribute(self, key, value):
            self.attributes[key] = value

    @contextmanager
    def managed_span(group, name, tracer=None, **attributes):
        span = FakeSpan()
        span.group = group
        span.name = name
        span.attributes.update(attributes)
        yield span

    @contextmanager
    def span_cm(name, tracer=None, record_exception=True, **attributes):
        span = FakeSpan()
        span.name = name
        span.attributes.update(attributes)
        yield span

    def trace_fn(group, name, tracer=None):
        def decorator(func):
            return func

        return decorator

    def is_span_group_enabled(group):
        return group in enabled

    def safe_set_span_attributes(span, attributes, redact_keys=None):
        for key, value in attributes.items():
            span.set_attribute(key, value)

    nemo = types.ModuleType("nemo")
    nemo.__path__ = []
    lens = types.ModuleType("nemo.lens")
    lens.__path__ = []
    lens.SpanRegistry = FakeSpanRegistry
    lens.is_span_group_enabled = is_span_group_enabled
    lens.managed_span = managed_span
    lens.safe_set_span_attributes = safe_set_span_attributes
    lens.span_cm = span_cm
    lens.trace_fn = trace_fn
    nemo.lens = lens

    monkeypatch.setitem(sys.modules, "nemo", nemo)
    monkeypatch.setitem(sys.modules, "nemo.lens", lens)
    return calls


def _module_function_ast(path, name):
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    return next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _global_vars_function_ast(name):
    return _module_function_ast(GLOBAL_VARS_PATH, name)


def _run_real_lens_shim_probe():
    from nemo.lens import SpanRegistry

    SpanRegistry.clear()
    spec = importlib.util.spec_from_file_location(MODULE, TELEMETRY_PATH)
    telemetry = importlib.util.module_from_spec(spec)
    sys.modules[MODULE] = telemetry
    spec.loader.exec_module(telemetry)

    assert telemetry._AVAILABLE is True
    assert SpanRegistry.namespaces() == ["megatron"]
    assert SpanRegistry.groups() == EXPECTED_GROUPS
    presets = SpanRegistry.presets()
    assert {name: presets[name] for name in EXPECTED_PRESETS} == EXPECTED_PRESETS

    class RecordingSpan:
        def __init__(self):
            self.attributes = {}

        def is_recording(self):
            return True

        def set_attribute(self, key, value):
            self.attributes[key] = value

    span = RecordingSpan()
    telemetry.set_attributes(
        span, {"count": 3, "missing": None, "nested": {"unsupported": True}, "password": "secret"}
    )
    assert span.attributes == {"count": 3, "password": "[REDACTED]"}


def test_group_constants_and_presets(monkeypatch):
    _block_lens_import(monkeypatch)
    telemetry = _import_fresh(monkeypatch)

    assert telemetry.GROUPS == EXPECTED_GROUPS
    assert telemetry.PRESETS == EXPECTED_PRESETS
    assert "all" not in telemetry.PRESETS


def test_no_lens_fallbacks_are_no_op(monkeypatch):
    _block_lens_import(monkeypatch)
    telemetry = _import_fresh(monkeypatch)

    assert telemetry.is_enabled(telemetry.JOB) is False

    with telemetry.managed_span(telemetry.JOB, "megatron.test") as span:
        assert span is None

    with telemetry.span_cm("megatron.test") as span:
        assert span is None

    def original():
        return "ok"

    assert telemetry.trace_fn(telemetry.JOB, "megatron.test")(original) is original
    assert telemetry.set_attributes(None, {"key": "value"}) is None
    assert telemetry.set_current_span_attributes({"key": "value"}) is None
    assert telemetry.setup(object()) is None


def test_import_registers_groups_with_lens(monkeypatch):
    calls = _install_fake_lens(monkeypatch)
    telemetry = _import_fresh(monkeypatch)

    assert calls == [("megatron", EXPECTED_GROUPS, EXPECTED_PRESETS)]
    assert telemetry.GROUPS == EXPECTED_GROUPS
    assert telemetry.PRESETS == EXPECTED_PRESETS


def test_installed_lens_missing_required_symbol_fails_import(monkeypatch):
    _install_fake_lens(monkeypatch)
    monkeypatch.delattr(sys.modules["nemo.lens"], "safe_set_span_attributes")

    with pytest.raises(ImportError, match="safe_set_span_attributes"):
        _import_fresh(monkeypatch)


def test_installed_lens_missing_setup_surface_fails_visibly(monkeypatch):
    _install_fake_lens(monkeypatch)
    telemetry = _import_fresh(monkeypatch)

    with pytest.raises(ImportError, match="NemoLensConfig"):
        telemetry.setup(object())


def test_lens_helpers_are_proxied(monkeypatch):
    _install_fake_lens(monkeypatch, enabled_groups={"megatron.job"})
    telemetry = _import_fresh(monkeypatch)

    assert telemetry.is_enabled(telemetry.JOB) is True
    assert telemetry.is_enabled(telemetry.TRAIN) is False

    with telemetry.managed_span(telemetry.JOB, "megatron.test", answer=42) as span:
        assert span.group == telemetry.JOB
        assert span.name == "megatron.test"
        assert span.attributes["answer"] == 42
        telemetry.set_attributes(span, {"phase": "setup"})
        assert span.attributes["phase"] == "setup"


def test_attribute_setters_contain_failures(monkeypatch):
    _install_fake_lens(monkeypatch)
    telemetry = _import_fresh(monkeypatch)

    def fail(*args, **kwargs):
        raise RuntimeError("attribute failure")

    monkeypatch.setattr(telemetry, "_safe_set_span_attributes", fail)
    assert telemetry.set_attributes(object(), {"phase": "setup"}) is None

    opentelemetry = types.ModuleType("opentelemetry")
    trace = types.ModuleType("opentelemetry.trace")
    trace.get_current_span = fail
    opentelemetry.trace = trace
    monkeypatch.setitem(sys.modules, "opentelemetry", opentelemetry)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", trace)
    assert telemetry.set_current_span_attributes({"phase": "setup"}) is None


def test_setup_uses_current_lens_signature(monkeypatch):
    _install_fake_lens(monkeypatch)
    lens = sys.modules["nemo.lens"]
    handle = types.SimpleNamespace(is_exporting=False, shutdown=lambda: None)

    class FakeConfig:
        enabled = False
        logs_enabled = False
        service_name = "nemo"

        @classmethod
        def from_env(cls, **kwargs):
            assert kwargs == {"prefix": "MEGATRON_OTEL", "fallback_prefix": "NEMO_LENS"}
            return cls()

    def strict_setup(config, resource_attributes=None):
        assert config.enabled is True
        assert config.service_name == "trainer-service"
        assert resource_attributes == {"service.name": "trainer-service", "trainer": "mapped"}
        return handle

    lens.NemoLensConfig = FakeConfig
    lens.setup_telemetry = strict_setup
    resource_module = types.ModuleType("megatron.core.telemetry.resource_attrs")
    resource_module.compose_telemetry_resource_attrs = lambda _args, *, include_training: {
        "service.name": "trainer-service",
        "trainer": "mapped",
    }
    monkeypatch.setitem(sys.modules, resource_module.__name__, resource_module)
    telemetry = _import_fresh(monkeypatch)
    args = types.SimpleNamespace(otel_enabled=True, otel_service_name=None)

    result = telemetry.setup(args)
    assert result is not handle
    assert result.is_exporting is False
    result.shutdown()


def test_setup_does_not_map_resources_when_disabled(monkeypatch):
    _install_fake_lens(monkeypatch)
    lens = sys.modules["nemo.lens"]
    handle = types.SimpleNamespace(is_exporting=False)

    class FakeConfig:
        enabled = False
        logs_enabled = False
        service_name = "nemo"

        @classmethod
        def from_env(cls, **_kwargs):
            return cls()

    def strict_setup(config, resource_attributes=None):
        assert config.enabled is False
        assert resource_attributes == {}
        return handle

    lens.NemoLensConfig = FakeConfig
    lens.setup_telemetry = strict_setup
    telemetry = _import_fresh(monkeypatch)
    args = types.SimpleNamespace(otel_enabled=False, otel_service_name=None)

    assert telemetry.setup(args) is handle


def test_current_span_attributes_use_lazy_lookup(monkeypatch):
    _install_fake_lens(monkeypatch)
    telemetry = _import_fresh(monkeypatch)

    with telemetry.span_cm("megatron.test") as span:
        opentelemetry = types.ModuleType("opentelemetry")
        trace = types.ModuleType("opentelemetry.trace")
        trace.get_current_span = lambda: span
        opentelemetry.trace = trace
        monkeypatch.setitem(sys.modules, "opentelemetry", opentelemetry)
        monkeypatch.setitem(sys.modules, "opentelemetry.trace", trace)

        telemetry.set_current_span_attributes({"phase": "current"})

    assert span.attributes["phase"] == "current"


def test_current_span_attributes_ignore_real_nonrecording_span(monkeypatch, telemetry_module):
    from opentelemetry import trace

    class GuardedNonRecordingSpan(trace.NonRecordingSpan):
        def set_attribute(self, key, value):
            raise AssertionError("a non-recording span must not be mutated")

    span = GuardedNonRecordingSpan(
        trace.SpanContext(
            trace_id=1,
            span_id=1,
            is_remote=False,
            trace_flags=trace.TraceFlags.DEFAULT,
            trace_state=trace.TraceState(),
        )
    )
    monkeypatch.setattr(trace, "get_current_span", lambda: span)

    assert span.is_recording() is False
    telemetry_module.set_current_span_attributes({"phase": "ignored"})


def test_shim_registers_with_installed_lens():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--real-lens-shim-probe"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr


def test_call_sites_do_not_import_lens_helpers_directly():
    repo = REPO_ROOT
    shim = TELEMETRY_PATH
    forbidden_modules = {"nemo.lens.helpers", "nemo.lens.state"}
    forbidden_top_level_names = {
        "SpanRegistry",
        "is_span_group_enabled",
        "managed_span",
        "safe_set_span_attributes",
        "span_cm",
        "trace_fn",
    }

    offenders = []
    for path in (repo / "megatron").rglob("*.py"):
        if path == shim:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = node.module or ""
            imported_names = {alias.name for alias in node.names}
            if module in forbidden_modules:
                offenders.append(f"{path.relative_to(repo)} imports from {module}")
            if module == "nemo.lens" and imported_names & forbidden_top_level_names:
                names = ", ".join(sorted(imported_names & forbidden_top_level_names))
                offenders.append(f"{path.relative_to(repo)} imports {names} from nemo.lens")
            if module == "nemo.lens.groups" and "SpanRegistry" in imported_names:
                offenders.append(f"{path.relative_to(repo)} imports SpanRegistry from {module}")

    assert offenders == []


def test_call_sites_mutate_span_attributes_only_through_shim():
    offenders = []
    for path in (REPO_ROOT / "megatron").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in {"set_attribute", "set_attributes"}:
                continue
            if (
                node.func.attr == "set_attributes"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "_otel"
            ):
                continue
            offenders.append(
                f"{path.relative_to(REPO_ROOT)}:{node.lineno} calls {node.func.attr} directly"
            )

    assert offenders == []


def test_call_sites_do_not_use_legacy_safe_setter_name():
    offenders = []
    for path in (REPO_ROOT / "megatron").rglob("*.py"):
        if "_otel.safe_set_span_attributes" in path.read_text():
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []


def test_call_sites_do_not_use_unregistered_group_literals():
    repo = REPO_ROOT
    call = re.compile(
        r"(?:_otel_managed_span|_otel_trace_fn|_otel_sg_enabled|"
        r"_otel\.managed_span|_otel\.trace_fn|_otel\.is_enabled)"
        r"\(\s*(['\"])(?P<group>[^'\"]+)\1"
    )

    offenders = []
    for path in (repo / "megatron").rglob("*.py"):
        text = path.read_text()
        for match in call.finditer(text):
            group = match.group("group")
            if group not in EXPECTED_GROUPS:
                offenders.append(f"{path.relative_to(repo)} uses unregistered group {group!r}")

    assert offenders == []


def test_global_setup_delegates_optional_lens_policy_to_the_shim():
    function = _global_vars_function_ast("_set_telemetry")
    imports = [node for node in ast.walk(function) if isinstance(node, ast.ImportFrom)]
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]

    assert all(not (node.module or "").startswith("nemo.lens") for node in imports)
    assert any(
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "_otel"
        and call.func.attr == "setup"
        for call in calls
    )


def _run_real_lens_setup_probe():
    import os

    import nemo.lens as lens
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    for key in (
        "OTEL_RESOURCE_ATTRIBUTES",
        "OTEL_SERVICE_NAME",
        "MEGATRON_OTEL_ENABLED",
        "NEMO_LENS_ENABLED",
        "NEMO_LENS_LOGS_ENABLED",
        "MEGATRON_OTEL_LOGS_ENABLED",
    ):
        os.environ.pop(key, None)
    os.environ["MEGATRON_OTEL_METRICS_ENABLED"] = "false"
    original_carrier = os.environ.get("OTEL_RESOURCE_ATTRIBUTES")
    exporter = InMemorySpanExporter()
    real_setup = lens.setup_telemetry

    def setup(config, resource_attributes=None):
        assert config.service_name == "trainer-smoke"
        assert resource_attributes == {"service.name": "trainer-smoke"}
        return real_setup(config, span_exporter=exporter, resource_attributes=resource_attributes)

    lens.setup_telemetry = setup
    resource_module = types.ModuleType("megatron.core.telemetry.resource_attrs")
    resource_module.compose_telemetry_resource_attrs = lambda _args, *, include_training: {
        "service.name": "trainer-smoke"
    }
    sys.modules[resource_module.__name__] = resource_module
    spec = importlib.util.spec_from_file_location(MODULE, TELEMETRY_PATH)
    telemetry = importlib.util.module_from_spec(spec)
    sys.modules[MODULE] = telemetry
    spec.loader.exec_module(telemetry)
    args = types.SimpleNamespace(otel_enabled=True, otel_service_name="trainer-smoke")
    handle = telemetry.setup(args)
    try:
        assert handle.is_exporting is True
        with handle.tracer.start_as_current_span("service-name-probe"):
            pass
        from opentelemetry import trace

        trace.get_tracer_provider().force_flush()
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].resource.attributes["service.name"] == "trainer-smoke"
    finally:
        handle.shutdown()
    assert os.environ.get("OTEL_RESOURCE_ATTRIBUTES") == original_carrier


def test_obsolete_instrumentation_is_removed_without_shim_migration():
    activation_path = (
        REPO_ROOT / "megatron/core/pipeline_parallel/fine_grained_activation_offload.py"
    )
    activation_source = activation_path.read_text()
    assert "telemetry as _otel" not in activation_source
    for name in ("offload", "reload"):
        function = _module_function_ast(activation_path, name)
        assert not function.decorator_list

    ddp_path = REPO_ROOT / "megatron/core/distributed/distributed_data_parallel.py"
    assert not _module_function_ast(ddp_path, "start_grad_sync").decorator_list

    async_source = (REPO_ROOT / "megatron/training/async_utils.py").read_text()
    assert "_tag_current_span_call_idx" not in async_source
    assert "otel_bootstrap" not in async_source
    assert "build_otel_worker_bootstrap" not in async_source

    training_source = TRAINING_PATH.read_text()
    assert "_otel_mark_goodput" not in training_source
    assert "is_goodput_span" not in training_source


def test_real_lens_setup_smoke():
    pytest.importorskip("nemo.lens")
    code = (
        f"import runpy; namespace = runpy.run_path({str(Path(__file__).resolve())!r}); "
        "namespace['_run_real_lens_setup_probe']()"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    assert sys.argv[1:] == ["--real-lens-shim-probe"]
    _run_real_lens_shim_probe()
