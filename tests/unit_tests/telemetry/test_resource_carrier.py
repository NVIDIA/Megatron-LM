# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the trainer Resource carrier lifecycle."""

import ast
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import nemo.lens as lens
import nemo.lens.logging_bridge as logging_bridge
import pytest
from nemo.lens.resources.attributes import (
    format_otel_resource_attributes,
    publish_otel_resource_attributes,
)

GLOBAL_VARS_PATH = Path(__file__).resolve().parents[3] / "megatron/training/global_vars.py"
RESOURCE_ENV = "OTEL_RESOURCE_ATTRIBUTES"
COMPOSED = {"service.name": "trainer-service", "nv.dl.rank": 3, "nv.dl.role": "trainer"}


def _load_shutdown_function(handle):
    tree = ast.parse(GLOBAL_VARS_PATH.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_shutdown_telemetry"
    )
    namespace = {"_GLOBAL_TELEMETRY_HANDLE": handle}
    exec(compile(ast.Module(body=[function], type_ignores=[]), GLOBAL_VARS_PATH, "exec"), namespace)
    return namespace


def _configure_boundary(
    monkeypatch,
    *,
    enabled=True,
    logs_enabled=False,
    setup_error=None,
    logging_error=None,
    shutdown_error=None,
):
    calls = {"events": []}

    class Config:
        service_name = "nemo"

        @classmethod
        def from_env(cls, **kwargs):
            calls["from_env"] = kwargs
            config = cls()
            config.enabled = enabled
            config.logs_enabled = logs_enabled
            return config

    def setup_telemetry(config, resource_attributes=None):
        calls["events"].append(("setup", os.environ.get(RESOURCE_ENV)))
        calls["setup"] = resource_attributes
        calls["config_service_name"] = config.service_name
        if setup_error is not None:
            raise setup_error
        return handle

    def shutdown():
        calls["events"].append(("shutdown", os.environ.get(RESOURCE_ENV)))
        calls["shutdown"] = calls.get("shutdown", 0) + 1
        if shutdown_error is not None:
            raise shutdown_error

    handle = SimpleNamespace(is_exporting=True, shutdown=shutdown, tracer="tracer")
    monkeypatch.setattr(lens, "NemoLensConfig", Config)
    monkeypatch.setattr(lens, "setup_telemetry", setup_telemetry)

    def setup_logging_bridge():
        calls["events"].append(("logging", os.environ.get(RESOURCE_ENV)))
        if logging_error is not None:
            raise logging_error

    monkeypatch.setattr(logging_bridge, "setup_logging_bridge", setup_logging_bridge)

    resource_attrs = types.ModuleType("megatron.core.telemetry.resource_attrs")

    def compose_resource_attrs(args, *, include_training):
        calls["mapped"] = calls.get("mapped", 0) + 1
        assert args.rank == 3
        assert include_training is True
        return dict(COMPOSED)

    resource_attrs.compose_telemetry_resource_attrs = compose_resource_attrs
    monkeypatch.setitem(sys.modules, "megatron.core.telemetry.resource_attrs", resource_attrs)
    return calls, handle


def _args(**overrides):
    values = {"rank": 3, "local_rank": 1, "otel_enabled": True, "otel_service_name": None}
    values.update(overrides)
    return SimpleNamespace(**values)


def _set_original(monkeypatch, present, original):
    if present:
        monkeypatch.setenv(RESOURCE_ENV, original)
    else:
        monkeypatch.delenv(RESOURCE_ENV, raising=False)


def _assert_original(present, original):
    assert (RESOURCE_ENV in os.environ) is present
    assert os.environ.get(RESOURCE_ENV) == original


@pytest.mark.parametrize(
    ("present", "original"), [(False, None), (True, ""), (True, "parent=value%20with%20spaces")]
)
def test_enabled_setup_publishes_exact_resource_until_shutdown(
    monkeypatch, telemetry_module, present, original
):
    _set_original(monkeypatch, present, original)
    calls, handle = _configure_boundary(monkeypatch)
    carrier = format_otel_resource_attributes(COMPOSED)

    result = telemetry_module.setup(_args())

    assert result is not handle
    assert result.tracer == "tracer"
    assert calls["from_env"] == {"prefix": "MEGATRON_OTEL", "fallback_prefix": "NEMO_LENS"}
    assert calls["setup"] == COMPOSED
    assert calls["config_service_name"] == "trainer-service"
    assert calls["events"] == [("setup", carrier)]
    assert os.environ[RESOURCE_ENV] == carrier

    with publish_otel_resource_attributes({"nv.dl.role": "checkpoint_worker"}):
        assert "nv.dl.role=checkpoint_worker" in os.environ[RESOURCE_ENV]
    assert os.environ[RESOURCE_ENV] == carrier

    result.shutdown()
    result.shutdown()

    assert calls["shutdown"] == 1
    assert calls["events"][-1] == ("shutdown", carrier)
    _assert_original(present, original)


def test_disabled_setup_does_not_map_publish_or_wrap(monkeypatch, telemetry_module):
    monkeypatch.setenv(RESOURCE_ENV, "original=carrier")
    calls, handle = _configure_boundary(monkeypatch, enabled=False)

    result = telemetry_module.setup(_args(otel_enabled=False))

    assert result is handle
    assert "mapped" not in calls
    assert calls["setup"] == {}
    assert os.environ[RESOURCE_ENV] == "original=carrier"


@pytest.mark.parametrize("failure_point", ["setup", "logging", "wrapping"])
def test_setup_failures_restore_exact_environment(monkeypatch, telemetry_module, failure_point):
    monkeypatch.setenv(RESOURCE_ENV, "")
    error = RuntimeError(f"{failure_point} failed")
    calls, _ = _configure_boundary(
        monkeypatch,
        logs_enabled=failure_point == "logging",
        setup_error=error if failure_point == "setup" else None,
        logging_error=error if failure_point == "logging" else None,
    )
    if failure_point == "wrapping":
        monkeypatch.setattr(
            telemetry_module,
            "_ManagedTrainerTelemetry",
            lambda *args, **kwargs: (_ for _ in ()).throw(error),
        )

    with pytest.raises(RuntimeError, match=f"{failure_point} failed"):
        telemetry_module.setup(_args())

    assert os.environ[RESOURCE_ENV] == ""
    assert calls.get("shutdown", 0) == (0 if failure_point == "setup" else 1)


def test_logging_failure_preserves_original_error_when_shutdown_fails(
    monkeypatch, telemetry_module
):
    monkeypatch.delenv(RESOURCE_ENV, raising=False)
    calls, _ = _configure_boundary(
        monkeypatch,
        logs_enabled=True,
        logging_error=RuntimeError("logging failed"),
        shutdown_error=RuntimeError("shutdown failed"),
    )

    with pytest.raises(RuntimeError, match="logging failed"):
        telemetry_module.setup(_args())

    assert calls["shutdown"] == 1
    assert RESOURCE_ENV not in os.environ


def test_rejects_overlapping_active_setup_before_mutating_environment(
    monkeypatch, telemetry_module
):
    monkeypatch.setenv(RESOURCE_ENV, "original=carrier")
    calls, _ = _configure_boundary(monkeypatch)
    result = telemetry_module.setup(_args())
    carrier = os.environ[RESOURCE_ENV]

    with pytest.raises(RuntimeError, match="already active"):
        telemetry_module.setup(_args())

    assert os.environ[RESOURCE_ENV] == carrier
    assert calls["mapped"] == 1
    assert len([event for event in calls["events"] if event[0] == "setup"]) == 1
    result.shutdown()
    assert os.environ[RESOURCE_ENV] == "original=carrier"


def test_global_shutdown_releases_handle_and_is_idempotent(monkeypatch, telemetry_module):
    monkeypatch.delenv(RESOURCE_ENV, raising=False)
    calls, _ = _configure_boundary(monkeypatch)
    result = telemetry_module.setup(_args())
    namespace = _load_shutdown_function(result)

    namespace["_shutdown_telemetry"]()
    namespace["_shutdown_telemetry"]()

    assert calls["shutdown"] == 1
    assert namespace["_GLOBAL_TELEMETRY_HANDLE"] is None
    assert RESOURCE_ENV not in os.environ
