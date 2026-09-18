# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Checkpoint helpers keep application work intact without Lens installed."""

import ast
import builtins
import importlib.util
import sys
from pathlib import Path

import pytest

TELEMETRY_PATH = Path(__file__).resolve().parents[3] / "megatron/core/telemetry/telemetry.py"
TRAINING_PATH = Path(__file__).resolve().parents[3] / "megatron/training/training.py"


def _checkpoint_poll_block():
    tree = ast.parse(TRAINING_PATH.read_text(), filename=str(TRAINING_PATH))
    train = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    loop = next(node for node in train.body if isinstance(node, ast.While))
    with_index = next(
        index
        for index, node in enumerate(loop.body)
        if isinstance(node, ast.With)
        and "SPAN_CHECKPOINT_SAVE_FINALIZE" in ast.unparse(node.items[0].context_expr)
    )
    return loop.body[with_index - 1 : with_index + 1]


@pytest.fixture
def telemetry_without_lens(monkeypatch):
    real_import = builtins.__import__

    def no_lens(name, *args, **kwargs):
        if name == "nemo.lens" or name.startswith("nemo.lens."):
            raise ModuleNotFoundError("Lens hidden for fallback regression", name="nemo.lens")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_lens)
    name = "_checkpoint_no_lens_probe"
    spec = importlib.util.spec_from_file_location(name, TELEMETRY_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    assert module._AVAILABLE is False
    return module


@pytest.mark.parametrize("helper", ["checkpoint_exposed_save_span", "checkpoint_save_span"])
def test_checkpoint_save_context_without_lens_still_executes_body(telemetry_without_lens, helper):
    events = []
    with getattr(telemetry_without_lens, helper)(41) as span:
        assert span is None
        events.append("application work")
    assert events == ["application work"]


def test_checkpoint_poll_without_lens_preserves_application_call(telemetry_without_lens):
    events = []

    def callback(*, blocking):
        events.append(blocking)

    namespace = {"_otel": telemetry_without_lens, "maybe_finalize_async_save": callback}
    exec(
        compile(ast.Module(body=_checkpoint_poll_block(), type_ignores=[]), TRAINING_PATH, "exec"),
        namespace,
    )
    assert events == [False]


def test_checkpoint_poll_without_lens_preserves_application_failure(telemetry_without_lens):
    failure = RuntimeError("checkpoint failed")

    def callback(*, blocking):
        assert blocking is False
        raise failure

    namespace = {"_otel": telemetry_without_lens, "maybe_finalize_async_save": callback}
    with pytest.raises(RuntimeError) as caught:
        exec(
            compile(
                ast.Module(body=_checkpoint_poll_block(), type_ignores=[]), TRAINING_PATH, "exec"
            ),
            namespace,
        )
    assert caught.value is failure
