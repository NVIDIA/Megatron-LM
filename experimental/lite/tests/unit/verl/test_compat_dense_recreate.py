# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import contextmanager
import sys
from types import ModuleType, SimpleNamespace

import pytest

from verl_mlite import compat


def _install_fake_reload_modules(monkeypatch, original):
    fp8_utils = ModuleType("verl.utils.vllm.vllm_fp8_utils")
    fp8_utils.prepare_quanted_weights_for_loading = original
    dsv4_utils = ModuleType("verl.utils.vllm.vllm_dsv4_fp8_utils")
    dsv4_utils.is_deepseek_v4_model = lambda model: True
    vllm_config = ModuleType("vllm.config")
    events = []

    @contextmanager
    def set_current_vllm_config(config):
        events.append(("enter", config))
        try:
            yield
        finally:
            events.append(("exit", config))

    vllm_config.set_current_vllm_config = set_current_vllm_config
    monkeypatch.setitem(sys.modules, fp8_utils.__name__, fp8_utils)
    monkeypatch.setitem(sys.modules, dsv4_utils.__name__, dsv4_utils)
    monkeypatch.setitem(sys.modules, vllm_config.__name__, vllm_config)
    monkeypatch.setattr(compat, "_vllm_importable", lambda: True)
    return fp8_utils, events


def test_dense_recreate_runs_under_model_runner_vllm_config(monkeypatch) -> None:
    original_calls = []

    def original(model_runner):
        original_calls.append(("prepare", model_runner.model))
        return "state"

    fp8_utils, events = _install_fake_reload_modules(
        monkeypatch, original
    )
    monkeypatch.setattr(
        compat,
        "_recreate_dense_fp8_linear_params",
        lambda model: events.append(("recreate", model)) or 1,
    )

    assert compat._patch_verl_dsv4_prepare_recreates_dense()
    runner = SimpleNamespace(model=object(), vllm_config=object())
    assert fp8_utils.prepare_quanted_weights_for_loading(runner) == "state"
    assert events == [
        ("enter", runner.vllm_config),
        ("recreate", runner.model),
        ("exit", runner.vllm_config),
    ]
    assert original_calls == [("prepare", runner.model)]


def test_dense_recreate_precedes_verl_online_loader_attachment(monkeypatch) -> None:
    order = []

    def original(model_runner):
        order.append("attach-online-loaders")
        return True

    fp8_utils, _ = _install_fake_reload_modules(monkeypatch, original)
    monkeypatch.setattr(
        compat,
        "_recreate_dense_fp8_linear_params",
        lambda model: order.append("recreate") or 1,
    )

    assert compat._patch_verl_dsv4_prepare_recreates_dense()
    runner = SimpleNamespace(model=object(), vllm_config=object())
    assert fp8_utils.prepare_quanted_weights_for_loading(runner) is True
    assert order == ["recreate", "attach-online-loaders"]


def test_dense_recreate_failure_is_not_silently_ignored(monkeypatch) -> None:
    fp8_utils, _ = _install_fake_reload_modules(monkeypatch, lambda model_runner: True)
    monkeypatch.setattr(
        compat,
        "_recreate_dense_fp8_linear_params",
        lambda model: (_ for _ in ()).throw(RuntimeError("recreate failed")),
    )

    assert compat._patch_verl_dsv4_prepare_recreates_dense()
    runner = SimpleNamespace(model=object(), vllm_config=object())
    with pytest.raises(RuntimeError, match="recreate failed"):
        fp8_utils.prepare_quanted_weights_for_loading(runner)
