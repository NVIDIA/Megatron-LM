# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
from types import ModuleType, SimpleNamespace

import pytest

from megatron.core.ops.ssm.kda import backends


def test_select_fla_backend(monkeypatch):
    sentinel = object()
    calls = []

    def fake_require(module, *symbols, **kwargs):
        calls.append((module, symbols, kwargs))
        return SimpleNamespace(chunk_kda=sentinel)

    monkeypatch.setattr(backends, "require", fake_require)

    assert backends.select_kda_recurrence("fla") is sentinel
    assert calls == [("fla.ops.kda", ("chunk_kda",), {"needed_by": "FLA Kimi Delta Attention"})]


def test_select_cudnn_backend_checks_version_and_frost(monkeypatch):
    sentinel = object()
    adapter = ModuleType("megatron.core.ops.ssm.kda.cudnn_backend")
    adapter.cudnn_chunk_kda = sentinel
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    calls = []

    def fake_require(module, *symbols, **kwargs):
        calls.append((module, symbols, kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(backends, "require", fake_require)

    assert backends.select_kda_recurrence("cudnn") is sentinel
    assert calls[0][0] == "cudnn.linear_attention.ops"
    assert calls[0][2]["min_version"] == "1.32.0"
    assert calls[0][2]["dist"] == "nvidia-cudnn-frontend"
    assert calls[1][0] == "cudnn.linear_attention.frost.common.piece_chain"


def test_select_kda_backend_rejects_unknown_name():
    with pytest.raises(ValueError, match="expected 'fla' or 'cudnn'"):
        backends.select_kda_recurrence("unknown")
