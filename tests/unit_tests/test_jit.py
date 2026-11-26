# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib.util
from pathlib import Path

import pytest
import torch

import megatron.core.utils as utils

JIT_PATH = Path(__file__).resolve().parents[2] / "megatron" / "core" / "jit.py"


def _load_fresh_jit_module():
    """Execute a private copy of megatron/core/jit.py.

    The shared ``megatron.core.jit`` module is left untouched, so decorators already
    applied by other modules keep their references.
    """
    spec = importlib.util.spec_from_file_location("_test_megatron_core_jit", JIT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _legacy_script(func):
    return func


@pytest.mark.parametrize(
    "torch_min_version_result,has_script,expected",
    [
        # PyTorch without the public torch.jit.script API (pytorch/pytorch#203621).
        (True, False, "compile"),
        (True, True, "compile"),
        (False, True, "script"),
        (ImportError, False, "noop"),
    ],
)
def test_jit_fuser_selection(monkeypatch, torch_min_version_result, has_script, expected):
    """jit_fuser must not require torch.jit.script on PyTorch >= 2.2."""

    def is_torch_min_version(*args, **kwargs):
        if torch_min_version_result is ImportError:
            raise ImportError("version unavailable")
        return torch_min_version_result

    monkeypatch.setattr(utils, "is_torch_min_version", is_torch_min_version)
    if has_script:
        monkeypatch.setattr(torch.jit, "script", _legacy_script, raising=False)
    else:
        monkeypatch.delattr(torch.jit, "script", raising=False)

    module = _load_fresh_jit_module()

    expected_fuser = {
        "compile": torch.compile,
        "script": _legacy_script,
        "noop": module.noop_decorator,
    }[expected]
    assert module.jit_fuser is expected_fuser

    module.disable_jit_fuser()

    def function():
        return 1

    assert module.jit_fuser(function) is function
