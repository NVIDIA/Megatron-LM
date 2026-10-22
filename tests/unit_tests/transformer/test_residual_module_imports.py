# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Legacy residual imports, tuple specs, and pickle paths preserve object identity."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_IMPORT_CHECK = r"""
import importlib
import pickle
import sys

prefix = "megatron.core.transformer."
first = (
    ("transformer_layer",)
    if sys.argv[1] == "legacy-first"
    else (
        "hyper_connection_transformer_layer",
        "wide_residual_transformer_layer",
        "wide_residual",
        "residual_recompute",
    )
)
for name in first:
    importlib.import_module(prefix + name)

from megatron.core.transformer.spec_utils import ModuleSpec, get_module

aliases = [
    ("transformer_layer", "HyperConnectionTransformerLayer", "hyper_connection_transformer_layer"),
]
for old_name, symbol, new_name in aliases:
    old_path, new_path = prefix + old_name, prefix + new_name
    canonical = getattr(importlib.import_module(new_path), symbol)
    # Tuple specs use vars(module), so a lazy getattr alone is insufficient.
    assert get_module(ModuleSpec(module=(old_path, symbol))) is canonical
    legacy = importlib.import_module(old_path)
    assert getattr(legacy, symbol) is canonical
    assert vars(legacy)[symbol] is canonical
    assert pickle.loads(pickle.dumps(canonical)) is canonical
    legacy_pickle = b"c" + old_path.encode() + b"\n" + symbol.encode() + b"\n."
    assert pickle.loads(legacy_pickle) is canonical

package = importlib.import_module("megatron.core.transformer")
for symbol, new_name in (
    ("HyperConnectionTransformerLayer", "hyper_connection_transformer_layer"),
    ("WideResidualTransformerLayer", "wide_residual_transformer_layer"),
    ("TransformerLayer", "transformer_layer"),
    ("TransformerLayerSubmodules", "transformer_layer"),
):
    assert getattr(package, symbol) is getattr(importlib.import_module(prefix + new_name), symbol)
"""


@pytest.mark.parametrize("import_order", ["legacy-first", "canonical-first"])
def test_residual_import_paths_preserve_identity_in_fresh_interpreter(import_order):
    # A fresh interpreter exercises package initialization rather than cached pytest imports.
    environment = dict(os.environ, PYTHONOPTIMIZE="0", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-B", "-c", _IMPORT_CHECK, import_order],
        cwd=Path(__file__).resolve().parents[3],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
