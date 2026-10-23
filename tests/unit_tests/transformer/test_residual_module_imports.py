# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Canonical residual modules and the legacy mHC export preserve class identity."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_IMPORT_CHECK = r"""
import importlib
import sys
import typing

prefix = "megatron.core.transformer."
first = {
    "common-first": ("transformer_layer",),
    "canonical-first": (
        "hyper_connection_transformer_layer",
        "wide_residual_transformer_layer",
        "wide_residual",
        "residual_recompute",
    ),
    "connection-first": ("residual_connection", "residual_recompute"),
    "recompute-first": ("residual_recompute", "residual_connection"),
}[sys.argv[1]]
for name in first:
    importlib.import_module(prefix + name)

connection = importlib.import_module(prefix + "residual_connection")
recompute = importlib.import_module(prefix + "residual_recompute")
for method in (
    connection.ResidualConnection._checkpoint_read,
    connection.ResidualConnection._checkpoint_write,
):
    assert typing.get_type_hints(method)["context"] is recompute.ResidualStreamRecomputeContext

package = importlib.import_module("megatron.core.transformer")
for symbol, new_name in (
    ("HyperConnectionTransformerLayer", "hyper_connection_transformer_layer"),
    ("WideResidualTransformerLayer", "wide_residual_transformer_layer"),
    ("TransformerLayer", "transformer_layer"),
    ("TransformerLayerSubmodules", "transformer_layer"),
):
    assert getattr(package, symbol) is getattr(importlib.import_module(prefix + new_name), symbol)

from megatron.core.transformer.transformer_layer import HyperConnectionTransformerLayer

legacy = importlib.import_module(prefix + "transformer_layer")
canonical = importlib.import_module(prefix + "hyper_connection_transformer_layer")
assert HyperConnectionTransformerLayer is canonical.HyperConnectionTransformerLayer
assert legacy.HyperConnectionTransformerLayer is canonical.HyperConnectionTransformerLayer
assert not hasattr(legacy, "_missing_residual_symbol")

class DownstreamHyperConnectionLayer(HyperConnectionTransformerLayer):
    pass

assert DownstreamHyperConnectionLayer.__bases__ == (canonical.HyperConnectionTransformerLayer,)
"""


@pytest.mark.parametrize(
    "import_order", ["common-first", "canonical-first", "connection-first", "recompute-first"]
)
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
