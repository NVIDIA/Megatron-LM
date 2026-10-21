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
    ("transformer_layer", "wide_residual_layer")
    if sys.argv[1] == "legacy-first"
    else ("hyper_connection_transformer_layer", "wide_residual_transformer_layer", "wide_residual_policy")
)
for name in first:
    importlib.import_module(prefix + name)

from megatron.core.transformer.spec_utils import ModuleSpec, get_module

aliases = [
    ("transformer_layer", "HyperConnectionTransformerLayer", "hyper_connection_transformer_layer"),
    ("wide_residual_layer", "LearnedWideResidualRetention", "wide_residual"),
    ("wide_residual_layer", "StreamwiseSigmoidMap", "wide_residual"),
    ("wide_residual_layer", "StreamwiseSigmoidResidualReadout", "wide_residual"),
    ("wide_residual_layer", "StreamwiseSigmoidWideResidualRead", "wide_residual"),
    ("wide_residual_layer", "_active_vector", "wide_residual"),
    ("wide_residual_layer", "_make_padded_vector", "wide_residual"),
    ("wide_residual_layer", "_mark_replicated_tp_parameter", "wide_residual"),
    ("wide_residual_layer", "_mark_residual_map_parameter", "wide_residual"),
    ("wide_residual_layer", "_mark_retention_parameter", "wide_residual"),
    ("wide_residual_layer", "expand_wide_residual_stream", "wide_residual"),
    ("wide_residual_layer", "build_wide_residual_readout", "wide_residual"),
    ("wide_residual_layer", "StreamwiseSigmoidWideResidualConnection", "wide_residual_policy"),
    ("wide_residual_layer", "load_residual_state", "wide_residual_policy"),
    ("wide_residual_layer", "wide_residual_recompute_enabled", "wide_residual_policy"),
    ("wide_residual_layer", "WideResidualTransformerLayer", "wide_residual_transformer_layer"),
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

for symbol, new_name in (
    ("_MIN_CONTROLLER_NUMEL", "wide_residual"),
    ("_RESIDUAL_STATE_PREFIXES", "wide_residual_policy"),
):
    legacy = importlib.import_module(prefix + "wide_residual_layer")
    canonical_module = importlib.import_module(prefix + new_name)
    assert vars(legacy)[symbol] is vars(canonical_module)[symbol]

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
