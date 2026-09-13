# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for ModelOpt calibration prefill compatibility."""

import ast
from pathlib import Path


def _load_helper(relative_path):
    source_path = Path(__file__).parents[3] / relative_path
    source_tree = ast.parse(source_path.read_text(), filename=source_path)
    helper = next(
        node
        for node in source_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_megatron_prefill_for_calibration"
    )
    namespace = {"inspect": __import__("inspect")}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), source_path, "exec"), namespace)
    return namespace


def test_prefill_omits_unsupported_skip_return_logits():
    """Legacy ModelOpt receives only the parameters its plugin exposes."""
    for relative_path in (
        "examples/post_training/modelopt/quantize.py",
        "examples/post_training/modelopt/prune.py",
    ):
        namespace = _load_helper(relative_path)
        calls = []

        def legacy_prefill(model, input_ids):
            calls.append((model, input_ids))
            return "legacy-result"

        namespace["megatron_prefill"] = legacy_prefill
        result = namespace["_megatron_prefill_for_calibration"]("model", "input-ids")

        assert result == "legacy-result"
        assert calls == [("model", "input-ids")]


def test_prefill_skips_logits_when_supported():
    """Newer ModelOpt avoids retaining calibration logits."""
    for relative_path in (
        "examples/post_training/modelopt/quantize.py",
        "examples/post_training/modelopt/prune.py",
    ):
        namespace = _load_helper(relative_path)
        calls = []

        def modern_prefill(model, input_ids, *, skip_return_logits=False):
            calls.append((model, input_ids, skip_return_logits))
            return "modern-result"

        namespace["megatron_prefill"] = modern_prefill
        result = namespace["_megatron_prefill_for_calibration"]("model", "input-ids")

        assert result == "modern-result"
        assert calls == [("model", "input-ids", True)]
