# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression tests for optional ModelOpt imports and config selection."""

import ast
from pathlib import Path

ROOT = Path(__file__).parents[3]
ENTRYPOINTS = (ROOT / "pretrain_gpt.py", ROOT / "pretrain_hybrid.py")


def _import_guard(path: Path) -> ast.Try:
    module = ast.parse(path.read_text())
    return next(
        node
        for node in module.body
        if isinstance(node, ast.Try)
        and any(
            isinstance(stmt, ast.ImportFrom) and stmt.module == "megatron.post_training.arguments"
            for stmt in node.body
        )
    )


def test_modelopt_import_guards_only_suppress_missing_modelopt():
    """Internal import failures must not silently disable ModelOpt."""
    for path in ENTRYPOINTS:
        guard = _import_guard(path)
        assert len(guard.handlers) == 1
        handler = guard.handlers[0]
        assert isinstance(handler.type, ast.Name) and handler.type.id == "ImportError"
        assert handler.name == "error"
        assert any(isinstance(node, ast.Raise) for node in ast.walk(handler))


def test_entrypoints_apply_modelopt_model_config():
    """Both entrypoints must route ModelOpt-enabled runs through the stock builders' ModelOpt config."""
    for path in ENTRYPOINTS:
        calls = [
            node for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, ast.Call)
        ]
        assert any(
            isinstance(call.func, ast.Name)
            and call.func.id == "should_use_modelopt_config"
            and call.args
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == "args"
            for call in calls
        )
        assert any(
            isinstance(call.func, ast.Name) and call.func.id == "apply_modelopt_model_config"
            for call in calls
        )
