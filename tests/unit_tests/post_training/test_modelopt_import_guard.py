# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression tests for optional ModelOpt imports and config selection."""

import ast
import inspect
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


def test_modelopt_configs_resolve_specialized_builders():
    """ModelOpt configs must retain their specialized builders and converter hook."""
    from megatron.post_training.model_builder import (
        ModelOptGPTModelBuilder,
        ModelOptHybridModelBuilder,
        ModelOptHybridModelConfig,
        ModelOptModelConfig,
    )

    assert ModelOptModelConfig.builder.endswith("ModelOptGPTModelBuilder")
    assert ModelOptHybridModelConfig.builder.endswith("ModelOptHybridModelBuilder")
    assert "modelopt_gpt_hybrid_builder" in inspect.getsource(ModelOptGPTModelBuilder.build_model)
    assert ModelOptGPTModelBuilder.build_model is ModelOptHybridModelBuilder.build_model


def test_entrypoints_select_modelopt_configs():
    """Both entrypoints must pass the ModelOpt config class when enabled."""
    gpt = (ROOT / "pretrain_gpt.py").read_text()
    hybrid = (ROOT / "pretrain_hybrid.py").read_text()
    assert "maybe_enable_modelopt(args)" in gpt
    assert "gpt_config_from_args(args, model_config_cls=ModelOptModelConfig)" in gpt
    assert "hybrid_config_from_args(args, model_config_cls=ModelOptHybridModelConfig)" in hybrid
