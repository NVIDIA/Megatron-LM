# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Architecture guard for model HF checkpoint loaders."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_MODEL_CHECKPOINTS = (
    "qwen3_moe/lite/checkpoint.py",
    "qwen3_5/lite/checkpoint.py",
    "kimi_k2/lite/checkpoint.py",
    "glm5/lite/checkpoint.py",
    "deepseek_v4/lite/checkpoint.py",
)


@pytest.mark.parametrize("relative_path", _MODEL_CHECKPOINTS)
def test_model_hf_loaders_only_configure_the_primitive(relative_path: str) -> None:
    model_root = Path(__file__).resolve().parents[3] / "megatron" / "lite" / "model"
    source = (model_root / relative_path).read_text()
    tree = ast.parse(source)
    loader = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "load_hf_weights"
    )

    assert not any(
        isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.With, ast.Dict))
        for node in ast.walk(loader)
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_load"
        for node in ast.walk(loader)
    )
    assert not {
        "SafeTensorReader",
        "StreamingStateLoader",
        "TensorLoadSink",
    }.intersection(source)
