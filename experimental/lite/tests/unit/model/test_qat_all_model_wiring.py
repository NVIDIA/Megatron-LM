# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts for model-level QAT composition and checkpoint loading."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

MODEL_ROOT = Path(__file__).parents[3] / "megatron" / "lite" / "model"
MODEL_NAMES = ("qwen3_moe", "qwen3_5", "deepseek_v4", "glm5", "kimi_k2")


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_every_model_applies_qat_before_optimizer_construction(model_name: str):
    protocol_path = MODEL_ROOT / model_name / "lite" / "protocol.py"
    tree = ast.parse(protocol_path.read_text())
    impl_config = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ImplConfig"
    )
    fields = {
        node.target.id
        for node in impl_config.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert "qat" in fields

    build_model = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_model"
    )
    calls = [node for node in ast.walk(build_model) if isinstance(node, ast.Call)]
    qat_call = next(
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "apply_qat_to_chunks"
    )
    optimizer_calls = [
        node
        for node in calls
        if (
            isinstance(node.func, ast.Name)
            and (
                "optimizer" in node.func.id
                or node.func.id.startswith("_build_dist_opt")
            )
        )
        or (isinstance(node.func, ast.Attribute) and "optimizer" in node.func.attr)
    ]
    assert optimizer_calls
    assert qat_call.lineno < min(node.lineno for node in optimizer_calls)


def test_checkpoint_primitive_canonicalizes_qat_master_weight():
    primitive_path = MODEL_ROOT.parent / "primitive" / "ckpt" / "hf_weights.py"
    tree = ast.parse(primitive_path.read_text())
    canonicalizer_import = next(
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "megatron.lite.primitive.quantization.qat"
        and any(alias.name == "canonical_state_key" for alias in node.names)
    )
    assert canonicalizer_import

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "canonical_state_key"
    ]
    assert calls, "the canonicalizer must be used by the production load path"

    for model_name in MODEL_NAMES:
        checkpoint = (MODEL_ROOT / model_name / "lite" / "checkpoint.py").read_text()
        assert "canonical_state_key" not in checkpoint
