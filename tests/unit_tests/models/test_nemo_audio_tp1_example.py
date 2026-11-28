# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import ast
import importlib.util
import sys
from pathlib import Path

_EXAMPLE = (
    Path(__file__).resolve().parents[3] / "examples" / "audio" / "run_nemo_transformer_audio_tp1.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location("nemo_audio_tp1_example_under_test", _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_importing_tp1_example_does_not_touch_model_parallel():
    before = {name for name in sys.modules if name == "megatron" or name.startswith("megatron.")}
    module = _load_example()
    after = {name for name in sys.modules if name == "megatron" or name.startswith("megatron.")}
    assert callable(module.main)
    assert before == after


def test_example_main_is_called_only_under_name_guard():
    tree = ast.parse(_EXAMPLE.read_text())
    main_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "main"
    ]
    assert len(main_calls) == 1
    guarded = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    ]
    assert len(guarded) == 1
    called_names = [
        stmt.value.func.id
        for stmt in guarded[0].body
        if isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Name)
    ]
    assert called_names == ["main"]

    text = _EXAMPLE.read_text()
    assert 'attn_impl="sdpa"' in text
    assert 'pre_encode="conv"' in text
    assert "PackedAudioEmbeddings" in text
    assert "initialize_model_parallel" in text
    assert "get_tensor_model_parallel_world_size" in text
    assert "flash_attn_varlen_func" in text
