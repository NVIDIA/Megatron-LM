# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""HF-save entry-point contract for every registered Lite model protocol."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from megatron.lite.model.registry import TRAIN_RUNTIME_MODULES


LITE_ROOT = Path(__file__).resolve().parents[3]
_REGISTERED_PROTOCOLS = sorted(TRAIN_RUNTIME_MODULES.items())


@pytest.mark.parametrize(
    ("runtime_name", "module_name"),
    _REGISTERED_PROTOCOLS,
    ids=[runtime_name for runtime_name, _ in _REGISTERED_PROTOCOLS],
)
def test_registered_protocol_exposes_hf_save(
    runtime_name: str, module_name: str
) -> None:
    protocol_path = LITE_ROOT / Path(*module_name.split(".")).with_suffix(".py")
    tree = ast.parse(protocol_path.read_text())
    functions = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "save_hf_weights" in functions, (
        f"{runtime_name} ({module_name}) cannot honor save_contents=['hf_model']"
    )


@pytest.mark.parametrize("model_name", ["kimi_k2", "qwen3_moe"])
def test_new_hf_save_protocols_delegate_all_arguments(
    model_name: str, monkeypatch: pytest.MonkeyPatch, transformer_engine_import_stub
) -> None:
    transformer_engine_import_stub()
    protocol = importlib.import_module(f"megatron.lite.model.{model_name}.lite.protocol")
    checkpoint = importlib.import_module(f"megatron.lite.model.{model_name}.lite.checkpoint")
    calls = []
    monkeypatch.setattr(
        checkpoint,
        "save_hf_weights",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    chunks, model_cfg, parallel_state = object(), object(), object()

    protocol.save_hf_weights(
        chunks,
        "/tmp/hf-save-contract",
        model_cfg,
        parallel_state,
    )

    assert calls == [
        (
            (chunks, "/tmp/hf-save-contract", model_cfg, parallel_state),
            {},
        )
    ]
