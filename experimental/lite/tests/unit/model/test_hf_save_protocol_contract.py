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
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "save_hf_weights" in functions, (
        f"{runtime_name} ({module_name}) cannot honor save_contents=['hf_model']"
    )


@pytest.mark.parametrize("model_name", ["kimi_k2", "qwen3_moe"])
def test_new_hf_save_protocols_delegate_all_arguments(
    model_name: str, monkeypatch: pytest.MonkeyPatch, transformer_engine_import_stub
) -> None:
    transformer_engine_import_stub()
    protocol = importlib.import_module(
        f"megatron.lite.model.{model_name}.lite.protocol"
    )
    checkpoint = importlib.import_module(
        f"megatron.lite.model.{model_name}.lite.checkpoint"
    )
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


# Engine-side ``save_contents=['hf_model']`` unconditionally forwards
# ``export_dtype`` / ``target`` / ``resync_config`` — protocols that don't
# consume them must still accept them without raising ``TypeError``.
_ENGINE_EXPORT_KWARGS = {
    "export_dtype": "bfloat16",
    "target": "mxfp4",
    "resync_config": {"expert_dtype": "fp4"},
}


@pytest.mark.parametrize(
    "model_name", ["kimi_k2", "qwen3_moe", "qwen3_5", "deepseek_v4"]
)
def test_hf_save_protocols_accept_engine_export_kwargs(
    model_name: str, monkeypatch: pytest.MonkeyPatch, transformer_engine_import_stub
) -> None:
    transformer_engine_import_stub()
    if model_name == "deepseek_v4":
        # DS4 protocol drags in megatron.core via the CSA module at import time.
        pytest.importorskip(
            "megatron.core",
            reason="deepseek_v4 protocol needs megatron.core in the test env",
        )
    protocol = importlib.import_module(
        f"megatron.lite.model.{model_name}.lite.protocol"
    )
    checkpoint = importlib.import_module(
        f"megatron.lite.model.{model_name}.lite.checkpoint"
    )
    calls = []

    def _record(*args, **kwargs):
        calls.append((args, kwargs))

    # kimi_k2 / qwen3_moe use function-local imports of the checkpoint
    # writer; qwen3_5 aliases it at module load as ``_save_hf_weights_impl``.
    # Patch both surfaces so the test targets whichever the protocol uses.
    monkeypatch.setattr(checkpoint, "save_hf_weights", _record)
    if hasattr(protocol, "_save_hf_weights_impl"):
        monkeypatch.setattr(protocol, "_save_hf_weights_impl", _record)

    chunks, model_cfg, parallel_state = object(), object(), object()

    protocol.save_hf_weights(
        chunks,
        "/tmp/hf-save-kwargs",
        model_cfg,
        parallel_state,
        **_ENGINE_EXPORT_KWARGS,
    )

    assert calls == [
        (
            (chunks, "/tmp/hf-save-kwargs", model_cfg, parallel_state),
            _ENGINE_EXPORT_KWARGS,
        )
    ]


@pytest.mark.parametrize("model_name", ["qwen3_moe", "qwen3_5"])
def test_hf_save_checkpoint_warns_on_unused_export_kwargs(
    model_name: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    transformer_engine_import_stub,
) -> None:
    """Protocols without an MXFP4/block-FP8 save path warn (with kwargs printed)
    when engine-level export kwargs arrive, and never forward them to the
    shared writer.  Stays fully in-memory: the writer is monkeypatched."""
    transformer_engine_import_stub()
    checkpoint = importlib.import_module(
        f"megatron.lite.model.{model_name}.lite.checkpoint"
    )
    calls = []
    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.save_hf_weights",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    class _Config:
        vocab_size = 32

    model, parallel_state = object(), object()
    with caplog.at_level("WARNING", logger=checkpoint.__name__):
        checkpoint.save_hf_weights(
            model,
            "/tmp/hf-save-drop",
            _Config(),
            parallel_state,
            **_ENGINE_EXPORT_KWARGS,
        )

    assert len(calls) == 1
    forwarded_kwargs = calls[0][1]
    for key in _ENGINE_EXPORT_KWARGS:
        assert key not in forwarded_kwargs
    assert forwarded_kwargs.get("vocab_size") == 32

    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING"
    ]
    assert any("ignoring unsupported kwargs" in msg for msg in warnings)
    joined = "\n".join(warnings)
    for key in _ENGINE_EXPORT_KWARGS:
        assert key in joined
