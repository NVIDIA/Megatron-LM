# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts for MXFP4 resync exports of the non-Qwen3-MoE QAT models."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from megatron.lite.primitive.quantization.mxfp4 import (
    MXFP4_BLOCK_SIZE,
    dequantize_mxfp4,
    quantize_mxfp4,
)


pytestmark = pytest.mark.mlite


@pytest.mark.parametrize("model_name", ["qwen3_5", "glm5", "kimi_k2"])
def test_mxfp4_resync_export_matches_primitive_and_skips_release_weights(
    monkeypatch, model_name, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    checkpoint = importlib.import_module(f"megatron.lite.model.{model_name}.lite.checkpoint")
    hf_weights = importlib.import_module("megatron.lite.primitive.ckpt.hf_weights")
    weight = torch.linspace(-4, 4, MXFP4_BLOCK_SIZE * 2, dtype=torch.float32).reshape(2, -1)
    source = [
        ("model.layers.0.mlp.experts.0.up_proj.weight", weight),
        ("model.embed_tokens.weight", weight.clone()),
        ("lm_head.weight", weight.clone()),
        ("model.layers.0.mlp.gate.weight", weight.clone()),
    ]

    def fake_export(model, spec, ps, **kwargs):
        assert kwargs == {"vocab_size": 128}
        yield from source

    monkeypatch.setattr(hf_weights, "export_hf_weights", fake_export)
    exported = dict(checkpoint.export_hf_weights(object(), SimpleNamespace(vocab_size=128), object(), target="mxfp4"))
    prefix = "model.layers.0.mlp.experts.0.up_proj"
    packed, scale = quantize_mxfp4(weight)
    assert torch.equal(exported[f"{prefix}.weight"], packed.view(torch.uint8))
    assert torch.equal(exported[f"{prefix}.weight_scale"], scale.view(torch.uint8))
    assert exported[f"{prefix}.weight"].shape[-1] * 2 == weight.shape[-1]
    assert exported[f"{prefix}.weight_scale"].shape[-1] == weight.shape[-1] // MXFP4_BLOCK_SIZE
    assert torch.equal(dequantize_mxfp4(exported[f"{prefix}.weight"].view(torch.int8), exported[f"{prefix}.weight_scale"].view(torch.float8_e8m0fnu)), dequantize_mxfp4(packed, scale))
    for ignored in ("model.embed_tokens.weight", "lm_head.weight", "model.layers.0.mlp.gate.weight"):
        assert torch.equal(exported[ignored], dict(source)[ignored])
        assert f"{ignored[:-7]}.weight_scale" not in exported


@pytest.mark.parametrize("model_name", ["qwen3_5", "glm5", "kimi_k2"])
def test_mxfp4_resync_export_rejects_unsupported_target(
    monkeypatch, model_name, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    checkpoint = importlib.import_module(f"megatron.lite.model.{model_name}.lite.checkpoint")
    hf_weights = importlib.import_module("megatron.lite.primitive.ckpt.hf_weights")
    monkeypatch.setattr(hf_weights, "export_hf_weights", lambda *args, **kwargs: iter(()))
    with pytest.raises(ValueError, match="resync target"):
        list(checkpoint.export_hf_weights(object(), SimpleNamespace(vocab_size=128), object(), target="block_fp8"))
