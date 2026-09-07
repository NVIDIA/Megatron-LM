# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU unit tests for QAT wiring on qwen3_moe (MoE 3D expert weights + protocol)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from megatron.lite.model.qwen3_moe.lite.checkpoint import export_hf_weights
from megatron.lite.primitive.ckpt.hf_weights import _resolve_param_name
from megatron.lite.primitive.quantization.qat import (
    QATSpec,
    _fake_quant_weight_tensor,
    apply_qat_to_chunks,
    canonical_state_key,
    normalize_qat_spec,
)
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.backends.mlite.runtime import _build_impl_cfg

pytestmark = pytest.mark.mlite


class _GroupedExpertLinear(nn.Module):
    """Minimal stand-in for ``te.GroupedLinear`` with a 3D stacked weight."""

    def __init__(self, num_experts: int, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, out_features, in_features))


class _ParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)


def _toy_qwen3_moe_chunk():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Module()
            self.embed.embedding = nn.Embedding(128, 64)
            layer = nn.Module()
            layer.attn = nn.Module()
            layer.attn.qkv = _ParallelLinear(64, 96)
            layer.attn.proj = _ParallelLinear(96, 64)
            layer.moe = nn.Module()
            layer.moe.router = nn.Module()
            layer.moe.router.gate = nn.Linear(64, 8, bias=False)
            layer.moe.experts = nn.Module()
            layer.moe.experts.fc1 = _GroupedExpertLinear(4, 64, 128)
            layer.moe.experts.fc2 = _GroupedExpertLinear(4, 32, 64)
            self.layers = nn.ModuleList([layer])
            self.lm_head = nn.Module()
            self.lm_head.col = _ParallelLinear(64, 128)

        def forward(self, x):
            h = self.embed.embedding(x)
            h = self.layers[0].attn.proj(self.layers[0].attn.qkv(h))
            h = h @ self.layers[0].moe.router.gate.weight.T
            return self.lm_head.col(h)

    return Toy().to(torch.bfloat16)


def test_qwen3_moe_qat_quantizes_experts_and_attn_skips_router_embed_head():
    torch.manual_seed(0)
    chunk = _toy_qwen3_moe_chunk()
    spec = QATSpec(enabled=True, format="fp8", group_size=-1)
    stats = apply_qat_to_chunks([chunk], spec)

    experts = chunk.layers[0].moe.experts
    assert parametrize.is_parametrized(experts.fc1, "weight")
    assert parametrize.is_parametrized(experts.fc2, "weight")
    assert parametrize.is_parametrized(chunk.layers[0].attn.qkv.linear, "weight")
    assert parametrize.is_parametrized(chunk.layers[0].attn.proj.linear, "weight")
    assert not parametrize.is_parametrized(chunk.layers[0].moe.router.gate, "weight")
    assert not hasattr(chunk.embed.embedding, "parametrizations")
    assert not parametrize.is_parametrized(chunk.lm_head.col.linear, "weight")
    assert stats["quantized_modules"] == 4  # qkv + proj + fc1 + fc2


def test_qwen3_moe_expert_ste_forward_matches_manual_fake_quant():
    torch.manual_seed(1)
    experts = _GroupedExpertLinear(3, 32, 64).to(torch.bfloat16)
    spec = QATSpec(enabled=True, format="fp8", group_size=-1)
    apply_qat_to_chunks([experts], spec)

    master = experts.parametrizations.weight.original
    ref = _fake_quant_weight_tensor(master, spec)
    assert torch.equal(experts.weight, ref)

    x = torch.randn(2, 3, 32, dtype=torch.bfloat16)
    out = torch.einsum("bei,eoi->beo", x, experts.weight)
    ref_out = torch.einsum("bei,eoi->beo", x, ref)
    assert torch.equal(out, ref_out)


def test_qwen3_moe_canonical_state_key_maps_grouped_expert_master():
    torch.manual_seed(2)
    chunk = _toy_qwen3_moe_chunk()
    apply_qat_to_chunks([chunk], QATSpec(enabled=True, format="int8", group_size=-1))

    state = chunk.state_dict()
    logical = "layers.0.moe.experts.fc1.weight"
    assert logical not in state
    assert not any(logical in key for key in state)

    canonical = {canonical_state_key(key): key for key in state}
    actual = _resolve_param_name(logical, state)
    assert actual == canonical[logical]
    assert actual.endswith("parametrizations.weight.original")

    real = torch.randn_like(
        chunk.layers[0].moe.experts.fc1.parametrizations.weight.original
    )
    resolved = {actual: real}
    for name, param in chunk.named_parameters():
        if name == actual:
            param.data.copy_(resolved[name])
    torch.testing.assert_close(
        chunk.layers[0].moe.experts.fc1.parametrizations.weight.original,
        real,
        rtol=0,
        atol=0,
    )


def test_qwen3_moe_mxfp4_resync_exports_compressed_tensors_stream(
    monkeypatch,
):
    from megatron.lite.primitive.ckpt import hf_weights as hf_weights_module

    expert = torch.arange(64, dtype=torch.float32).reshape(2, 32) - 31
    router = torch.arange(64, dtype=torch.float32).reshape(2, 32)
    source = [
        ("model.layers.0.mlp.experts.0.gate_proj.weight", expert),
        ("model.layers.0.self_attn.q_proj.weight", expert.clone()),
        ("model.layers.0.mlp.gate.weight", router),
        ("model.layers.0.input_layernorm.weight", torch.ones(32)),
    ]
    captured = {}

    def fake_export(model, spec, ps, **kwargs):
        captured["kwargs"] = kwargs
        yield from source

    monkeypatch.setattr(hf_weights_module, "export_hf_weights", fake_export)

    exported = dict(
        export_hf_weights(
            object(),
            SimpleNamespace(vocab_size=128),
            object(),
            target="mxfp4",
            limit=4,
        )
    )

    prefix = "model.layers.0.mlp.experts.0.gate_proj"
    dense_prefix = "model.layers.0.self_attn.q_proj"
    assert captured["kwargs"] == {"limit": 4, "vocab_size": 128}
    assert exported[f"{prefix}.weight"].dtype == torch.uint8
    assert exported[f"{prefix}.weight"].shape == (2, 16)
    assert exported[f"{prefix}.weight_scale"].dtype == torch.uint8
    assert exported[f"{prefix}.weight_scale"].shape == (2, 1)
    assert exported[f"{dense_prefix}.weight"].dtype == torch.uint8
    assert exported[f"{dense_prefix}.weight"].shape == (2, 16)
    assert exported[f"{dense_prefix}.weight_scale"].dtype == torch.uint8
    assert exported[f"{dense_prefix}.weight_scale"].shape == (2, 1)
    assert torch.equal(exported["model.layers.0.mlp.gate.weight"], router)
    assert torch.equal(
        exported["model.layers.0.input_layernorm.weight"],
        torch.ones(32),
    )


@pytest.mark.parametrize(
    ("arm", "impl_qat"),
    [
        ("bf16_baseline", None),
        ("fp8_rollout_qat_off", {"enabled": False, "format": "fp8"}),
        ("fp8_rollout_qat_on", {"enabled": True, "format": "fp8", "group_size": -1}),
    ],
)
def test_qwen3_moe_dapo_three_arm_impl_cfg_resolves(
    transformer_engine_import_stub, arm, impl_qat
):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite import protocol as qwen3_moe_protocol

    rt_cfg = MegatronLiteConfig(
        model_name="qwen3_moe",
        impl_cfg={"qat": impl_qat} if impl_qat is not None else {},
    )
    impl_cfg = _build_impl_cfg(qwen3_moe_protocol, rt_cfg)
    assert hasattr(impl_cfg, "qat")
    normalized = normalize_qat_spec(impl_cfg.qat)
    if arm == "bf16_baseline":
        assert normalized.enabled is False
    elif arm == "fp8_rollout_qat_off":
        assert normalized.enabled is False
        assert normalized.format == "fp8_e4m3"
    else:
        assert normalized.enabled is True
        assert normalized.format == "fp8_e4m3"
        assert normalized.group_size == -1
