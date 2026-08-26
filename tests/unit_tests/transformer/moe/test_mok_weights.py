# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core import fp8_utils
from megatron.core.transformer.moe.megakernel import factory
from megatron.core.transformer.moe.megakernel.mok import weights
from megatron.core.transformer.transformer_config import TransformerConfig


def _mok_transformer_config(**overrides):
    values = {
        "num_layers": 2,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_moe_experts": 8,
        "moe_ffn_hidden_size": 256,
        "moe_shared_expert_intermediate_size": 256,
        "expert_model_parallel_size": 4,
        "gated_linear_unit": True,
        "activation_func": F.silu,
        "gradient_accumulation_fusion": True,
        "moe_megakernel_backend": "mok",
    }
    values.update(overrides)
    return TransformerConfig(**values)


def test_bf16_shared_expert_config_reuses_original_config():
    config = SimpleNamespace(fp8_param=False)

    assert weights.prepare_shared_expert_config(config) is config


def test_mxfp8_shared_expert_config_expresses_bf16_module(monkeypatch, recwarn):
    original_recipe = object()
    config = SimpleNamespace(
        fp8="hybrid", fp8_param=True, quant_recipe=original_recipe, sentinel=object()
    )
    monkeypatch.setattr(weights, "_SHARED_EXPERT_BF16_WARNING_EMITTED", False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    shared_config = weights.prepare_shared_expert_config(config)

    assert shared_config is not config
    assert config.fp8 == "hybrid"
    assert config.fp8_param
    assert config.quant_recipe is original_recipe
    assert shared_config.fp8 is None
    assert not shared_config.fp8_param
    assert shared_config.quant_recipe is original_recipe
    assert shared_config.sentinel is config.sentinel
    assert len(recwarn) == 1
    assert "shared experts in BF16" in str(recwarn[0].message)

    weights.prepare_shared_expert_config(config)
    assert len(recwarn) == 1


def test_mok_shared_expert_init_context_disables_outer_fp8(monkeypatch):
    config = SimpleNamespace(moe_megakernel_backend="mok")
    sentinel = object()
    calls = []

    def fake_disabled_context(candidate, *, is_init):
        calls.append((candidate, is_init))
        return sentinel

    monkeypatch.setattr(fp8_utils, "get_fp8_disabled_context", fake_disabled_context)

    actual = factory.megakernel_shared_expert_init_context(config)

    assert actual is sentinel
    assert calls == [(config, True)]


def test_mok_backend_accepts_bf16_routed_parameters():
    config = _mok_transformer_config()

    assert config.moe_megakernel_backend == "mok"
    assert config.fp8 is None
    assert not config.fp8_param


def test_mok_backend_accepts_native_mxfp8_parameters():
    config = _mok_transformer_config(fp8="hybrid", fp8_recipe="mxfp8", fp8_param=True)

    assert config.fp8_recipe == "mxfp8"
    assert config.fp8_param


def test_mok_backend_rejects_mxfp8_without_fp8_parameters():
    with pytest.raises(ValueError, match="fp8_param=True"):
        _mok_transformer_config(fp8="hybrid", fp8_recipe="mxfp8", fp8_param=False)


@pytest.mark.parametrize(
    "overrides",
    [
        {"recompute_granularity": "selective", "recompute_modules": ["moe"]},
        {"moe_layer_recompute": True},
    ],
)
def test_mok_backend_rejects_whole_moe_recompute(overrides):
    with pytest.raises(ValueError, match="whole-MoE activation recomputation"):
        _mok_transformer_config(**overrides)
