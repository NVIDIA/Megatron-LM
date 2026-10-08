# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest
import torch.nn as nn

from megatron.lite.model.qwen3_5.config import Qwen35Config
from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
from megatron.lite.model.registry import resolve_model_type_from_hf, resolve_runtime_model_name

LITE_ROOT = Path(__file__).resolve().parents[3]


def _tiny_qwen3_hf_dict() -> dict:
    return {
        "model_type": "qwen3_moe",
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 1,
        "vocab_size": 64,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 8,
        "rope_parameters": {"rope_theta": 12345.0},
    }


def _tiny_qwen35_text_config() -> dict:
    return {
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "num_hidden_layers": 2,
        "vocab_size": 64,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 8,
        "shared_expert_intermediate_size": 8,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 4,
        "linear_num_value_heads": 2,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 2,
        "num_nextn_predict_layers": 1,
        "layer_types": ["linear_attention", "full_attention", "full_attention"],
        "rope_parameters": {"partial_rotary_factor": 1.0, "mrope_section": [1, 1, 0]},
    }


def _tiny_dense_qwen35_text_config() -> dict:
    config = _tiny_qwen35_text_config()
    config.update(
        {
            "model_type": "qwen3_5",
            "intermediate_size": 24,
            "num_nextn_predict_layers": 0,
            "layer_types": ["linear_attention", "full_attention"],
        }
    )
    for field in (
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
    ):
        config.pop(field)
    return config


def test_registry_resolves_qwen_lite_model_names():
    assert resolve_model_type_from_hf({"model_type": "qwen3_moe"}) == "qwen3"
    assert resolve_model_type_from_hf({"model_type": "qwen3_5"}) == "qwen3_5"
    assert resolve_model_type_from_hf({"model_type": "qwen3_5_moe"}) == "qwen3_5"
    assert resolve_runtime_model_name("qwen3", "lite") == "qwen3"
    assert resolve_runtime_model_name("qwen3_moe", "lite") == "qwen3_moe"
    assert resolve_runtime_model_name("qwen3_5", "lite") == "qwen3_5"


def test_qwen3_config_from_hf_dict_derives_head_dim_and_rope_theta():
    cfg = Qwen3MoEConfig._from_hf_dict(_tiny_qwen3_hf_dict())

    assert cfg.hidden_size == 16
    assert cfg.head_dim == 4
    assert cfg.layer_types == ["full_attention"]
    assert cfg.rope_theta == 12345.0


def test_qwen3_config_rejects_invalid_expert_topk():
    hf = _tiny_qwen3_hf_dict()
    hf["num_experts_per_tok"] = 3

    with pytest.raises(ValueError, match="num_experts_per_tok"):
        Qwen3MoEConfig._from_hf_dict(hf)


def test_qwen35_config_from_text_config_splits_mtp_layer_types():
    cfg = Qwen35Config._from_hf_dict(
        {"model_type": "qwen3_5_moe", "text_config": _tiny_qwen35_text_config()}
    )

    assert cfg.layer_types == ["linear_attention", "full_attention"]
    assert cfg.mtp_layer_types == ["full_attention"]
    assert cfg.rotary_dim == 4
    assert cfg.mrope_section == [1, 1, 0]
    assert cfg.hf_text_prefix == "model.language_model"


def test_qwen35_dense_config_uses_nested_text_variant_and_intermediate_size():
    cfg = Qwen35Config._from_hf_dict(
        {"model_type": "qwen3_5", "text_config": _tiny_dense_qwen35_text_config()}
    )

    assert cfg.model_type == "qwen3_5"
    assert not cfg.is_moe
    assert cfg.intermediate_size == 24
    assert cfg.hf_text_prefix == "model.language_model"


def test_qwen35_standalone_text_config_uses_model_hf_prefix():
    cfg = Qwen35Config._from_hf_dict(_tiny_dense_qwen35_text_config())

    assert cfg.model_type == "qwen3_5"
    assert cfg.hf_text_prefix == "model"


def test_qwen35_dense_config_requires_intermediate_size():
    hf = _tiny_dense_qwen35_text_config()
    hf.pop("intermediate_size")

    with pytest.raises(ValueError, match="intermediate_size"):
        Qwen35Config._from_hf_dict({"model_type": "qwen3_5", "text_config": hf})


def test_qwen35_dense_and_moe_layers_select_distinct_ffn_branches(
    transformer_engine_import_stub, monkeypatch
):
    transformer_engine_import_stub()

    from megatron.lite.model.qwen3_5.lite import model as qwen35_model
    from megatron.lite.model.qwen3_5.lite.protocol import MODULE_MAP

    class Attention(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

    class Linear(nn.Module):
        def __init__(self, in_features, out_features, ps, **kwargs):
            super().__init__()
            del ps, kwargs
            self.linear = nn.Linear(in_features, out_features, bias=False)

        def forward(self, x):
            return self.linear(x)

    class MoE(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    monkeypatch.setattr(qwen35_model, "FullAttention", Attention)
    monkeypatch.setattr(qwen35_model, "ColumnParallelLinear", Linear)
    monkeypatch.setattr(qwen35_model, "RowParallelLinear", Linear)
    monkeypatch.setattr(qwen35_model, "MoELayer", MoE)
    monkeypatch.setattr(qwen35_model.te, "RMSNorm", lambda *args, **kwargs: nn.Identity())

    ps = object()
    dense = Qwen35Config(
        model_type="qwen3_5",
        num_hidden_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=12,
        layer_types=["full_attention"],
    )
    moe = Qwen35Config(
        num_hidden_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=4,
        shared_expert_intermediate_size=4,
        layer_types=["full_attention"],
    )

    dense_layer = qwen35_model.Qwen35Layer(dense, ps, 0)
    moe_layer = qwen35_model.Qwen35Layer(moe, ps, 0)

    assert isinstance(dense_layer.mlp, qwen35_model.DenseMLP)
    assert dense_layer.moe is None
    assert dense_layer.mlp_norm is None
    assert moe_layer.mlp is None
    assert isinstance(moe_layer.moe, MoE)
    assert isinstance(moe_layer.mlp_norm, nn.Identity)
    assert MODULE_MAP["mlp"](dense_layer) is dense_layer.mlp
    assert MODULE_MAP["moe"](dense_layer) is None
    assert MODULE_MAP["router"](dense_layer) is None
    assert MODULE_MAP["experts"](dense_layer) is None


def test_qwen_lite_protocols_build_configs_from_hf_dicts(transformer_engine_import_stub):
    transformer_engine_import_stub()

    from megatron.lite.model.qwen3_5.lite import protocol as qwen35_protocol
    from megatron.lite.model.qwen3_moe.lite import protocol as qwen3_protocol

    qwen3_cfg = qwen3_protocol.build_model_config(_tiny_qwen3_hf_dict(), vocab_size=128)
    qwen35_cfg = qwen35_protocol.build_model_config(
        {"model_type": "qwen3_5_moe", "text_config": _tiny_qwen35_text_config()}, vocab_size=128
    )

    assert qwen3_cfg.vocab_size == 128
    assert qwen35_cfg.vocab_size == 128
    assert qwen35_cfg.layer_type_at(0) == "linear_attention"
    assert qwen35_cfg.layer_type_at(1) == "full_attention"


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        (None, ("1", "1", "1")),
        ("auto", ("1", "1", "1")),
        ("flash", ("1", "0", "0")),
        ("fused", ("0", "1", "0")),
        ("unfused", ("0", "0", "1")),
        ("local", ("0", "0", "1")),
    ],
)
def test_qwen35_attention_backend_override_resets_te_environment(
    transformer_engine_import_stub, monkeypatch, backend, expected
):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_5.lite.model import _apply_attention_backend_override

    for name in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.setenv(name, "polluted")

    _apply_attention_backend_override(backend)

    assert tuple(
        os.environ[name]
        for name in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN")
    ) == expected


def test_qwen35_attention_backend_override_rejects_unknown_value(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_5.lite.model import _apply_attention_backend_override

    with pytest.raises(ValueError, match="attention_backend_override"):
        _apply_attention_backend_override("unknown")


def test_qwen_lite_protocols_reexport_checkpoint_hook_names():
    protocol_paths = [
        LITE_ROOT / "megatron/lite/model/qwen3_moe/lite/protocol.py",
        LITE_ROOT / "megatron/lite/model/qwen3_5/lite/protocol.py",
    ]

    for path in protocol_paths:
        tree = ast.parse(path.read_text())
        exported = _string_list_assignment(tree, "__all__")
        checkpoint_imports = _checkpoint_import_names(tree)

        assert "EXPERT_CLASSIFIER" in exported
        assert "PLACEMENT_FN" in exported
        assert "EXPERT_CLASSIFIER" in checkpoint_imports
        assert "PLACEMENT_FN" in checkpoint_imports


def _string_list_assignment(tree: ast.Module, name: str) -> set[str]:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            return set()
        return {item.value for item in node.value.elts if isinstance(item, ast.Constant)}
    return set()


def _checkpoint_import_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module is None or not node.module.endswith(".lite.checkpoint"):
            continue
        names.update(alias.name for alias in node.names)
    return names
