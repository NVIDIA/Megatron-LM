# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from megatron.lite.model.qwen3_5.config import Qwen35Config
from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
from megatron.lite.model.registry import resolve_model_type_from_hf, resolve_runtime_model_name

pytestmark = pytest.mark.mlite

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


def test_registry_resolves_qwen_lite_model_names():
    assert resolve_model_type_from_hf({"model_type": "qwen3_moe"}) == "qwen3"
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


def test_qwen_lite_protocols_build_configs_from_hf_dicts():
    pytest.importorskip("transformer_engine.pytorch")

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
