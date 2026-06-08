from __future__ import annotations

import pytest

from megatron.lite.model.qwen3_5.config import Qwen35Config
from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
from megatron.lite.model.registry import (
    resolve_model_type_from_hf,
    resolve_runtime_model_name,
)


pytestmark = pytest.mark.mlite


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
        "rope_parameters": {
            "partial_rotary_factor": 1.0,
            "mrope_section": [1, 1, 0],
        },
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
        {
            "model_type": "qwen3_5_moe",
            "text_config": _tiny_qwen35_text_config(),
        }
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
        {"model_type": "qwen3_5_moe", "text_config": _tiny_qwen35_text_config()},
        vocab_size=128,
    )

    assert qwen3_cfg.vocab_size == 128
    assert qwen35_cfg.vocab_size == 128
    assert qwen35_cfg.layer_type_at(0) == "linear_attention"
    assert qwen35_cfg.layer_type_at(1) == "full_attention"
