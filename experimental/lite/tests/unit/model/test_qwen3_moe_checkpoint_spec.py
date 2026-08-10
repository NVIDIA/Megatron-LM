# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
from megatron.lite.model.qwen3_moe.lite.checkpoint import Qwen3MoEWeightSpec


def _config(*, num_nextn_predict_layers: int) -> Qwen3MoEConfig:
    return Qwen3MoEConfig(
        num_hidden_layers=1,
        layer_types=["full_attention"],
        num_experts=2,
        num_experts_per_tok=1,
        num_nextn_predict_layers=num_nextn_predict_layers,
    )


def test_weight_map_omits_mtp_embedding_when_config_has_no_mtp() -> None:
    weight_map = Qwen3MoEWeightSpec(
        _config(num_nextn_predict_layers=0)
    ).weight_map()

    assert "mtp_embed.embedding.weight" not in weight_map


def test_weight_map_includes_mtp_embedding_when_config_has_mtp() -> None:
    weight_map = Qwen3MoEWeightSpec(
        _config(num_nextn_predict_layers=1)
    ).weight_map()

    assert weight_map["mtp_embed.embedding.weight"] == [
        "model.embed_tokens.weight"
    ]
