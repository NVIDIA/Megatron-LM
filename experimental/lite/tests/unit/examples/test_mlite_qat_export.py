# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Contracts for the MLite-owned QAT rollout exporter."""

from __future__ import annotations

import pytest
import torch
from megatron.lite.primitive.quantization.mxfp4 import MXFP4_BLOCK_SIZE

pytestmark = pytest.mark.optional


def _qat_config(**overrides):
    config = {
        "enable": True,
        "apply_modelopt_fake_quant": False,
        "mode": "mxfp4",
        "group_size": MXFP4_BLOCK_SIZE,
        "ignore_patterns": ["lm_head", "embed_tokens", "re:.*mlp.gate$"],
    }
    config.update(overrides)
    return config


def test_qat_export_stays_lazy() -> None:
    from verl_mlite.qat_export import export_qat_weights

    consumed = []

    def source():
        consumed.append(True)
        yield "model.layers.0.mlp.down_proj.weight", torch.ones(2, MXFP4_BLOCK_SIZE)

    exported = export_qat_weights(source(), _qat_config())

    assert consumed == []
    next(exported)
    assert consumed == [True]


def test_qat_export_uses_safe_exclusions_when_config_omits_them() -> None:
    """An enabled engine must not quantize fragile HF head/embed/router weights."""
    from verl_mlite.qat_export import export_qat_weights

    head = torch.full((1, MXFP4_BLOCK_SIZE), 1.0, dtype=torch.bfloat16)
    embed = torch.full((1, MXFP4_BLOCK_SIZE), 2.0, dtype=torch.bfloat16)
    router = torch.full((1, MXFP4_BLOCK_SIZE), 3.0, dtype=torch.bfloat16)
    dense = torch.arange(MXFP4_BLOCK_SIZE, dtype=torch.bfloat16).reshape(1, -1)
    config = _qat_config()
    config.pop("ignore_patterns")

    exported = dict(
        export_qat_weights(
            iter(
                [
                    ("model.lm_head.weight", head),
                    ("model.embed_tokens.weight", embed),
                    ("model.layers.0.mlp.gate.weight", router),
                    ("model.layers.0.mlp.down_proj.weight", dense),
                ]
            ),
            config,
        )
    )

    assert exported["model.lm_head.weight"] is head
    assert exported["model.embed_tokens.weight"] is embed
    assert exported["model.layers.0.mlp.gate.weight"] is router
    assert "model.lm_head.weight_scale" not in exported
    assert "model.embed_tokens.weight_scale" not in exported
    assert "model.layers.0.mlp.gate.weight_scale" not in exported
    assert exported["model.layers.0.mlp.down_proj.weight"].dtype == torch.uint8
    assert "model.layers.0.mlp.down_proj.weight_scale" in exported


def test_qat_export_real_mxfp4_matches_modelopt_reference_encoding() -> None:
    """Exercise the real exporter, not an engine-level fake, against ModelOpt's grid."""
    from verl_mlite.qat_export import export_qat_weights

    # One E8M0 scale of 1 and E2M1 midpoint values: ModelOpt rounds every tie
    # down, so 5.0 serializes as 4.0 rather than 6.0.
    weight = torch.tensor(
        [
            [
                0.0,
                0.25,
                0.75,
                1.25,
                1.75,
                2.5,
                3.5,
                5.0,
                -0.25,
                -0.75,
                -1.25,
                -1.75,
                -2.5,
                -3.5,
                -5.0,
                -6.0,
            ]
            * 2
        ],
        dtype=torch.bfloat16,
    )

    exported = dict(
        export_qat_weights(
            iter([("model.layers.0.mlp.down_proj.weight", weight)]), _qat_config()
        )
    )

    packed = exported["model.layers.0.mlp.down_proj.weight"]
    scale = exported["model.layers.0.mlp.down_proj.weight_scale"]
    assert torch.equal(
        packed,
        torch.tensor(
            [[0x00, 0x21, 0x43, 0x65, 0x98, 0xBA, 0xDC, 0xFE] * 2],
            dtype=torch.uint8,
        ),
    )
    assert torch.equal(scale, torch.tensor([[127]], dtype=torch.uint8))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"apply_modelopt_fake_quant": True}, "apply_modelopt_fake_quant=False"),
        ({"mode": "w4a16", "group_size": 16}, "only supports mode='mxfp4'"),
        ({"group_size": 16}, "group_size=32"),
    ],
)
def test_qat_export_rejects_unsupported_contract(overrides, message) -> None:
    from verl_mlite.qat_export import export_qat_weights

    with pytest.raises(ValueError, match=message):
        export_qat_weights(iter(()), _qat_config(**overrides))
