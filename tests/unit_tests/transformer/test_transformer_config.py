# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.transformer.transformer_config import TransformerConfig


def _make_transformer_config(**kwargs) -> TransformerConfig:
    return TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=4, **kwargs)


def _make_overlap_config(mtp_num_layers: int | None) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        expert_model_parallel_size=2,
        moe_token_dispatcher_type="alltoall",
        overlap_moe_expert_parallel_comm=True,
        bf16=True,
        mtp_num_layers=mtp_num_layers,
    )


@pytest.mark.parametrize("mtp_num_layers", [None, 0, 1])
def test_ep_a2a_overlap_accepts_supported_mtp_layer_counts(mtp_num_layers: int | None):
    config = _make_overlap_config(mtp_num_layers)

    assert config.mtp_num_layers == mtp_num_layers


@pytest.mark.parametrize("mtp_num_layers", [-1, 2])
def test_ep_a2a_overlap_rejects_unsupported_mtp_layer_counts(mtp_num_layers: int):
    with pytest.raises(AssertionError, match="MTP supports at most one layer"):
        _make_overlap_config(mtp_num_layers)


def test_gdp_num_householder_defaults_to_three():
    config = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=4)

    assert config.gdp_num_householder == 3


def test_gdp_num_householder_accepts_positive_values():
    config = TransformerConfig(
        num_layers=1, hidden_size=128, num_attention_heads=4, gdp_num_householder=5
    )

    assert config.gdp_num_householder == 5


@pytest.mark.parametrize("num_householder", [0, -1])
def test_gdp_num_householder_rejects_non_positive_values(num_householder: int):
    with pytest.raises(ValueError, match="gdp_num_householder must be positive"):
        TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            gdp_num_householder=num_householder,
        )


def test_mxfp8_2d_quantization_accepts_mxfp8_recipe():
    config = _make_transformer_config(fp8="e4m3", fp8_recipe="mxfp8", mxfp8_2d_quantization=True)

    assert config.mxfp8_2d_quantization


@pytest.mark.parametrize(
    ("config_kwargs", "match"),
    [
        ({"fp8_recipe": "mxfp8"}, "together with fp8 mode"),
        ({"fp8": "e4m3", "fp8_recipe": "delayed"}, "requires fp8_recipe='mxfp8'"),
        (
            {
                "fp8": "e4m3",
                "fp8_recipe": "mxfp8",
                "moe_single_grouped_weight": True,
            },
            "does not support moe_single_grouped_weight",
        ),
    ],
)
def test_mxfp8_2d_quantization_rejects_incompatible_fp8_config(config_kwargs, match):
    with pytest.raises(ValueError, match=match):
        _make_transformer_config(mxfp8_2d_quantization=True, **config_kwargs)
