# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Configuration gates for the independent dev-based GLM53 integration."""

import pytest

from megatron.core.transformer.transformer_config import MLATransformerConfig


def _dsa_config(**kwargs):
    values = dict(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        experimental_attention_variant="dsa",
        add_bias_linear=False,
        dsa_indexer_topk=32,
        dsa_indexer_kpool=4,
        dsa_indexer_loss_coeff=0.0,
    )
    values.update(kwargs)
    return MLATransformerConfig(**values)


@pytest.mark.parametrize("quantized_scores", [False, True])
def test_hf_pool_compression_is_independent_of_score_quantization(quantized_scores):
    config = _dsa_config(
        dsa_indexer_kpool_key_dtype_compression=True,
        dsa_indexer_kpool_use_quantization=quantized_scores,
    )
    assert config.dsa_indexer_kpool_key_dtype_compression is True
    assert config.dsa_indexer_kpool_use_quantization is quantized_scores


@pytest.mark.parametrize(
    "overrides",
    [
        {"dsa_indexer_kpool": 1},
        {
            "dsa_indexer_weights_proj_use_quantization": False,
            "dsa_indexer_weights_proj_output_dtype": "fp32",
        },
        {"dsa_indexer_scoring_relu": False},
    ],
)
def test_hf_scoring_rejects_unimplemented_precision_paths(overrides):
    with pytest.raises(ValueError, match="dsa_indexer_hf_scoring requires"):
        _dsa_config(dsa_indexer_hf_scoring=True, **overrides)
