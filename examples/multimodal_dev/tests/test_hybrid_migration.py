# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused tests for the multimodal_dev HybridModel migration."""

from types import SimpleNamespace

import pytest

from examples.multimodal_dev.models.qwen35_vl import factory
from examples.multimodal_dev.models.qwen35_vl.configuration import (
    get_qwen35_vl_language_config,
)
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_linear_attention_pattern,
)
from megatron.core.models.hybrid.hybrid_layer_allocation import (
    Symbols,
    get_hybrid_total_layer_count,
    parse_hybrid_pattern,
    validate_segment_layers,
)

# The four-block cadence documented in the README and emitted by
# scripts/run_qwen35_vl.sh: three GatedDeltaNet blocks then one full-attention
# block, each block contributing an attention layer and an MLP layer.
MOE_PATTERN_4_BLOCKS = "GEGEGE*E"
DENSE_PATTERN_4_BLOCKS = "G-G-G-*-"
# The former GPT path built the same cadence from --linear-attention-freq 4.
QWEN35_LINEAR_ATTENTION_FREQ = 4


def _args(**overrides):
    values = {
        "hybrid_layer_pattern": "GEGEGE*E/*E",
        "untie_embeddings_and_output_weights": True,
        "padded_vocab_size": 1024,
        "max_position_embeddings": 4096,
        "image_token_id": 42,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_factory_requires_hybrid_layer_pattern():
    with pytest.raises(ValueError, match="requires --hybrid-layer-pattern"):
        factory.build_model(
            _args(hybrid_layer_pattern=None),
            language_config=SimpleNamespace(),
            vision_config=SimpleNamespace(),
        )


def test_factory_passes_hybrid_pattern(monkeypatch):
    captured = {}

    def fake_model(**kwargs):
        captured.update(kwargs)
        return "model"

    monkeypatch.setattr(
        "examples.multimodal_dev.models.qwen35_vl.model.Qwen35VLModel", fake_model
    )
    result = factory.build_model(
        _args(),
        language_config=SimpleNamespace(),
        vision_config=SimpleNamespace(),
    )

    assert result == "model"
    assert captured["hybrid_layer_pattern"] == "GEGEGE*E/*E"
    assert captured["share_embeddings_and_output_weights"] is False


@pytest.mark.parametrize("pattern", [MOE_PATTERN_4_BLOCKS, DENSE_PATTERN_4_BLOCKS])
def test_pattern_reproduces_gpt_block_structure(pattern):
    """Each hybrid block is one attention layer followed by one MLP layer."""
    layers = validate_segment_layers(pattern)

    attention_layers = layers[0::2]
    mlp_layers = layers[1::2]
    assert all(s in (Symbols.GDN, Symbols.ATTENTION) for s in attention_layers)
    assert all(s in (Symbols.MLP, Symbols.MOE) for s in mlp_layers)
    assert len(attention_layers) == len(mlp_layers)
    # The GPT path applied the same MLP kind to every block.
    assert len(set(mlp_layers)) == 1


@pytest.mark.parametrize("pattern", [MOE_PATTERN_4_BLOCKS, DENSE_PATTERN_4_BLOCKS])
def test_pattern_matches_gpt_linear_attention_layout(pattern):
    """GDN / full-attention placement matches --linear-attention-freq 4."""
    attention_layers = validate_segment_layers(pattern)[0::2]

    gpt_config = SimpleNamespace(
        num_layers=len(attention_layers),
        linear_attention_freq=QWEN35_LINEAR_ATTENTION_FREQ,
        experimental_attention_variant="gated_delta_net",
    )
    # get_linear_attention_pattern returns 1 for linear attention, 0 for SDPA.
    expected = [
        Symbols.GDN if is_linear else Symbols.ATTENTION
        for is_linear in get_linear_attention_pattern(gpt_config)
    ]
    assert attention_layers == expected


def test_mtp_pattern_replicates_final_block():
    """GPT MTP copied the final decoder layer; the MTP pattern must match it."""
    parsed = parse_hybrid_pattern(f"{MOE_PATTERN_4_BLOCKS}/{MOE_PATTERN_4_BLOCKS[-2:]}")

    assert parsed.mtp_num_depths == 1
    assert parsed.mtp_pattern == parsed.main_pattern[-2:]


def test_language_config_num_layers_is_hybrid_layer_count():
    """The variant config must be expressed in hybrid layers, not GPT blocks."""
    # The proxy variant is four Qwen blocks, i.e. eight hybrid layers.
    config = get_qwen35_vl_language_config("proxy")

    assert config.num_layers == get_hybrid_total_layer_count(MOE_PATTERN_4_BLOCKS)
