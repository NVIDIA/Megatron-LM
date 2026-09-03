# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.models.hybrid import (
    ArchitectureMetadata,
    MTPSplit,
    PipelineSplit,
    scan_hybrid_layer_config_list,
)
from megatron.core.ssm.mamba_layer_config import MambaLayerConfig
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.moe.moe_layer_config import MoELayerConfig


class _MTPSplitSubclass(MTPSplit):
    pass


def _root_config() -> TransformerConfig:
    return TransformerConfig(num_layers=3, hidden_size=64, num_attention_heads=4)


def _layer(config_type):
    return config_type.from_config(_root_config())


def test_scanner_returns_raw_marker_positions():
    mamba = _layer(MambaLayerConfig)
    attention = _layer(AttentionLayerConfig)
    moe = _layer(MoELayerConfig)
    architecture = [
        mamba,
        PipelineSplit,
        attention,
        PipelineSplit,
        mamba,
        PipelineSplit,
        attention,
        MTPSplit,
        moe,
        MTPSplit,
        moe,
    ]

    assert scan_hybrid_layer_config_list(architecture, pp_size=2) == ArchitectureMetadata(
        decoder_layer_count=4,
        mtp_num_depths=2,
        pipeline_split_indices=(1, 3, 5),
        mtp_split_indices=(7, 9),
        pipeline_segment_count=4,
        inferred_vpp_size=2,
    )


def test_scanner_allows_decoder_only_and_leading_mtp_split():
    mamba = _layer(MambaLayerConfig)

    assert scan_hybrid_layer_config_list([mamba], pp_size=4) == ArchitectureMetadata(
        decoder_layer_count=1,
        mtp_num_depths=0,
        pipeline_split_indices=(),
        mtp_split_indices=(),
        pipeline_segment_count=1,
        inferred_vpp_size=None,
    )
    assert scan_hybrid_layer_config_list([MTPSplit, mamba]) == ArchitectureMetadata(
        decoder_layer_count=0,
        mtp_num_depths=1,
        pipeline_split_indices=(),
        mtp_split_indices=(0,),
        pipeline_segment_count=1,
        inferred_vpp_size=None,
    )


def test_scanner_rejects_an_empty_architecture():
    with pytest.raises(ValueError, match="must not be empty"):
        scan_hybrid_layer_config_list([])


@pytest.mark.parametrize(
    "architecture",
    [
        lambda decoder, head: [decoder, MTPSplit],
        lambda decoder, head: [decoder, MTPSplit, MTPSplit, head],
        lambda decoder, head: [decoder, MTPSplit, head, MTPSplit],
    ],
    ids=["trailing-first-split", "consecutive-splits", "trailing-later-split"],
)
def test_scanner_rejects_empty_mtp_depths(architecture):
    decoder = _layer(MambaLayerConfig)
    head = _layer(AttentionLayerConfig)

    with pytest.raises(ValueError, match="MTP depth .* is empty"):
        scan_hybrid_layer_config_list(architecture(decoder, head))


def test_scanner_rejects_pipeline_split_after_mtp():
    decoder = _layer(MambaLayerConfig)
    head = _layer(AttentionLayerConfig)

    with pytest.raises(ValueError, match="PipelineSplit.*after the first MTPSplit"):
        scan_hybrid_layer_config_list([decoder, MTPSplit, head, PipelineSplit])


@pytest.mark.parametrize(
    "architecture",
    [
        lambda first, second: [PipelineSplit, first, PipelineSplit, second],
        lambda first, second: [first, PipelineSplit, PipelineSplit, second],
        lambda first, second: [first, PipelineSplit, second, PipelineSplit],
    ],
    ids=["leading", "consecutive", "trailing"],
)
def test_scanner_rejects_empty_pipeline_segments(architecture):
    with pytest.raises(ValueError, match="Pipeline segment .* is empty"):
        scan_hybrid_layer_config_list(
            architecture(_layer(MambaLayerConfig), _layer(AttentionLayerConfig)), pp_size=1
        )


def test_scanner_requires_explicit_segments_to_be_divisible_by_pp_size():
    architecture = [
        _layer(MambaLayerConfig),
        PipelineSplit,
        _layer(AttentionLayerConfig),
        PipelineSplit,
        _layer(MoELayerConfig),
    ]

    with pytest.raises(ValueError, match="not evenly divisible by pp_size=2"):
        scan_hybrid_layer_config_list(architecture, pp_size=2)


@pytest.mark.parametrize("pp_size", [0, -1])
def test_scanner_rejects_nonpositive_pp_size(pp_size):
    with pytest.raises(ValueError, match="pp_size must be positive"):
        scan_hybrid_layer_config_list([_layer(MambaLayerConfig)], pp_size=pp_size)


@pytest.mark.parametrize(
    "invalid_entry",
    [
        object(),
        TransformerConfig(num_layers=1, hidden_size=64, num_attention_heads=4),
        MTPSplit(),
        PipelineSplit(),
        _MTPSplitSubclass,
        [_layer(MambaLayerConfig)],
    ],
    ids=[
        "object",
        "base-transformer-config",
        "mtp-instance",
        "pipeline-instance",
        "marker-subclass",
        "nested-list",
    ],
)
def test_scanner_requires_supported_configs_or_exact_marker_classes(invalid_entry):
    with pytest.raises(ValueError, match="Invalid hybrid layer config entry at index 1"):
        scan_hybrid_layer_config_list([_layer(MambaLayerConfig), invalid_entry])


def test_scanner_accepts_identity_aligned_mtp_heads():
    attention = _layer(AttentionLayerConfig)
    moe = _layer(MoELayerConfig)
    head = [attention, moe]

    assert scan_hybrid_layer_config_list(
        [_layer(MambaLayerConfig), MTPSplit, *head, MTPSplit, *head]
    ).mtp_split_indices == (1, 4)


@pytest.mark.parametrize(
    "second_head",
    [
        lambda attention, moe: [attention],
        lambda attention, moe: [moe, attention],
        lambda attention, moe: [
            AttentionLayerConfig.from_config(attention),
            MoELayerConfig.from_config(moe),
        ],
    ],
    ids=["different-length", "different-order", "equal-values-distinct-objects"],
)
def test_scanner_rejects_mtp_heads_that_do_not_reuse_the_template_objects(second_head):
    attention = _layer(AttentionLayerConfig)
    moe = _layer(MoELayerConfig)

    with pytest.raises(ValueError, match="reuse the same layer config objects in the same order"):
        scan_hybrid_layer_config_list(
            [
                _layer(MambaLayerConfig),
                MTPSplit,
                attention,
                moe,
                MTPSplit,
                *second_head(attention, moe),
            ]
        )
