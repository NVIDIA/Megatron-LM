# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Config-time check that every PP/VPP chunk of a DSA layout starts on a computing layer."""

import pytest
import torch

from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.experimental_attention_variant.dsa import is_dsa_skip_topk_layer
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout
from megatron.core.transformer.transformer_config import (
    TransformerConfig,
    _validate_dsa_index_share_pipeline_layout,
)

# A 78-layer DSA model with 7 repeated MTP layers, dsa_indexer_topk_freq=4 and
# dsa_indexer_skip_topk_offset=3: top-k is computed on layers 1, 2, 3, 7, 11, ..., 75 and on
# the repeated MTP layer 79.
NUM_LAYERS = 78
MTP_NUM_LAYERS = 7
TOPK_FREQ = 4
SKIP_TOPK_OFFSET = 3

VALID_LAYOUTS = {
    # PP4 layouts with 18/24/24/12 and 22/24/24/8 decoder layers per stage.
    "Et*18|t*24|t*24|t*12m*7L": 4,
    "Et*22|t*24|t*24|t*8m*7L": 4,
    # PP4 x VPP2 with an MTP-only last chunk, and a variant with MTP beside decoders.
    "Et*10|t*12|t*12|t*12|t*12|t*12|t*8|m*7L": 4,
    "Et*10|t*12|t*12|t*12|t*12|t*12|t*4|t*4m*7L": 4,
}

INVALID_LAYOUTS = {
    # PP chunk 1 starts on layer 21, a skip layer whose source 19 is on PP rank 0.
    "Et*20|t*24|t*24|t*10m*7L": (4, "PP rank 1, VPP stage 0 starts on decoder layer 21"),
    # The last decoder chunk (PP rank 3, VPP stage 1) starts on skip layer 77.
    "Et*10|t*12|t*12|t*12|t*12|t*12|t*6|t*2m*7L": (
        4,
        "PP rank 3, VPP stage 1 starts on decoder layer 77",
    ),
    # A VPP stage-1 chunk on PP rank 0 starting on skip layer 49 (source 47).
    "Et*10|t*12|t*12|t*14|t*12|t*10|t*8|m*7L": (
        4,
        "PP rank 0, VPP stage 1 starts on decoder layer 49",
    ),
}


def _expected_chunk_starts(layout: PipelineParallelLayerLayout):
    """Independent reference: first decoder layer of each (pp, vp) chunk in flatten order."""
    starts = {}
    layer = 0
    for vp_stage in range(layout.virtual_pipeline_model_parallel_size):
        for pp_rank in range(layout.pipeline_model_parallel_size):
            chunk = layout.layout[pp_rank][vp_stage]
            if LayerType.decoder in chunk:
                starts[(pp_rank, vp_stage)] = layer + 1
            layer += chunk.count(LayerType.decoder)
    assert layer == NUM_LAYERS
    return starts


@pytest.mark.parametrize("layout_str,pp", sorted(VALID_LAYOUTS.items()))
def test_valid_index_share_layouts_pass(layout_str, pp):
    layout = PipelineParallelLayerLayout(layout_str, pp)
    _validate_dsa_index_share_pipeline_layout(
        layout, num_layers=NUM_LAYERS, topk_freq=TOPK_FREQ, skip_topk_offset=SKIP_TOPK_OFFSET
    )
    for (pp_rank, vp_stage), start in _expected_chunk_starts(layout).items():
        assert not is_dsa_skip_topk_layer(start, SKIP_TOPK_OFFSET, TOPK_FREQ), (pp_rank, vp_stage)


def test_vpp2_layout_chunk_starts_are_the_computing_layers():
    layout = PipelineParallelLayerLayout("Et*10|t*12|t*12|t*12|t*12|t*12|t*8|m*7L", 4)
    assert layout.virtual_pipeline_model_parallel_size == 2
    starts = _expected_chunk_starts(layout)
    assert starts == {
        (0, 0): 1,
        (1, 0): 11,
        (2, 0): 23,
        (3, 0): 35,
        (0, 1): 47,
        (1, 1): 59,
        (2, 1): 71,
    }
    # The MTP-only chunk (PP rank 3, VPP stage 1) starts on MTP layer 79, which computes.
    assert layout.layout[3][1].count(LayerType.mtp) == MTP_NUM_LAYERS
    assert not is_dsa_skip_topk_layer(NUM_LAYERS + 1, SKIP_TOPK_OFFSET, TOPK_FREQ)


@pytest.mark.parametrize("layout_str,case", sorted(INVALID_LAYOUTS.items()))
def test_invalid_layouts_are_rejected(layout_str, case):
    pp, message = case
    layout = PipelineParallelLayerLayout(layout_str, pp)
    with pytest.raises(ValueError, match=message):
        _validate_dsa_index_share_pipeline_layout(
            layout, num_layers=NUM_LAYERS, topk_freq=TOPK_FREQ, skip_topk_offset=SKIP_TOPK_OFFSET
        )


def test_mtp_only_chunk_starting_on_a_skip_layer_is_rejected():
    # 6 decoder layers with offset 3 and frequency 4: MTP layer 7 computes, but with 5 decoder
    # layers MTP layer 6 is a skip layer whose source (layer 3) sits in another chunk.
    good = PipelineParallelLayerLayout("Et*2|t*4|m*2L", 3)
    _validate_dsa_index_share_pipeline_layout(
        good, num_layers=6, topk_freq=TOPK_FREQ, skip_topk_offset=SKIP_TOPK_OFFSET
    )
    bad = PipelineParallelLayerLayout("Et*2|t*3|m*2L", 3)
    with pytest.raises(ValueError, match="starts on MTP layer 6"):
        _validate_dsa_index_share_pipeline_layout(
            bad, num_layers=5, topk_freq=TOPK_FREQ, skip_topk_offset=SKIP_TOPK_OFFSET
        )


def test_empty_and_embedding_only_chunks_are_ignored():
    layout = PipelineParallelLayerLayout("E|t*6|t*4|L", 4)
    _validate_dsa_index_share_pipeline_layout(
        layout, num_layers=10, topk_freq=TOPK_FREQ, skip_topk_offset=SKIP_TOPK_OFFSET
    )


def _make_dsa_layout_config(layout, pp, **overrides):
    kwargs = dict(
        num_layers=NUM_LAYERS,
        hidden_size=128,
        num_attention_heads=4,
        experimental_attention_variant="dsa",
        add_bias_linear=False,
        dsa_indexer_topk_freq=TOPK_FREQ,
        dsa_indexer_skip_topk_offset=SKIP_TOPK_OFFSET,
        mtp_num_layers=MTP_NUM_LAYERS,
        pipeline_model_parallel_size=pp,
        pipeline_model_parallel_layout=layout,
        pipeline_dtype=torch.bfloat16,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_transformer_config_accepts_a_vpp2_layout_with_an_mtp_only_chunk():
    config = _make_dsa_layout_config("Et*10|t*12|t*12|t*12|t*12|t*12|t*8|m*7L", 4)
    assert config.virtual_pipeline_model_parallel_size == 2
    assert config.mtp_standalone is False


def test_transformer_config_rejects_a_vpp_chunk_on_a_skip_layer():
    with pytest.raises(ValueError, match="PP rank 3, VPP stage 1 starts on decoder layer 77"):
        _make_dsa_layout_config("Et*10|t*12|t*12|t*12|t*12|t*12|t*6|t*2m*7L", 4)


@pytest.mark.parametrize(
    "overrides",
    [
        # No cross-layer sharing: every layer computes its own top-k.
        dict(dsa_indexer_topk_freq=1),
        # Not a DSA model.
        dict(experimental_attention_variant=None, dsa_indexer_topk_freq=1),
    ],
)
def test_layout_check_only_applies_to_dsa_index_sharing(overrides):
    config = _make_dsa_layout_config("Et*20|t*24|t*24|t*10m*7L", 4, **overrides)
    assert config.pipeline_model_parallel_layout.virtual_pipeline_model_parallel_size == 1
