# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for loss-stage resolution in PipelineParallelLayerLayout (MTP loss split).

These tests are pure-logic: they construct the layout directly and pass explicit pp_rank /
vp_stage, so they need neither a GPU nor an initialized process group.
"""

import pytest

from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout

# Target recipe layout: rank0=E, rank1/2=text, rank3/4/5=mtp0/1/2, rank6/7=loss x2.
SPLIT_LAYOUT = "E|t|t|m|m|m|LL|LL"
RECIPE_LAYOUT = "E|(t|)*2m|m|m|LL|LL"  # the form used in the yaml; expands to SPLIT_LAYOUT
LEGACY_LAYOUT = "E|t|t|m|m|m|L"  # single loss stage handles everything


def test_recipe_layout_expands_as_expected():
    split = PipelineParallelLayerLayout.parse_str_to_list(SPLIT_LAYOUT)
    recipe = PipelineParallelLayerLayout.parse_str_to_list(RECIPE_LAYOUT)
    assert recipe == split
    assert len(split) == 8


def test_split_layout_counts():
    layout = PipelineParallelLayerLayout(SPLIT_LAYOUT, pipeline_model_parallel_size=8)
    assert layout.flatten_layout.count(LayerType.embedding) == 1
    assert layout.flatten_layout.count(LayerType.decoder) == 2
    assert layout.flatten_layout.count(LayerType.mtp) == 3
    assert layout.flatten_layout.count(LayerType.loss) == 4


def test_get_loss_stages_order():
    layout = PipelineParallelLayerLayout(SPLIT_LAYOUT, pipeline_model_parallel_size=8)
    assert layout.get_loss_stages() == [(6, 0, 2), (7, 0, 2)]
    assert layout.is_loss_split() is True


def test_chunk_assignment_split():
    layout = PipelineParallelLayerLayout(SPLIT_LAYOUT, pipeline_model_parallel_size=8)
    # rank6 owns the high MTP chunks (mtp1, mtp2), not the final stage.
    owned6, final6 = layout.get_loss_chunk_assignment(mtp_num_layers=3, pp_rank=6, vp_stage=0)
    assert owned6 == [2, 3]
    assert final6 is False
    # rank7 owns the low chunks (main, mtp0) and is the final stage (owns chunk 0).
    owned7, final7 = layout.get_loss_chunk_assignment(mtp_num_layers=3, pp_rank=7, vp_stage=0)
    assert owned7 == [0, 1]
    assert final7 is True


def test_chunk_assignment_legacy_single_loss_stage():
    layout = PipelineParallelLayerLayout(LEGACY_LAYOUT, pipeline_model_parallel_size=7)
    assert layout.is_loss_split() is False
    owned, final = layout.get_loss_chunk_assignment(mtp_num_layers=3, pp_rank=6, vp_stage=0)
    assert owned == [0, 1, 2, 3]  # single stage handles all chunks
    assert final is True


def test_validate_split_layout_ok():
    layout = PipelineParallelLayerLayout(SPLIT_LAYOUT, pipeline_model_parallel_size=8)
    # Should not raise: 2 + 2 loss slots == 1 + mtp_num_layers (4).
    layout.validate_layer_layout(num_layers=2, mtp_num_layers=3)


def test_validate_split_layout_slot_mismatch_raises():
    # 1 + 2 = 3 loss slots, but 1 + mtp_num_layers = 4 -> must fail.
    layout = PipelineParallelLayerLayout("E|t|t|m|m|m|L|LL", pipeline_model_parallel_size=8)
    with pytest.raises(AssertionError, match="total .*number of loss slots"):
        layout.validate_layer_layout(num_layers=2, mtp_num_layers=3)


def test_validate_legacy_layout_exempt():
    layout = PipelineParallelLayerLayout(LEGACY_LAYOUT, pipeline_model_parallel_size=7)
    # Single loss stage is exempt from the slot-sum check.
    layout.validate_layer_layout(num_layers=2, mtp_num_layers=3)
