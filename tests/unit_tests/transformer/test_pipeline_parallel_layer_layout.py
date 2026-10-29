# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""A custom pipeline layout builds the stage of the pipeline rank the caller passes."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.transformer_block import TransformerBlock, get_num_layers_to_build
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from tests.unit_tests.test_utilities import Utils

# Two pipeline ranks with two virtual stages each. The stages are listed in
# (vp_stage, pp_rank) order: (0, 0), (0, 1), (1, 0), (1, 1).
INTERLEAVED_LAYOUT = "Et|tt|t|tL"
# (layer offset, number of decoder layers) of each [pp_rank][vp_stage].
INTERLEAVED_STAGES = [[(0, 1), (3, 1)], [(1, 2), (4, 1)]]

# Two pipeline ranks without virtual stages, split unevenly.
UNEVEN_LAYOUT = "Ettt|tL"
UNEVEN_STAGES = [[(0, 3)], [(3, 1)]]

LAYOUTS = [
    pytest.param(INTERLEAVED_LAYOUT, 5, INTERLEAVED_STAGES, id="interleaved"),
    pytest.param(UNEVEN_LAYOUT, 4, UNEVEN_STAGES, id="uneven"),
]


def _make_config(layout, num_layers):
    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=64,
        num_attention_heads=4,
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.bfloat16,
        pipeline_model_parallel_layout=layout,
        use_cpu_initialization=True,
    )


def _vp_stages(config, stages):
    """The vp_stage each model chunk passes: None without interleaving."""
    if config.virtual_pipeline_model_parallel_size is None:
        return [None]
    return list(range(len(stages)))


def _forbid_global_pipeline_state(monkeypatch):
    """Make reads of the global pipeline rank and virtual pipeline size fail."""

    def forbidden(*args, **kwargs):
        raise RuntimeError("the global pipeline rank or virtual pipeline size was read")

    monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_rank", forbidden)
    monkeypatch.setattr(parallel_state, "get_virtual_pipeline_model_parallel_world_size", forbidden)


@pytest.mark.parametrize("layout, num_layers, expected", LAYOUTS)
def test_explicit_pp_rank_selects_the_stage(monkeypatch, layout, num_layers, expected):
    """With pp_rank given, counts and offsets come from the layout alone."""
    config = _make_config(layout, num_layers)
    _forbid_global_pipeline_state(monkeypatch)

    for pp_rank, stages in enumerate(expected):
        for vp_stage, (offset, count) in zip(_vp_stages(config, stages), stages):
            assert get_transformer_layer_offset(config, vp_stage, pp_rank) == offset
            assert get_num_layers_to_build(config, vp_stage, pp_rank) == count
            assert config.pipeline_model_parallel_layout.get_layer_id_list(
                layer_type=LayerType.decoder, vp_stage=vp_stage, pp_rank=pp_rank
            ) == list(range(offset, offset + count))


@pytest.mark.parametrize("layout, num_layers, expected", LAYOUTS)
def test_default_pp_rank_comes_from_the_global_grid(monkeypatch, layout, num_layers, expected):
    """Without pp_rank, the global grid's pipeline rank selects the stage."""
    config = _make_config(layout, num_layers)
    # initialize_model_parallel sets the global virtual pipeline size from the same config.
    vp_size = config.virtual_pipeline_model_parallel_size
    monkeypatch.setattr(
        parallel_state, "get_virtual_pipeline_model_parallel_world_size", lambda: vp_size
    )

    for pp_rank, stages in enumerate(expected):
        monkeypatch.setattr(
            parallel_state, "get_pipeline_model_parallel_rank", lambda rank=pp_rank: rank
        )
        for vp_stage, (offset, count) in zip(_vp_stages(config, stages), stages):
            assert get_transformer_layer_offset(config, vp_stage) == offset
            assert get_num_layers_to_build(config, vp_stage) == count


@pytest.mark.parametrize(
    "layout, num_layers, expected, global_vp_size",
    [
        pytest.param(INTERLEAVED_LAYOUT, 5, INTERLEAVED_STAGES, None, id="interleaved"),
        pytest.param(UNEVEN_LAYOUT, 4, UNEVEN_STAGES, 2, id="uneven"),
    ],
)
def test_layout_uses_its_own_virtual_pipeline_size(
    monkeypatch, layout, num_layers, expected, global_vp_size
):
    """The layout's virtual pipeline size, not the global grid's, decides how vp_stage is used."""
    config = _make_config(layout, num_layers)
    pipeline_layout = config.pipeline_model_parallel_layout
    monkeypatch.setattr(
        parallel_state, "get_virtual_pipeline_model_parallel_world_size", lambda: global_vp_size
    )

    for pp_rank, stages in enumerate(expected):
        for vp_stage, (offset, count) in zip(_vp_stages(config, stages), stages):
            assert pipeline_layout.get_layer_offset(LayerType.decoder, vp_stage, pp_rank) == offset
            assert (
                pipeline_layout.get_num_layers_to_build(LayerType.decoder, vp_stage, pp_rank)
                == count
            )
        if config.virtual_pipeline_model_parallel_size is not None:
            with pytest.raises(AssertionError, match="vp_stage must be passed"):
                pipeline_layout.get_num_layers_to_build(LayerType.decoder, None, pp_rank)
            with pytest.raises(AssertionError, match="vp_stage must be passed"):
                pipeline_layout.get_layer_offset(LayerType.decoder, None, pp_rank)


class TestTransformerBlockOnItsOwnPipelineGrid:
    """A block on its own PP=2 grid builds its stage while the global grid has PP=1."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("forbid_global_reads", [True, False])
    def test_layer_numbers_follow_the_block_pipeline_group(self, monkeypatch, forbid_global_reads):
        pp_size = 2
        if Utils.world_size % pp_size != 0:
            pytest.skip(f"needs a world size divisible by {pp_size}")
        # The global grid has one pipeline stage without virtual stages, so reading it would
        # select stage (0, 0) on every rank.
        Utils.initialize_model_parallel()
        grid = HyperCommGrid([1, 1, pp_size, Utils.world_size // pp_size], ["tp", "cp", "pp", "dp"])
        pg_collection = ProcessGroupCollection(
            tp=grid.create_pg("tp"), cp=grid.create_pg("cp"), pp=grid.create_pg("pp")
        )
        pp_rank = pg_collection.pp.rank()
        config = _make_config(INTERLEAVED_LAYOUT, 5)

        with monkeypatch.context() as patch:
            if forbid_global_reads:
                _forbid_global_pipeline_state(patch)
            for vp_stage, (offset, count) in enumerate(INTERLEAVED_STAGES[pp_rank]):
                block = TransformerBlock(
                    config,
                    get_gpt_layer_local_spec(),
                    pg_collection=pg_collection,
                    vp_stage=vp_stage,
                )
                layer_numbers = [layer.layer_number for layer in block.layers]
                assert layer_numbers == list(
                    range(offset + 1, offset + count + 1)
                ), f"{pp_rank=} {vp_stage=}"
