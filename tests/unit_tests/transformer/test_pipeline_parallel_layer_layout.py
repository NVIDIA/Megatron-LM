# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""A pipeline stage builds the layers of the pipeline rank the caller passes. Omitting the rank
falls back to the global pipeline rank with a warning, unless the model has one stage."""

import warnings

import pytest
import torch

from megatron.core import parallel_state, process_groups_config
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.process_groups_config import ProcessGroupCollection, ProcessGroupFallbackWarning
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.transformer_block import TransformerBlock, get_num_layers_to_build
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from tests.unit_tests.test_utilities import Utils, forbid_global_process_groups

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

# Two pipeline ranks without a custom layout: four layers split evenly.
EVEN_STAGES = [[(0, 2)], [(2, 2)]]
PLACEMENTS = LAYOUTS + [pytest.param(None, 4, EVEN_STAGES, id="even-split")]

# Every function that places layers on a pipeline stage, with the argument it falls back on.
_PLACEMENT_FALLBACKS = {
    ("get_transformer_layer_offset", "pp_rank"),
    ("get_num_layers_to_build", "pp_rank"),
    ("PipelineParallelLayerLayout.get_layer_offset", "pp_rank"),
    ("PipelineParallelLayerLayout.get_num_layers_to_build", "pp_rank"),
}


@pytest.fixture(autouse=True)
def fresh_warning_registry(monkeypatch):
    """Each test observes the first fallback warning of every owner."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


def _fallbacks(record):
    """Return the (owner, argument) pairs of the fallback warnings in ``record``.

    Also checks that each warning carries the removal version and points at this file, which
    omitted the argument.
    """
    pairs = set()
    for w in record:
        if not issubclass(w.category, ProcessGroupFallbackWarning):
            continue
        message = str(w.message)
        assert "deprecated since Megatron Core 0.21 and will be removed in 0.23" in message
        assert w.filename == __file__
        owner, _, rest = message.partition(" was called without `")
        pairs.add((owner, rest.partition("`")[0]))
    return pairs


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


@pytest.mark.parametrize("layout, num_layers, expected", PLACEMENTS)
def test_explicit_pp_rank_selects_the_stage(monkeypatch, layout, num_layers, expected):
    """With pp_rank given, counts and offsets come from the config alone, without a warning."""
    config = _make_config(layout, num_layers)
    _forbid_global_pipeline_state(monkeypatch)

    with forbid_global_process_groups(), warnings.catch_warnings():
        warnings.simplefilter("error", ProcessGroupFallbackWarning)
        for pp_rank, stages in enumerate(expected):
            for vp_stage, (offset, count) in zip(_vp_stages(config, stages), stages):
                assert get_transformer_layer_offset(config, vp_stage, pp_rank) == offset
                assert get_num_layers_to_build(config, vp_stage, pp_rank) == count
                if layout is not None:
                    assert config.pipeline_model_parallel_layout.get_layer_id_list(
                        layer_type=LayerType.decoder, vp_stage=vp_stage, pp_rank=pp_rank
                    ) == list(range(offset, offset + count))


@pytest.mark.parametrize("layout, num_layers, expected", PLACEMENTS)
def test_omitted_pp_rank_warns_and_uses_the_global_rank(monkeypatch, layout, num_layers, expected):
    """Without pp_rank, the global pipeline rank selects the stage, and each function warns."""
    config = _make_config(layout, num_layers)
    pipeline_layout = config.pipeline_model_parallel_layout

    for pp_rank, stages in enumerate(expected):
        # parallel_state.set_pipeline_model_parallel_rank overrides the rank the same way.
        monkeypatch.setattr(
            parallel_state, "get_pipeline_model_parallel_rank", lambda rank=pp_rank: rank
        )
        for vp_stage, (offset, count) in zip(_vp_stages(config, stages), stages):
            process_groups_config._warned_global_process_group_fallbacks.clear()
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                assert get_transformer_layer_offset(config, vp_stage) == offset
                assert get_num_layers_to_build(config, vp_stage) == count
                if layout is not None:
                    assert pipeline_layout.get_layer_id_list(
                        layer_type=LayerType.decoder, vp_stage=vp_stage
                    ) == list(range(offset, offset + count))
            expected_fallbacks = {
                pair
                for pair in _PLACEMENT_FALLBACKS
                if layout is not None or not pair[0].startswith("PipelineParallelLayerLayout")
            }
            assert _fallbacks(record) == expected_fallbacks


@pytest.mark.parametrize("layout", [None, "EtttL"])
def test_single_pipeline_stage_needs_no_pp_rank(layout):
    """With one pipeline stage, every rank is stage 0, so no rank is read and nothing warns."""
    config = TransformerConfig(
        num_layers=3,
        hidden_size=64,
        num_attention_heads=4,
        pipeline_model_parallel_layout=layout,
        use_cpu_initialization=True,
    )

    with forbid_global_process_groups(), warnings.catch_warnings():
        warnings.simplefilter("error", ProcessGroupFallbackWarning)
        assert get_transformer_layer_offset(config) == 0
        assert get_num_layers_to_build(config) == 3
        if layout is not None:
            assert config.pipeline_model_parallel_layout.get_layer_id_list() == [0, 1, 2]


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


class TestOmittedPipelineRankOnTheGlobalGrid:
    """Without pp_rank, a stage of the global PP=2 grid builds that stage, with a warning."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("layout, num_layers, expected", PLACEMENTS)
    def test_omitted_pp_rank_matches_the_global_pipeline_rank(self, layout, num_layers, expected):
        pp_size = 2
        if Utils.world_size % pp_size != 0:
            pytest.skip(f"needs a world size divisible by {pp_size}")
        config = _make_config(layout, num_layers)
        Utils.initialize_model_parallel(
            pipeline_model_parallel_size=pp_size,
            virtual_pipeline_model_parallel_size=config.virtual_pipeline_model_parallel_size,
        )
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        stages = expected[pp_rank]

        with pytest.warns(ProcessGroupFallbackWarning) as record:
            for vp_stage, (offset, count) in zip(_vp_stages(config, stages), stages):
                assert get_transformer_layer_offset(config, vp_stage) == offset
                assert get_num_layers_to_build(config, vp_stage) == count
        assert _fallbacks(record) == {
            ("get_transformer_layer_offset", "pp_rank"),
            ("get_num_layers_to_build", "pp_rank"),
        }


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
