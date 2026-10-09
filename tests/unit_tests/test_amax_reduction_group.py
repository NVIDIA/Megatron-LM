# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""FP8/FP4 contexts take the amax reduction group from the caller's process-group collection."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.core import fp4_utils, fp8_utils
from megatron.core.enums import Fp8Recipe
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.amax_reduction_group_utils import (
    forbid_global_amax_group,
    global_amax_group,
    record_amax_groups,
    te_quantization_params,
    with_copied_amax_groups,
)
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not fp8_utils.HAVE_TE, reason="requires Transformer Engine")


def _config(**kwargs):
    return TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4, **kwargs)


class TestAmaxReductionGroupHelper:

    def test_selects_the_collection_field(self):
        from megatron.core.process_groups_config import amax_reduction_group

        tp_cp, tp_dp_cp = object(), object()
        pg_collection = ProcessGroupCollection(tp_cp=tp_cp, tp_dp_cp=tp_dp_cp)

        assert amax_reduction_group(pg_collection, tp_only_amax_red=True) is tp_cp
        assert amax_reduction_group(pg_collection, tp_only_amax_red=False) is tp_dp_cp

    @pytest.mark.parametrize("tp_only_amax_red", [False, True])
    def test_rejects_an_unset_or_none_field(self, tp_only_amax_red):
        from megatron.core.process_groups_config import amax_reduction_group

        field_name = "tp_cp" if tp_only_amax_red else "tp_dp_cp"
        other_name = "tp_dp_cp" if tp_only_amax_red else "tp_cp"
        for pg_collection in (
            ProcessGroupCollection(**{other_name: object()}),
            ProcessGroupCollection(**{field_name: None, other_name: object()}),
        ):
            with pytest.raises(ValueError, match=field_name):
                amax_reduction_group(pg_collection, tp_only_amax_red)


class TestAmaxReductionGroupOnGrids:

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "grid",
        [
            dict(tensor_model_parallel_size=2),
            dict(context_parallel_size=2),
            dict(tensor_model_parallel_size=2, gtp_remat_size=2),
        ],
        ids=["tp2", "cp2", "tp2-gtp2"],
    )
    @pytest.mark.parametrize("tp_only_amax_red", [False, True])
    def test_standard_collection_matches_global_group(self, grid, tp_only_amax_red):
        from megatron.core.process_groups_config import amax_reduction_group

        if Utils.world_size != 4:
            pytest.skip("the grid sizes below assume 4 ranks")
        Utils.initialize_model_parallel(**grid)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        group = amax_reduction_group(pg_collection, tp_only_amax_red)

        assert group is global_amax_group(tp_only_amax_red)
        # TP x CP is 2 on every grid; the full group spans DP and the gtp_remat peers.
        assert group.size() == (2 if tp_only_amax_red else 4)

    @pytest.mark.parametrize("recipe", [Fp8Recipe.delayed, Fp8Recipe.tensorwise])
    @pytest.mark.parametrize("tp_only_amax_red", [False, True])
    def test_fp8_context_reduces_over_the_collection_group(self, recipe, tp_only_amax_red):
        Utils.initialize_model_parallel(tensor_model_parallel_size=min(2, Utils.world_size))
        pg_collection = with_copied_amax_groups()
        expected = pg_collection.tp_cp if tp_only_amax_red else pg_collection.tp_dp_cp
        assert expected is not global_amax_group(tp_only_amax_red)
        config = _config(fp8="e4m3", fp8_recipe=recipe, tp_only_amax_red=tp_only_amax_red)

        with forbid_global_amax_group(), record_amax_groups() as groups:
            fp8_utils.get_fp8_context(config, 0, pg_collection=pg_collection)

        assert len(groups) == 1 and groups[0] is expected

    @pytest.mark.parametrize("tp_only_amax_red", [False, True])
    def test_fp4_context_reduces_over_the_collection_group(self, tp_only_amax_red):
        Utils.initialize_model_parallel(tensor_model_parallel_size=min(2, Utils.world_size))
        pg_collection = with_copied_amax_groups()
        expected = pg_collection.tp_cp if tp_only_amax_red else pg_collection.tp_dp_cp
        config = _config(fp4="e2m1", tp_only_amax_red=tp_only_amax_red)

        with forbid_global_amax_group(), record_amax_groups() as groups:
            fp4_utils.get_fp4_context(config, 0, pg_collection=pg_collection)

        assert len(groups) == 1 and groups[0] is expected

    def test_te_module_recipe_reduces_over_the_collection_group(self):
        """Training and evaluation recipes can select different amax groups."""
        from megatron.core.extensions import transformer_engine as te_extension

        Utils.initialize_model_parallel(tensor_model_parallel_size=min(2, Utils.world_size))
        pg_collection = with_copied_amax_groups()
        qparams = te_quantization_params(training_tp_only=False, evaluation_tp_only=True)

        with forbid_global_amax_group(), record_amax_groups() as groups:
            te_extension._get_fp8_autocast_for_quant_params(
                qparams, True, pg_collection=pg_collection
            )
            te_extension._get_fp8_autocast_for_quant_params(
                qparams, False, pg_collection=pg_collection
            )

        assert len(groups) == 2
        assert groups[0] is pg_collection.tp_dp_cp
        assert groups[1] is pg_collection.tp_cp

    def test_custom_grid_differs_from_global_grid(self):
        """A model on its own grid reduces over its own TP x DP x CP ranks."""
        world_size = Utils.world_size
        if world_size < 4 or world_size % 2:
            pytest.skip("needs an even number of ranks, at least 4")
        # Global grid: TP=1 and PP=2, so the global amax group is the DP group of a stage.
        Utils.initialize_model_parallel(pipeline_model_parallel_size=2)
        # Custom grid: PP is the fastest dimension, so its TP groups cross the global stages.
        grid = HyperCommGrid([2, world_size // 2, 1, 1], ["pp", "tp", "cp", "dp"])
        pg_collection = ProcessGroupCollection(
            tp_cp=grid.create_pg(["tp", "cp"]), tp_dp_cp=grid.create_pg(["tp", "cp", "dp"])
        )
        custom_ranks = list(range(torch.distributed.get_rank() % 2, world_size, 2))
        global_ranks = torch.distributed.get_process_group_ranks(global_amax_group(False))
        assert custom_ranks != global_ranks
        config = _config(fp8="e4m3", fp8_recipe=Fp8Recipe.delayed)

        with record_amax_groups() as groups:
            fp8_utils.get_fp8_context(config, pg_collection=pg_collection)

        assert len(groups) == 1
        assert torch.distributed.get_process_group_ranks(groups[0]) == custom_ranks

    @pytest.mark.parametrize("tp_only_amax_red", [False, True])
    def test_missing_collection_group_raises(self, tp_only_amax_red):
        from megatron.core.extensions import transformer_engine as te_extension

        Utils.initialize_model_parallel(tensor_model_parallel_size=min(2, Utils.world_size))
        field_name = "tp_cp" if tp_only_amax_red else "tp_dp_cp"
        other_name = "tp_dp_cp" if tp_only_amax_red else "tp_cp"
        pg_collection = ProcessGroupCollection(
            **{other_name: global_amax_group(not tp_only_amax_red)}
        )
        config = _config(
            fp8="e4m3", fp8_recipe=Fp8Recipe.tensorwise, tp_only_amax_red=tp_only_amax_red
        )
        qparams = te_quantization_params(tp_only_amax_red, tp_only_amax_red)

        with pytest.raises(ValueError, match=field_name):
            fp8_utils.get_fp8_context(config, pg_collection=pg_collection)
        with pytest.raises(ValueError, match=field_name):
            te_extension._get_fp8_autocast_for_quant_params(
                qparams, True, pg_collection=pg_collection
            )

    def test_init_context_does_not_need_an_amax_group(self):
        Utils.initialize_model_parallel()
        config = _config(fp8="e4m3", fp8_recipe=Fp8Recipe.tensorwise, fp8_param=True)

        with (
            forbid_global_amax_group(),
            patch.object(
                fp8_utils.transformer_engine.pytorch, "fp8_model_init", return_value=MagicMock()
            ) as fp8_model_init,
        ):
            fp8_utils.get_fp8_context(
                config, 0, is_init=True, pg_collection=ProcessGroupCollection()
            )

        fp8_model_init.assert_called_once()

    @pytest.mark.parametrize("tp_only_amax_red", [False, True])
    def test_omitted_collection_uses_the_global_group(self, tp_only_amax_red):
        """Callers that pass no collection keep the global amax reduction group."""
        from megatron.core.extensions import transformer_engine as te_extension

        Utils.initialize_model_parallel(tensor_model_parallel_size=min(2, Utils.world_size))
        expected = global_amax_group(tp_only_amax_red)
        fp8_config = _config(
            fp8="e4m3", fp8_recipe=Fp8Recipe.tensorwise, tp_only_amax_red=tp_only_amax_red
        )
        fp4_config = _config(fp4="e2m1", tp_only_amax_red=tp_only_amax_red)
        qparams = te_quantization_params(tp_only_amax_red, tp_only_amax_red)

        with record_amax_groups() as groups:
            fp8_utils.get_fp8_context(fp8_config)
            fp4_utils.get_fp4_context(fp4_config)
            te_extension._get_fp8_autocast_for_quant_params(qparams, True)

        assert len(groups) == 3
        assert all(group is expected for group in groups)
