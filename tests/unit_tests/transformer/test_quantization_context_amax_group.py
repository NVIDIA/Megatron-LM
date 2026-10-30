# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Transformer layers, blocks and MTP layers reduce amaxes over their own collection."""

import pytest
import torch

from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    MultiTokenPredictionLayer,
)
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.amax_reduction_group_utils import (
    forbid_global_amax_group,
    global_amax_group,
    record_amax_groups,
    with_copied_amax_groups,
)
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not HAVE_TE, reason="requires Transformer Engine")

QUANTIZATION = {
    "fp8-delayed": dict(fp8="e4m3", fp8_recipe=Fp8Recipe.delayed),
    "fp8-tensorwise": dict(fp8="e4m3", fp8_recipe=Fp8Recipe.tensorwise),
    "fp4": dict(fp4="e2m1"),
}
HIDDEN = 64


def _config(**kwargs):
    return TransformerConfig(
        num_layers=2,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        use_cpu_initialization=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        **kwargs,
    )


def _bare_module(cls, config, pg_collection):
    """Create ``cls`` without building its submodules."""
    module = cls.__new__(cls)
    torch.nn.Module.__init__(module)
    module.config = config
    module.pg_collection = pg_collection
    return module


class TestTransformerLayerAmaxReductionGroup:

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("quantization", ["fp8-tensorwise", "fp4"])
    def test_inner_context_uses_the_layer_collection_on_a_custom_grid(self, quantization):
        world_size = Utils.world_size
        if world_size < 4 or world_size % 2:
            pytest.skip("needs an even number of ranks, at least 4")
        # Global grid: TP=1 and PP=2, so the global amax group is the DP group of a stage.
        Utils.initialize_model_parallel(pipeline_model_parallel_size=2)
        # The layer's grid has PP as the fastest dimension, so its TP groups cross the stages.
        grid = HyperCommGrid([2, world_size // 2, 1, 1], ["pp", "tp", "cp", "dp"])
        pg_collection = ProcessGroupCollection(
            tp_cp=grid.create_pg(["tp", "cp"]), tp_dp_cp=grid.create_pg(["tp", "cp", "dp"])
        )
        layer = _bare_module(TransformerLayer, _config(**QUANTIZATION[quantization]), pg_collection)
        layer.layer_number = 1

        with record_amax_groups() as groups:
            layer.get_inner_quantization_context()

        assert len(groups) == 1
        custom_ranks = list(range(torch.distributed.get_rank() % 2, world_size, 2))
        assert torch.distributed.get_process_group_ranks(groups[0]) == custom_ranks
        assert groups[0] is pg_collection.tp_dp_cp


class TestTransformerBlockAmaxReductionGroup:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("recompute", [False, True], ids=["no-recompute", "full-recompute"])
    @pytest.mark.parametrize("quantization", list(QUANTIZATION))
    def test_forward_contexts_use_the_block_collection(self, quantization, recompute):
        pg_collection = with_copied_amax_groups(ProcessGroupCollection.use_mpu_process_groups())
        recompute_kwargs = (
            dict(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1)
            if recompute
            else {}
        )
        config = _config(**QUANTIZATION[quantization], **recompute_kwargs)
        block = TransformerBlock(
            config, get_gpt_layer_with_transformer_engine_spec(), pg_collection=pg_collection
        ).cuda()
        block.train()
        hidden_states = torch.randn(16, 2, HIDDEN, device="cuda", dtype=torch.bfloat16)

        with forbid_global_amax_group(), record_amax_groups() as groups:
            block(hidden_states=hidden_states, attention_mask=None)

        # Delayed scaling opens one context around the stack; other recipes open one per layer.
        assert len(groups) == (1 if quantization == "fp8-delayed" else config.num_layers)
        assert all(group is pg_collection.tp_dp_cp for group in groups)


class TestMultiTokenPredictionAmaxReductionGroup:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("quantization", ["fp8-tensorwise", "fp4"])
    def test_inner_context_uses_the_layer_collection(self, quantization):
        pg_collection = with_copied_amax_groups()
        layer = _bare_module(
            MultiTokenPredictionLayer, _config(**QUANTIZATION[quantization]), pg_collection
        )

        with forbid_global_amax_group(), record_amax_groups() as groups:
            layer.get_inner_quantization_context()

        assert len(groups) == 1 and groups[0] is pg_collection.tp_dp_cp

    def test_projection_and_layer_contexts_use_the_layer_collection(self):
        pg_collection = with_copied_amax_groups()
        layer = _bare_module(
            MultiTokenPredictionLayer, _config(**QUANTIZATION["fp8-tensorwise"]), pg_collection
        )
        layer.mtp_layer_pattern = None
        layer.mhc_enabled = False
        layer._concat_embeddings = lambda hidden_states, decoder_input: hidden_states
        layer.mtp_model_layer = lambda hidden_states, **kwargs: (hidden_states, None)
        layer._postprocess = lambda hidden_states: hidden_states
        hidden_states = torch.randn(16, 2, HIDDEN)

        with forbid_global_amax_group(), record_amax_groups() as groups:
            layer._proj_and_transformer_layer(hidden_states, hidden_states)

        assert len(groups) == 2
        assert all(group is pg_collection.tp_dp_cp for group in groups)

    def test_default_block_collection_carries_the_amax_reduction_groups(self):
        config = _config(mtp_num_layers=1, **QUANTIZATION["fp8-tensorwise"])
        spec = get_gpt_mtp_block_spec(
            config=config,
            spec=get_gpt_layer_with_transformer_engine_spec(),
            use_transformer_engine=True,
        )
        block = MultiTokenPredictionBlock(config=config, spec=spec)

        with forbid_global_amax_group(), record_amax_groups() as groups:
            block.layers[0].get_inner_quantization_context()

        assert len(groups) == 1 and groups[0] is global_amax_group(False)
