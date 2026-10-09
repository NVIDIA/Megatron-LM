# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CUDA graph runners reduce amaxes over the graphed module's process-group collection."""

from types import SimpleNamespace

import pytest

from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.transformer.cuda_graphs import _CudaGraphRunner
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.amax_reduction_group_utils import (
    forbid_global_amax_group,
    record_amax_groups,
    with_copied_amax_groups,
)
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not HAVE_TE, reason="requires Transformer Engine")


class TestCudaGraphRunnerAmaxReductionGroup:

    def setup_method(self, method):
        Utils.initialize_model_parallel(tensor_model_parallel_size=min(2, Utils.world_size))

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("quantization", ["fp8", "fp4"])
    def test_quantization_context_uses_the_base_module_collection(self, quantization):
        pg_collection = with_copied_amax_groups()
        if quantization == "fp8":
            quantization_kwargs = dict(fp8="e4m3", fp8_recipe=Fp8Recipe.tensorwise)
        else:
            quantization_kwargs = dict(fp4="e2m1")
        config = TransformerConfig(
            num_layers=2, hidden_size=64, num_attention_heads=4, **quantization_kwargs
        )
        runner = object.__new__(_CudaGraphRunner)
        runner.fp8_runtime_enabled = quantization == "fp8"
        runner.fp4_runtime_enabled = quantization == "fp4"
        runner.base_module = SimpleNamespace(
            config=config, layer_number=2, pg_collection=pg_collection
        )

        with forbid_global_amax_group(), record_amax_groups() as groups:
            runner.get_quantization_context()

        assert len(groups) == 1 and groups[0] is pg_collection.tp_dp_cp
