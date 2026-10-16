# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""HybridStack layers reduce amaxes over the stack's process-group collection."""

import pytest
import torch

from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols, validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.amax_reduction_group_utils import (
    forbid_global_amax_group,
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


class TestHybridStackAmaxReductionGroup:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("quantization", list(QUANTIZATION))
    def test_forward_contexts_use_the_stack_collection(self, quantization):
        pg_collection = with_copied_amax_groups(
            ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'pp', 'cp'])
        )
        config = TransformerConfig(
            hidden_size=256,
            num_layers=2,
            num_attention_heads=4,
            # 16 Mamba heads keep the input projection width a multiple of 16, as FP8 requires.
            mamba_head_dim=32,
            use_cpu_initialization=True,
            params_dtype=torch.bfloat16,
            bf16=True,
            **QUANTIZATION[quantization],
        )
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION
        stack = HybridStack(
            config,
            hybrid_stack_spec.submodules,
            layer_config_list=validate_segment_layers(layer_pattern, config),
            pp_layer_offset=0,
            pg_collection=pg_collection,
        ).cuda()
        hidden_states = torch.randn(32, 2, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        attention_mask = torch.ones((2, 1, 32, 32), dtype=torch.bool, device="cuda")

        with forbid_global_amax_group(), record_amax_groups() as groups:
            stack(hidden_states, attention_mask=attention_mask)

        # Delayed scaling opens one context around the stack; other recipes open one per layer.
        assert len(groups) == (1 if quantization == "fp8-delayed" else len(layer_pattern))
        assert all(group is pg_collection.tp_dp_cp for group in groups)
        # CUDA graph runners take the quantization context's collection from each layer.
        assert isinstance(stack.layers[0], MambaLayer)
        assert all(layer.pg_collection is pg_collection for layer in stack.layers)
