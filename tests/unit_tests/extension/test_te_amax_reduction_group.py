# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Transformer Engine wrappers reduce amaxes over their owner's process-group collection."""

import pytest
import torch

from megatron.core.extensions.transformer_engine import (
    HAVE_TE,
    TEColumnParallelLinear,
    TELayerNormColumnParallelLinear,
    TELinear,
    TELMHeadColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.amax_reduction_group_utils import (
    forbid_global_amax_group,
    record_amax_groups,
    te_quantization_params,
    with_copied_amax_groups,
)
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not HAVE_TE, reason="requires Transformer Engine")

HIDDEN = 64
TOKENS = 32


def _init_method(weight):
    torch.nn.init.normal_(weight, std=0.02)


def _config(**kwargs):
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        params_dtype=torch.bfloat16,
        bf16=True,
        tensor_model_parallel_size=Utils.world_size // 2,
        **kwargs,
    )


def _build(kind, config, pg_collection):
    common = dict(
        config=config, init_method=_init_method, bias=False, skip_bias_add=False, is_expert=False
    )
    if kind == "column":
        return TEColumnParallelLinear(
            HIDDEN,
            HIDDEN,
            gather_output=False,
            tp_group=pg_collection.tp,
            pg_collection=pg_collection,
            **common,
        )
    if kind == "row":
        return TERowParallelLinear(
            HIDDEN,
            HIDDEN,
            input_is_parallel=True,
            tp_group=pg_collection.tp,
            pg_collection=pg_collection,
            **common,
        )
    if kind == "layernorm_column":
        return TELayerNormColumnParallelLinear(
            HIDDEN,
            HIDDEN,
            gather_output=False,
            tp_group=pg_collection.tp,
            pg_collection=pg_collection,
            **common,
        )
    if kind == "duplicated":
        return TELinear(
            HIDDEN,
            HIDDEN,
            parallel_mode="duplicated",
            skip_weight_param_allocation=False,
            pg_collection=pg_collection,
            **common,
        )
    if kind == "inference_duplicated":
        from megatron.core.tensor_parallel.inference_layers import InferenceLinear

        return InferenceLinear(
            HIDDEN,
            HIDDEN,
            parallel_mode="duplicated",
            skip_weight_param_allocation=False,
            pg_collection=pg_collection,
            **common,
        )
    raise ValueError(kind)


class TestTEWrapperAmaxReductionGroup:

    def setup_method(self, method):
        if Utils.world_size < 4 or Utils.world_size % 2:
            pytest.skip("needs an even number of ranks, at least 4")
        # TP is half the world, so tp_cp and tp_dp_cp are different groups.
        Utils.initialize_model_parallel(tensor_model_parallel_size=Utils.world_size // 2)
        model_parallel_cuda_manual_seed(123)
        self.pg_collection = with_copied_amax_groups(
            ProcessGroupCollection.use_mpu_process_groups()
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("kind", ["column", "row", "layernorm_column", "duplicated"])
    def test_module_recipe_uses_the_layer_collection(self, kind):
        module = _build(kind, _config(), self.pg_collection).cuda()
        # Training reduces over TP x DP x CP and evaluation only over TP x CP.
        module.te_quant_params = te_quantization_params(
            training_tp_only=False, evaluation_tp_only=True
        )
        in_features = HIDDEN // self.pg_collection.tp.size() if kind == "row" else HIDDEN
        x = torch.randn(TOKENS, 1, in_features, device="cuda", dtype=torch.bfloat16)

        with forbid_global_amax_group(), record_amax_groups() as groups:
            module.train()
            module(x)
            module.eval()
            module(x)

        assert len(groups) == 2
        assert groups[0] is self.pg_collection.tp_dp_cp
        assert groups[1] is self.pg_collection.tp_cp

    def test_inference_linear_training_recipe_uses_the_layer_collection(self):
        # InferenceLinear runs TELinear's forward only in training mode.
        module = _build("inference_duplicated", _config(), self.pg_collection).cuda()
        module.te_quant_params = te_quantization_params(
            training_tp_only=False, evaluation_tp_only=True
        )
        x = torch.randn(TOKENS, 1, HIDDEN, device="cuda", dtype=torch.bfloat16)

        with forbid_global_amax_group(), record_amax_groups() as groups:
            module.train()
            module(x)

        assert len(groups) == 1 and groups[0] is self.pg_collection.tp_dp_cp

    def test_grouped_linear_recipe_uses_the_layer_collection(self):
        from megatron.core.extensions.transformer_engine import TEGroupedLinear

        config = _config(num_moe_experts=2, moe_grouped_gemm=True, add_bias_linear=False)
        module = TEGroupedLinear(
            2,
            HIDDEN,
            HIDDEN,
            parallel_mode=None,
            config=config,
            init_method=_init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=True,
            pg_collection=self.pg_collection,
        ).cuda()
        module.te_quant_params = te_quantization_params(
            training_tp_only=False, evaluation_tp_only=False
        )
        m_splits = [TOKENS // 2, TOKENS // 2]
        x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)

        with forbid_global_amax_group(), record_amax_groups() as groups:
            module(x, m_splits)

        assert len(groups) == 1 and groups[0] is self.pg_collection.tp_dp_cp

    @pytest.mark.skipif(
        not (torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 10),
        reason="the MXFP8 output projection requires Blackwell (SM >= 10)",
    )
    def test_lm_head_uses_the_layer_collection(self):
        config = _config(fp8="hybrid", fp8_recipe="mxfp8", fp8_output_proj=True)
        module = TELMHeadColumnParallelLinear(
            HIDDEN,
            2 * HIDDEN,
            config=config,
            init_method=_init_method,
            bias=False,
            tp_group=self.pg_collection.tp,
            pg_collection=self.pg_collection,
        ).cuda()
        x = torch.randn(TOKENS, 1, HIDDEN, device="cuda", dtype=torch.bfloat16)

        with forbid_global_amax_group(), record_amax_groups() as groups:
            module(x)

        assert len(groups) == 1 and groups[0] is self.pg_collection.tp_dp_cp
