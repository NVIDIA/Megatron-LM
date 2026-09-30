# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core import config
from megatron.core.transformer.moe.fused_a2a import reset_hybrid_ep_buffer
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe.test_token_dispatcher import (
    MoEModelTestContainer,
    is_deep_ep_available,
    is_hybrid_ep_available,
    permute_fusion_params,
)


@pytest.mark.skipif(
    not is_deep_ep_available() and not is_hybrid_ep_available(),
    reason="Deep EP and Hybrid EP are not available",
)
class TestFlexDispatcher:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        reset_hybrid_ep_buffer()
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2)])
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    @pytest.mark.parametrize("moe_flex_dispatcher_backend", ["deepep", "hybridep"])
    @pytest.mark.parametrize("moe_permute_fusion_into_hybridep", [True, False])
    def test_capacity_forward_backward(
        self,
        tp_size,
        ep_size,
        permute_fusion,
        moe_flex_dispatcher_backend,
        moe_permute_fusion_into_hybridep,
    ):
        if moe_flex_dispatcher_backend == "deepep" and not is_deep_ep_available():
            pytest.skip("Deep EP is not available")
        if moe_flex_dispatcher_backend == "hybridep" and not is_hybrid_ep_available():
            pytest.skip("Hybrid EP is not available")
        if moe_permute_fusion_into_hybridep:
            if permute_fusion or moe_flex_dispatcher_backend != "hybridep":
                pytest.skip(
                    "moe_permute_fusion_into_hybridep skipped because permute_fusion or hybridep is not set"
                )
        if permute_fusion:
            config.ENABLE_EXPERIMENTAL = True
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_token_drop_policy="probs",
            moe_expert_capacity_factor=0.5,
            moe_pad_expert_input_to_capacity=False,
            moe_permute_fusion=permute_fusion,
            hidden_size=1024,
            moe_flex_dispatcher_backend=moe_flex_dispatcher_backend,
            moe_permute_fusion_into_hybridep=moe_permute_fusion_into_hybridep,
            test_dtype=torch.bfloat16,
        )
        container.dispatcher_capacity_test()
        config.ENABLE_EXPERIMENTAL = False

    @pytest.mark.skipif(
        not is_te_min_version("1.7.0"), reason="TE 1.7.0 is required for MoE with FP8."
    )
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2)])
    @pytest.mark.parametrize("permute_fusion", [True])
    @pytest.mark.parametrize("moe_flex_dispatcher_backend", ["deepep", "hybridep"])
    @pytest.mark.parametrize("moe_permute_fusion_into_hybridep", [True, False])
    def test_router_padding_for_fp8_forward_backward(
        self,
        tp_size,
        ep_size,
        permute_fusion,
        moe_flex_dispatcher_backend,
        moe_permute_fusion_into_hybridep,
    ):
        if moe_flex_dispatcher_backend == "deepep" and not is_deep_ep_available():
            pytest.skip("Deep EP is not available")
        if moe_flex_dispatcher_backend == "hybridep" and not is_hybrid_ep_available():
            pytest.skip("Hybrid EP is not available")
        if moe_permute_fusion_into_hybridep:
            if permute_fusion or moe_flex_dispatcher_backend != "hybridep":
                pytest.skip(
                    "moe_permute_fusion_into_hybridep skipped because permute_fusion or hybridep is not set"
                )
        if permute_fusion:
            config.ENABLE_EXPERIMENTAL = True
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=32,
            moe_router_topk=4,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_pad_expert_input_to_capacity=False,
            moe_permute_fusion=permute_fusion,
            hidden_size=1024,
            moe_flex_dispatcher_backend=moe_flex_dispatcher_backend,
            moe_permute_fusion_into_hybridep=moe_permute_fusion_into_hybridep,
            test_dtype=torch.bfloat16,
        )
        container.dispatcher_router_padding_for_fp8_test()
        config.ENABLE_EXPERIMENTAL = False
