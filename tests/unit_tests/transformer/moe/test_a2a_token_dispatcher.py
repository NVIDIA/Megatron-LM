# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe.test_token_dispatcher import (
    MoEModelTestContainer,
    permute_fusion_params,
)


def test_placeholder():
    """This is here because otherwise there's no other test in this module (all disabled) and pytest would fail."""
    pass


class TestAlltoAllDispatcher:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2), (1, 1)])
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    def test_forward_backward(self, tp_size, ep_size, permute_fusion):
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_permute_fusion=permute_fusion,
        )
        container.dispatcher_dropless_test()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2), (1, 1)])
    def test_a2aseq_forward_backward(self, tp_size, ep_size):
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall_seq",
            moe_permute_fusion=False,
        )
        container.dispatcher_dropless_test()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2), (1, 1)])
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    def test_capacity_forward_backward(self, tp_size, ep_size, permute_fusion):
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_token_drop_policy="probs",
            moe_expert_capacity_factor=0.5,
            moe_pad_expert_input_to_capacity=False,
            moe_permute_fusion=permute_fusion,
        )
        container.dispatcher_capacity_test()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2), (1, 1)])
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    def test_capacity_padding_forward_backward(self, tp_size, ep_size, permute_fusion):
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_token_drop_policy="probs",
            moe_expert_capacity_factor=0.6,
            moe_pad_expert_input_to_capacity=True,
            moe_permute_fusion=permute_fusion,
        )
        container.dispatcher_drop_and_pad_test()
