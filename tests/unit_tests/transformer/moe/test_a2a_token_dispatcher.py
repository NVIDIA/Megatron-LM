# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.moe.moe_utils import permute, unpermute
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe.test_token_dispatcher import MoEModelTestContainer

class TestAlltoAllDispatcher:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [
        (1, 8),
        (8, 1),
        (4, 2),
        (1, 1),
    ])
    def test_forward_backward(self, tp_size, ep_size):
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
        )
        container.dispatcher_dropless_test()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [
        (1, 8),
        (8, 1),
        (4, 2),
        (1, 1),
    ])
    def test_capacity_forward_backward(self, tp_size, ep_size):
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
        )
        container.dispacher_capacity_test()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [
        (1, 8),
        (8, 1),
        (4, 2),
        (1, 1)
    ])
    def test_capacity_padding_forward_backward(self, tp_size, ep_size):
        import time
        time.sleep(5)
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
            moe_pad_expert_input_to_capacity=True,
        )
        container.dispatcher_drop_and_pad_test()

