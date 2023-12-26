# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest

import torch

from megatron.core.transformer.moe.base_moe_layer import Router, ZeroDropTopKRouter
from megatron.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils
from megatron.core.transformer.transformer_config import TransformerConfig


class TestZeroDropTop2Router:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        print("done intializing")
        num_moe_experts = 4
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_router_type="top2",
        )
        self.router = ZeroDropTopKRouter(
            num_local_experts=num_moe_experts,
            local_expert_indices=range(num_moe_experts),
            config=transformer_config,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.router, Router)

        num_weights = sum([p.numel() for p in self.router.parameters()])
        assert num_weights == 12 * 4, num_weights

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_forward(self):
        self.router = self.router.cuda()
        # [num tokens, hidden size]
        hidden_states = torch.randn((32, self.router.config.hidden_size))
        hidden_states = hidden_states.cuda()
        scores, indices = self.router(hidden_states)
        print(scores.shape, indices.shape)
        assert scores.shape == (32, 2)
        assert indices.shape == (32, 2)
        print(
            (indices == 0).sum(), (indices == 1).sum(), (indices == 2).sum(), (indices == 3).sum()
        )
        assert (indices == 0).sum() == 15, (indices == 0).sum()
        assert (indices == 1).sum() == 18, (indices == 1).sum()
        assert (indices == 2).sum() == 18, (indices == 2).sum()
        assert (indices == 3).sum() == 13, (indices == 3).sum()
