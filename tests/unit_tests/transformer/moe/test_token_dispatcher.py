# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest

import torch

from megatron.core.transformer.moe.router import Router, TopKRouter
from megatron.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils
from megatron.core.transformer.transformer_config import TransformerConfig


class TestDroplessDispatcher:
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
        self.router = TopKRouter(
            num_local_experts=num_moe_experts,
            local_expert_indices=range(num_moe_experts),
            k=2,
            routing_type="top",
            config=transformer_config,
        )
        self.token_dispatcher = self.router.token_dispatcher

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_forward(self):
        self.router = self.router.cuda()
        # [bs, seql, hidden size]
        hidden_states = torch.randn((32, 8, self.router.config.hidden_size))
        hidden_states = hidden_states.cuda()
        scores, indices = self.router(hidden_states)
        assert scores.shape == (256, 2), "Scores shape is not correct"
        assert indices.shape == (256, 2), "Indices shape is not correct"
        print(
            (indices == 0).sum(), (indices == 1).sum(), (indices == 2).sum(), (indices == 3).sum()
        )
        (
            permuted_local_hidden_states,
            tokens_per_expert,
            local_probs,
            revert_indices,
            global_local_map,
        ) = self.token_dispatcher.dispatch(hidden_states, scores, indices)
        probs = torch.ones_like(local_probs) / 2
        restored_hidden_states, restored_bias = self.token_dispatcher.restore(
            permuted_local_hidden_states,
            probs,
            revert_indices,
            global_local_map,
            bias=torch.zeros_like(permuted_local_hidden_states),
        )

        assert torch.allclose(
            restored_hidden_states, hidden_states
        ), "Restored hidden states do not match original hidden states"
