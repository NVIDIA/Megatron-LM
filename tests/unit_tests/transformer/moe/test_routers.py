# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest

import torch

from megatron.core.transformer.moe.router import Router
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec


class TestTop2Router:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        print("done intializing")
        num_moe_experts = 4
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0,
        )
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_moe_experts, moe_grouped_gemm=False
        )
        self.sequential_mlp = MoELayer(
            self.transformer_config, transformer_layer_spec.submodules.mlp.submodules
        )
        self.router = self.sequential_mlp.router

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.router, Router)

        num_weights = sum([p.numel() for p in self.router.parameters()])
        assert num_weights == 12 * 4, num_weights

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("moe_router_pre_softmax", [
        (True),
        (False),
    ])
    def test_router_forward(self, moe_router_pre_softmax):
        with torch.no_grad():
            self.router = self.router.cuda()
            self.router.config.moe_router_pre_softmax = moe_router_pre_softmax
            # [num tokens, hidden size]
            hidden_states = torch.randn((32, 2, self.router.config.hidden_size))
            hidden_states = hidden_states.cuda()
            scores, indices = self.router(hidden_states)
            print(scores.shape, indices.shape)
            assert scores.shape == (64, 2)
            assert indices.shape == (64, 2)
            print(
                (indices == 0).sum(), (indices == 1).sum(), (indices == 2).sum(), (indices == 3).sum()
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_aux_loss(self):
        self.sequential_mlp = self.sequential_mlp.cuda()
        
        # Without aux loss
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size))
        hidden_states = hidden_states.cuda()
        out = self.sequential_mlp(hidden_states)[0]
        out.sum().mul_(0).backward()
        assert self.sequential_mlp.router.weight.grad.abs().sum() == 0
        
        # With aux loss
        self.transformer_config.moe_aux_loss_coeff = 1
        out = self.sequential_mlp(hidden_states)[0]
        out.sum().mul_(0).backward()
        assert self.sequential_mlp.router.weight.grad.abs().sum() > 0
        
        # With Z loss
        self.transformer_config.moe_aux_loss_coeff = 0
        self.transformer_config.moe_z_loss_coeff = 1
        self.sequential_mlp.router.weight.grad.fill_(0)
        out = self.sequential_mlp(hidden_states)[0]
        out.sum().mul_(0).backward()
        assert self.sequential_mlp.router.weight.grad.abs().sum() > 0