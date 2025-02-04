# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import get_updated_expert_bias
from megatron.core.transformer.moe.router import Router
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


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
        transformer_layer_spec = get_gpt_layer_local_spec(
            num_experts=num_moe_experts, moe_grouped_gemm=False
        )
        self.sequential_mlp = MoELayer(
            self.transformer_config, transformer_layer_spec.submodules.mlp.submodules
        )
        self.router = self.sequential_mlp.router

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_constructor(self):
        assert isinstance(self.router, Router)

        num_weights = sum([p.numel() for p in self.router.parameters()])
        assert num_weights == 12 * 4, num_weights

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("moe_router_pre_softmax", [(True), (False)])
    @pytest.mark.parametrize("score_function", ["sigmoid", "softmax"])
    def test_router_forward(self, moe_router_pre_softmax, score_function):
        with torch.no_grad():
            self.router = self.router.cuda()
            self.router.config.moe_router_pre_softmax = moe_router_pre_softmax
            self.router.config.moe_router_score_function = score_function
            # [num tokens, hidden size]
            hidden_states = torch.randn((32, 2, self.router.config.hidden_size))
            hidden_states = hidden_states.cuda()
            scores, indices = self.router(hidden_states)

    @pytest.mark.internal
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


class TestDeviceLimitedTop2Router:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=8)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        print("done intializing")
        num_moe_experts = 8
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            expert_model_parallel_size=8,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk_limited_devices=2,
            moe_router_pre_softmax=True,
            moe_router_topk=2,
            moe_aux_loss_coeff=0,
        )
        transformer_layer_spec = get_gpt_layer_local_spec(
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
        assert num_weights == 12 * 8, num_weights

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("moe_router_pre_softmax", [(True), (False)])
    @pytest.mark.parametrize("score_function", ["sigmoid", "softmax"])
    def test_router_forward(self, moe_router_pre_softmax, score_function):
        with torch.no_grad():
            self.router = self.router.cuda()
            self.router.config.moe_router_pre_softmax = moe_router_pre_softmax
            self.router.config.moe_router_score_function = score_function
            if moe_router_pre_softmax:
                self.router.config.moe_router_topk_scaling_factor = 16.0
            # [num tokens, hidden size]
            hidden_states = torch.randn((32, 2, self.router.config.hidden_size))
            hidden_states = hidden_states.cuda()
            scores, indices = self.router(hidden_states)
            print(scores.shape, indices.shape)
            assert scores.shape == (64, 8)
            assert indices.shape == (64, 8)
            print(
                (indices == 0).sum(),
                (indices == 1).sum(),
                (indices == 2).sum(),
                (indices == 3).sum(),
            )


class TestAuxLossFreeTop2Router:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=8)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        print("done intializing")
        num_moe_experts = 8
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            expert_model_parallel_size=8,
            moe_router_load_balancing_type="none",  # No aux loss
            moe_router_score_function="sigmoid",  # Using sigmoid scoring
            moe_router_enable_expert_bias=True,  # Enable expert bias
            moe_router_bias_update_rate=0.1,  # Set bias update rate
            moe_router_topk=2,
        )
        transformer_layer_spec = get_gpt_layer_local_spec(
            num_experts=num_moe_experts, moe_grouped_gemm=False
        )
        self.moe_layer = MoELayer(
            self.transformer_config, transformer_layer_spec.submodules.mlp.submodules
        )
        self.router = self.moe_layer.router
        assert self.router.expert_bias is not None
        assert self.router.local_tokens_per_expert is not None

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_router_forward_aux_free(self):
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size))
        hidden_states = hidden_states.cuda()
        self.router = self.router.cuda()

        # First forward pass
        initial_bias = self.router.expert_bias.clone()
        scores1, indices1 = self.router(hidden_states)
        initial_tokens = self.router.local_tokens_per_expert.clone()
        updated_bias = get_updated_expert_bias(
            self.router.local_tokens_per_expert,
            self.router.expert_bias,
            self.router.config.moe_router_bias_update_rate,
        )

        # Verify expert bias was updated
        assert not torch.equal(initial_bias, updated_bias), "Expert bias should be updated"

        # Basic output checks
        assert scores1.shape == (64, 8), "Router scores shape mismatch"
        assert indices1.shape == (64, 8), "Router indices shape mismatch"

        # Print some debug info
        print("Updated bias after first forward pass:", updated_bias)
