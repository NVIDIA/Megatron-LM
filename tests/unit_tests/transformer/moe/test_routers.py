# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import get_updated_expert_bias, router_gating_linear
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
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
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
            hidden_states = hidden_states.cuda().bfloat16()
            scores, indices = self.router(hidden_states)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_aux_loss(self):
        self.sequential_mlp = self.sequential_mlp.cuda()

        # Without aux loss
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size))
        hidden_states = hidden_states.cuda().bfloat16()
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

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_router_dtype(self):
        self.router = self.router.cuda()
        self.sequential_mlp = self.sequential_mlp.cuda()
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size), dtype=torch.bfloat16)
        hidden_states = hidden_states.cuda()

        # Test with default setting (bf16)
        self.router.config.moe_router_dtype = None
        with torch.no_grad():
            scores, routing_map = self.router(hidden_states)
            out = self.sequential_mlp(hidden_states)
            assert scores.dtype == torch.bfloat16, "Router output should be bf16 by default"
            assert out[0].dtype == torch.bfloat16

        # Test with fp32 enabled
        self.router.config.moe_router_dtype = 'fp32'
        with torch.no_grad():
            scores, routing_map = self.router(hidden_states)
            out = self.sequential_mlp(hidden_states)
            assert scores.dtype == torch.float32, "Router output should be fp32 when enabled"
            assert out[0].dtype == torch.bfloat16
            self.sequential_mlp.config.moe_token_dispatcher_type = "alltoall"
            out = self.sequential_mlp(hidden_states)
            assert out[0].dtype == torch.bfloat16
            self.sequential_mlp.config.moe_token_dispatcher_type = "allgather"

        # Test with fp64 enabled
        self.router.config.moe_router_dtype = 'fp64'
        with torch.no_grad():
            scores, routing_map = self.router(hidden_states)
            out = self.sequential_mlp(hidden_states)
            assert scores.dtype == torch.float64, "Router output should be fp64 when enabled"
            assert out[0].dtype == torch.bfloat16

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_force_load_balancing(self):
        hidden_states = torch.randn(
            (32, 2, self.router.config.hidden_size), device="cuda", dtype=torch.bfloat16
        )
        hidden_states.requires_grad = True

        # First forward pass with normal routing
        normal_scores, normal_routing_map = self.router(hidden_states)

        # Second forward pass with force load balancing
        self.router.config.moe_router_force_load_balancing = True
        force_scores, force_routing_map = self.router(hidden_states)

        assert normal_scores.shape == force_scores.shape
        assert normal_routing_map.shape == force_routing_map.shape
        assert torch.equal(normal_scores, force_scores) == False

        # Backward pass for force load balancing
        self.router.zero_grad()
        force_scores.sum().backward()
        assert hidden_states.grad is not None
        assert self.router.weight.grad.norm() > 0

        self.router.config.moe_router_force_load_balancing = False


class TestGroupLimitedRouter:
    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=8,
            context_parallel_size=1,
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        print("done intializing")

        num_moe_experts = 16
        self.transformer_config = TransformerConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=8,
            context_parallel_size=1,
            num_moe_experts=num_moe_experts,
            moe_router_topk=4,
            moe_router_group_topk=2,
            moe_router_num_groups=8,
            moe_router_pre_softmax=True,
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=0,
            moe_token_dispatcher_type="alltoall",
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
        )

        # init MoE layer
        transformer_layer_spec = get_gpt_layer_local_spec(
            num_experts=num_moe_experts, moe_grouped_gemm=False
        )
        self.moe_layer = MoELayer(
            self.transformer_config, transformer_layer_spec.submodules.mlp.submodules
        ).cuda()
        self.router = self.moe_layer.router

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_constructor(self):
        assert isinstance(self.router, Router)

        num_weights = sum([p.numel() for p in self.router.parameters()])
        assert (
            num_weights
            == self.transformer_config.hidden_size * self.transformer_config.num_moe_experts
        ), num_weights

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("moe_router_group_topk,moe_router_num_groups", [(3, 8), (2, 4)])
    @pytest.mark.parametrize("moe_router_pre_softmax", [(True), (False)])
    @pytest.mark.parametrize("score_function", ["sigmoid", "softmax"])
    def test_router_forward(
        self, moe_router_group_topk, moe_router_num_groups, moe_router_pre_softmax, score_function
    ):
        with torch.no_grad():
            self.router.config.moe_router_group_topk = moe_router_group_topk
            self.router.config.moe_router_num_groups = moe_router_num_groups
            self.router.config.moe_router_pre_softmax = moe_router_pre_softmax
            self.router.config.moe_router_score_function = score_function
            if moe_router_pre_softmax:
                self.router.config.moe_router_topk_scaling_factor = 16.0

            seq_len = 2
            batch_size = 2
            num_tokens = seq_len * batch_size
            # hidden_states shape: [seq_len, batch_size, hidden_size]
            hidden_states = (
                torch.randn((seq_len, batch_size, self.router.config.hidden_size)).cuda().bfloat16()
            )
            scores, routing_map = self.router(hidden_states)
            assert scores.shape == (num_tokens, self.router.config.num_moe_experts), scores.shape
            assert routing_map.shape == (
                num_tokens,
                self.router.config.num_moe_experts,
            ), routing_map.shape

            group_routing_map = (
                routing_map.reshape(num_tokens, moe_router_num_groups, -1).max(dim=-1).values
            )
            assert torch.all(group_routing_map.sum(dim=-1) <= moe_router_group_topk)


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
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
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
        hidden_states = hidden_states.cuda().bfloat16()
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


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("router_dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_router_gating_linear(router_dtype):
    tols = dict(rtol=2.0e-2, atol=1.0e-3)

    ref_inp = torch.randn((4096, 7168), dtype=torch.bfloat16, device="cuda")
    ref_weight = torch.randn((256, 7168), dtype=torch.bfloat16, device="cuda")
    ref_inp.requires_grad = True
    ref_weight.requires_grad = True
    bwd_input = torch.randn((4096, 256), dtype=router_dtype, device="cuda")

    ref_output = torch.nn.functional.linear(ref_inp.to(router_dtype), ref_weight.to(router_dtype))
    ref_output.backward(bwd_input)

    inp = ref_inp.detach()
    weight = ref_weight.detach()
    inp.requires_grad = True
    weight.requires_grad = True
    output = router_gating_linear(inp, weight, router_dtype)
    output.backward(bwd_input)

    assert output.dtype == router_dtype
    assert ref_inp.grad.dtype == ref_inp.dtype
    assert ref_weight.grad.dtype == ref_weight.dtype
    assert torch.allclose(output, ref_output, **tols)
    assert torch.allclose(inp.grad, ref_inp.grad, **tols)
    assert torch.allclose(weight.grad, ref_weight.grad, **tols)
