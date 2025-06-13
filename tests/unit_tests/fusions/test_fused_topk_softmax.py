# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import copy
import random

import pytest
import torch
import torch.nn.functional as F

from megatron.core import config
from megatron.core.fusions.fused_topk_routing import fused_topk_softmax_without_capacity
from megatron.core.transformer.moe.moe_utils import topk_softmax_with_capacity


class TestFusedTopkGating:

    def setup_method(self, method):
        # enable experimental feature
        if config.ENABLE_EXPERIMENTAL is False:
            config.ENABLE_EXPERIMENTAL = True

    def teardown_method(self, method):
        # disable experimental feature
        if config.ENABLE_EXPERIMENTAL is True:
            config.ENABLE_EXPERIMENTAL = False

    @pytest.mark.experimental
    @pytest.mark.parametrize("num_tokens", [2048, 4096])
    @pytest.mark.parametrize("num_experts", [32, 64])
    @pytest.mark.parametrize("topk", [6, 8])
    @pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
    @pytest.mark.parametrize("group_config", [
        (None, None),  
        (1, 1), 
        (2, 1),       
        (2, 2),       
        (4, 1),       
        (4, 2),       
        (4, 3),       
        (4, 4),       
    ])
    def test_fused_topk_gating_forward(self, num_tokens, num_experts, topk, score_function, group_config):
        num_groups, group_topk = group_config
        
        # Skip invalid combinations
        if topk > num_experts:
            pytest.skip(f"topk ({topk}) cannot be greater than num_experts ({num_experts})")
        # Check if group configuration is valid
        if num_groups is not None:
            # 1. Number of experts must be divisible by number of groups
            if num_experts % num_groups != 0:
                pytest.skip(f"num_experts ({num_experts}) must be divisible by num_groups ({num_groups})")
            
            experts_per_group = num_experts // num_groups
            
            # 2. Number of experts selected per group cannot exceed experts in group
            if group_topk > experts_per_group:
                pytest.skip(f"group_topk ({group_topk}) cannot be greater than experts_per_group ({experts_per_group})")
            
            # 3. Selected groups must be large enough to ensure enough experts are selected
            if group_topk * experts_per_group < topk:
                pytest.skip(f"group_topk ({group_topk}) * experts_per_group ({experts_per_group}) must be >= topk ({topk})")
        
        # Create input data
        logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device='cuda')
        logits.requires_grad = True
        
        expert_bias = None
        if score_function == "sigmoid" and random.choice([True, False]):
            expert_bias = torch.randn(num_experts, dtype=torch.float32, device='cuda')
        
        scaling_factor = random.uniform(0.5, 2.0) if random.choice([True, False]) else None
        
        logits_pytorch = copy.deepcopy(logits)
        expert_bias_pytorch = copy.deepcopy(expert_bias) if expert_bias is not None else None
        
        # Test fused version
        topk_masked_gates, topk_map, tokens_per_expert = fused_topk_softmax_without_capacity(
            logits=logits,
            topk=topk,
            capacity_factor=None,
            pad_to_capacity=False,
            drop_policy=None,
            use_pre_softmax=True,
            num_groups=num_groups,
            group_topk=group_topk,
            scaling_factor=scaling_factor,
            deterministic_mode=False,
            score_function=score_function,
            expert_bias=expert_bias,
        )
        
        # Test PyTorch version
        topk_masked_gates_pytorch, topk_map_pytorch, tokens_per_expert_pytorch = topk_softmax_with_capacity(
            logits=logits_pytorch,
            topk=topk,
            capacity_factor=None,
            pad_to_capacity=False,
            drop_policy=None,
            use_pre_softmax=True,
            num_groups=num_groups,
            group_topk=group_topk,
            scaling_factor=scaling_factor,
            deterministic_mode=False,
            score_function=score_function,
            expert_bias=expert_bias_pytorch,
        )
        
        # Verify forward pass results
        assert torch.allclose(topk_masked_gates, topk_masked_gates_pytorch, rtol=1e-6, atol=1e-6)
        assert torch.equal(topk_map, topk_map_pytorch)
        assert torch.equal(tokens_per_expert, tokens_per_expert_pytorch)

    @pytest.mark.experimental
    @pytest.mark.parametrize("num_tokens", [2048, 4096])
    @pytest.mark.parametrize("num_experts", [32, 64])
    @pytest.mark.parametrize("topk", [6, 8])
    @pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
    @pytest.mark.parametrize("group_config", [
        (None, None),  
        (1, 1),       
        (2, 1),       
        (2, 2),       
        (4, 1),       
        (4, 2),       
        (4, 3),       
        (4, 4),       
    ])
    def test_fused_topk_gating_backward(self, num_tokens, num_experts, topk, score_function, group_config):
        num_groups, group_topk = group_config
        
        # Skip invalid combinations
        if topk > num_experts:
            pytest.skip(f"topk ({topk}) cannot be greater than num_experts ({num_experts})")
        
        # Check if group configuration is valid
        if num_groups is not None:
            # 1. Number of experts must be divisible by number of groups
            if num_experts % num_groups != 0:
                pytest.skip(f"num_experts ({num_experts}) must be divisible by num_groups ({num_groups})")
            
            experts_per_group = num_experts // num_groups
            
            # 2. Number of experts selected per group cannot exceed experts in group
            if group_topk > experts_per_group:
                pytest.skip(f"group_topk ({group_topk}) cannot be greater than experts_per_group ({experts_per_group})")
            
            # 3. Selected groups must be large enough to ensure enough experts are selected
            if group_topk * experts_per_group < topk:
                pytest.skip(f"group_topk ({group_topk}) * experts_per_group ({experts_per_group}) must be >= topk ({topk})")
        
        # Create input data
        logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device='cuda')
        logits.requires_grad = True
        
        expert_bias = None
        if score_function == "sigmoid" and random.choice([True, False]):
            expert_bias = torch.randn(num_experts, dtype=torch.float32, device='cuda')
            expert_bias.requires_grad = True
        
        scaling_factor = random.uniform(0.5, 2.0) if random.choice([True, False]) else None
        
        logits_pytorch = copy.deepcopy(logits)
        expert_bias_pytorch = copy.deepcopy(expert_bias) if expert_bias is not None else None
        
        # Test fused version
        topk_masked_gates, topk_map, tokens_per_expert = fused_topk_softmax_without_capacity(
            logits=logits,
            topk=topk,
            capacity_factor=None,
            pad_to_capacity=False,
            drop_policy=None,
            use_pre_softmax=True,
            num_groups=num_groups,
            group_topk=group_topk,
            scaling_factor=scaling_factor,
            deterministic_mode=False,
            score_function=score_function,
            expert_bias=expert_bias,
        )
        
        # Test PyTorch version
        topk_masked_gates_pytorch, topk_map_pytorch, tokens_per_expert_pytorch = topk_softmax_with_capacity(
            logits=logits_pytorch,
            topk=topk,
            capacity_factor=None,
            pad_to_capacity=False,
            drop_policy=None,
            use_pre_softmax=True,
            num_groups=num_groups,
            group_topk=group_topk,
            scaling_factor=scaling_factor,
            deterministic_mode=False,
            score_function=score_function,
            expert_bias=expert_bias_pytorch,
        )
        
        # Use PyTorch version output as gradient input
        grad_output = topk_masked_gates_pytorch.detach()
        
        # Calculate gradients
        loss = (topk_masked_gates * grad_output).sum()
        loss.backward()
        
        loss_pytorch = (topk_masked_gates_pytorch * grad_output).sum()
        loss_pytorch.backward()
        
        # Verify backward pass results
        assert torch.allclose(logits.grad, logits_pytorch.grad, rtol=1e-6, atol=1e-6)
        
        # Note: Do not verify expert_bias gradients as they are not updated in actual computation

    @pytest.mark.experimental
    @pytest.mark.parametrize("input_dtype", [torch.float32, torch.float64])
    def test_fused_topk_gating_dtypes(self, input_dtype):
        if input_dtype == torch.float32:
            tols = dict(rtol=1.0e-6, atol=1.0e-6)
        elif input_dtype == torch.float64:
            tols = dict(rtol=1.0e-9, atol=1.0e-7)
        else:
            raise ValueError(f"Invalid input dtype: {input_dtype}")
        
        num_tokens, num_experts, topk = 4096, 64, 8
        
        logits = torch.randn(num_tokens, num_experts, dtype=input_dtype, device='cuda')
        logits.requires_grad = True
        
        logits_pytorch = copy.deepcopy(logits)
        
        # Test fused version
        topk_masked_gates, topk_map, tokens_per_expert = fused_topk_softmax_without_capacity(
            logits=logits,
            topk=topk,
            capacity_factor=None,
            pad_to_capacity=False,
            drop_policy=None,
            use_pre_softmax=True,
            num_groups=None,
            group_topk=None,
            scaling_factor=None,
            deterministic_mode=False,
            score_function="softmax",
            expert_bias=None,
        )
        
        # Test PyTorch version
        topk_masked_gates_pytorch, topk_map_pytorch, tokens_per_expert_pytorch = topk_softmax_with_capacity(
            logits=logits_pytorch,
            topk=topk,
            capacity_factor=None,
            pad_to_capacity=False,
            drop_policy=None,
            use_pre_softmax=True,
            num_groups=None,
            group_topk=None,
            scaling_factor=None,
            deterministic_mode=False,
            score_function="softmax",
            expert_bias=None,
        )
        
        # Verify results
        assert topk_masked_gates.dtype == input_dtype
        assert torch.allclose(topk_masked_gates, topk_masked_gates_pytorch, **tols)
        assert torch.equal(topk_map, topk_map_pytorch)

    @pytest.mark.experimental
    def test_fused_topk_gating_edge_cases(self):
        # Test edge cases
        num_tokens, num_experts = 1, 1
        topk = 1
        
        logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device='cuda')
        
        # Test case with topk=1
        topk_masked_gates, topk_map, tokens_per_expert = fused_topk_softmax_without_capacity(
            logits=logits,
            topk=topk,
            capacity_factor=None,
            pad_to_capacity=False,
            drop_policy=None,
            use_pre_softmax=True,
            num_groups=None,
            group_topk=None,
            scaling_factor=None,
            deterministic_mode=False,
            score_function="softmax",
            expert_bias=None,
        )
        
        # Verify only one expert is selected
        assert topk_map.sum().item() == num_tokens
        assert tokens_per_expert.sum().item() == num_tokens
        
        # Verify probability sum is 1 (for each token)
        gates_sum = topk_masked_gates.sum(dim=1)
        assert torch.allclose(gates_sum, torch.ones(num_tokens, device='cuda'), rtol=1e-6)

    @pytest.mark.experimental
    def test_fused_topk_gating_with_scaling(self):
        # Test scaling factor
        num_tokens, num_experts, topk = 8, 6, 3
        scaling_factor = 2.0
        
        logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device='cuda')
        
        # Without scaling factor
        topk_masked_gates_no_scale, _, _ = fused_topk_softmax_without_capacity(
            logits=logits.clone(),
            topk=topk,
            capacity_factor=None,
            pad_to_capacity=False,
            drop_policy=None,
            use_pre_softmax=True,
            num_groups=None,
            group_topk=None,
            scaling_factor=None,
            deterministic_mode=False,
            score_function="softmax",
            expert_bias=None,
        )
        
        # With scaling factor
        topk_masked_gates_with_scale, _, _ = fused_topk_softmax_without_capacity(
            logits=logits.clone(),
            topk=topk,
            capacity_factor=None,
            pad_to_capacity=False,
            drop_policy=None,
            use_pre_softmax=True,
            num_groups=None,
            group_topk=None,
            scaling_factor=scaling_factor,
            deterministic_mode=False,
            score_function="softmax",
            expert_bias=None,
        )
        
        # Verify scaling factor effect
        expected_scaled = topk_masked_gates_no_scale * scaling_factor
        assert torch.allclose(topk_masked_gates_with_scale, expected_scaled, rtol=1e-6)