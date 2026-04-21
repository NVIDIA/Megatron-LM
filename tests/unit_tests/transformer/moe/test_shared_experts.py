# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import dataclasses

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.parallel_state import get_tensor_model_parallel_world_size
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def is_deep_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_DEEP_EP

    return HAVE_DEEP_EP


if is_deep_ep_available():
    TOKEN_DISPATCHER_TYPES = ["alltoall", "flex"]
else:
    TOKEN_DISPATCHER_TYPES = ["alltoall"]


class TestSharedExperts:
    def setup_method(self, method):
        self.config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=16,
            moe_shared_expert_intermediate_size=32,
            moe_shared_expert_overlap=False,
            moe_token_dispatcher_type="alltoall",
            use_cpu_initialization=True,
            activation_func=torch.nn.functional.silu,
            gated_linear_unit=True,
            bias_activation_fusion=True,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=4,
            add_bias_linear=False,
        )

    def get_moe_layer(self, **kargs) -> MoELayer:
        transformer_layer_spec = get_gpt_layer_local_spec(
            num_experts=self.config.num_moe_experts, moe_grouped_gemm=False
        )
        new_config = dataclasses.replace(self.config, **kargs)
        if get_tensor_model_parallel_world_size() > 1:
            new_config.sequence_parallel = True
        if new_config.moe_token_dispatcher_type == "flex":
            new_config.moe_enable_deepep = True
        moe_layer = MoELayer(new_config, transformer_layer_spec.submodules.mlp.submodules)
        moe_layer.cuda()
        return moe_layer

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("dispatcher_type", TOKEN_DISPATCHER_TYPES)
    @pytest.mark.parametrize("tp_size, ep_size", [[1, 1], [4, 1], [1, 4], [2, 4]])
    def test_shared_expert_forward_backward(self, dispatcher_type: str, tp_size, ep_size):
        """
        Tests that the MoELayer with and without shared expert overlap produce
        identical outputs and gradients.
        """
        if tp_size == 1 and ep_size == 1 and dispatcher_type == "flex":
            pytest.skip("Flex dispatcher is not supported for tp=1, ep=1")
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        # Create MoE layer with shared expert overlap enabled.
        model_parallel_cuda_manual_seed(123)
        moe_layer_overlap = self.get_moe_layer(
            moe_shared_expert_overlap=True, moe_token_dispatcher_type=dispatcher_type
        )

        # Create MoE layer with shared expert overlap disabled.
        model_parallel_cuda_manual_seed(123)
        moe_layer_no_overlap = self.get_moe_layer(
            moe_shared_expert_overlap=False, moe_token_dispatcher_type=dispatcher_type
        )
        moe_layer_no_overlap.load_state_dict(moe_layer_overlap.state_dict())

        # Sanity check that the weights are identical.
        for p_overlap, p_no_overlap in zip(
            moe_layer_overlap.parameters(), moe_layer_no_overlap.parameters()
        ):
            assert torch.equal(p_overlap, p_no_overlap)

        # Verify attributes of the MoE layers.
        num_weights_overlap = sum([p.numel() for p in moe_layer_overlap.parameters()])
        num_weights_no_overlap = sum([p.numel() for p in moe_layer_no_overlap.parameters()])
        assert num_weights_overlap == num_weights_no_overlap

        assert moe_layer_overlap.shared_experts is not None
        assert moe_layer_overlap.shared_experts.stream is not None
        assert moe_layer_overlap.token_dispatcher.shared_experts is not None

        assert moe_layer_no_overlap.shared_experts is not None
        assert moe_layer_no_overlap.token_dispatcher.shared_experts is None

        # Create a dummy input tensor.
        hidden_states = torch.ones(
            (32, 2, self.config.hidden_size), requires_grad=True, device="cuda"
        )
        hidden_states_no_overlap = hidden_states.clone().detach().requires_grad_(True)

        # Forward pass.
        output_overlap, _ = moe_layer_overlap(hidden_states)
        output_no_overlap, _ = moe_layer_no_overlap(hidden_states_no_overlap)
        assert torch.allclose(
            output_overlap, output_no_overlap
        ), f"max diff: {torch.max(torch.abs(output_overlap - output_no_overlap))}"

        # Backward pass.
        output_overlap.mean().backward()
        output_no_overlap.mean().backward()

        # Check gradients.
        for p_overlap, p_no_overlap in zip(
            moe_layer_overlap.parameters(), moe_layer_no_overlap.parameters()
        ):
            assert torch.allclose(
                p_overlap.grad, p_no_overlap.grad
            ), f"max diff: {torch.max(torch.abs(p_overlap.grad - p_no_overlap.grad))}"
