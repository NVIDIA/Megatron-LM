# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import copy
import dataclasses

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import get_capacity
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


class MoEModelTestContainer:
    def __init__(
        self,
        tp_size,
        ep_size,
        pp_size,
        cp_size=1,
        moe_tp_size=None,
        data_parallel_random_init=False,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_router_load_balancing_type="aux_loss",
        moe_token_dispatcher_type="alltoall",
        moe_expert_capacity_factor=None,
        moe_pad_expert_input_to_capacity=False,
        moe_aux_loss_coeff=0.1,
        **kwargs,
    ):
        self.num_local_experts = num_moe_experts // ep_size
        if moe_tp_size is None:
            moe_tp_size = tp_size
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            expert_model_parallel_size=ep_size,
            context_parallel_size=cp_size,
            expert_tensor_parallel_size=moe_tp_size,
        )
        _set_random_seed(seed_=123, data_parallel_random_init=data_parallel_random_init)
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )
        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        self.config = TransformerConfig(
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            pipeline_model_parallel_size=pp_size,
            context_parallel_size=cp_size,
            expert_tensor_parallel_size=moe_tp_size,
            moe_router_topk=moe_router_topk,
            num_moe_experts=num_moe_experts,
            moe_router_load_balancing_type=moe_router_load_balancing_type,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_expert_capacity_factor=moe_expert_capacity_factor,
            moe_pad_expert_input_to_capacity=moe_pad_expert_input_to_capacity,
            moe_aux_loss_coeff=moe_aux_loss_coeff,
            num_layers=1,
            moe_grouped_gemm=kwargs.get("moe_grouped_gemm", False),
            hidden_size=kwargs.get("hidden_size", 16),
            num_attention_heads=kwargs.get("num_attention_heads", 8),
            use_cpu_initialization=kwargs.get("use_cpu_initialization", True),
            sequence_parallel=tp_size > 1,
            add_bias_linear=kwargs.get("add_bias_linear", False),
            moe_permute_fusion=kwargs.get("moe_permute_fusion", False),
            moe_enable_deepep=kwargs.get("moe_enable_deepep", False),
        )

        # init moe layer
        self.moe_layer = self.new_moe_layer()

    def new_moe_layer(self, **kargs):
        transformer_layer_spec = get_gpt_layer_local_spec(
            num_experts=self.config.num_moe_experts, moe_grouped_gemm=self.config.moe_grouped_gemm
        )
        new_config = dataclasses.replace(self.config, **kargs)
        moe_layer = MoELayer(new_config, transformer_layer_spec.submodules.mlp.submodules).cuda()
        moe_layer.set_layer_number(0)
        return moe_layer

    def __del__(self):
        torch.distributed.barrier()
        torch.cuda.synchronize()
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def dispatcher_dropless_test(self):
        moe_layer = self.moe_layer
        bs = 32
        seql = 8
        # TODO: Find why setting manual seed can cause the test to fail
        # Manual seed to differentiate input data for each rank
        # rank = torch.distributed.get_rank()
        # torch.manual_seed(1000 + rank)
        hidden_states = torch.randn((bs, seql, moe_layer.config.hidden_size))
        hidden_states = hidden_states.cuda()
        # Permute and then unpermute data are supposed to restore original data
        ans = hidden_states
        hidden_states.requires_grad = True
        probs, indices = moe_layer.router(hidden_states)
        probs = torch.ones_like(probs) / moe_layer.router.topk

        (permuted_local_hidden_states, tokens_per_expert) = (
            moe_layer.token_dispatcher.token_permutation(hidden_states, probs, indices)
        )

        restored_hidden_states, restored_bias = moe_layer.token_dispatcher.token_unpermutation(
            permuted_local_hidden_states
        )

        # reduce across TP rank equals to multiply data by a scale of ETP
        scale = moe_layer.config.expert_tensor_parallel_size
        restored_hidden_states = restored_hidden_states / scale

        assert torch.allclose(
            restored_hidden_states, ans
        ), "Restored hidden states do not match original hidden states"

        # check if the grad of the hidden states is same as the hidden states
        torch.autograd.backward(restored_hidden_states, hidden_states)
        assert torch.allclose(
            hidden_states.grad, ans
        ), "Restored hidden states do not match original hidden states"

    @pytest.mark.internal
    def dispatcher_capacity_test(self):
        moe_layer = self.moe_layer
        num_tokens = 16
        hidden_states = torch.randn((num_tokens, moe_layer.config.hidden_size))
        hidden_states = hidden_states.cuda()
        hidden_states.requires_grad = True
        probs, indices = moe_layer.router(hidden_states)

        # Create the answer.
        prob_mask = probs != 0
        probs = torch.ones_like(probs) * prob_mask / moe_layer.router.topk
        local_probss = probs
        restored_hidden_states_answer = hidden_states * local_probss.sum(dim=1).unsqueeze(1)

        (permuted_local_hidden_states, tokens_per_expert) = (
            moe_layer.token_dispatcher.token_permutation(hidden_states, probs, indices)
        )

        # Check tokens per expert not exceed the capacity.
        capacity = get_capacity(
            num_tokens * self.config.moe_router_topk,
            self.config.num_moe_experts,
            self.config.moe_expert_capacity_factor,
        )
        assert torch.all(
            tokens_per_expert
            <= capacity
            * self.config.expert_model_parallel_size
            * self.config.tensor_model_parallel_size
        ), "Tokens per expert exceed the capacity"

        permuted_local_hidden_states /= moe_layer.config.tensor_model_parallel_size

        restored_hidden_states, restored_bias = moe_layer.token_dispatcher.token_unpermutation(
            permuted_local_hidden_states
        )
        assert torch.allclose(
            restored_hidden_states, restored_hidden_states_answer
        ), "Restored hidden states does not match"

        # check if the grad of the hidden states is same as the hidden states
        torch.autograd.backward(restored_hidden_states, hidden_states)
        assert torch.allclose(
            hidden_states.grad, restored_hidden_states_answer
        ), "Gradient of hidden states should be same as hidden states"

    @pytest.mark.internal
    def dispatcher_drop_and_pad_test(self):
        """Test if the tokens are dropped and padded correctly.

        Since the probs of padded tokens are 0, the combined results for
        dispatching with or without padding should be the same.
        """
        moe_layer = self.new_moe_layer(moe_pad_expert_input_to_capacity=False)

        num_tokens = 16
        hidden_states = torch.randn((num_tokens, moe_layer.config.hidden_size)).cuda()
        hidden_states.requires_grad = True

        probs_1, indices_1 = moe_layer.router(hidden_states)
        (permuted_input_1, tokens_per_expert) = moe_layer.token_dispatcher.token_permutation(
            hidden_states, probs_1, indices_1
        )
        forward_answer, restored_bias = moe_layer.token_dispatcher.token_unpermutation(
            permuted_input_1
        )
        torch.autograd.backward(forward_answer, forward_answer)
        backward_answer = hidden_states.grad.clone()
        hidden_states.grad = None
        torch.cuda.synchronize()
        # End

        moe_layer_2 = self.new_moe_layer(moe_pad_expert_input_to_capacity=True)
        moe_layer_2.load_state_dict(moe_layer.state_dict())

        probs_2, indices_2 = moe_layer_2.router(hidden_states)
        (permuted_input_2, tokens_per_expert) = moe_layer_2.token_dispatcher.token_permutation(
            hidden_states, probs_2, indices_2
        )
        restored_hidden_states, restored_bias = moe_layer_2.token_dispatcher.token_unpermutation(
            permuted_input_2
        )

        # # Check tokens per expert equals to the capacity.
        capacity = get_capacity(
            num_tokens * self.config.moe_router_topk,
            self.config.num_moe_experts,
            self.config.moe_expert_capacity_factor,
        )
        assert torch.all(
            tokens_per_expert
            == capacity
            * self.config.expert_model_parallel_size
            * self.config.tensor_model_parallel_size
        ), "Tokens per expert should be the same as the capacity"
        assert torch.allclose(
            restored_hidden_states, forward_answer
        ), "Restored hidden states does not match"

        # check if the grad of the hidden states is same as the hidden states
        torch.autograd.backward(restored_hidden_states, restored_hidden_states)
        assert torch.allclose(
            hidden_states.grad, backward_answer
        ), "Gradient of hidden states should be same as hidden states"

    def set_params(self):
        # TODO: Set consistent parameters for various parallelisms.
        raise NotImplementedError

    def destroy(self):
        Utils.destroy_model_parallel()


permute_fusion_params = [False]
if is_te_min_version("1.14.0"):
    permute_fusion_params.append(True)


class TestAllgatherDispatcher:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.parametrize("tp_size,ep_size", [(8, 1), (1, 8), (2, 4), (1, 1)])
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    def test_forward_backward(self, tp_size, ep_size, permute_fusion):
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="allgather",
            moe_permute_fusion=permute_fusion,
        )

        container.dispatcher_dropless_test()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    @pytest.mark.parametrize(
        "tp_size,ep_size,moe_tp_size", [(1, 1, 8), (1, 2, 4), (1, 4, 2), (2, 2, 4)]
    )
    def test_moe_tp_forward_backward(self, tp_size, ep_size, moe_tp_size, permute_fusion):
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            moe_tp_size=moe_tp_size,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="allgather",
            sequence_parallel=True,
            moe_grouped_gemm=True,
            moe_permute_fusion=permute_fusion,
            use_cpu_initialization=False,
        )

        container.dispatcher_dropless_test()


def is_deep_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_DEEP_EP

    return HAVE_DEEP_EP


@pytest.mark.skipif(not is_deep_ep_available(), reason="Deep EP is not available")
class TestFlexDispatcher:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.parametrize("tp_size,ep_size", [(8, 1), (1, 8), (2, 4)])
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    def test_forward_backward(self, tp_size, ep_size, permute_fusion):
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_permute_fusion=permute_fusion,
            hidden_size=4,
            moe_enable_deepep=True,
        )
        container.dispatcher_dropless_test()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2)])
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    def test_capacity_forward_backward(self, tp_size, ep_size, permute_fusion):
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
            hidden_size=4,
            moe_enable_deepep=True,
        )
        container.dispatcher_capacity_test()
