# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
from megatron.core import parallel_state

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import permute, unpermute
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


class MoEModelTestContainer:
    def __init__(
        self,
        tp_size,
        ep_size,
        pp_size,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_router_load_balancing_type="aux_loss",
        moe_token_dispatcher_type="alltoall",
        **kwargs,
    ):
        self.num_local_experts = num_moe_experts // ep_size
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            expert_model_parallel_size=ep_size,
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)
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
            moe_router_topk=moe_router_topk,
            num_moe_experts=num_moe_experts,
            moe_router_load_balancing_type=moe_router_load_balancing_type,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            num_layers=1,
            hidden_size=kwargs.get("hidden_size", 1024),
            num_attention_heads=kwargs.get("num_attention_heads", 8),
            use_cpu_initialization=kwargs.get("use_cpu_initialization", True),
            sequence_parallel=kwargs.get("sequence_parallel", False),
            add_bias_linear=kwargs.get("add_bias_linear", False),
        )

        # init moe layer
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_moe_experts, moe_grouped_gemm=False
        )
        self.moe_layer = MoELayer(
            self.config, transformer_layer_spec.submodules.mlp.submodules
        ).cuda()

    def set_params(self):
        # TODO: Set consistent parameters for various parallelisms.
        raise NotImplementedError

    def destroy(self):
        Utils.destroy_model_parallel()


class TestAllgatherDispatcher:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_tp_forward(self):
        container = MoEModelTestContainer(
            tp_size=8,
            ep_size=1,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="allgather",
            sequence_parallel=True,
        )
        moe_layer = container.moe_layer
        # [bs, seql, hidden size]
        hidden_states = torch.randn((32, 8, moe_layer.router.config.hidden_size))
        hidden_states = hidden_states.cuda()
        hidden_states.requires_grad = True
        scores, indices = moe_layer.router(hidden_states)
        assert scores.shape == (256, moe_layer.router.topk), "Scores shape is not correct"
        assert indices.shape == (256, moe_layer.router.topk), "Indices shape is not correct"
        scores = torch.ones_like(scores) / 2
        (
            permuted_local_hidden_states,
            tokens_per_expert,
        ) = moe_layer.token_dispatcher.token_permutation(hidden_states, scores, indices)
        permuted_local_hidden_states /= moe_layer.config.tensor_model_parallel_size
        restored_hidden_states, restored_bias = moe_layer.token_dispatcher.token_unpermutation(
            permuted_local_hidden_states, bias=torch.zeros_like(permuted_local_hidden_states),
        )

        assert torch.allclose(
            restored_hidden_states, hidden_states
        ), "Restored hidden states do not match original hidden states"

        # check if the grad of the hidden states is same as the hidden states
        torch.autograd.backward(restored_hidden_states, restored_hidden_states)
        assert torch.allclose(
            hidden_states.grad, hidden_states
        ), "Gradient of hidden states should be same as hidden states"
        container.destroy()


class TestAlltoAllDispatcher:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_ep_forward_backward(self):
        container = MoEModelTestContainer(
            tp_size=1,
            ep_size=8,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
        )
        moe_layer = container.moe_layer
        # [bs, seql, hidden size]
        hidden_states = torch.randn((32, 8, moe_layer.config.hidden_size))
        hidden_states = hidden_states.cuda()
        hidden_states.requires_grad = True
        scores, indices = moe_layer.router(hidden_states)
        assert scores.shape == (256, moe_layer.router.topk), "Scores shape is not correct"
        assert indices.shape == (256, moe_layer.router.topk), "Indices shape is not correct"
        scores = torch.ones_like(scores) / moe_layer.router.topk

        (
            permuted_local_hidden_states,
            tokens_per_expert,
        ) = moe_layer.token_dispatcher.token_permutation(hidden_states, scores, indices)

        print(f"Dispatched tokens per expert: {tokens_per_expert}")

        restored_hidden_states, restored_bias = moe_layer.token_dispatcher.token_unpermutation(
            permuted_local_hidden_states
        )
        assert torch.allclose(
            restored_hidden_states, hidden_states
        ), "Restored hidden states do not match original hidden states"

        # check if the grad of the hidden states is same as the hidden states
        torch.autograd.backward(restored_hidden_states, restored_hidden_states)
        assert torch.allclose(
            hidden_states.grad, hidden_states
        ), "Gradient of hidden states should be same as hidden states"

        container.destroy()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_tp_forward_backward(self):
        container = MoEModelTestContainer(
            tp_size=8,
            ep_size=1,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
            sequence_parallel=True,
        )
        moe_layer = container.moe_layer

        hidden_states = torch.randn((32, 8, moe_layer.config.hidden_size))
        hidden_states = hidden_states.cuda()
        hidden_states.requires_grad = True
        scores, indices = moe_layer.router(hidden_states)
        assert scores.shape == (256 * moe_layer.config.tensor_model_parallel_size, moe_layer.router.topk), "Scores shape is not correct"
        assert indices.shape == (256 * moe_layer.config.tensor_model_parallel_size, moe_layer.router.topk), "Indices shape is not correct"
        scores = torch.ones_like(scores) / moe_layer.router.topk

        ## Uncomment these lines to assist in bug location.
        # hidden_states = torch.ones_like(hidden_states) * torch.distributed.get_rank()
        # hidden_states.requires_grad = True
        # indices = torch.ones_like(indices) * torch.distributed.get_rank()
        # print(permuted_local_hidden_states)

        (
            permuted_local_hidden_states,
            tokens_per_expert,
        ) = moe_layer.token_dispatcher.token_permutation(hidden_states, scores, indices)

        # print(f"Dispatched tokens per expert: {tokens_per_expert}")

        permuted_local_hidden_states /= moe_layer.config.tensor_model_parallel_size

        restored_hidden_states, restored_bias = moe_layer.token_dispatcher.token_unpermutation(
            permuted_local_hidden_states
        )

        assert torch.allclose(
            restored_hidden_states, hidden_states
        ), "Restored hidden states do not match original hidden states"

        # check if the grad of the hidden states is same as the hidden states
        torch.autograd.backward(restored_hidden_states, restored_hidden_states)
        assert torch.allclose(
            hidden_states.grad, hidden_states
        ), "Gradient of hidden states should be same as hidden states"

        container.destroy()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_tp_ep_forward_backward(self):
        container = MoEModelTestContainer(
            tp_size=4,
            ep_size=2,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
            sequence_parallel=True,
        )
        moe_layer = container.moe_layer

        hidden_states = torch.randn((32, 8, moe_layer.config.hidden_size))
        hidden_states = hidden_states.cuda()
        hidden_states.requires_grad = True
        scores, indices = moe_layer.router(hidden_states)
        assert scores.shape == (256 * moe_layer.config.tensor_model_parallel_size, moe_layer.router.topk), "Scores shape is not correct"
        assert indices.shape == (256 * moe_layer.config.tensor_model_parallel_size, moe_layer.router.topk), "Indices shape is not correct"
        scores = torch.ones_like(scores) / moe_layer.router.topk

        ## Uncomment these lines to assist in bug location.
        # hidden_states = torch.ones_like(hidden_states) * torch.distributed.get_rank()
        # hidden_states.requires_grad = True
        # indices = torch.ones_like(indices) * torch.distributed.get_rank()
        # print(permuted_local_hidden_states)

        (
            permuted_local_hidden_states,
            tokens_per_expert,
        ) = moe_layer.token_dispatcher.token_permutation(hidden_states, scores, indices)

        print(f"Dispatched tokens per expert: {tokens_per_expert}")

        permuted_local_hidden_states /= moe_layer.config.tensor_model_parallel_size

        restored_hidden_states, restored_bias = moe_layer.token_dispatcher.token_unpermutation(
            permuted_local_hidden_states
        )

        assert torch.allclose(
            restored_hidden_states, hidden_states
        ), "Restored hidden states do not match original hidden states"

        # check if the grad of the hidden states is same as the hidden states
        torch.autograd.backward(restored_hidden_states, restored_hidden_states)
        assert torch.allclose(
            hidden_states.grad, hidden_states
        ), "Gradient of hidden states should be same as hidden states"

        container.destroy()
        
    def test_permute_and_unpermute(self):
        tokens = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3], [0, 0, 0]], dtype=torch.float32)
        indices = torch.tensor([[0, 4], [4, 3], [4, 2], [1, 1]])
        probs = torch.ones_like(indices) / 2
        permuted_tokens, sorted_indices = permute(tokens, indices, 2)
        print(permuted_tokens, sorted_indices)
        unpermuted_tokens = unpermute(permuted_tokens, sorted_indices, probs=probs, topk=2)
        print(unpermuted_tokens)
        assert torch.allclose(tokens, unpermuted_tokens)


if __name__ == "__main__":

    GMLP_test = TestAlltoAllDispatcher()
    GMLP_test.setup_method(None)
    GMLP_test.test_ep_forward_backward()
