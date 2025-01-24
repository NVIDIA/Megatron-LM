# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestSharedExperts:

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    def test_gpu_forward(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        print("done intializing")
        num_moe_experts = 2
        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            moe_shared_expert_intermediate_size=32,
            use_cpu_initialization=True,
            activation_func=torch.nn.functional.silu,
            gated_linear_unit=True,
            bias_activation_fusion=True,
            moe_router_load_balancing_type="sinkhorn",
            moe_router_topk=1,
            add_bias_linear=False,
        )
        transformer_layer_spec = get_gpt_layer_local_spec(
            num_experts=num_moe_experts, moe_grouped_gemm=False
        )
        self.moe_layer = MoELayer(
            transformer_config, transformer_layer_spec.submodules.mlp.submodules
        )

        assert isinstance(self.moe_layer, MoELayer)

        num_weights = sum([p.numel() for p in self.moe_layer.parameters()])
        assert num_weights == 3480 + 1152
        assert self.moe_layer.shared_experts is not None
        assert self.moe_layer.shared_experts.stream is None
        assert self.moe_layer.token_dispatcher.shared_experts is None

        moe_layer = self.moe_layer
        moe_layer.cuda()
        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((32, 2, moe_layer.config.hidden_size))
        hidden_states = hidden_states.cuda()
        output, _ = moe_layer(hidden_states)
        assert output.shape[0] == 32
        assert output.shape[1] == 2
        assert output.shape[2] == moe_layer.config.hidden_size
        assert output.dtype == torch.float32
        assert output.device.type == 'cuda'


class TestSharedExpertsOverlap:

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    def test_gpu_forward(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        print("done intializing")
        num_moe_experts = 2
        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            moe_shared_expert_intermediate_size=32,
            moe_shared_expert_overlap=True,
            moe_token_dispatcher_type="alltoall",
            use_cpu_initialization=True,
            activation_func=torch.nn.functional.silu,
            gated_linear_unit=True,
            bias_activation_fusion=True,
            moe_router_load_balancing_type="sinkhorn",
            moe_router_topk=1,
            add_bias_linear=False,
        )
        transformer_layer_spec = get_gpt_layer_local_spec(
            num_experts=num_moe_experts, moe_grouped_gemm=False
        )
        self.moe_layer = MoELayer(
            transformer_config, transformer_layer_spec.submodules.mlp.submodules
        )

        assert isinstance(self.moe_layer, MoELayer)

        num_weights = sum([p.numel() for p in self.moe_layer.parameters()])
        assert num_weights == 3480 + 1152
        assert self.moe_layer.shared_experts is not None
        assert self.moe_layer.shared_experts.stream is not None
        assert self.moe_layer.token_dispatcher.shared_experts is not None

        moe_layer = self.moe_layer
        moe_layer.cuda()
        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((32, 2, moe_layer.config.hidden_size))
        hidden_states = hidden_states.cuda()
        output, _ = moe_layer(hidden_states)
        assert output.shape[0] == 32
        assert output.shape[1] == 2
        assert output.shape[2] == moe_layer.config.hidden_size
        assert output.dtype == torch.float32
        assert output.device.type == 'cuda'
