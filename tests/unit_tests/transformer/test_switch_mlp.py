# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest

import torch

from megatron.core.transformer.switch_mlp import SwitchMLP
from tests.unit_tests.test_utilities import Utils
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import gpt_layer_with_transformer_engine_spec_moe

class TestParallelSwitchMLP:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1,1)
        model_parallel_cuda_manual_seed(123)
        print("done intializing")
        transformer_config = TransformerConfig(num_layers=2, hidden_size=12, num_attention_heads=4, num_moe_experts= 2, use_cpu_initialization=True)
        self.switch_mlp = SwitchMLP(transformer_config,
                       gpt_layer_with_transformer_engine_spec_moe.submodules.mlp.submodules)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.switch_mlp, SwitchMLP)

        num_weights = sum([p.numel() for p in self.switch_mlp.parameters()])
        assert num_weights == 2450


    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_forward(self):
        switch_mlp = self.switch_mlp
        switch_mlp.cuda()
        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((32, 2, switch_mlp.config.hidden_size))
        hidden_states = hidden_states.cuda()
        output, output_bias = switch_mlp(hidden_states)
        assert output.shape[0] == 32
        assert output.shape[1] == 2
        assert output.shape[2] == switch_mlp.config.hidden_size
        assert output_bias.shape[2] == switch_mlp.config.hidden_size
        assert output.dtype == torch.float32
        assert output.device.type == 'cuda'
        assert output_bias.device.type == 'cuda'

