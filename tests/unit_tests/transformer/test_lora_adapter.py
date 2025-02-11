# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.lora_adapter import LoraAdapter
from megatron.core.transformer.transformer_config import TransformerConfig

from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize('expert_tensor_parallel_size', [1, 2])
@pytest.mark.parametrize('pipeline_model_parallel_size', [1, 2])
@pytest.mark.parametrize('tensor_model_parallel_size', [1, 2])
class TestLoraAdapterWithLoraLayers:

    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self, expert_tensor_parallel_size, pipeline_model_parallel_size, tensor_model_parallel_size):
        Utils.initialize_model_parallel(
            expert_tensor_parallel_size=expert_tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            tensor_model_parallel_size=tensor_model_parallel_size,
        )
        model_parallel_cuda_manual_seed(123)
        self.input_size = 64
        self.output_size = 64
        self.rank = 16
        self.alpha = 32
        self.config = TransformerConfig(
            num_layers=1,
            add_bias_linear=False,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=expert_tensor_parallel_size if expert_tensor_parallel_size > 1 else None,
            expert_model_parallel_size=expert_tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            tensor_model_parallel_size=tensor_model_parallel_size,
            sequence_parallel=tensor_model_parallel_size > 1,
            pipeline_dtype=torch.float32,
        )
        base_layer = ColumnParallelLinear(
            input_size=self.input_size,
            output_size=self.output_size,
            config=self.config,
            bias=False,
            skip_bias_add=True,
            init_method=torch.nn.init.zeros_,
        )
        self.lora_adapter = LoraAdapter(
            base_layer,
            config=self.config,
            rank=self.rank,
            alpha=self.alpha,
            dropout=0.01,
        )
        yield
        Utils.destroy_model_parallel()

    def test_constructor(self):
        parallel_output_size = int(self.output_size / self.config.tensor_model_parallel_size)
        assert _get_nparams(self.lora_adapter) == self.input_size * parallel_output_size \
            + self.input_size * self.rank \
            + self.rank * parallel_output_size
    
        INPUT_INDEX = 1
        OUTPUT_INDEX = 0
        assert self.lora_adapter.base_layer.weight.shape[INPUT_INDEX] == self.lora_adapter.lora_a.weight.shape[INPUT_INDEX]
        assert self.lora_adapter.base_layer.weight.shape[OUTPUT_INDEX] == self.lora_adapter.lora_b.weight.shape[OUTPUT_INDEX]

    def test_load_state_dict(self):
        model = torch.nn.Module()
        model.add_module("output_layer", self.lora_adapter)
        parallel_output_size = int(self.output_size / self.config.tensor_model_parallel_size)
        state_dict = {"output_layer.weight": torch.ones(parallel_output_size, self.input_size)}
        missing_keys, unexpected_keys = model.load_state_dict(state_dict)

        assert missing_keys == []
        assert unexpected_keys == []
        assert self.lora_adapter.base_layer.weight.all()
        assert torch.any(self.lora_adapter.lora_a.weight)
        assert self.lora_adapter.lora_b.weight.sum() == 0


class TestLoraAdapterWithUnknownBaseLayer:

    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

        self.input_size = 64
        self.output_size = 64
        self.rank = 16
        self.alpha = 32
        self.transformer_config = TransformerConfig(
            num_layers=1,
            add_bias_linear=False,
            hidden_size=12,
            num_attention_heads=4,
        )
        self.base_layer = torch.nn.Linear(self.input_size, self.output_size, bias=False)
        self.lora_adapter = LoraAdapter(
            self.base_layer,
            config=self.transformer_config,
            rank=self.rank,
            alpha=self.alpha,
            dropout=0.01,
        )
        yield
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert _get_nparams(self.lora_adapter) == self.input_size * self.output_size


def _get_nparams(module: torch.nn.Module):
    return sum([p.numel() for p in module.parameters()])
