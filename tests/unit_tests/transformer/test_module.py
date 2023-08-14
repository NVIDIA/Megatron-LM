# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest

import torch

from megatron.core.transformer.module import Float16Module, MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

DEVICE_CAPABILITY = None
if torch.cuda.is_available():
    DEVICE_CAPABILITY = torch.cuda.get_device_capability()


class DummyModule(MegatronModule):
    # def __init__(self, config: TransformerConfig, share_embeddings_and_output_weights=True):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)

        self.linear = torch.nn.modules.Linear(in_features=2, out_features=1)

    def forward(self, x):
        return self.linear(x)

class TestMegatronModule:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1,1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True)
        self.megatron_module = DummyModule(config=transformer_config).cuda()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()   

    def test_megatron_module(self):
        megatron_module = self.megatron_module
        assert megatron_module
        assert megatron_module.config.hidden_size == 12
        assert megatron_module.config.ffn_hidden_size == 48
        assert megatron_module.linear.weight.dtype == torch.float32

        x = torch.ones((2, 2)).cuda()
        assert megatron_module(x).dtype == torch.float32

        # TODO: test bad configs actually fail
        # failed_module = megatron_module
        # failed_module.fp16 = True
        # failed_module.bf16 = True


class TestFloat16Module:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1,1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True)
        self.megatron_module = DummyModule(config=self.transformer_config).cuda()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()   
        
    def test_fp16_module(self):
        transformer_config = self.transformer_config
        megatron_module = self.megatron_module
        transformer_config.fp16 = True
        fp16_module = Float16Module(config=transformer_config, module=megatron_module)

        assert fp16_module
        assert fp16_module.config.hidden_size == 12
        assert fp16_module.config.ffn_hidden_size == 48
        assert fp16_module.module.linear.weight.dtype == torch.float16

        x = torch.ones((2, 2)).cuda()
        # inputs are converted to fp16 then outputs are converted to fp32
        assert fp16_module(x).dtype == torch.float32

    pytest.mark.skipif(
        not DEVICE_CAPABILITY or DEVICE_CAPABILITY[0] < 8, reason='bfloat16 is not supported on this device'
    )

    def test_bf16_module(self):
        transformer_config = self.transformer_config
        megatron_module = self.megatron_module
        transformer_config.bf16 = True
        bf16_module = Float16Module(config=transformer_config, module=megatron_module)

        assert bf16_module
        assert bf16_module.config.hidden_size == 12
        assert bf16_module.config.ffn_hidden_size == 48
        assert bf16_module.module.linear.weight.dtype == torch.bfloat16

        x = torch.ones((2, 2)).cuda()
        # inputs are converted to bf16 then outputs are converted to fp32
        assert bf16_module(x).dtype == torch.float32

