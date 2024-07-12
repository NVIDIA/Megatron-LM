# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest

import torch
import types

from megatron.core.transformer.parallel_mlp import ParallelMLP
from megatron.global_vars import set_args

from deepspeed.accelerator import get_accelerator
device_name = get_accelerator().device_name()

@pytest.fixture
def mlp(transformer_config):
    mlp_args = types.SimpleNamespace(
        swiglu=False,
        openai_gelu=True,
        onnx_safe=False,
        bias_gelu_fusion=False,
        transformer_impl="",
        cache_fp8_weight=False,
        fp8_interval=False,
        cache_fp8_weight_fwd=False
    )
    set_args(mlp_args)
    return ParallelMLP(transformer_config)


class TestParallelMLP:
    def test_constructor(self, mlp):
        assert isinstance(mlp, ParallelMLP)

        num_weights = sum([p.numel() for p in mlp.parameters()])
        assert num_weights == 1212

    def test_cpu_forward(self, mlp, transformer_config):
        # [sequence length, micro batch size, hidden size]
        hidden_states = torch.ones((32, 2, transformer_config.hidden_size))
        output, output_bias = mlp(hidden_states)
        assert output.shape[0] == 32
        assert output.shape[1] == 2
        assert output.shape[2] == transformer_config.hidden_size
        assert output_bias == None
        assert output.dtype == torch.float32

    @pytest.mark.skipif(not get_accelerator().is_available(), reason="accelerator not available")
    def test_accelerator_forward(self, mlp, transformer_config):
        mlp.to(device_name)
        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((32, 2, transformer_config.hidden_size))
        hidden_states = hidden_states.to(device_name)
        output, output_bias = mlp(hidden_states)
        assert output.shape[0] == 32
        assert output.shape[1] == 2
        assert output.shape[2] == transformer_config.hidden_size
        assert output_bias == None
        assert output.dtype == torch.float32
        assert output.device.type == device_name

