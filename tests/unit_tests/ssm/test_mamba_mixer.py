# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestMambaMixer:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            hidden_size=256,  # The Mamba layer places several constraints on this
            # Need to specify num_attention_heads and num_layers or TransformerConfig
            # will generate errors.
            num_layers=1,
            num_attention_heads=1,
            use_cpu_initialization=True,
        )
        modules = mamba_stack_spec.submodules.mamba_layer.submodules.mixer.submodules
        self.mixer = MambaMixer(transformer_config, modules, transformer_config.hidden_size)
        self.mixer_no_mem_eff_path = MambaMixer(
            transformer_config, modules, transformer_config.hidden_size, use_mem_eff_path=False
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("use_mem_eff_path", [True, False])
    def test_gpu_forward(self, use_mem_eff_path):
        if use_mem_eff_path:
            mixer = self.mixer
        else:
            mixer = self.mixer_no_mem_eff_path
        mixer.cuda()
        micro_batch_size = 2
        sequence_length = 32
        hidden_states = torch.ones((sequence_length, micro_batch_size, mixer.config.hidden_size))
        hidden_states = hidden_states.cuda()
        output, bias = mixer(hidden_states)
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == mixer.config.hidden_size
        assert output.dtype == torch.float32
