# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.ssm.mamba_block import MambaStack
from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils


class TestMambaBlock:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def get_mamba_block(self, hybrid_override_pattern):
        transformer_config = TransformerConfig(
            hidden_size=256,  # The Mamba layer places several constraints on this
            # Need to specify num_attention_heads and num_layers or TransformerConfig
            # will generate errors.
            num_layers=len(hybrid_override_pattern),
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        modules = mamba_stack_spec.submodules
        return MambaStack(
            transformer_config, modules, hybrid_override_pattern=hybrid_override_pattern
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_gpu_forward(self):
        hybrid_override_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP
        block = self.get_mamba_block(hybrid_override_pattern)
        block.cuda()
        micro_batch_size = 2
        sequence_length = 32
        hidden_states = torch.ones((sequence_length, micro_batch_size, block.config.hidden_size))
        hidden_states = hidden_states.cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        )
        attention_mask = attention_mask.cuda()
        output = block(hidden_states, attention_mask=attention_mask)
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == block.config.hidden_size
        assert output.dtype == torch.float32

    def test_layer_types(self):
        """
        Make sure that the layer types specified with hybrid_override_pattern
        were honored.
        """
        hybrid_override_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP
        block = self.get_mamba_block(hybrid_override_pattern)
        layers = block.layers
        # Note that this matches the order specified by hybrid_override_pattern in setup_method
        assert type(layers[0]) == MambaLayer
        assert type(layers[1]) == TransformerLayer
        assert type(layers[1].self_attention) == SelfAttention
        assert type(layers[2]) == TransformerLayer
        assert type(layers[2].mlp) == MLP

    def test_invalid_layer_types_cause_failure(self):
        invalid_symbol = '+'
        assert invalid_symbol not in Symbols.VALID  # sanity check.
        hybrid_override_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP + invalid_symbol
        # _allocate_override() in mamba_hybrid_layer_allocation.py throws a ValueError.
        with pytest.raises(ValueError):
            block = self.get_mamba_block(hybrid_override_pattern)
