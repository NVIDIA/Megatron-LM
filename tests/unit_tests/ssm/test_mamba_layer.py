# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.internal
class TestMambaLayer:

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
        modules = mamba_stack_spec.submodules.mamba_layer.submodules
        model_comm_pgs = ModelCommProcessGroups.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.layer = MambaLayer(transformer_config, modules, model_comm_pgs=model_comm_pgs)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_gpu_forward(self):
        layer = self.layer
        layer.cuda()
        micro_batch_size = 2
        sequence_length = 32
        hidden_states = torch.ones((sequence_length, micro_batch_size, layer.config.hidden_size))
        hidden_states = hidden_states.cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        )
        attention_mask = attention_mask.cuda()
        output = layer(hidden_states, attention_mask=attention_mask)
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == layer.config.hidden_size
        assert output.dtype == torch.float32

    def test_mamba_layer_recompute_disabled(self):
        """Test Mamba layer with recomputation disabled."""
        layer = self.layer
        layer.cuda()
        
        # Ensure recomputation is disabled
        assert not layer.mamba_layer_recompute
        
        micro_batch_size = 2
        sequence_length = 32
        hidden_states = torch.ones((sequence_length, micro_batch_size, layer.config.hidden_size))
        hidden_states = hidden_states.cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        )
        attention_mask = attention_mask.cuda()
        
        output = layer(hidden_states, attention_mask=attention_mask)
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == layer.config.hidden_size
        assert output.dtype == torch.float32

    def test_mamba_layer_recompute_enabled(self):
        """Test Mamba layer with recomputation enabled."""
        # Create config with recomputation enabled
        transformer_config = TransformerConfig(
            hidden_size=256,
            num_layers=1,
            num_attention_heads=1,
            use_cpu_initialization=True,
            recompute_granularity='selective',
            recompute_modules=['mamba'],
        )
        modules = mamba_stack_spec.submodules.mamba_layer.submodules
        model_comm_pgs = ModelCommProcessGroups.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        layer = MambaLayer(transformer_config, modules, model_comm_pgs=model_comm_pgs)
        layer.cuda()
        
        # Ensure recomputation is enabled
        assert layer.mamba_layer_recompute
        assert transformer_config.recompute_granularity == 'selective'
        assert 'mamba' in transformer_config.recompute_modules
        
        micro_batch_size = 2
        sequence_length = 32
        hidden_states = torch.ones((sequence_length, micro_batch_size, layer.config.hidden_size))
        hidden_states = hidden_states.cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        )
        attention_mask = attention_mask.cuda()
        
        output = layer(hidden_states, attention_mask=attention_mask)
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == layer.config.hidden_size
        assert output.dtype == torch.float32

    def test_mamba_layer_recompute_vs_no_recompute_consistency(self):
        """Test that recomputed and non-recomputed outputs are consistent."""
        micro_batch_size = 2
        sequence_length = 32
        
        # Test with the same random seed for reproducibility
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)
        # Create layer without recomputation
        transformer_config_no_recompute = TransformerConfig(
            hidden_size=256,
            num_layers=1,
            num_attention_heads=1,
            use_cpu_initialization=True,
        )
        modules = mamba_stack_spec.submodules.mamba_layer.submodules
        model_comm_pgs = ModelCommProcessGroups.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        layer_no_recompute = MambaLayer(
            transformer_config_no_recompute, modules, model_comm_pgs=model_comm_pgs
        )
        layer_no_recompute.cuda()
        
        # Create layer with recomputation enabled
        transformer_config_recompute = TransformerConfig(
            hidden_size=256,
            num_layers=1,
            num_attention_heads=1,
            use_cpu_initialization=True,
            recompute_granularity='selective',
            recompute_modules=['mamba'],
        )
        layer_recompute = MambaLayer(
            transformer_config_recompute, modules, model_comm_pgs=model_comm_pgs
        )
        layer_recompute.cuda()
        
        # Copy weights from no-recompute layer to recompute layer
        layer_recompute.load_state_dict(layer_no_recompute.state_dict())
        
        # Prepare input
        hidden_states = torch.randn((sequence_length, micro_batch_size, 256))
        hidden_states = hidden_states.cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        )
        attention_mask = attention_mask.cuda()
        
        # Forward pass without recomputation
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)
        output_no_recompute = layer_no_recompute(hidden_states, attention_mask=attention_mask)
        
        # Forward pass with recomputation
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)
        output_recompute = layer_recompute(hidden_states, attention_mask=attention_mask)
        
        # Check that outputs are close (allowing for numerical differences due to recomputation)
        torch.testing.assert_close(output_no_recompute, output_recompute)

