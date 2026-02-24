# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_block import MambaStack
from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils


@pytest.mark.internal
class TestMambaBlock:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def get_pg_collection(self):
        return ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'pp', 'cp'])

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
            transformer_config,
            modules,
            hybrid_override_pattern=hybrid_override_pattern,
            pg_collection=self.get_pg_collection(),
        )

    def get_dsa_mamba_block(self, hybrid_override_pattern):
        config = MLATransformerConfig(
            num_layers=len(hybrid_override_pattern),
            hidden_size=256,
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type='rope',
            rotary_base=10000,
            rotary_percent=1.0,
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=32,
        )
        modules = mamba_stack_spec.submodules
        return MambaStack(
            config,
            modules,
            hybrid_override_pattern=hybrid_override_pattern,
            pg_collection=self.get_pg_collection(),
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_gpu_forward(self):
        """Test GPU forward pass."""
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
        assert isinstance(layers[0], MambaLayer)
        assert isinstance(layers[1], TransformerLayer)
        assert isinstance(layers[1].self_attention, SelfAttention)
        assert isinstance(layers[2], TransformerLayer)
        assert isinstance(layers[2].mlp, MLP)

    def test_invalid_layer_types_cause_failure(self):
        invalid_symbol = '+'
        assert invalid_symbol not in Symbols.VALID  # sanity check.
        hybrid_override_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP + invalid_symbol
        # _allocate_override() in mamba_hybrid_layer_allocation.py throws a ValueError.
        with pytest.raises(ValueError):
            block = self.get_mamba_block(hybrid_override_pattern)

    def test_dsa_layer_types(self):
        """S symbol creates a TransformerLayer with MLASelfAttention."""
        pattern = Symbols.MAMBA + Symbols.DSA_ATTENTION + Symbols.MAMBA
        block = self.get_dsa_mamba_block(pattern)
        layers = block.layers
        assert isinstance(layers[0], MambaLayer)
        assert isinstance(layers[1], TransformerLayer)
        assert isinstance(layers[1].self_attention, MLASelfAttention)
        assert isinstance(layers[1].self_attention.core_attention, DSAttention)
        assert isinstance(layers[2], MambaLayer)

    def test_mixed_attention_and_dsa_layer_types(self):
        """* and S in the same block create different attention types."""
        pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.DSA_ATTENTION + Symbols.MAMBA
        block = self.get_dsa_mamba_block(pattern)
        layers = block.layers
        assert isinstance(layers[0], MambaLayer)
        assert isinstance(layers[1], TransformerLayer)
        assert isinstance(layers[1].self_attention, SelfAttention)
        assert isinstance(layers[2], TransformerLayer)
        assert isinstance(layers[2].self_attention, MLASelfAttention)
        assert isinstance(layers[2].self_attention.core_attention, DSAttention)
        assert isinstance(layers[3], MambaLayer)
