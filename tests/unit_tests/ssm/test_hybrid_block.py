# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.hybrid.hybrid_block import HybridStack, HyperConnectionHybridLayer
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols, validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils


@pytest.mark.internal
class TestHybridBlock:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def get_pg_collection(self):
        return ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'pp', 'cp'])

    def get_mamba_block(self, layer_pattern, enable_hyper_connections=False):
        layer_type_list = validate_segment_layers(layer_pattern)
        mhc_kwargs = (
            {"enable_hyper_connections": True, "hidden_dropout": 0.0, "mhc_sinkhorn_iterations": 5}
            if enable_hyper_connections
            else {}
        )
        transformer_config = TransformerConfig(
            hidden_size=256,  # The Mamba layer places several constraints on this
            # Need to specify num_attention_heads and num_layers or TransformerConfig
            # will generate errors.
            num_layers=len(layer_type_list),
            num_attention_heads=4,
            use_cpu_initialization=True,
            **mhc_kwargs,
        )
        modules = hybrid_stack_spec.submodules
        return HybridStack(
            transformer_config,
            modules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            pg_collection=self.get_pg_collection(),
        )

    def get_dsa_mamba_block(self, layer_pattern, enable_hyper_connections=False):
        layer_type_list = validate_segment_layers(layer_pattern)
        mhc_kwargs = (
            {"enable_hyper_connections": True, "hidden_dropout": 0.0, "mhc_sinkhorn_iterations": 5}
            if enable_hyper_connections
            else {}
        )
        transformer_config = MLATransformerConfig(
            hidden_size=256,  # The Mamba layer places several constraints on this
            # Need to specify num_attention_heads and num_layers or TransformerConfig
            # will generate errors.
            num_layers=len(layer_type_list),
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
            **mhc_kwargs,
        )
        modules = hybrid_stack_spec.submodules
        return HybridStack(
            transformer_config,
            modules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            pg_collection=self.get_pg_collection(),
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_gpu_forward(self):
        """Test GPU forward pass."""
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP
        block = self.get_mamba_block(layer_pattern)
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
        Make sure that the layer types specified with layer_pattern
        were honored.
        """
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP
        block = self.get_mamba_block(layer_pattern)
        layers = block.layers
        # Note that this matches the order specified by layer_pattern above
        assert isinstance(layers[0], MambaLayer)
        assert isinstance(layers[1], TransformerLayer)
        assert isinstance(layers[1].self_attention, SelfAttention)
        assert isinstance(layers[2], TransformerLayer)
        assert isinstance(layers[2].mlp, MLP)

    def test_hyper_connection_layer_wrappers(self):
        """mHC wraps each hybrid layer while preserving the layer type underneath."""
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP
        block = self.get_mamba_block(layer_pattern, enable_hyper_connections=True)
        layers = block.layers
        assert all(isinstance(layer, HyperConnectionHybridLayer) for layer in layers)
        assert isinstance(layers[0].inner_layer, MambaLayer)
        assert isinstance(layers[1].inner_layer, TransformerLayer)
        assert isinstance(layers[1].inner_layer.self_attention, SelfAttention)
        assert isinstance(layers[2].inner_layer, TransformerLayer)
        assert isinstance(layers[2].inner_layer.mlp, MLP)

    def test_hyper_connection_recompute_plan_for_hybrid_layers(self):
        """HybridStack creates per-layer mHC recompute managers when requested."""
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP
        layer_type_list = validate_segment_layers(layer_pattern)
        transformer_config = TransformerConfig(
            hidden_size=256,
            num_layers=len(layer_type_list),
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            hidden_dropout=0.0,
            mhc_sinkhorn_iterations=5,
            recompute_granularity="selective",
            recompute_modules=["core_attn", "mhc"],
        )
        block = HybridStack(
            transformer_config,
            hybrid_stack_spec.submodules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            pg_collection=self.get_pg_collection(),
        )

        managers, block_ends = block._build_mhc_recompute_layer_plan(use_mhc_recompute=True)
        assert len(managers) == len(block.layers)
        assert all(manager is not None for manager in managers)
        assert block_ends[-1] is True

    def test_hyper_connection_gpu_forward(self):
        """mHC-enabled HybridStack expands internally and contracts back at the output."""
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP
        block = self.get_mamba_block(layer_pattern, enable_hyper_connections=True)
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

    def test_hyper_connection_gdn_gpu_forward(self):
        """mHC runs through GDN, attention, and Mamba hybrid layers."""
        layer_pattern = Symbols.GDN + Symbols.ATTENTION + Symbols.MAMBA
        layer_type_list = validate_segment_layers(layer_pattern)
        transformer_config = TransformerConfig(
            hidden_size=256,
            num_layers=len(layer_type_list),
            num_attention_heads=4,
            use_cpu_initialization=True,
            activation_func=torch.nn.functional.silu,
            enable_hyper_connections=True,
            hidden_dropout=0.0,
            mhc_sinkhorn_iterations=5,
        )
        block = HybridStack(
            transformer_config,
            hybrid_stack_spec.submodules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            pg_collection=self.get_pg_collection(),
        )
        block.cuda()
        micro_batch_size = 2
        sequence_length = 32
        hidden_states = torch.ones((sequence_length, micro_batch_size, block.config.hidden_size))
        hidden_states = hidden_states.cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()
        output = block(hidden_states, attention_mask=attention_mask)
        assert output.shape == (sequence_length, micro_batch_size, block.config.hidden_size)

    def test_hyper_connection_dsa_layer_wrappers(self):
        """mHC wraps DeepSeek-style DSA and MLP split layers."""
        layer_pattern = Symbols.MAMBA + Symbols.DS_ATTENTION + Symbols.MLP
        block = self.get_dsa_mamba_block(layer_pattern, enable_hyper_connections=True)
        layers = block.layers
        assert all(isinstance(layer, HyperConnectionHybridLayer) for layer in layers)
        assert isinstance(layers[0].inner_layer, MambaLayer)
        assert isinstance(layers[1].inner_layer, TransformerLayer)
        assert isinstance(layers[1].inner_layer.self_attention, MLASelfAttention)
        assert isinstance(layers[1].inner_layer.self_attention.core_attention, DSAttention)
        assert isinstance(layers[2].inner_layer, TransformerLayer)
        assert isinstance(layers[2].inner_layer.mlp, MLP)

    def test_hyper_connection_pipeline_boundary_shapes(self):
        """HybridStack keeps n-stream tensors between PP stages and contracts at the end."""
        layer_type_list = validate_segment_layers(Symbols.MAMBA)
        transformer_config = TransformerConfig(
            hidden_size=256,
            num_layers=len(layer_type_list),
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            hidden_dropout=0.0,
            mhc_sinkhorn_iterations=5,
        )
        modules = hybrid_stack_spec.submodules
        first_stage = HybridStack(
            transformer_config,
            modules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            post_process=False,
            pg_collection=self.get_pg_collection(),
        ).cuda()
        last_stage = HybridStack(
            transformer_config,
            modules,
            pre_process=False,
            layer_type_list=layer_type_list,
            pp_layer_offset=1,
            post_process=True,
            pg_collection=self.get_pg_collection(),
        ).cuda()

        micro_batch_size = 2
        sequence_length = 32
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, transformer_config.hidden_size), device='cuda'
        )
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool, device='cuda'
        )

        pp_hidden = first_stage(hidden_states, attention_mask=attention_mask)
        assert pp_hidden.shape == (
            sequence_length,
            micro_batch_size,
            transformer_config.hidden_size * transformer_config.num_residual_streams,
        )

        last_stage.set_input_tensor(pp_hidden.detach())
        output = last_stage(hidden_states, attention_mask=attention_mask)
        assert output.shape == (sequence_length, micro_batch_size, transformer_config.hidden_size)

    def test_invalid_layer_types_cause_failure(self):
        invalid_symbol = '+'
        assert invalid_symbol not in Symbols.VALID_LAYERS  # sanity check.
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP + invalid_symbol
        # validate_segment_layers() in hybrid_layer_allocation.py throws a ValueError.
        with pytest.raises(ValueError):
            block = self.get_mamba_block(layer_pattern)

    def test_gdn_layer_types(self):
        """
        Make sure that G creates a TransformerLayer wrapping GatedDeltaNet,
        while * creates a TransformerLayer wrapping SelfAttention.
        """
        layer_pattern = Symbols.GDN + Symbols.ATTENTION + Symbols.MAMBA
        block = self.get_mamba_block(layer_pattern)
        layers = block.layers
        assert isinstance(layers[0], TransformerLayer)
        assert isinstance(layers[0].self_attention, GatedDeltaNet)
        assert isinstance(layers[1], TransformerLayer)
        assert isinstance(layers[1].self_attention, SelfAttention)
        assert isinstance(layers[2], MambaLayer)

    def test_gdn_gpu_forward(self):
        """Test GPU forward pass with GDN, attention, and Mamba layers."""
        layer_pattern = Symbols.GDN + Symbols.ATTENTION + Symbols.MAMBA
        layer_type_list = validate_segment_layers(layer_pattern)
        transformer_config = TransformerConfig(
            hidden_size=256,
            num_layers=len(layer_type_list),
            num_attention_heads=4,
            use_cpu_initialization=True,
            activation_func=torch.nn.functional.silu,
        )
        modules = hybrid_stack_spec.submodules
        block = HybridStack(
            transformer_config,
            modules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            pg_collection=self.get_pg_collection(),
        )
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

    def test_dsa_layer_types(self):
        """D symbol creates a TransformerLayer with MLASelfAttention."""
        layer_pattern = Symbols.MAMBA + Symbols.DS_ATTENTION + Symbols.MAMBA
        block = self.get_dsa_mamba_block(layer_pattern)
        layers = block.layers
        assert isinstance(layers[0], MambaLayer)
        assert isinstance(layers[1], TransformerLayer)
        assert isinstance(layers[1].self_attention, MLASelfAttention)
        assert isinstance(layers[1].self_attention.core_attention, DSAttention)
        assert isinstance(layers[2], MambaLayer)

    def test_mixed_attention_and_dsa_layer_types(self):
        """* and D in the same block fail."""
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.DS_ATTENTION + Symbols.MAMBA
        with pytest.raises(ValueError):
            block = self.get_dsa_mamba_block(layer_pattern)
