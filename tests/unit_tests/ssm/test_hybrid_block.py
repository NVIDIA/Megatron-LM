# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.hybrid.fine_grained_callables import build_hybrid_stack_callables
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import (
    Symbols,
    get_layer_type_list_physical_count,
    validate_segment_layers,
)
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

    def get_pg_collection(self, required_pgs=None):
        if required_pgs is None:
            required_pgs = ['tp', 'pp', 'cp']
        return ProcessGroupCollection.use_mpu_process_groups(required_pgs=required_pgs)

    def get_mamba_block(self, layer_pattern):
        layer_type_list = validate_segment_layers(layer_pattern)
        transformer_config = TransformerConfig(
            hidden_size=256,  # The Mamba layer places several constraints on this
            # Need to specify num_attention_heads and num_layers or TransformerConfig
            # will generate errors.
            num_layers=len(layer_type_list),
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        modules = hybrid_stack_spec.submodules
        return HybridStack(
            transformer_config,
            modules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            pg_collection=self.get_pg_collection(),
        )

    def get_dsa_mamba_block(self, layer_pattern):
        layer_type_list = validate_segment_layers(layer_pattern)
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
        )
        modules = hybrid_stack_spec.submodules
        return HybridStack(
            transformer_config,
            modules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            pg_collection=self.get_pg_collection(),
        )

    def get_attention_mlp_block(self, layer_pattern):
        layer_type_list = validate_segment_layers(layer_pattern)
        transformer_config = TransformerConfig(
            hidden_size=256,
            num_layers=get_layer_type_list_physical_count(layer_type_list),
            num_attention_heads=4,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            use_cpu_initialization=True,
        )
        return HybridStack(
            transformer_config,
            hybrid_stack_spec.submodules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            pg_collection=self.get_pg_collection(),
        )

    def get_attention_moe_block(self, layer_pattern):
        layer_type_list = validate_segment_layers(layer_pattern)
        transformer_config = TransformerConfig(
            hidden_size=256,
            num_layers=get_layer_type_list_physical_count(layer_type_list),
            num_attention_heads=4,
            ffn_hidden_size=256,
            num_moe_experts=8,
            expert_model_parallel_size=1,
            moe_router_topk=2,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            hidden_dropout=0.0,
            attention_dropout=0.0,
            use_cpu_initialization=True,
        )
        return HybridStack(
            transformer_config,
            hybrid_stack_spec.submodules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            pg_collection=self.get_pg_collection(
                required_pgs=[
                    'tp',
                    'pp',
                    'cp',
                    'tp_cp',
                    'tp_dp_cp',
                    'ep',
                    'expt_tp',
                    'tp_ep',
                    'expt_dp',
                ]
            ),
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

    def test_group_layer_type_builds_nested_hybrid_stack(self):
        """Bracketed groups build an inner HybridStack with physical layer numbering."""
        layer_type_list = validate_segment_layers("M[M*]-")
        transformer_config = TransformerConfig(
            hidden_size=256,
            num_layers=get_layer_type_list_physical_count(layer_type_list),
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        block = HybridStack(
            transformer_config,
            hybrid_stack_spec.submodules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            pg_collection=self.get_pg_collection(),
        )
        assert isinstance(block.layers[0], MambaLayer)
        assert isinstance(block.layers[1], HybridStack)
        assert isinstance(block.layers[1].layers[0], MambaLayer)
        assert isinstance(block.layers[1].layers[1], TransformerLayer)
        assert isinstance(block.layers[2], TransformerLayer)
        assert [layer.layer_number for layer in block.layers[1].layers] == [2, 3]
        assert block.layers[2].layer_number == 4

    def test_group_sharded_state_dict_uses_logical_layer_keys(self):
        """Grouped attention+MLP layers share one Transformer-compatible checkpoint key."""
        layer_type_list = validate_segment_layers("[*-]")
        transformer_config = TransformerConfig(
            hidden_size=256,
            num_layers=get_layer_type_list_physical_count(layer_type_list),
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        block = HybridStack(
            transformer_config,
            hybrid_stack_spec.submodules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            logical_layer_offset=0,
            pg_collection=self.get_pg_collection(),
        )

        sharded_state_dict = block.sharded_state_dict(prefix="decoder.")
        sharded_keys = {value.key for value in sharded_state_dict.values() if hasattr(value, "key")}

        assert "decoder.layers.0.self_attention.linear_qkv.weight" in sharded_keys
        assert "decoder.layers.0.mlp.linear_fc1.weight" in sharded_keys
        assert "decoder.layers.1.mlp.linear_fc1.weight" not in sharded_keys
        assert "decoder.final_layernorm.weight" in sharded_keys
        assert "decoder.final_norm.weight" not in sharded_keys

    def test_group_forward_matches_equivalent_flat_layers(self):
        """A bracket group is only a scheduling/checkpoint boundary, not new math."""
        flat_block = self.get_attention_mlp_block("*-")
        group_block = self.get_attention_mlp_block("[*-]")

        group_block.layers[0].layers[0].load_state_dict(flat_block.layers[0].state_dict())
        group_block.layers[0].layers[1].load_state_dict(flat_block.layers[1].state_dict())
        group_block.final_norm.load_state_dict(flat_block.final_norm.state_dict())

        flat_block.cuda().eval()
        group_block.cuda().eval()
        sequence_length = 16
        micro_batch_size = 2
        hidden_states = torch.randn(
            sequence_length,
            micro_batch_size,
            flat_block.config.hidden_size,
            device="cuda",
        )
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length),
            dtype=bool,
            device="cuda",
        )

        with torch.no_grad():
            flat_output = flat_block(hidden_states.clone(), attention_mask=attention_mask)
            group_output = group_block(hidden_states.clone(), attention_mask=attention_mask)

        torch.testing.assert_close(group_output, flat_output, rtol=0, atol=0)

    def test_group_overlap_callables_keep_ep_moe_split_visible(self):
        """EP-overlap scheduling still sees dispatch/experts/combine inside a group."""
        block = self.get_attention_moe_block("[*E]")

        forward_callables, bwd_dw_callable_map, is_moe, num_local_experts = (
            build_hybrid_stack_callables(block.layers[0], layer_type=block.layer_type_list[0])
        )

        pre_dispatch, dispatch, experts, combine, mtp_post_process = forward_callables
        assert callable(pre_dispatch)
        assert callable(dispatch)
        assert callable(experts)
        assert callable(combine)
        assert mtp_post_process is None
        assert is_moe
        assert num_local_experts == 8
        assert "attn" in bwd_dw_callable_map
        assert "mlp" in bwd_dw_callable_map

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
