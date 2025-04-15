# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import Router
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


class TestMoELayerInit:
    def setup_method(self, method):
        pass

    @pytest.mark.skipif(
        not is_te_min_version("1.7.0.dev0"),
        reason="Expert with TE Linear is only supported in TE 1.7.0 and later.",
    )
    @pytest.mark.parametrize("moe_token_dispatcher_type", ["allgather", "alltoall"])
    @pytest.mark.parametrize("num_moe_experts", [1, 2])
    @pytest.mark.parametrize("grouped_gemm", [True, False])
    def test_te_moe_layer(self, num_moe_experts, moe_token_dispatcher_type, grouped_gemm):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        self.transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=grouped_gemm,
            moe_ffn_hidden_size=128,
            add_bias_linear=False,
        )
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
        )
        moe_layer = MoELayer(
            self.transformer_config, transformer_layer_spec.submodules.mlp.submodules
        )
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("moe_token_dispatcher_type", ["allgather", "alltoall"])
    @pytest.mark.parametrize("num_moe_experts", [1, 2])
    @pytest.mark.parametrize("grouped_gemm", [True, False])
    def test_legacy_moe_layer(self, num_moe_experts, moe_token_dispatcher_type, grouped_gemm):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        num_moe_experts = 4
        self.transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=grouped_gemm,
            add_bias_linear=False,
        )
        transformer_layer_spec = get_gpt_layer_local_spec(
            num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
        )
        moe_layer = MoELayer(
            self.transformer_config, transformer_layer_spec.submodules.mlp.submodules
        )
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("moe_token_dispatcher_type", ["alltoall", "allgather"])
    @pytest.mark.parametrize("grouped_gemm", [True, False])
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 1), (2, 2)])
    def test_moe_with_late_initialize(
        self, moe_token_dispatcher_type, grouped_gemm, tp_size, ep_size
    ):
        num_moe_experts = 4
        hidden_size = 12
        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            add_bias_linear=False,
            moe_grouped_gemm=grouped_gemm,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            sequence_parallel=tp_size > 1,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
        )

        # Fake initialization as NeMo does
        Utils.fake_initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        moe_layer = MoELayer(
            transformer_config, transformer_layer_spec.submodules.mlp.submodules
        ).cuda()

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        input_data = torch.randn(
            16, 4, hidden_size, device=torch.cuda.current_device(), dtype=torch.bfloat16
        )
        output = moe_layer(input_data)

        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestInterleaveTransformerBlock:

    @pytest.mark.parametrize("moe_layer_freq", [2, eval("[0,1,1,1]"), eval("[0]*2+[1]*2")])
    def test_interleave_transformer_block(self, moe_layer_freq):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=4,
            moe_layer_freq=moe_layer_freq,
            moe_ffn_hidden_size=256,
            use_cpu_initialization=True,
            num_moe_experts=2,
        )
        self.parallel_transformer_block = TransformerBlock(
            self.transformer_config, get_gpt_decoder_block_spec(self.transformer_config, False)
        )

        # Check if the moe layer is interleaved correctly
        if isinstance(self.transformer_config.moe_layer_freq, int):
            moe_layer_pattern = [
                1 if (i % self.transformer_config.moe_layer_freq == 0) else 0
                for i in range(self.transformer_config.num_layers)
            ]
        else:
            moe_layer_pattern = self.transformer_config.moe_layer_freq

        for i, layer in enumerate(self.parallel_transformer_block.layers):
            is_moe_layer = isinstance(layer.mlp, MoELayer)
            assert is_moe_layer == moe_layer_pattern[i]

        # Test forward pass
        parallel_transformer_block = self.parallel_transformer_block
        config: TransformerConfig = parallel_transformer_block.config
        sequence_length = 32
        micro_batch_size = 2
        parallel_transformer_block.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((sequence_length, micro_batch_size, config.hidden_size))
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()
        hidden_states = parallel_transformer_block(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        assert hidden_states.shape[0] == sequence_length
        assert hidden_states.shape[1] == micro_batch_size
        assert hidden_states.shape[2] == config.hidden_size

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
