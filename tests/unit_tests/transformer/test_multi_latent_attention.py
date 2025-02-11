# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import os
from importlib.metadata import version
from inspect import signature

import pytest
import torch
import transformer_engine as te

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.multi_latent_attention import MLASelfAttention, MultiLatentAttention
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils


class TestParallelMLAAttention:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = MLATransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_head_dim=128,
            v_head_dim=128,
            qk_pos_emb_head_dim=64,
            rotary_base=10000,
            max_position_embeddings=32,
        )
        self.parallel_attention = MLASelfAttention(
            self.transformer_config,
            get_gpt_layer_with_transformer_engine_spec(
                multi_latent_attention=True
            ).submodules.self_attention.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_input_params_forward(self):
        """
        Test to ensure that MultiLatentAttention has all parameters
        required by the Attention class's forward method.
        """
        # Extract parameters from the forward methods of both Attention and MultiLatentAttention
        attn_params = set(signature(Attention.forward).parameters.keys())
        mla_params = set(signature(MultiLatentAttention.forward).parameters.keys())

        # Identify parameters that are in Attention but missing in MultiLatentAttention
        missing_params = attn_params - mla_params
        assert not missing_params, f"Missing parameters in MultiLatentAttention: {missing_params}"

    def test_constructor(self):
        assert isinstance(self.parallel_attention, MLASelfAttention)
        assert self.parallel_attention.layer_number == 1

        num_weights = sum([p.numel() for p in self.parallel_attention.parameters()])
        assert num_weights == 65036

    def test_cpu_forward(self):
        # we can't currently do this because the global memory buffer is on GPU
        pass

    def test_gpu_forward(self):
        if is_te_min_version("1.10.0"):
            # use flash attention for hopper, future may support fused attention for ampere
            os.environ['NVTE_FUSED_ATTN'] = "0"
            os.environ['NVTE_FLASH_ATTN'] = "1"

            config = self.parallel_attention.config
            sequence_length = 32
            micro_batch_size = 2

            self.parallel_attention.cuda()

            # [sequence length, batch size, hidden size]
            hidden_states = torch.ones(
                (sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size)
            )
            hidden_states = hidden_states.cuda()

            attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

            output, bias = self.parallel_attention(hidden_states, attention_mask)

            assert config.recompute_granularity is None
            assert output.shape[0] == sequence_length
            assert output.shape[1] == micro_batch_size
            assert output.shape[2] == config.hidden_size
            assert bias.shape[0] == config.hidden_size

    def test_checkpointed_gpu_forward(self):
        if is_te_min_version("1.10.0"):
            # use flash attention for hopper, future may support fused attention for ampere
            os.environ['NVTE_FUSED_ATTN'] = "1"
            os.environ['NVTE_FLASH_ATTN'] = "0"

            transformer_config = self.transformer_config
            transformer_config.recompute_granularity = 'selective'
            checkpointed_parallel_attention = MLASelfAttention(
                transformer_config,
                get_gpt_layer_with_transformer_engine_spec(
                    multi_latent_attention=True
                ).submodules.self_attention.submodules,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
            )
            config = checkpointed_parallel_attention.config

            sequence_length = 32
            micro_batch_size = 2

            checkpointed_parallel_attention.cuda()

            # [sequence length, batch size, hidden size]
            hidden_states = torch.ones(
                (
                    sequence_length,
                    micro_batch_size,
                    checkpointed_parallel_attention.config.hidden_size,
                )
            )
            hidden_states = hidden_states.cuda()

            attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

            output, bias = checkpointed_parallel_attention(hidden_states, attention_mask)

            assert config.recompute_granularity == 'selective'
            assert output.shape[0] == sequence_length
            assert output.shape[1] == micro_batch_size
            assert output.shape[2] == config.hidden_size
            assert bias.shape[0] == config.hidden_size


class TestTensorParallelMLAAttention:

    def setup_method(self, method):
        self.tensor_parallel_size = 2
        Utils.initialize_model_parallel(self.tensor_parallel_size, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = MLATransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_head_dim=128,
            v_head_dim=128,
            qk_pos_emb_head_dim=64,
            rotary_base=10000,
            max_position_embeddings=64,
            tensor_model_parallel_size=self.tensor_parallel_size,
            sequence_parallel=True,
        )
        self.parallel_attention = MLASelfAttention(
            self.transformer_config,
            get_gpt_layer_with_transformer_engine_spec(
                multi_latent_attention=True
            ).submodules.self_attention.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_gpu_forward(self):
        if is_te_min_version("1.10.0"):
            # use flash attention for hopper, future may support fused attention for ampere
            os.environ['NVTE_FUSED_ATTN'] = "0"
            os.environ['NVTE_FLASH_ATTN'] = "1"

            config = self.parallel_attention.config
            sequence_length = 64
            sub_sequence_length = sequence_length // self.tensor_parallel_size
            micro_batch_size = 2

            self.parallel_attention.cuda()

            # [sequence length, batch size, hidden size]
            hidden_states = torch.ones(
                (sub_sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size)
            )
            hidden_states = hidden_states.cuda()

            attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

            output, bias = self.parallel_attention(hidden_states, attention_mask)

            assert config.recompute_granularity is None
            assert output.shape[0] == sub_sequence_length
            assert output.shape[1] == micro_batch_size
            assert output.shape[2] == config.hidden_size
            assert bias.shape[0] == config.hidden_size
