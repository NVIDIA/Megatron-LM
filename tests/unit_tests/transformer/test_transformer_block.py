# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import os
import pytest

import torch
from megatron.core import dist_checkpointing

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.transformer_block import TransformerBlock
from tests.unit_tests.test_utilities import Utils
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec

class TestParallelTransformerBlock:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1,1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True)
        self.parallel_transformer_block = TransformerBlock(self.transformer_config,
                                                           get_gpt_layer_with_transformer_engine_spec())

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        parallel_transformer_block = self.parallel_transformer_block
        assert isinstance(parallel_transformer_block, TransformerBlock)
        num_weights = sum([p.numel() for p in parallel_transformer_block.parameters()])
        assert num_weights == 3792
        assert parallel_transformer_block.num_layers_per_pipeline_rank == 2
        assert len(parallel_transformer_block.layers) == 2
        layer_0: TransformerLayer = parallel_transformer_block._get_layer(0)
        assert layer_0.layer_number == 1
        layer_1: TransformerLayer = parallel_transformer_block._get_layer(1)
        assert layer_1.layer_number == 2

    def test_gpu_forward(self):
        parallel_transformer_block = self.parallel_transformer_block
        config: TransformerConfig = parallel_transformer_block.config

        sequence_length = 32
        micro_batch_size = 2
        parallel_transformer_block.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((sequence_length, micro_batch_size, config.hidden_size))
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        hidden_states = parallel_transformer_block(hidden_states=hidden_states, attention_mask=attention_mask)
        assert hidden_states.shape[0] == sequence_length
        assert hidden_states.shape[1] == micro_batch_size
        assert hidden_states.shape[2] == config.hidden_size

    def test_gpu_forward_full_checkpoint(self):
        transformer_config = self.transformer_config
        config = transformer_config
        config.recompute_granularity = 'full'
        config.recompute_method = 'block'
        config.recompute_num_layers = config.num_layers
        full_transformer_block = TransformerBlock(config,
                                                  get_gpt_layer_with_transformer_engine_spec())
        assert full_transformer_block.config.recompute_granularity == 'full'
        assert full_transformer_block.config.recompute_method == 'block'

        sequence_length = 32
        micro_batch_size = 2
        full_transformer_block.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((sequence_length, micro_batch_size, config.hidden_size))
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        hidden_states = full_transformer_block(hidden_states=hidden_states, attention_mask=attention_mask)
        assert hidden_states.shape[0] == sequence_length
        assert hidden_states.shape[1] == micro_batch_size
        assert hidden_states.shape[2] == config.hidden_size

    def test_gpu_forward_selective_checkpoint(self):
        transformer_config = self.transformer_config
        config = transformer_config
        config.recompute_granularity = 'selective'
        selective_transformer_block = TransformerBlock(config,
                                                       get_gpt_layer_with_transformer_engine_spec())
        assert selective_transformer_block.config.recompute_granularity == 'selective'
        assert selective_transformer_block.checkpoint_core_attention

        sequence_length = 32
        micro_batch_size = 2
        selective_transformer_block.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((sequence_length, micro_batch_size, config.hidden_size))
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        hidden_states = selective_transformer_block(hidden_states=hidden_states, attention_mask=attention_mask)
        assert hidden_states.shape[0] == sequence_length
        assert hidden_states.shape[1] == micro_batch_size
        assert hidden_states.shape[2] == config.hidden_size
