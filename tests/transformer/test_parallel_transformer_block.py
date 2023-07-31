# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest

import torch

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.parallel_transformer_layer import ParallelTransformerLayer
from megatron.core.transformer.parallel_transformer_block import ParallelTransformerBlock


@pytest.fixture
def parallel_transformer_block(transformer_config):
    return ParallelTransformerBlock(transformer_config)


class TestParallelTransformerBlock:
    def test_constructor(self, parallel_transformer_block: ParallelTransformerBlock):
        assert isinstance(parallel_transformer_block, ParallelTransformerBlock)
        num_weights = sum([p.numel() for p in parallel_transformer_block.parameters()])
        assert num_weights == 3792
        assert parallel_transformer_block.num_layers_per_pipeline_rank == 2
        assert len(parallel_transformer_block.layers) == 2
        layer_0: ParallelTransformerLayer = parallel_transformer_block._get_layer(0)
        assert layer_0.layer_number == 1
        layer_1: ParallelTransformerLayer = parallel_transformer_block._get_layer(1)
        assert layer_1.layer_number == 2

    def test_gpu_forward(self, parallel_transformer_block: ParallelTransformerBlock):
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

    def test_gpu_forward_full_checkpoint(self, transformer_config: TransformerConfig):
        config = transformer_config
        config.recompute_granularity = 'full'
        config.recompute_method = 'block'
        config.recompute_num_layers = config.num_layers
        full_transformer_block = ParallelTransformerBlock(config)
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

    def test_gpu_forward_selective_checkpoint(self, transformer_config: TransformerConfig):
        config = transformer_config
        config.recompute_granularity = 'selective'
        selective_transformer_block = ParallelTransformerBlock(config)
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
