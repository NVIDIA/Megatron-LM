# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


import pytest

import torch

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.parallel_transformer_layer import ParallelTransformerLayer


@pytest.fixture
def parallel_transformer_layer(transformer_config):
    return ParallelTransformerLayer(transformer_config)


class TestParallelTransformerLayer:
    def test_constructor(self, parallel_transformer_layer):
        assert isinstance(parallel_transformer_layer, ParallelTransformerLayer)
        assert parallel_transformer_layer.layer_number == 1

        num_weights = sum([p.numel() for p in parallel_transformer_layer.parameters()])
        assert num_weights == 1884

    def test_gpu_forward(self, parallel_transformer_layer):
        config: TransformerConfig = parallel_transformer_layer.config
        sequence_length = 32
        micro_batch_size = 2
        parallel_transformer_layer.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((sequence_length, micro_batch_size, config.hidden_size))
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        hidden_states = parallel_transformer_layer(hidden_states=hidden_states, attention_mask=attention_mask)
        assert hidden_states.shape[0] == sequence_length
        assert hidden_states.shape[1] == micro_batch_size
        assert hidden_states.shape[2] == config.hidden_size
