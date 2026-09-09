# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


import pytest
import torch

from megatron.core.transformer.attention import CrossAttention

""" 

@pytest.fixture
def core_attention(transformer_config):
    return CrossAttention(transformer_config)


class TestCoreAttention:
    def test_constructor(self, core_attention):
        assert isinstance(core_attention, CrossAttention)
        assert core_attention.layer_number == 1

        num_weights = sum([p.numel() for p in core_attention.parameters()])
        assert num_weights == 0

    def test_cpu_forward(self, core_attention):
        # we can't currently do this because the global memory buffer is on GPU
        pass

    def test_gpu_forward(self, core_attention):

        # destroy_global_memory_buffer()
        # _set_global_memory_buffer()
        # model_parallel_cuda_manual_seed(123)

        core_attention.cuda()
        config = core_attention.config
        sequence_length = 32
        micro_batch_size = 2
        # query_layer (float): [sequence_length, micro_batch_size, num_attention_heads, hidden_size / num_attention_heads]
        query_layer = torch.ones(
            (
                sequence_length,
                micro_batch_size,
                config.num_attention_heads,
                config.hidden_size // config.num_attention_heads,
            )
        ).cuda()

        key_layer = torch.ones_like(query_layer).cuda()

        value_layer = torch.ones_like(query_layer).cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        context_layer = core_attention(
            query_layer=query_layer, key_layer=key_layer, value_layer=value_layer, attention_mask=attention_mask
        )

        assert context_layer.shape[0] == sequence_length
        assert context_layer.shape[1] == micro_batch_size
        assert context_layer.shape[2] == config.hidden_size
        assert context_layer.device.type == 'cuda'
        assert context_layer.dtype == torch.float32

"""
