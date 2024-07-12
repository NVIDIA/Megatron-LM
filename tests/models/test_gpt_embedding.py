# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest

import torch
import types

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_embedding import GPTEmbedding
from megatron.global_vars import set_args

from deepspeed.accelerator import get_accelerator
device_name = get_accelerator().device_name()

@pytest.fixture
def gpt_embedding(transformer_config):
    args = types.SimpleNamespace(params_dtype=torch.float32, embed_layernorm=False)
    set_args(args)
    embedding = GPTEmbedding(config=transformer_config, vocab_size=100, max_sequence_length=4)
    return embedding


class TestGPTEmbedding:
    def test_constructor(self, gpt_embedding: GPTEmbedding):
        assert isinstance(gpt_embedding, GPTEmbedding)
        num_weights = sum([p.numel() for p in gpt_embedding.parameters()])
        assert num_weights == 1248

    def test_zero_parameters(self, gpt_embedding: GPTEmbedding):
        sum_weights = sum([p.sum() for p in gpt_embedding.parameters()])
        assert sum_weights != 0
        gpt_embedding.zero_parameters()
        sum_weights = sum([p.sum() for p in gpt_embedding.parameters()])
        assert sum_weights == 0

    def test_cpu_forward(self, gpt_embedding: GPTEmbedding):
        input_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64).repeat((2, 1))
        position_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64).repeat((2, 1))
        embeddings = gpt_embedding(input_ids, position_ids)
        assert embeddings.device.type == 'cpu'
        assert embeddings.shape[0] == gpt_embedding.max_sequence_length
        assert embeddings.shape[1] == input_ids.shape[0]
        assert embeddings.shape[2] == gpt_embedding.config.hidden_size

    def test_accelerator_forward(self, gpt_embedding: GPTEmbedding):
        gpt_embedding.to(device_name)
        input_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64).repeat((2, 1)).to(device_name)
        position_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64).repeat((2, 1)).to(device_name)
        embeddings = gpt_embedding(input_ids, position_ids)
        assert embeddings.device.type == device_name
        assert embeddings.shape[0] == gpt_embedding.max_sequence_length
        assert embeddings.shape[1] == input_ids.shape[0]
        assert embeddings.shape[2] == gpt_embedding.config.hidden_size
