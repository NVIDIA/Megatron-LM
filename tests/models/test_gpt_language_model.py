# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest

import torch

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_language_model import GPTLanguageModel


@pytest.fixture
def gpt_language_model(transformer_config):
    language_model = GPTLanguageModel(config=transformer_config, vocab_size=100, max_sequence_length=4)
    return language_model


class TestGPTLanguageModel:
    def test_constructor(self, gpt_language_model: GPTLanguageModel):
        assert isinstance(gpt_language_model, GPTLanguageModel)

        assert gpt_language_model.max_sequence_length == 4

        num_weights = sum([p.numel() for p in gpt_language_model.parameters()])
        assert num_weights == 5040

    def test_set_input_tensor(self, gpt_language_model: GPTLanguageModel):
        config: TransformerConfig = gpt_language_model.config
        sequence_length = gpt_language_model.max_sequence_length
        micro_batch_size = 2

        # [sequence length, batch size, hidden size]
        input_tensor = torch.ones((sequence_length, micro_batch_size, config.hidden_size))

        gpt_language_model.set_input_tensor(input_tensor)

        assert gpt_language_model.encoder.input_tensor.shape[0] == sequence_length
        assert gpt_language_model.encoder.input_tensor.shape[1] == micro_batch_size
        assert gpt_language_model.encoder.input_tensor.shape[2] == config.hidden_size

    def test_gpu_forward(self, gpt_language_model: GPTLanguageModel):
        config: TransformerConfig = gpt_language_model.config
        sequence_length = gpt_language_model.max_sequence_length
        micro_batch_size = 2

        gpt_language_model.cuda()

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        hidden_states = gpt_language_model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        assert hidden_states.shape[0] == sequence_length
        assert hidden_states.shape[1] == micro_batch_size
        assert hidden_states.shape[2] == config.hidden_size

    def test_state_dict_for_save_checkpoint(self, gpt_language_model: GPTLanguageModel):
        pass

    def test_load_state_dict(self, gpt_language_model: GPTLanguageModel):
        pass

