# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest

import torch
import types

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.global_vars import set_args

from deepspeed.accelerator import get_accelerator
device_name = get_accelerator().device_name()

@pytest.fixture
def gpt_model(transformer_config):
    args = types.SimpleNamespace(params_dtype=torch.float32, embed_layernorm=False)
    set_args(args)
    language_model = GPTModel(config=transformer_config, vocab_size=100, max_sequence_length=4)
    return language_model


class TestGPTModel:
    @pytest.mark.xfail(device_name=='hpu', reason="TELayerNorm is not defined in HPU")
    def test_constructor(self, gpt_model: GPTModel):
        assert isinstance(gpt_model, GPTModel)

        assert gpt_model.max_sequence_length == 4

        num_weights = sum([p.numel() for p in gpt_model.parameters()])
        assert num_weights == 5040

    @pytest.mark.xfail(device_name=='hpu', reason="TELayerNorm is not defined in HPU")
    def test_set_input_tensor(self, gpt_model: GPTModel):
        config: TransformerConfig = gpt_model.config
        sequence_length = gpt_model.max_sequence_length
        micro_batch_size = 2

        # [sequence length, batch size, hidden size]
        input_tensor = torch.ones((sequence_length, micro_batch_size, config.hidden_size))

        gpt_model.set_input_tensor(input_tensor)

        assert gpt_model.decoder.input_tensor.shape[0] == sequence_length
        assert gpt_model.decoder.input_tensor.shape[1] == micro_batch_size
        assert gpt_model.decoder.input_tensor.shape[2] == config.hidden_size

    @pytest.mark.xfail(device_name=='hpu', reason="TELayerNorm is not defined in HPU")
    def test_post_process_forward(self, gpt_model: GPTModel):
        config: TransformerConfig = gpt_model.config
        sequence_length = gpt_model.max_sequence_length
        micro_batch_size = 2

        gpt_model.to(device_name)

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).to(device_name)
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).to(device_name)
        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).to(device_name)

        logits = gpt_model.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == gpt_model.vocab_size

    @pytest.mark.xfail(device_name=='hpu', reason="TELayerNorm is not defined in HPU")
    def test_no_post_process_forward(self, gpt_model: GPTModel):
        pass

    @pytest.mark.xfail(device_name=='hpu', reason="TELayerNorm is not defined in HPU")
    def test_no_preprocess_forward(self, gpt_model: GPTModel):
        pass

    @pytest.mark.xfail(device_name=='hpu', reason="TELayerNorm is not defined in HPU")
    def test_state_dict_for_save_checkpoint(self, gpt_model: GPTModel):
        pass

    @pytest.mark.xfail(device_name=='hpu', reason="TELayerNorm is not defined in HPU")
    def test_load_state_dict(self, gpt_model: GPTModel):
        pass

