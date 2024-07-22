# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from megatron.core.device_utils import get_current_device
import pytest

import torch
import os

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.bert.bert_model import BertModel
from tests.unit_tests.test_utilities import Utils
from megatron.core.tensor_parallel.random import model_parallel_device_manual_seed
from megatron.core.models.bert.bert_layer_specs import bert_layer_with_transformer_engine_spec

class TestBertModel:

    def setup_method(self, method):
        os.environ['NVTE_ALLOW_NONDETERMINISTIC_ALGO'] = '0' #Bert does not support flash attention
        tp = 1
        pp = 1
        Utils.initialize_model_parallel(tp, pp)
        model_parallel_device_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4,
            use_cpu_initialization=True, perform_initialization=True,
            tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp, pipeline_dtype=torch.bfloat16
        )
        self.bert_model = BertModel(
            config=transformer_config, num_tokentypes=0,
            transformer_layer_spec=bert_layer_with_transformer_engine_spec, vocab_size=100, max_sequence_length=4
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.bert_model, BertModel)

        assert self.bert_model.max_sequence_length == 4

        num_weights = sum([p.numel() for p in self.bert_model.parameters()])
        assert num_weights == 6702

    def test_set_input_tensor(self):
        config: TransformerConfig = self.bert_model.config
        sequence_length = self.bert_model.max_sequence_length
        micro_batch_size = 2

        # [sequence length, batch size, hidden size]
        input_tensor = torch.ones((sequence_length, micro_batch_size, config.hidden_size))

        self.bert_model.set_input_tensor(input_tensor)

        assert self.bert_model.encoder.input_tensor.shape[0] == sequence_length
        assert self.bert_model.encoder.input_tensor.shape[1] == micro_batch_size
        assert self.bert_model.encoder.input_tensor.shape[2] == config.hidden_size

    def test_post_process_forward(self):
        config: TransformerConfig = self.bert_model.config
        sequence_length = self.bert_model.max_sequence_length
        micro_batch_size = 2

        self.bert_model.to(device=get_current_device())

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).to(device=get_current_device())
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).to(device=get_current_device())
        attention_mask = torch.ones((micro_batch_size, sequence_length), dtype=bool).to(device=get_current_device())

        logits = self.bert_model.forward(input_ids=input_ids, attention_mask=attention_mask)

        assert logits[0].shape[0] == micro_batch_size
        assert logits[0].shape[1] == sequence_length
        assert logits[0].shape[2] == self.bert_model.vocab_size

    def test_no_post_process_forward(self):
        pass

    def test_no_preprocess_forward(self):
        pass

    def test_state_dict_for_save_checkpoint(self):
        pass

    def test_load_state_dict(self):
        pass

