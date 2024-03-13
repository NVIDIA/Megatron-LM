# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest

import torch

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from tests.unit_tests.test_utilities import Utils
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec

class TestGPTModel:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1,1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True)
        self.gpt_model = GPTModel(config=transformer_config, transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(), vocab_size=100, max_sequence_length=4)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.gpt_model, GPTModel)

        assert self.gpt_model.max_sequence_length == 4

        num_weights = sum([p.numel() for p in self.gpt_model.parameters()])
        assert num_weights == 6240

    def test_set_input_tensor(self):
        config: TransformerConfig = self.gpt_model.config
        sequence_length = self.gpt_model.max_sequence_length
        micro_batch_size = 2

        # [sequence length, batch size, hidden size]
        input_tensor = torch.ones((sequence_length, micro_batch_size, config.hidden_size))

        self.gpt_model.set_input_tensor(input_tensor)

        assert self.gpt_model.decoder.input_tensor.shape[0] == sequence_length
        assert self.gpt_model.decoder.input_tensor.shape[1] == micro_batch_size
        assert self.gpt_model.decoder.input_tensor.shape[2] == config.hidden_size

    def test_post_process_forward(self):
        config: TransformerConfig = self.gpt_model.config
        sequence_length = self.gpt_model.max_sequence_length
        micro_batch_size = 2

        self.gpt_model.cuda()

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        logits = self.gpt_model.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == self.gpt_model.vocab_size

    def test_no_post_process_forward(self):
        pass

    def test_no_preprocess_forward(self):
        pass

    def test_state_dict_for_save_checkpoint(self):
        expected_state_dict_keys = ['embedding.word_embeddings.weight', 'embedding.position_embeddings.weight', 'decoder.layers.0.self_attention.linear_proj.weight', 'decoder.layers.0.self_attention.linear_proj.bias', 'decoder.layers.0.self_attention.linear_proj._extra_state', 'decoder.layers.0.self_attention.linear_qkv.layer_norm_weight', 'decoder.layers.0.self_attention.linear_qkv.layer_norm_bias', 'decoder.layers.0.self_attention.linear_qkv.weight', 'decoder.layers.0.self_attention.linear_qkv.bias', 'decoder.layers.0.self_attention.linear_qkv._extra_state', 'decoder.layers.0.mlp.linear_fc1.layer_norm_weight', 'decoder.layers.0.mlp.linear_fc1.layer_norm_bias', 'decoder.layers.0.mlp.linear_fc1.weight', 'decoder.layers.0.mlp.linear_fc1.bias', 'decoder.layers.0.mlp.linear_fc1._extra_state', 'decoder.layers.0.mlp.linear_fc2.weight', 'decoder.layers.0.mlp.linear_fc2.bias', 'decoder.layers.0.mlp.linear_fc2._extra_state', 'decoder.layers.1.self_attention.linear_proj.weight', 'decoder.layers.1.self_attention.linear_proj.bias', 'decoder.layers.1.self_attention.linear_proj._extra_state', 'decoder.layers.1.self_attention.linear_qkv.layer_norm_weight', 'decoder.layers.1.self_attention.linear_qkv.layer_norm_bias', 'decoder.layers.1.self_attention.linear_qkv.weight', 'decoder.layers.1.self_attention.linear_qkv.bias', 'decoder.layers.1.self_attention.linear_qkv._extra_state', 'decoder.layers.1.mlp.linear_fc1.layer_norm_weight', 'decoder.layers.1.mlp.linear_fc1.layer_norm_bias', 'decoder.layers.1.mlp.linear_fc1.weight', 'decoder.layers.1.mlp.linear_fc1.bias', 'decoder.layers.1.mlp.linear_fc1._extra_state', 'decoder.layers.1.mlp.linear_fc2.weight', 'decoder.layers.1.mlp.linear_fc2.bias', 'decoder.layers.1.mlp.linear_fc2._extra_state', 'decoder.final_layernorm.weight', 'decoder.final_layernorm.bias', 'output_layer.weight']
        actual_state_dict_keys = list(self.gpt_model.sharded_state_dict().keys())
        assert actual_state_dict_keys == expected_state_dict_keys, f"The actual and expected sharded state dict keys dont match. The actual keys are : {actual_state_dict_keys} while we expected {expected_state_dict_keys}"

    def test_load_state_dict(self):
        pass

