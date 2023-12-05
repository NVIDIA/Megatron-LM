# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


import pytest

import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from tests.unit_tests.test_utilities import Utils


class TestParallelTransformerLayer:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1,1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True)
        self.parallel_transformer_layer = TransformerLayer(transformer_config,
                                                           get_gpt_layer_with_transformer_engine_spec().submodules)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        parallel_transformer_layer = self.parallel_transformer_layer
        assert isinstance(parallel_transformer_layer, TransformerLayer)
        assert parallel_transformer_layer.layer_number == 1

        num_weights = sum([p.numel() for p in parallel_transformer_layer.parameters()])
        assert num_weights == 1884

    def test_gpu_forward(self):
        parallel_transformer_layer = self.parallel_transformer_layer
        config: TransformerConfig = parallel_transformer_layer.config
        sequence_length = 32
        micro_batch_size = 2
        parallel_transformer_layer.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((sequence_length, micro_batch_size, config.hidden_size))
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        hidden_states, context = parallel_transformer_layer(hidden_states=hidden_states, attention_mask=attention_mask)
        assert hidden_states.shape[0] == sequence_length
        assert hidden_states.shape[1] == micro_batch_size
        assert hidden_states.shape[2] == config.hidden_size

    @pytest.mark.parametrize('tp_pp', [(4, 2), (1, 1), (8, 1), (2, 2)])
    def test_sharded_state_dict(self, tp_pp):
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(*tp_pp)

        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=8, use_cpu_initialization=True)
        parallel_transformer_layer = TransformerLayer(transformer_config,
                                                      get_gpt_layer_with_transformer_engine_spec().submodules)

        sharded_state_dict = parallel_transformer_layer.sharded_state_dict()

        extra_states = {k: v for k, v in sharded_state_dict.items() if k.endswith('extra_state')}
        sharded_tensors = {k: v for k, v in sharded_state_dict.items() if not k.endswith('extra_state')}
        assert all(isinstance(t, ShardedObject) for t in extra_states.values())
        assert all(isinstance(t, ShardedTensor) for t in sharded_tensors.values())

        # Test all local shapes
        tensor_local_shapes = {k: v.local_shape for k, v in sharded_tensors.items()}
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        assert tensor_local_shapes == get_tensor_shapes_for_tp(transformer_config, tp_size)

        # Test all global shapes. Prepend num layers in front of expected shapes
        tensor_global_shapes = {k: v.global_shape for k, v in sharded_tensors.items()}
        expected_global_shapes = {k: (transformer_config.num_layers, *v)
                                  for k, v in get_tensor_shapes_for_tp(transformer_config, 1).items()}
        assert tensor_global_shapes == expected_global_shapes

        # Test ShardedTensor keys
        for state_dict_key, sh_ten in sharded_tensors.items():
            assert state_dict_key == f'0.{sh_ten.key}'

        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(1, 1)


def get_tensor_shapes_for_tp(transformer_config, tp_size):
    hs = transformer_config.hidden_size
    return {
        '0.mlp.linear_fc1.layer_norm_weight': (hs,),
        '0.mlp.linear_fc1.layer_norm_bias': (hs,),
        '0.mlp.linear_fc1.weight': (hs * 4 // tp_size, hs),
        '0.mlp.linear_fc1.bias': (hs * 4 // tp_size,),
        '0.mlp.linear_fc2.weight': (hs, hs * 4 // tp_size),
        '0.mlp.linear_fc2.bias': (hs,),
        '0.self_attention.linear_proj.weight': (hs, hs // tp_size),
        '0.self_attention.linear_proj.bias': (hs,),
        '0.self_attention.linear_qkv.layer_norm_weight': (hs,),
        '0.self_attention.linear_qkv.layer_norm_bias': (hs,),
        '0.self_attention.linear_qkv.weight': (hs * 3 // tp_size, hs),
        '0.self_attention.linear_qkv.bias': (hs * 3 // tp_size,),
    }
