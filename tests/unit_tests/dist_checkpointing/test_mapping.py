# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest

import torch

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import is_main_replica, \
    ShardedTensorFactory, ShardedObject, apply_factories, apply_factory_merges
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

class TestShardedTensor:

    # def setup_method(self, method):
    #     Utils.initialize_model_parallel(1,1)
    #     transformer_config = TransformerConfig(num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True)
    #     self.gpt_embedding = GPTEmbedding(config=transformer_config, vocab_size=100, max_sequence_length=4, add_position_embedding=True)
    #
    # def teardown_method(self, method):
    #     Utils.destroy_model_parallel()
    
    def test_from_rank_offsets_constructor(self, dtype=torch.float, device='cuda'):
        data = torch.ones((1, 3, 7, 9), dtype=dtype, device=device)
        shape = data.shape
        rank_offsets = [
            (0, 0, 10),
            (2, 3, 6)
        ]
        sh_ten = ShardedTensor.from_rank_offsets('keyA', data, *rank_offsets)

        assert isinstance(sh_ten, ShardedTensor)
        assert sh_ten.dtype is dtype
        assert sh_ten.local_shape == shape
        assert sh_ten.global_shape == (shape[0] * 10, shape[1], shape[2] * 6, shape[3])
        assert sh_ten.global_offset == (0, 0, shape[2] * 3, 0)
        assert sh_ten.axis_fragmentations == (10, 1, 6, 1)

class TestShardedTensorFactory:
    def test_build_and_merge(self):
        def build_fn(key, tensor):
            return {
                'level2_a': ShardedTensor.from_rank_offsets(key + 'part1', tensor + 1),
                'level2_b': ShardedTensor.from_rank_offsets(key + 'part2', tensor + 2)
            }

        # state_dict will be modified in-place
        def get_state_dict():
            return {
                'level1': ShardedTensorFactory('a', torch.arange(3), build_fn, lambda x: x['level2_b'])
            }
        state_dict = get_state_dict()
        apply_factories(state_dict)
        assert torch.allclose(state_dict['level1']['level2_a'].data, torch.tensor([1, 2, 3]))
        assert torch.allclose(state_dict['level1']['level2_b'].data, torch.tensor([2, 3, 4]))

        # Simulate loading
        state_dict['level1']['level2_a'] = state_dict['level1']['level2_a'].data
        state_dict['level1']['level2_b'] = state_dict['level1']['level2_b'].data

        loaded_state_dict = apply_factory_merges(state_dict, get_state_dict())
        assert torch.allclose(loaded_state_dict['level1'], torch.tensor([2, 3, 4]))


def test_is_main_replica():
    assert is_main_replica(0)
    assert is_main_replica((0,))
    assert is_main_replica((0, 0))
    assert not is_main_replica(1)
    assert not is_main_replica(2)
    assert not is_main_replica((1,))
    assert not is_main_replica((1, 0))
    assert not is_main_replica((1, 1, 1))
