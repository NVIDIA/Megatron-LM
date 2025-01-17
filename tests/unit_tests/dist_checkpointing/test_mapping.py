# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.mapping import (
    ShardedObject,
    ShardedTensorFactory,
    apply_factories,
    apply_factory_merges,
    is_main_replica,
)
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
        rank_offsets = [(0, 0, 10), (2, 3, 6)]
        sh_ten = ShardedTensor.from_rank_offsets('keyA', data, *rank_offsets)

        assert isinstance(sh_ten, ShardedTensor)
        assert sh_ten.dtype is dtype
        assert sh_ten.local_shape == shape
        assert sh_ten.global_shape == (shape[0] * 10, shape[1], shape[2] * 6, shape[3])
        assert sh_ten.global_offset == (0, 0, shape[2] * 3, 0)
        assert sh_ten.axis_fragmentations == (10, 1, 6, 1)

    def test_from_rank_offsets_flat_constructor(self, dtype=torch.float, device='cuda'):
        data = torch.arange(28, dtype=dtype, device=device).reshape((1, 4, 7))
        shape = data.shape
        rank_offsets = [(1, 0, 2), (2, 3, 5)]
        flattened_range = slice(4, 9)
        flat_data = data.flatten()[flattened_range]
        sh_ten = ShardedTensor.from_rank_offsets_flat(
            'keyA', flat_data, data.shape, *rank_offsets, flattened_range=flattened_range
        )

        # The main attributes properties are unchanged
        assert isinstance(sh_ten, ShardedTensor)
        assert sh_ten.dtype is dtype
        assert sh_ten.local_shape == shape
        assert sh_ten.global_shape == (shape[0], shape[1] * 2, shape[2] * 5)
        assert sh_ten.global_offset == (0, 0, shape[2] * 3)
        assert sh_ten.axis_fragmentations == (1, 2, 5)

        assert torch.all(sh_ten.data == torch.arange(4, 9, device=device))

    def test_metadata_integrity_violation(self):
        data = torch.ones((1, 3, 7, 9), device='meta')
        rank_offsets = [(0, 0, 10), (2, 3, 6)]
        sh_ten = ShardedTensor.from_rank_offsets('keyA', data, *rank_offsets)
        sh_ten.validate_metadata_integrity()
        with pytest.raises(CheckpointingException):
            sh_ten.local_shape = (1, 2, 7, 9)
            sh_ten.validate_metadata_integrity()

        sh_ten = ShardedTensor.from_rank_offsets('keyA', data, *rank_offsets)
        with pytest.raises(CheckpointingException):
            sh_ten.global_offset = (0, 1, 0)
            sh_ten.validate_metadata_integrity()

        with pytest.raises(CheckpointingException):
            sh_ten = ShardedTensor.from_rank_offsets_flat(
                'keyA', data, data.shape, *rank_offsets, flattened_range=slice(4, 9)
            )

        sh_ten = ShardedTensor.from_rank_offsets_flat(
            'keyA', data.flatten()[4:9], data.shape, *rank_offsets, flattened_range=slice(4, 9)
        )
        assert sh_ten.local_shape == (1, 3, 7, 9)
        with pytest.raises(CheckpointingException):
            sh_ten.local_shape = (5,)
            sh_ten.validate_metadata_integrity()

    def test_narrowing(self):
        data = torch.ones((1, 3, 7, 9))
        rank_offsets = [(0, 0, 10), (2, 3, 6)]
        sh_ten = ShardedTensor.from_rank_offsets('keyA', data, *rank_offsets)
        (narr_sh_ten,) = sh_ten.narrow(1, 1, 2)
        assert narr_sh_ten.local_shape == (1, 2, 7, 9)
        assert narr_sh_ten.global_shape == (10, 2, 42, 9)
        assert narr_sh_ten.global_offset == (0, 0, 21, 0)

        (narr_sh_ten,) = sh_ten.narrow(2, 3, 2)
        assert narr_sh_ten.local_shape == (1, 3, 2, 9)
        assert narr_sh_ten.global_shape == (10, 3, 12, 9)
        assert narr_sh_ten.global_offset == (0, 0, 6, 0)

    def test_flat_narrow(self):
        data = torch.arange(28).reshape((4, 7))
        rank_offsets = [(0, 1, 2), (1, 3, 5)]
        flattened_range = slice(4, 9)
        flat_data = data.flatten()[flattened_range]
        sh_ten = ShardedTensor.from_rank_offsets_flat(
            'keyA', flat_data, data.shape, *rank_offsets, flattened_range=flattened_range
        )

        # The main attributes properties are unchanged
        assert isinstance(sh_ten, ShardedTensor)
        assert torch.all(sh_ten.data == torch.arange(4, 9))

        (narrow_sh_ten,) = sh_ten.narrow(
            0, 0, 1
        )  # First seven elements of unflat, intersection has 3 elements
        assert torch.all(narrow_sh_ten.data == torch.arange(4, 7))
        assert narrow_sh_ten.local_shape == (1, 7)
        assert narrow_sh_ten.global_shape == (2, 35)
        assert narrow_sh_ten.global_offset == (1, 21)

        (narrow_sh_ten,) = sh_ten.narrow(
            0, 0, 3
        )  # First 21 elements of unflat, intersection has all 5 elements
        assert torch.all(narrow_sh_ten.data == torch.arange(4, 9))
        assert narrow_sh_ten.local_shape == (3, 7)
        assert narrow_sh_ten.global_shape == (6, 35)
        assert narrow_sh_ten.global_offset == (3, 21)

        narrow_sh_ten = sh_ten.narrow(0, 2, 1)  # empty intersection
        assert not narrow_sh_ten, narrow_sh_ten


class TestShardedTensorFactory:
    def test_build_and_merge(self):
        def build_fn(key, tensor, replica_id, flattened_range):
            assert flattened_range is None
            return {
                'level2_a': ShardedTensor.from_rank_offsets(
                    key + 'part1', tensor + 1, replica_id=replica_id
                ),
                'level2_b': ShardedTensor.from_rank_offsets(
                    key + 'part2', tensor + 2, replica_id=replica_id
                ),
            }

        # state_dict will be modified in-place
        def get_state_dict():
            return {
                'level1': ShardedTensorFactory(
                    'a', torch.arange(3), build_fn, lambda x: x['level2_b']
                )
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
