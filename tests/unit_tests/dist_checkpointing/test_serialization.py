# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest

import torch

from megatron.core.dist_checkpointing import ShardedTensor, save, load

from tests.unit_tests.dist_checkpointing import empty_dir, TempNamedDir
from tests.unit_tests.test_utilities import Utils

class TestSerialization:

    # def setup_method(self, method):
    #     Utils.initialize_model_parallel(1,1)
    #     transformer_config = TransformerConfig(num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True)
    #     self.gpt_embedding = GPTEmbedding(config=transformer_config, vocab_size=100, max_sequence_length=4, add_position_embedding=True)
    #
    # def teardown_method(self, method):
    #     Utils.destroy_model_parallel()
    
    def test_single_process_save_load(self, tmp_path_dist_ckpt):
        Utils.initialize_model_parallel(1,1)

        sharded_state_dict = {
            'sd_keyA': ShardedTensor.from_rank_offsets('keyA', torch.ones(2, 4), replica_id=Utils.rank),
            'sd_keyB': ShardedTensor.from_rank_offsets('keyB', torch.ones(3, 5, 7), replica_id=Utils.rank),
        }

        with TempNamedDir(tmp_path_dist_ckpt / 'test_single_process_save_load') as ckpt_dir:
            save(sharded_state_dict, ckpt_dir)

            assert (ckpt_dir / 'keyA').is_dir()
            assert (ckpt_dir / 'keyB').is_dir()
            assert not (ckpt_dir / 'keyC').exists()
            
            load_ssd = {
                'load_sd_keyA': ShardedTensor.from_rank_offsets('keyA', torch.ones(2, 4), replica_id=Utils.rank),
            }
            loaded_state_dict = load(load_ssd, ckpt_dir)
            
            assert set(loaded_state_dict.keys()) == {'load_sd_keyA'}
            assert isinstance(loaded_state_dict['load_sd_keyA'], torch.Tensor)
            assert loaded_state_dict['load_sd_keyA'].shape == (2, 4)

        Utils.destroy_model_parallel()


    def test_multi_process_save(self, tmp_path_dist_ckpt):
        Utils.initialize_model_parallel(2,4)

        state_dict = {
            'sd_keyA': ShardedTensor.from_rank_offsets('keyA', torch.ones(2, 4), (0, Utils.rank, Utils.world_size)),
            'sd_keyB': ShardedTensor.from_rank_offsets('keyB', torch.ones(3, 5, 7), (2, Utils.rank, Utils.world_size)),
        }

        with TempNamedDir(tmp_path_dist_ckpt / 'test_multi_process_save') as ckpt_dir:
            save(state_dict, ckpt_dir)

            assert (ckpt_dir / 'keyA').is_dir()
            assert (ckpt_dir / 'keyB').is_dir()
            assert not (ckpt_dir / 'keyC').exists()

        Utils.destroy_model_parallel()
