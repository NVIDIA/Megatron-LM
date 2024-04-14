# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import io

import numpy as np
import pytest
import torch
from torch.distributed.checkpoint import CheckpointException

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import ShardedTensor, save, load
from megatron.core.dist_checkpointing.core import CheckpointingException, \
    maybe_load_config
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.dist_checkpointing.mapping import ShardedTensorFactory, \
    ShardedObject
from megatron.core.dist_checkpointing.serialization import load_tensors_metadata

from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


class TestSerialization:
    def test_single_process_save_load(self, tmp_path_dist_ckpt):
        Utils.initialize_model_parallel(1,1)

        sharded_state_dict = {
            'sd_keyA': ShardedTensor.from_rank_offsets('keyA', torch.ones(2, 4), replica_id=Utils.rank),
            'sd_keyB': ShardedTensor.from_rank_offsets('keyB', torch.ones(3, 5, 7), replica_id=Utils.rank),
        }

        with TempNamedDir(tmp_path_dist_ckpt / 'test_single_process_save_load') as ckpt_dir:
            save(sharded_state_dict, ckpt_dir)
            torch.distributed.barrier()

            saved_config = maybe_load_config(ckpt_dir)
            if saved_config.sharded_backend == 'zarr':
                assert (ckpt_dir / 'keyA').is_dir()
                assert (ckpt_dir / 'keyB').is_dir()
                assert not (ckpt_dir / 'keyC').exists()
                assert not (ckpt_dir / 'sd_keyA').is_dir()

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

            saved_config = maybe_load_config(ckpt_dir)
            if saved_config.sharded_backend == 'zarr':
                assert (ckpt_dir / 'keyA').is_dir()
                assert (ckpt_dir / 'keyB').is_dir()
                assert not (ckpt_dir / 'keyC').exists()
                assert not (ckpt_dir / 'sd_keyA').is_dir()

        Utils.destroy_model_parallel()


    def test_partition_change_save_load(self, tmp_path_dist_ckpt, strategy=None):
        Utils.initialize_model_parallel(2,4)

        # ten_a: global shape (2, 4):
        ten_a_global = torch.tensor([[0, 1, 2, 3], [10, 11, 12, 13]])
        ten_a = torch.zeros(1, 1) + 10 * parallel_state.get_tensor_model_parallel_rank() + parallel_state.get_pipeline_model_parallel_rank()
        assert ten_a.shape == (1, 1)

        # ten_b: global shape (4, 5, 80), where (x, y, z) is (100x + z)
        ten_b = torch.zeros(4, 5, 10) + (torch.arange(10) + 10 * Utils.rank)
        ten_b += torch.arange(4).unsqueeze(-1).unsqueeze(-1) * 100
        assert ten_b.shape == (4, 5, 10)

        state_dict = {
            'sd_keyA': ShardedTensor.from_rank_offsets('keyA', ten_a,
                                                       (0, parallel_state.get_tensor_model_parallel_rank(), parallel_state.get_tensor_model_parallel_world_size()),
                                                       (1, parallel_state.get_pipeline_model_parallel_rank(), parallel_state.get_pipeline_model_parallel_world_size()),
                                                       replica_id=0),
            'sd_keyB': ShardedTensor.from_rank_offsets('keyB', ten_b, (2, Utils.rank, Utils.world_size)),
        }

        ten_a_global_shape = ten_a_global.shape
        ten_b_global_shape = (4, 5, 10 * 8)

        assert state_dict['sd_keyA'].local_shape == (1, 1)
        assert state_dict['sd_keyA'].global_shape == ten_a_global_shape
        assert state_dict['sd_keyB'].global_shape == ten_b_global_shape

        with TempNamedDir(tmp_path_dist_ckpt / 'test_partition_change_save_load') as ckpt_dir:
            save(state_dict, ckpt_dir, strategy)

            del ten_a, ten_b

            # without changing TPxPP, load tensors without any sharding
            load_sd = {
                'sd_keyA': ShardedTensor.from_rank_offsets('keyA',
                                                           torch.empty(ten_a_global_shape),
                                                           replica_id=Utils.rank),
                'sd_keyB': ShardedTensor.from_rank_offsets('keyB',
                                                           torch.empty(ten_b_global_shape),
                                                           replica_id=Utils.rank),
            }
            loaded_state_dict = load(load_sd, ckpt_dir)

            ten_a = loaded_state_dict['sd_keyA']
            ten_b = loaded_state_dict['sd_keyB']
            assert isinstance(ten_a, torch.Tensor)
            assert ten_a.shape == ten_a_global_shape
            assert torch.all(ten_a == ten_a_global)

            assert isinstance(ten_b, torch.Tensor)
            assert ten_b.shape == ten_b_global_shape
            assert np.all([
                val == 100 * x + z
                for x, x_row in enumerate(ten_b)
                for y, y_row in enumerate(x_row)
                for z, val in enumerate(y_row)
            ])

            del ten_a, ten_b

            # change TPxPP
            Utils.destroy_model_parallel()
            Utils.initialize_model_parallel(1,2)

            load_sd = {
                'sd_keyA': ShardedTensor.from_rank_offsets('keyA', torch.empty(2, 1),
                                                           (1, parallel_state.get_data_parallel_rank(), parallel_state.get_data_parallel_world_size()),
                                                           replica_id=parallel_state.get_pipeline_model_parallel_rank()),
                'sd_keyB': ShardedTensor.from_rank_offsets('keyB', torch.empty(5, 80),
                                                           (0, Utils.rank // 2, 4),
                                                           prepend_axis_num=1,
                                                           replica_id=Utils.rank % 2),
            }

            loaded_state_dict = load(load_sd, ckpt_dir)
            ten_a = loaded_state_dict['sd_keyA']
            ten_b = loaded_state_dict['sd_keyB']

            assert isinstance(ten_a, torch.Tensor)
            assert ten_a.shape == (2, 1)
            assert torch.all(ten_a[:, 0] == ten_a_global[:, parallel_state.get_data_parallel_rank()])

            assert isinstance(ten_b, torch.Tensor)
            assert ten_b.shape == (5, 10 * 8)
            assert torch.all(ten_b == torch.arange(80).unsqueeze(0).expand(5, 80) + Utils.rank // 2 * 100)

    def test_load_tensors_metadata(self, tmp_path_dist_ckpt):
        Utils.initialize_model_parallel(2,4)

        state_dict = {
            'sd_keyA': ShardedTensor.from_rank_offsets('keyA', torch.arange(10) + Utils.rank * 10, (0, Utils.rank, Utils.world_size)),
            'sd_keyB': ShardedTensor.from_rank_offsets('keyB', torch.ones(3, 5, 7), (2, Utils.rank, Utils.world_size)),
        }

        with TempNamedDir(tmp_path_dist_ckpt / 'test_load_tensors_metadata') as ckpt_dir:
            save(state_dict, ckpt_dir)

            del state_dict
            sharded_state_dict = load_tensors_metadata(ckpt_dir)
            # loaded dict keys are ShardedTensor keys!
            assert 'keyA' in sharded_state_dict
            assert 'sd_keyA' not in sharded_state_dict

            # Check metadata
            assert sharded_state_dict['keyA'].global_shape == (10 * Utils.world_size,)
            assert sharded_state_dict['keyB'].global_shape == (3, 5, 7 * Utils.world_size)
            assert sharded_state_dict['keyA'].local_shape == sharded_state_dict['keyA'].global_shape
            assert sharded_state_dict['keyB'].local_shape == sharded_state_dict['keyB'].global_shape
            assert sharded_state_dict['keyA'].global_offset == (0,)
            assert sharded_state_dict['keyB'].global_offset == (0, 0, 0)
            assert sharded_state_dict['keyA'].axis_fragmentations == (1,)
            assert sharded_state_dict['keyB'].axis_fragmentations == (1, 1, 1)
            assert sharded_state_dict['keyA'].replica_id == 0
            assert sharded_state_dict['keyB'].replica_id == 0

            # metadata dict can be loaded. We don't validate access because there are multiple replica_id=0
            state_dict = load(sharded_state_dict, ckpt_dir, validate_access_integrity=False)
            assert torch.all(state_dict['keyA'] == torch.arange(10 * Utils.world_size))

        Utils.destroy_model_parallel()

    def test_can_mix_sharded_tensors_and_factories(self, tmp_path_dist_ckpt):
        Utils.initialize_model_parallel(1, 1)

        def _build_fn(key, tensor, replica_id):
            return [
                ShardedTensor.from_rank_offsets(key + 'part1', tensor, replica_id=replica_id),
                ShardedTensor.from_rank_offsets(key + 'part2', tensor, replica_id=replica_id),
                ShardedTensor.from_rank_offsets(key + 'part3', tensor, replica_id=replica_id),
            ]

        # state dict can be modified by dist_checkpointing.save, so two copies
        def get_sharded_state_dict(base=0):
            return {'all': [
                ShardedTensor.from_rank_offsets('A', torch.arange(2) + base, replica_id=Utils.rank),
                ShardedTensor.from_rank_offsets('B', torch.arange(3) + base, replica_id=Utils.rank),
                ShardedTensor.from_rank_offsets('C', torch.arange(4) + base, replica_id=Utils.rank),
                ShardedTensorFactory('D', torch.arange(5) + base, _build_fn, sum, replica_id=Utils.rank),
            ]}

        with TempNamedDir(tmp_path_dist_ckpt / 'test_can_mix_sharded_tensors_and_factories') as ckpt_dir:
            save(get_sharded_state_dict(0), ckpt_dir)
            loaded_state_dict = load(get_sharded_state_dict(10), ckpt_dir)

        expected_sd = {
            'all': [
                torch.arange(2),
                torch.arange(3),
                torch.arange(4),
                torch.arange(5) * 3,  # sum of three parts, as specified in merge_fn
            ]
        }
        diffs = diff(loaded_state_dict, expected_sd)
        assert not any(map(bool, diffs)), diffs

        Utils.destroy_model_parallel()

    def test_load_error_msg(self, tmp_path_dist_ckpt):
        ckpt_dir_name = 'test_load_error_msg'
        Utils.initialize_model_parallel(1, 1)
        sh_ten = ShardedTensor.from_rank_offsets('keyA', torch.rand(10), replica_id=Utils.rank)
        state_dict = {'some_key': sh_ten}

        # Non-existent directory
        non_ex_path = f'/tmp/non-existent-path/{ckpt_dir_name}'
        with pytest.raises(CheckpointingException) as exc_info:
            load(state_dict, non_ex_path)
        assert f'directory {non_ex_path} does not exist' in str(exc_info.value)

        with TempNamedDir(tmp_path_dist_ckpt / ckpt_dir_name) as ckpt_dir:
            torch.distributed.barrier()
            # Empty directory - not a distributed checkpoint
            with pytest.raises(CheckpointingException) as exc_info:
                load(state_dict, ckpt_dir)
            assert f'is not a distributed checkpoint' in str(exc_info.value)

            # Missing Zarr arrays
            torch.distributed.barrier()
            save(state_dict, ckpt_dir)
            sh_ten.key = 'different_key'
            # TODO: remove torch exception
            with pytest.raises((CheckpointingException, CheckpointException)) as exc_info:
                load(state_dict, ckpt_dir)
            assert "different_key" in str(exc_info.value)

    def test_sharded_object_serialization(self, tmp_path_dist_ckpt):
        Utils.initialize_model_parallel(1, 1)
        with TempNamedDir(tmp_path_dist_ckpt / 'test_sh_obj') as ckpt_dir:
            state = {'some': 'dict'}
            state_serialized = io.BytesIO()
            torch.save(state, state_serialized)
            state_dict = {'some_key': ShardedObject('sh_obj_A', state_serialized, (1,), (0,),
                                                    replica_id=Utils.rank)}

            save(state_dict, ckpt_dir)
            del state, state_serialized, state_dict
            other_state = {'other': 'dictionary'}
            other_serialized = io.BytesIO()
            torch.save(other_state, other_serialized)
            state_dict = {'other_key': ShardedObject('sh_obj_A', other_serialized, (1,), (0,),
                                                     replica_id=Utils.rank)}
            load_state_dict = load(state_dict, ckpt_dir)
            assert 'other_key' in load_state_dict
            load_state_dict['other_key'].seek(0)
            loaded_state = torch.load(load_state_dict['other_key'])

            assert loaded_state == {'some': 'dict'}

        Utils.destroy_model_parallel()

    def test_tensor_shape_mismatch(self, tmp_path_dist_ckpt):
        Utils.initialize_model_parallel(2,4)

        # Global tensor is just a range(32) repeated twice over the first dimension
        local_tensor = torch.arange(4).unsqueeze(0).expand(2, 4) + Utils.rank * 4

        state_dict = {
            'rigid': ShardedTensor.from_rank_offsets('keyA', local_tensor, (1, Utils.rank, Utils.world_size)),
            'flexible': ShardedTensor.from_rank_offsets('keyB', local_tensor, (1, Utils.rank, Utils.world_size),
                                                        allow_shape_mismatch=True),
        }
        assert state_dict['rigid'].global_shape == (2, 32)
        assert state_dict['flexible'].global_shape == (2, 32)

        with TempNamedDir(tmp_path_dist_ckpt / 'test_tensor_shape_mismatch') as ckpt_dir:
            save(state_dict, ckpt_dir)

            pp_size = parallel_state.get_pipeline_model_parallel_world_size()
            pp_rank = parallel_state.get_pipeline_model_parallel_rank()
            tp_rank = parallel_state.get_tensor_model_parallel_rank()

            # Smaller coverage than expected (28 < 32)
            state_dict = {
                'rigid': ShardedTensor.from_rank_offsets('keyA', torch.ones(2, 7), (1, pp_rank, pp_size), replica_id=tp_rank),
            }
            with pytest.raises((CheckpointingException, CheckpointException)):
                load(state_dict, ckpt_dir)

            state_dict = {
                'flexible': ShardedTensor.from_rank_offsets('keyB', torch.ones(2, 7), (1, pp_rank, pp_size), replica_id=tp_rank,
                                                            allow_shape_mismatch=True),
            }
            loaded_state_dict = load(state_dict, ckpt_dir)
            assert torch.all(loaded_state_dict['flexible'] == torch.arange(7).unsqueeze(0).expand(2, 7) + pp_rank * 7)

            # Larger coverage than expected (36 > 32)
            state_dict = {
                'rigid': ShardedTensor.from_rank_offsets('keyA', torch.ones(2, 9), (1, pp_rank, pp_size), replica_id=tp_rank),
            }
            with pytest.raises((CheckpointingException, CheckpointException)):
                load(state_dict, ckpt_dir)

            state_dict = {
                'flexible': ShardedTensor.from_rank_offsets('keyB', torch.ones(2, 9), (1, pp_rank, pp_size), replica_id=tp_rank,
                                                            allow_shape_mismatch=True),
            }
            loaded_state_dict = load(state_dict, ckpt_dir)
            expected_tensor = torch.arange(9).unsqueeze(0).expand(2, 9) + pp_rank * 9

            if pp_rank >= (32 // 9):
                assert pp_rank == 3, pp_rank
                expected_tensor[:, 5:] = 0  # padding with 0s
            assert torch.all(loaded_state_dict['flexible'] == expected_tensor)

        Utils.destroy_model_parallel()