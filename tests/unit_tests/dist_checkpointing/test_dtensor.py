# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from typing import Optional

import pytest
import torch
import torch.distributed.checkpoint
from torch.distributed import DeviceMesh
from torch.distributed.checkpoint import FileSystemReader, BytesStorageMetadata, \
    TensorStorageMetadata, Metadata
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.tensor import DTensor, Shard
from torch.optim import Adam

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import ShardedTensor, load, \
    load_plain_tensors, save, ShardedObject, load_common_state_dict
from megatron.core.dist_checkpointing.core import save_config, \
    CheckpointingConfig
from megatron.core.dist_checkpointing.dict_utils import diff, nested_values
from megatron.core.dist_checkpointing.optimizer import (
    get_param_id_to_sharded_param_map,
    optim_state_to_sharding_state,
)
from megatron.core.dist_checkpointing.serialization import load_sharded_metadata
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import sharded_state_dict_default
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


# TODO: this file differs from `./models/test_mlp_glu.py` only with `glu=False` and introducing
#  another simple model (`initialize_layer_norm`). Merge those files.
def initialize_mlp(glu=False):
    model_parallel_cuda_manual_seed(123)
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    transformer_config = TransformerConfig(
        num_layers=pp_size,
        hidden_size=96,
        num_attention_heads=4,
        use_cpu_initialization=True,
        gated_linear_unit=glu,
    )
    return MLP(
        transformer_config, get_gpt_layer_with_transformer_engine_spec().submodules.mlp.submodules
    )

def initialize_layer_norm(glu=False):
    model_parallel_cuda_manual_seed(123)
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    transformer_config = TransformerConfig(
        num_layers=pp_size,
        hidden_size=96,
        num_attention_heads=4,
        use_cpu_initialization=True,
        gated_linear_unit=glu,
    )
    return TENorm(transformer_config, 96)



def get_pp_offsets():
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    return ((0, pp_rank, pp_size),)


class TestSimpleResharding:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("initialize_fn", [initialize_layer_norm, initialize_mlp])
    @pytest.mark.parametrize(
        "src_tp_pp,dest_tp_pp",
        [
            # changing PP is impossible because the number of layers must be the same
            ((2, 1), (2, 1)),
            ((2, 1), (8, 1)),
            ((8, 1), (1, 1)),
        ],
    )
    def test_parallel_reconfiguration_e2e(self, tmp_path_dist_ckpt, src_tp_pp, dest_tp_pp, initialize_fn):
        """Test module saving and loading with different TP/PP"""
        Utils.initialize_model_parallel(*src_tp_pp)

        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_dtensor_A'
        ) as ckpt_dir_A, TempNamedDir(
            tmp_path_dist_ckpt / 'test_dtensor_B'
        ) as ckpt_dir_B:
            # Save checkpoint A
            mlp_A = initialize_fn()
            sh_state_dict = sharded_state_dict_default(mlp_A, sharded_offsets=get_pp_offsets())

            save(sh_state_dict, ckpt_dir_A)
            Utils.destroy_model_parallel()

            # Load checkpoint A with different TP/PP and save as checkpoint B
            Utils.initialize_model_parallel(*dest_tp_pp)
            mlp_B = initialize_fn()
            sh_state_dict = sharded_state_dict_default(mlp_B, sharded_offsets=get_pp_offsets())
            state_dict = load(sh_state_dict, ckpt_dir_A)
            mlp_B.load_state_dict(state_dict)
            save(sharded_state_dict_default(mlp_B, sharded_offsets=get_pp_offsets()), ckpt_dir_B)
            Utils.destroy_model_parallel()

            # Test both checkpoints are equal
            Utils.initialize_model_parallel(1, 1)
            state_dict_A = load_plain_tensors(ckpt_dir_A)
            state_dict_B = load_plain_tensors(ckpt_dir_B)
            diffs = diff(state_dict_A, state_dict_B)
            assert not any(map(bool, diffs)), diffs


class TestNonTensors:
    def test_nontensor_objects(self, tmp_path_dist_ckpt):
        device_mesh = DeviceMesh("cuda", list(range(Utils.world_size)), _init_backend=True)
        state_dict = {
            f'a_local_data_{Utils.rank}_sd': [1, Utils.rank],
            'b_ten_sd': DTensor.from_local(torch.ones(1) * Utils.rank, device_mesh, [Shard(0)]),
            'c_fake_common_data': (Utils.rank, 77),
            'd_truly_common_data': (42, 37),
        }
        with TempNamedDir(
                tmp_path_dist_ckpt / 'test_nontensor_objects'
        ) as ckpt_dir:
            torch.distributed.checkpoint.save(state_dict, checkpoint_id=ckpt_dir)

            metadata = FileSystemReader(ckpt_dir).read_metadata()
            for i in range(Utils.world_size):
                assert isinstance(metadata.state_dict_metadata[f'a_local_data_{i}_sd'], BytesStorageMetadata)
            assert isinstance(metadata.state_dict_metadata['c_fake_common_data'], BytesStorageMetadata)
            assert isinstance(metadata.state_dict_metadata['d_truly_common_data'], BytesStorageMetadata)

            assert isinstance(metadata.state_dict_metadata['b_ten_sd'], TensorStorageMetadata)

            torch.distributed.barrier()
            # loaded_state_dict = {}
            # planner = _EmptyStateDictLoadPlanner()
            # torch.distributed.checkpoint.load(loaded_state_dict, checkpoint_id=ckpt_dir, planner=planner)
            # # There seems to be a bug, so can't just use `loaded_state_dict`
            # loaded_state_dict = planner.state_dict

            loaded_state_dict = {
                **{
                    f'a_local_data_{i}_sd': None
                    for i in range(Utils.world_size)
                },
                'c_fake_common_data': None,
                'd_truly_common_data': None,
                'b_ten_sd': torch.empty(1 * Utils.world_size),
            }
            torch.distributed.checkpoint.load(loaded_state_dict, checkpoint_id=ckpt_dir)

            print(loaded_state_dict)
            for i in range(Utils.world_size):
                assert loaded_state_dict[f'a_local_data_{i}_sd'] == (1, i)
            assert loaded_state_dict['c_fake_common_data'] == (0, 77)
            assert loaded_state_dict['d_truly_common_data'] == (42, 37)

            assert torch.all(loaded_state_dict['b_ten_sd'] == torch.arange(Utils.world_size))


    def test_sharded_objects(self, tmp_path_dist_ckpt):
        device_mesh = DeviceMesh("cuda", list(range(Utils.world_size)), _init_backend=True)
        state_dict = {
            f'a_local_data_{Utils.rank}_sd': ShardedObject(f'a_local_data_{Utils.rank}_sd', [1, 2, Utils.rank], (1,), (0,), replica_id=0),
            'b_ten_sd': ShardedTensor.from_rank_offsets('b_ten', torch.ones(3), (0, Utils.rank, Utils.world_size),
                                                        dtensor_ckpt_device_mesh=device_mesh, dtensor_ckpt_placements=[Shard(0)]),
            'c_fake_common_data': (Utils.rank, 77),
            'd_truly_common_data': (42, 37),
        }
        with TempNamedDir(
                tmp_path_dist_ckpt / 'test_sharded_objects'
        ) as ckpt_dir:
            save(state_dict, ckpt_dir)

            sharded_metadata = load_sharded_metadata(ckpt_dir)
            for i in range(Utils.world_size):
                assert isinstance(sharded_metadata[f'a_local_data_{i}_sd/shard_0_1'], ShardedObject)
                assert sharded_metadata[f'a_local_data_{i}_sd/shard_0_1'].key == f'a_local_data_{i}_sd'

            assert isinstance(sharded_metadata['b_ten'], ShardedTensor)
            
            common_state_dict = load_common_state_dict(ckpt_dir)
            assert common_state_dict['c_fake_common_data'] == (0, 77)
            assert common_state_dict['d_truly_common_data'] == (42, 37)
            
    def test_dcp_state_dict_structure(self, tmp_path_dist_ckpt):
        device_mesh = DeviceMesh("cuda", list(range(Utils.world_size)), _init_backend=True)
        state_dict = {
            'other': {
                'a': 1,
                'b': 2,
                'c': {'cc': {'ccc': 'cccc'}}
            },
            'mixed': {
                'a': 4,
                'b': DTensor.from_local(torch.ones(1) * Utils.rank, device_mesh, [Shard(0)]),
            },
            'just_tensors': {
                'a': DTensor.from_local(torch.ones(1) * Utils.rank, device_mesh, [Shard(0)]),
                'b': DTensor.from_local(torch.ones(1) * Utils.rank, device_mesh, [Shard(0)]),
            }
        }
        with TempNamedDir(
                tmp_path_dist_ckpt / 'test_dcp_state_dict_structure'
        ) as ckpt_dir:
            torch.distributed.checkpoint.save(state_dict, checkpoint_id=ckpt_dir)
            metadata = FileSystemReader(ckpt_dir).read_metadata()
            ckpt_keys = metadata.state_dict_metadata.keys()

            assert 'other.a' in ckpt_keys
            assert 'other.b' in ckpt_keys
            assert 'other.c.cc.ccc' in ckpt_keys  # DCP doesn't "cut" this tree at the `other.c` level
            assert 'mixed.a' in ckpt_keys
            assert 'mixed.b' in ckpt_keys
            assert 'just_tensors.a' in ckpt_keys
            assert 'just_tensors.b' in ckpt_keys
