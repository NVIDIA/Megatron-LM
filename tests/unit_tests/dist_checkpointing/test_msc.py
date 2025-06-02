# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import traceback
import torch

from megatron.core.device_utils import get_current_device, get_current_device_type, get_xla_model

try:
    from torch.distributed import DeviceMesh
    from torch.distributed._tensor import DTensor

    HAVE_DTENSOR = True
except ImportError:
    HAVE_DTENSOR = False

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import ShardedTensor, load, save
from megatron.core.dist_checkpointing.strategies.base import StrategyAction, get_default_strategy
from megatron.core.msc_utils import MultiStorageClientFeature
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils

xm = get_xla_model()

class TestSerializationWithMultiStorageClient:

    def setup_method(self, method):
        MultiStorageClientFeature.enable()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_process_save_load(self, tmp_path_dist_ckpt):
        Utils.initialize_model_parallel(2, 4)

        sharded_state_dict = {
            'sd_keyA': ShardedTensor.from_rank_offsets(
                'keyA', torch.ones(2, 4), replica_id=Utils.rank
            ),
            'sd_keyB': ShardedTensor.from_rank_offsets(
                'keyB', torch.ones(3, 5, 7), replica_id=Utils.rank
            ),
        }

        if HAVE_DTENSOR:
            device_type = "cpu" if xm else get_current_device_type()
            device = torch.device("cpu") if xm else get_current_device()
            mesh = DeviceMesh.from_group(
                parallel_state.get_data_parallel_group(with_context_parallel=True) if not xm else
                    parallel_state.get_data_parallel_group_gloo(with_context_parallel=True),
                device_type
            )
            sharded_state_dict['sd_keyD'] = ShardedTensor.from_rank_offsets(
                'keyD',
                DTensor.from_local(torch.ones(3, 5, 7, device=device), mesh)._local_tensor,
                replica_id=Utils.rank,
            )

        # sync=True to make sure other ranks wait for rank 0 to finish creating directory.
        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_single_process_save_load', sync=True,
            process_group=parallel_state.get_default_process_group()
        ) as ckpt_dir:
            save_strategy = get_default_strategy(StrategyAction.SAVE_SHARDED, 'torch_dist', 1)
            save(sharded_state_dict, ckpt_dir, save_strategy, 
                 process_group=parallel_state.get_default_process_group())
            torch.distributed.barrier(group=parallel_state.get_default_process_group())

            load_ssd = {
                'load_sd_keyA': ShardedTensor.from_rank_offsets(
                    'keyA', torch.ones(2, 4), replica_id=Utils.rank
                )
            }
            loaded_state_dict = load(load_ssd, ckpt_dir, 
                                     process_group=parallel_state.get_default_process_group())

            assert set(loaded_state_dict.keys()) == {'load_sd_keyA'}
            assert isinstance(loaded_state_dict['load_sd_keyA'], torch.Tensor)
            assert loaded_state_dict['load_sd_keyA'].shape == (2, 4)
       
        Utils.destroy_model_parallel()
