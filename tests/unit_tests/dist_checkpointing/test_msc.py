# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch

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
            mesh = DeviceMesh.from_group(
                parallel_state.get_data_parallel_group(with_context_parallel=True), "cuda"
            )
            sharded_state_dict['sd_keyD'] = ShardedTensor.from_rank_offsets(
                'keyD',
                DTensor.from_local(torch.ones(3, 5, 7), mesh)._local_tensor,
                replica_id=Utils.rank,
            )

        # sync=True to make sure other ranks wait for rank 0 to finish creating directory.
        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_single_process_save_load', sync=True
        ) as ckpt_dir:
            save_strategy = get_default_strategy(StrategyAction.SAVE_SHARDED, 'torch_dist', 1)
            save(sharded_state_dict, ckpt_dir, save_strategy)
            torch.distributed.barrier()

            load_ssd = {
                'load_sd_keyA': ShardedTensor.from_rank_offsets(
                    'keyA', torch.ones(2, 4), replica_id=Utils.rank
                )
            }
            loaded_state_dict = load(load_ssd, ckpt_dir)

            assert set(loaded_state_dict.keys()) == {'load_sd_keyA'}
            assert isinstance(loaded_state_dict['load_sd_keyA'], torch.Tensor)
            assert loaded_state_dict['load_sd_keyA'].shape == (2, 4)

        Utils.destroy_model_parallel()
