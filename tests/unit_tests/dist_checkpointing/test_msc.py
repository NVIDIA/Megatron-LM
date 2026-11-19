# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

try:
    from torch.distributed import DeviceMesh
    from torch.distributed._tensor import DTensor

    HAVE_DTENSOR = True
except ImportError:
    HAVE_DTENSOR = False

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import ShardedTensor, load, save
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy
from megatron.core.msc_utils import MultiStorageClientFeature
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


class TestSerializationWithMultiStorageClient:

    def setup_method(self, method):
        MultiStorageClientFeature.enable()

    def teardown_method(self, method):
        MultiStorageClientFeature.disable()
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
            save_strategy = TorchDistSaveShardedStrategy()
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

    def test_writer_options_reach_msc_writer(self, tmp_path_dist_ckpt, monkeypatch):
        Utils.initialize_model_parallel(2, 4)
        msc = MultiStorageClientFeature.import_package()
        writer_cls = msc.torch.MultiStorageFileSystemWriter
        options = object()
        received = []

        def recording_writer(path, *args, writer_options=None, **kwargs):
            received.append(writer_options)
            return writer_cls(path, *args, **kwargs)

        monkeypatch.setattr(msc.torch, 'MultiStorageFileSystemWriter', recording_writer)
        sharded_state_dict = {
            'sd_keyA': ShardedTensor.from_rank_offsets(
                'keyA', torch.arange(8.0).reshape(2, 4), replica_id=Utils.rank
            )
        }
        with TempNamedDir(tmp_path_dist_ckpt / 'test_writer_options', sync=True) as ckpt_dir:
            save(sharded_state_dict, ckpt_dir, TorchDistSaveShardedStrategy(writer_options=options))
            torch.distributed.barrier()
            assert received == [options]

            load_ssd = {
                'keyA': ShardedTensor.from_rank_offsets(
                    'keyA', torch.zeros(2, 4), replica_id=Utils.rank
                )
            }
            assert torch.equal(load(load_ssd, ckpt_dir)['keyA'], torch.arange(8.0).reshape(2, 4))

        MultiStorageClientFeature.disable()
        with pytest.raises(CheckpointingException, match='writer_options'):
            TorchDistSaveShardedStrategy(writer_options=options).save(
                sharded_state_dict, tmp_path_dist_ckpt / 'test_writer_options_without_msc'
            )
