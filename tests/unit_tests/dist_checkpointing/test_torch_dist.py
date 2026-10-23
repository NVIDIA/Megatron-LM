# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Tests for PyTorch DCP based checkpoint format. """

import pickle
from copy import deepcopy
from dataclasses import fields
from unittest import mock

import torch

from megatron.core.dist_checkpointing import ShardedTensor, load, save
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


class TestCachedMetadata:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_cached_metadata(self, tmp_path_dist_ckpt):
        Utils.initialize_model_parallel(2, 4)

        sharded_state_dict_non_cached = {
            'sd_keyA': ShardedTensor.from_rank_offsets(
                'keyA', torch.ones(2, 4), replica_id=Utils.rank
            ),
            'sd_keyB': ShardedTensor.from_rank_offsets(
                'keyB', torch.ones(3, 5, 7), replica_id=Utils.world_size - Utils.rank - 1
            ),
        }

        sharded_state_dict_cached = {
            'sd_keyA': ShardedTensor.from_rank_offsets(
                'keyA', torch.ones(2, 4), replica_id=Utils.rank
            ),
            'sd_keyB': ShardedTensor.from_rank_offsets(
                'keyB', torch.ones(3, 5, 7), replica_id=Utils.world_size - Utils.rank - 1
            ),
        }

        loaded_non_cached, loaded_cached = None, None
        md_non_cached, md_cached = None, None
        with TempNamedDir(tmp_path_dist_ckpt / 'ckpt_dir') as ckpt_dir:
            save(sharded_state_dict_non_cached, ckpt_dir, async_sharded_save=False)
            loaded_non_cached = load(sharded_state_dict_non_cached, ckpt_dir)
            md_path = ckpt_dir / '.metadata'
            with md_path.open('rb') as f:
                md_non_cached = pickle.load(f)

        save_strategy = deepcopy(TorchDistSaveShardedStrategy())
        save_strategy.use_cached_ckpt_structure = True
        # Run over 3 iterations with cached metadata enabled
        # The 3rd iteration will run with cached metadata
        # `ckpt_dir` at the 3rd iteration 2 will be maintained for comparison
        ckpt_dir = None
        for i in range(3):
            ckpt_dir = TempNamedDir(tmp_path_dist_ckpt / f'ckpt_dir_${i}_cached')
            save(
                sharded_state_dict_cached,
                ckpt_dir.__enter__(),
                save_strategy,
                async_sharded_save=False,
            )
            if i < 2:
                ckpt_dir.cleanup()
        loaded_cached = load(sharded_state_dict_cached, ckpt_dir.__enter__())
        md_path = ckpt_dir.__enter__() / '.metadata'

        with md_path.open('rb') as f:
            md_cached = pickle.load(f)

        # Check loaded state dict
        diffs = diff(loaded_non_cached, loaded_cached)

        assert not any(
            len(x) for x in diffs
        ), 'Cached metadata doesn\'t produce the same state_dict in loading'
        # Check metadata recorded in .metadata, torch.distributed.metadata.Metadata
        for field in fields(md_non_cached):
            if field.name not in ['storage_data', 'storage_meta']:
                diffs = diff(getattr(md_non_cached, field.name), getattr(md_cached, field.name))
                assert not any(
                    len(x) for x in diffs
                ), f'{field.name} is different in metadata from non-cached, cached metadata impls'
        ckpt_dir.cleanup()
        Utils.destroy_model_parallel()


class TestCPUTensors:
    def setup_method(self, method):
        Utils.initialize_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_cpu_tensors_dont_take_too_much_space(self, tmp_path_dist_ckpt):
        large_cuda_tensor = torch.ones(1_000_000, dtype=torch.float, device='cuda')
        large_cpu_tensor = torch.ones(1_000_000, dtype=torch.float)
        # Create small tensors which are a view of a large tensor
        sharded_state_dict = {
            'sd_keyA': ShardedTensor.from_rank_offsets(
                'keyA', large_cuda_tensor[:10], replica_id=Utils.rank
            ),
            'sd_keyB': ShardedTensor.from_rank_offsets(
                'keyB', large_cpu_tensor[:10], replica_id=Utils.rank
            ),
        }

        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_cpu_tensors_dont_take_too_much_space'
        ) as ckpt_dir:
            save(sharded_state_dict, ckpt_dir)

            distcp_files = [(ckpt_dir / '__0_0.distcp')]
            for file in distcp_files:
                assert file.exists()
                file_size = file.stat().st_size
                assert file_size < 10_000, file.name


class TestDCPCoordinationGroup:
    """DCP's plan / result exchange (gather_object, scatter_object_list) must not run on the NCCL
    default group: NCCL gather / scatter set up persistent P2P connections from the coordinator to
    every rank, which cost device memory linear in the world size for the rest of the run."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_sync_save_object_exchange_runs_on_gloo(self, tmp_path_dist_ckpt):
        Utils.initialize_model_parallel(1, 1)
        sharded_state_dict = {
            'sd_keyA': ShardedTensor.from_rank_offsets(
                'keyA', torch.arange(8.0).cuda() + Utils.rank, (0, Utils.rank, Utils.world_size)
            ),
            'sd_keyB': ShardedTensor.from_rank_offsets(
                'keyB', torch.ones(3, 5).cuda(), replica_id=Utils.rank
            ),
        }
        groups = []
        real_gather = torch.distributed.gather_object
        real_scatter = torch.distributed.scatter_object_list

        def gather_object(*args, **kwargs):
            groups.append(kwargs.get('group'))
            return real_gather(*args, **kwargs)

        def scatter_object_list(*args, **kwargs):
            groups.append(kwargs.get('group'))
            return real_scatter(*args, **kwargs)

        with (
            TempNamedDir(tmp_path_dist_ckpt / 'test_dcp_gloo') as ckpt_dir,
            mock.patch.object(torch.distributed, 'gather_object', gather_object),
            mock.patch.object(torch.distributed, 'scatter_object_list', scatter_object_list),
        ):
            save(sharded_state_dict, ckpt_dir, async_sharded_save=False)
            # all ranks call gather_object (non-coordinators send); the exchange happened
            assert groups
            for group in groups:
                assert group is not None
                assert torch.distributed.get_backend(group) == 'gloo'
                assert torch.distributed.get_world_size(group) == Utils.world_size
            loaded = load(sharded_state_dict, ckpt_dir)
            assert torch.equal(loaded['sd_keyA'].cpu(), torch.arange(8.0) + Utils.rank)
            assert torch.equal(loaded['sd_keyB'].cpu(), torch.ones(3, 5))
