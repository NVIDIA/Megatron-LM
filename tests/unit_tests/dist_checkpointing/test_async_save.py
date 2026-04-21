# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from unittest import mock

import pytest
import torch

from megatron.core.dist_checkpointing import ShardedTensor, load, save
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue
from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync
from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def write_data_os_err_mock_fn(
    transform_list, local_proc_idx, write_bucket, results_queue, count_queue, use_fsync, **kwargs
):
    """Raises an error on worker #2 during storage save"""
    try:
        if local_proc_idx == 2:
            raise OSError('worker #2 critical failure')
        output = (local_proc_idx, [])
    except Exception as e:
        output = (local_proc_idx, e)
    results_queue.put(output)
    count_queue.get()
    count_queue.task_done()


class TestAsyncSave:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize('persistent', [True, False])
    def test_async_is_equivalent_to_sync(self, tmp_path_dist_ckpt, persistent):
        Utils.initialize_model_parallel(2, 4)

        sharded_state_dict = {
            'sd_keyA': ShardedTensor.from_rank_offsets(
                'keyA', torch.ones(2, 4), replica_id=Utils.rank
            ),
            'sd_keyB': ShardedTensor.from_rank_offsets(
                'keyB', torch.ones(3, 5, 7), replica_id=Utils.world_size - Utils.rank - 1
            ),
        }

        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_equivalence_async'
        ) as async_ckpt_dir, TempNamedDir(
            tmp_path_dist_ckpt / 'test_equivalence_sync'
        ) as sync_ckpt_dir:
            # async
            async_calls = AsyncCallsQueue(persistent)
            async_request = save(sharded_state_dict, async_ckpt_dir, async_sharded_save=True)
            async_calls.schedule_async_request(async_request)

            # sync
            save(sharded_state_dict, sync_ckpt_dir, async_sharded_save=False)

            # finalize async
            async_calls.maybe_finalize_async_calls(blocking=True)

            # load and compare
            loaded_async_state_dict = load(sharded_state_dict, async_ckpt_dir)
            loaded_sync_state_dict = load(sharded_state_dict, sync_ckpt_dir)
            diffs = diff(loaded_async_state_dict, loaded_sync_state_dict)
            assert not any(map(bool, diffs)), diffs
            async_calls.close()

        Utils.destroy_model_parallel()

    @pytest.mark.parametrize('async_save', [False, True])
    @pytest.mark.parametrize('worker_fn', [write_data_os_err_mock_fn])
    def test_errors_are_reported(self, tmp_path_dist_ckpt, async_save, worker_fn):
        Utils.initialize_model_parallel(2, 4)
        sharded_state_dict = {
            f'key{i}': ShardedTensor.from_rank_offsets(f'key{i}_rank{Utils.rank}', torch.ones(2, 4))
            for i in range(4)  # make sure there is enough non-empty saving workers
        }

        with TempNamedDir(tmp_path_dist_ckpt / 'test_errors_are_reported') as ckpt_dir:
            async_calls = AsyncCallsQueue()
            save_strategy = TorchDistSaveShardedStrategy('torch_dist', 1, thread_count=8)

            try:
                orig_fn = FileSystemWriterAsync.write_preloaded_data
                FileSystemWriterAsync.write_preloaded_data = worker_fn
                with pytest.raises(RuntimeError) as exc_info:
                    if async_save:
                        async_request = save(
                            sharded_state_dict, ckpt_dir, save_strategy, async_sharded_save=True
                        )
                        async_calls.schedule_async_request(async_request)
                        async_calls.maybe_finalize_async_calls(blocking=True)
                    else:
                        save(sharded_state_dict, ckpt_dir, save_strategy)
                assert 'Worker failure' in str(exc_info.value)

            finally:
                FileSystemWriterAsync.write_preloaded_data = orig_fn

        Utils.destroy_model_parallel()
