# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import logging
import time
from unittest import mock

import pytest
import torch
from torch import multiprocessing as mp
from torch.distributed.checkpoint import CheckpointException

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
        if Utils.rank == 2 and local_proc_idx == 2:
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
    @pytest.mark.parametrize('abort', [True, False])
    def test_async_is_equivalent_to_sync(self, tmp_path_dist_ckpt, persistent, abort):
        Utils.initialize_model_parallel(2, 4)

        sharded_state_dict = {
            'sd_keyA': ShardedTensor.from_rank_offsets(
                'keyA', torch.ones(2, 4), replica_id=Utils.rank
            ),
            'sd_keyB': ShardedTensor.from_rank_offsets(
                'keyB', torch.ones(3, 5, 7), replica_id=Utils.world_size - Utils.rank - 1
            ),
        }

        with (
            TempNamedDir(tmp_path_dist_ckpt / 'test_equivalence_async') as async_ckpt_dir,
            TempNamedDir(tmp_path_dist_ckpt / 'test_equivalence_sync') as sync_ckpt_dir,
        ):
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
            async_calls.close(abort=abort)

        Utils.destroy_model_parallel()

    @pytest.mark.parametrize('async_save', [False, True])
    @pytest.mark.parametrize('worker_fn', [write_data_os_err_mock_fn])
    def test_errors_are_reported(self, tmp_path_dist_ckpt, async_save, worker_fn):
        Utils.initialize_model_parallel(2, 4)
        orig_fn = FileSystemWriterAsync.write_preloaded_data
        FileSystemWriterAsync.write_preloaded_data = worker_fn

        sharded_state_dict = {
            f'key{i}': ShardedTensor.from_rank_offsets(f'key{i}_rank{Utils.rank}', torch.ones(2, 4))
            for i in range(4)  # make sure there is enough non-empty saving workers
        }
        save_strategy = TorchDistSaveShardedStrategy('torch_dist', 1, thread_count=8)

        with (
            TempNamedDir(tmp_path_dist_ckpt / 'test_errors_are_reported') as ckpt_dir,
            pytest.raises(CheckpointException) as exc_info,
        ):
            if async_save:
                async_calls = AsyncCallsQueue()
                async_request = save(
                    sharded_state_dict, ckpt_dir, save_strategy, async_sharded_save=True
                )
                async_calls.schedule_async_request(async_request)
                async_calls.maybe_finalize_async_calls(blocking=True)
            else:
                save(sharded_state_dict, ckpt_dir, save_strategy)
        if Utils.rank == 0:
            assert 'Worker failure' in str(exc_info.value)
        else:
            assert 'Worker failure' not in str(exc_info.value)

        FileSystemWriterAsync.write_preloaded_data = orig_fn
        Utils.destroy_model_parallel()


class TestRetryLogic:

    def test_retry_on_transient_error_with_eventual_success(self, tmp_path_dist_ckpt, caplog):
        # Track call attempts
        call_count = {'count': 0}

        def mock_write_item_fail_twice(*args, **kwargs):
            # Raise exception two times, then succeed
            call_count['count'] += 1
            if call_count['count'] <= 2:
                raise OSError(f"Transient error on attempt {call_count['count']}")
            # Return None since the result just needs to be picklable for multiprocessing
            return None

        # Create mock queues
        results_queue = mp.SimpleQueue()
        count_queue = mp.JoinableQueue()
        count_queue.put(None)  # Add item for worker to consume

        write_bucket = (
            tmp_path_dist_ckpt / 'test_file.pt',
            'storage_key',
            ([(mock.MagicMock(), b'test')], []),  # (bytes_data, tensor_data)
        )

        with caplog.at_level(logging.WARNING):
            with mock.patch(
                'megatron.core.dist_checkpointing.strategies.filesystem_async._write_item',
                side_effect=mock_write_item_fail_twice,
            ):
                FileSystemWriterAsync.write_preloaded_data(
                    transform_list=[],
                    local_proc_idx=0,
                    write_bucket=write_bucket,
                    results_queue=results_queue,
                    count_queue=count_queue,
                    use_fsync=False,
                    max_ckpt_save_retries=3,
                    ckpt_save_retry_delay=0.1,
                )

        assert call_count['count'] == 3, "Should have attempted 3 times (2 failures + 1 success)"

        result = results_queue.get()
        proc_idx, results = result
        assert proc_idx == 0
        assert not isinstance(results, Exception), "Should have succeeded after retries"

        warning_logs = [record for record in caplog.records if record.levelname == 'WARNING']
        assert len(warning_logs) == 2, "Should have logged 2 warnings for failed attempts"
        assert 'failed on attempt 1/3' in warning_logs[0].message
        assert 'Retrying in 0.10 seconds' in warning_logs[0].message
        assert 'OSError: Transient error on attempt 1' in warning_logs[0].message

    def test_retry_exhaustion_logs_error(self, tmp_path_dist_ckpt, caplog):

        def mock_write_item_always_fail(*args, **kwargs):
            # Always fail
            raise ConnectionError("Persistent connection error")

        results_queue = mp.SimpleQueue()
        count_queue = mp.JoinableQueue()
        count_queue.put(None)

        write_bucket = (
            tmp_path_dist_ckpt / 'test_file.pt',
            'storage_key',
            ([(mock.MagicMock(), b'test')], []),
        )

        with caplog.at_level(logging.WARNING):
            with mock.patch(
                'megatron.core.dist_checkpointing.strategies.filesystem_async._write_item',
                side_effect=mock_write_item_always_fail,
            ):
                FileSystemWriterAsync.write_preloaded_data(
                    transform_list=[],
                    local_proc_idx=1,
                    write_bucket=write_bucket,
                    results_queue=results_queue,
                    count_queue=count_queue,
                    use_fsync=False,
                    max_ckpt_save_retries=3,
                    ckpt_save_retry_delay=0.05,
                )

        result = results_queue.get()
        proc_idx, exception = result
        assert proc_idx == 1
        assert isinstance(exception, ConnectionError)

        error_logs = [record for record in caplog.records if record.levelname == 'ERROR']
        assert len(error_logs) == 1, "Should have logged 1 error for final failure"
        assert 'failed after 3 attempts' in error_logs[0].message
        assert 'ConnectionError: Persistent connection error' in error_logs[0].message

        warning_logs = [record for record in caplog.records if record.levelname == 'WARNING']
        assert len(warning_logs) == 2, "Should have logged 2 warnings before final failure"

    def test_verify_retry_delay(self, tmp_path_dist_ckpt):
        call_times = []

        def mock_write_item_track_time(*args, **kwargs):
            call_times.append(time.time())
            if len(call_times) <= 2:
                raise TimeoutError(f"Timeout on attempt {len(call_times)}")
            return None

        results_queue = mp.SimpleQueue()
        count_queue = mp.JoinableQueue()
        count_queue.put(None)

        write_bucket = (
            tmp_path_dist_ckpt / 'test_file.pt',
            'storage_key',
            ([(mock.MagicMock(), b'test')], []),
        )

        retry_delay = 0.2

        with mock.patch(
            'megatron.core.dist_checkpointing.strategies.filesystem_async._write_item',
            side_effect=mock_write_item_track_time,
        ):
            FileSystemWriterAsync.write_preloaded_data(
                transform_list=[],
                local_proc_idx=2,
                write_bucket=write_bucket,
                results_queue=results_queue,
                count_queue=count_queue,
                use_fsync=False,
                max_ckpt_save_retries=3,
                ckpt_save_retry_delay=retry_delay,
            )

        assert len(call_times) == 3
        time_diff_1 = call_times[1] - call_times[0]
        time_diff_2 = call_times[2] - call_times[1]

        assert (
            time_diff_1 >= retry_delay * 0.9
        ), f"First retry delay too short: {time_diff_1}s < {retry_delay}s"
        assert (
            time_diff_2 >= retry_delay * 0.9
        ), f"Second retry delay too short: {time_diff_2}s < {retry_delay}s"

    def test_success_no_retry(self, tmp_path_dist_ckpt, caplog):
        call_count = {'count': 0}

        def mock_write_item_succeed(*args, **kwargs):
            """Mock that succeeds immediately"""
            call_count['count'] += 1
            return None

        results_queue = mp.SimpleQueue()
        count_queue = mp.JoinableQueue()
        count_queue.put(None)

        write_bucket = (
            tmp_path_dist_ckpt / 'test_file.pt',
            'storage_key',
            ([(mock.MagicMock(), b'test')], []),
        )

        with caplog.at_level(logging.WARNING):
            with mock.patch(
                'megatron.core.dist_checkpointing.strategies.filesystem_async._write_item',
                side_effect=mock_write_item_succeed,
            ):
                FileSystemWriterAsync.write_preloaded_data(
                    transform_list=[],
                    local_proc_idx=3,
                    write_bucket=write_bucket,
                    results_queue=results_queue,
                    count_queue=count_queue,
                    use_fsync=False,
                    max_ckpt_save_retries=3,
                    ckpt_save_retry_delay=0.1,
                )

        # Verify only one attempt
        assert call_count['count'] == 1, "Should have attempted only once on success"

        # Verify no warning or error logs
        warning_logs = [record for record in caplog.records if record.levelname == 'WARNING']
        error_logs = [record for record in caplog.records if record.levelname == 'ERROR']
        assert len(warning_logs) == 0, "Should have no warnings on immediate success"
        assert len(error_logs) == 0, "Should have no errors on immediate success"

        # Verify successful result
        result = results_queue.get()
        proc_idx, results = result
        assert proc_idx == 3
        assert not isinstance(results, Exception)

    def test_local_results_reset_between_retries(self, tmp_path_dist_ckpt):
        results_per_attempt = []

        def mock_write_item_collect_results(*args, **kwargs):
            # Use a simple dict instead of MagicMock since it needs to be picklable
            result = {'value': f"attempt_{len(results_per_attempt) + 1}"}

            if len(results_per_attempt) == 0:
                results_per_attempt.append([result])
                raise RuntimeError("First attempt failure")
            else:
                results_per_attempt.append([result])
                return result

        results_queue = mp.SimpleQueue()
        count_queue = mp.JoinableQueue()
        count_queue.put(None)

        write_bucket = (
            tmp_path_dist_ckpt / 'test_file.pt',
            'storage_key',
            ([(mock.MagicMock(), b'test')], []),
        )

        with mock.patch(
            'megatron.core.dist_checkpointing.strategies.filesystem_async._write_item',
            side_effect=mock_write_item_collect_results,
        ):
            FileSystemWriterAsync.write_preloaded_data(
                transform_list=[],
                local_proc_idx=4,
                write_bucket=write_bucket,
                results_queue=results_queue,
                count_queue=count_queue,
                use_fsync=False,
                max_ckpt_save_retries=2,
                ckpt_save_retry_delay=0.05,
            )

        result = results_queue.get()
        proc_idx, final_results = result
        assert proc_idx == 4
        assert len(final_results) == 1, "Should only have results from successful attempt"
        assert (
            final_results[0]['value'] == "attempt_2"
        ), "Should have results from second attempt only"
