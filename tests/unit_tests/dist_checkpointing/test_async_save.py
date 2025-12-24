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


class TestWriteItemWithRetry:
    """Tests for the write_item_with_retry local function"""

    def test_write_item_retry_success_first_attempt(self, tmp_path_dist_ckpt, caplog):
        """Test that write_item_with_retry succeeds on first attempt"""
        call_count = {'count': 0}

        def mock_write_item_succeed(*args, **kwargs):
            call_count['count'] += 1
            return None

        results_queue = mp.SimpleQueue()
        count_queue = mp.JoinableQueue()
        count_queue.put(None)

        # Create mock WriteItem with index attribute
        mock_write_item_obj = mock.MagicMock()
        mock_write_item_obj.index = 'test_key_0'

        write_bucket = (
            tmp_path_dist_ckpt / 'test_file.pt',
            'storage_key',
            ([(mock_write_item_obj, b'test_data')], []),  # (bytes_data, tensor_data)
        )

        with caplog.at_level(logging.WARNING):
            with mock.patch(
                'megatron.core.dist_checkpointing.strategies.filesystem_async._write_item',
                side_effect=mock_write_item_succeed,
            ):
                FileSystemWriterAsync.write_preloaded_data(
                    transform_list=[],
                    local_proc_idx=0,
                    write_bucket=write_bucket,
                    results_queue=results_queue,
                    count_queue=count_queue,
                    use_fsync=False,
                    max_item_retries=3,
                    item_retry_delay=1.0,
                )

        # Verify only one attempt per item
        assert call_count['count'] == 1, "Should call _write_item once for success"

        # Verify no warnings
        warning_logs = [record for record in caplog.records if record.levelname == 'WARNING']
        assert len(warning_logs) == 0, "Should have no warnings on immediate success"

        # Verify successful result
        result = results_queue.get()
        proc_idx, results = result
        assert proc_idx == 0
        assert not isinstance(results, Exception)
        assert len(results) == 1

    def test_write_item_retry_transient_failure(self, tmp_path_dist_ckpt, caplog):
        """Test that write_item_with_retry retries on transient failures"""
        call_count = {'count': 0}

        def mock_write_item_fail_twice(*args, **kwargs):
            call_count['count'] += 1
            if call_count['count'] <= 2:
                raise IOError(f"Transient I/O error on attempt {call_count['count']}")
            return None

        results_queue = mp.SimpleQueue()
        count_queue = mp.JoinableQueue()
        count_queue.put(None)

        mock_write_item_obj = mock.MagicMock()
        mock_write_item_obj.index = 'test_key_1'

        write_bucket = (
            tmp_path_dist_ckpt / 'test_file.pt',
            'storage_key',
            ([(mock_write_item_obj, b'test_data')], []),
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
                    max_item_retries=3,
                    item_retry_delay=1.0,
                )

        # Verify 3 attempts (2 failures + 1 success)
        assert call_count['count'] == 3, "Should retry twice and succeed on third attempt"

        # Verify warning logs
        warning_logs = [record for record in caplog.records if record.levelname == 'WARNING']
        assert len(warning_logs) == 2, "Should have 2 warnings for the 2 failed attempts"
        assert 'Write item test_key_1 failed on attempt 1/3' in warning_logs[0].message
        assert 'Retrying in 1.0s' in warning_logs[0].message

        # Verify successful result
        result = results_queue.get()
        proc_idx, results = result
        assert proc_idx == 0
        assert not isinstance(results, Exception)

    def test_write_item_retry_exhaustion(self, tmp_path_dist_ckpt, caplog):
        """Test that write_item_with_retry fails after exhausting retries"""

        def mock_write_item_always_fail(*args, **kwargs):
            raise PermissionError("Permission denied - persistent error")

        results_queue = mp.SimpleQueue()
        count_queue = mp.JoinableQueue()
        count_queue.put(None)

        mock_write_item_obj = mock.MagicMock()
        mock_write_item_obj.index = 'test_key_2'

        write_bucket = (
            tmp_path_dist_ckpt / 'test_file.pt',
            'storage_key',
            ([(mock_write_item_obj, b'test_data')], []),
        )

        with caplog.at_level(logging.WARNING):
            with mock.patch(
                'megatron.core.dist_checkpointing.strategies.filesystem_async._write_item',
                side_effect=mock_write_item_always_fail,
            ):
                FileSystemWriterAsync.write_preloaded_data(
                    transform_list=[],
                    local_proc_idx=0,
                    write_bucket=write_bucket,
                    results_queue=results_queue,
                    count_queue=count_queue,
                    use_fsync=False,
                    max_item_retries=3,
                    item_retry_delay=1.0,
                )

        # Verify error result
        result = results_queue.get()
        proc_idx, exception = result
        assert proc_idx == 0
        assert isinstance(exception, PermissionError)

        # Verify error log
        error_logs = [record for record in caplog.records if record.levelname == 'ERROR']
        assert len(error_logs) >= 1, "Should have at least one error log"
        # Find the specific error about write item failure
        item_error_logs = [log for log in error_logs if 'Failed to write item test_key_2' in log.message]
        assert len(item_error_logs) == 1, "Should have error log for write_item failure"
        assert 'after 3 attempts' in item_error_logs[0].message
        assert 'PermissionError' in item_error_logs[0].message

        # Verify warning logs (should have 2 warnings for non-final attempts)
        warning_logs = [record for record in caplog.records if record.levelname == 'WARNING']
        item_warning_logs = [log for log in warning_logs if 'Write item test_key_2' in log.message]
        assert len(item_warning_logs) == 2, "Should have 2 warnings before final failure"

    def test_write_item_retry_with_fsync(self, tmp_path_dist_ckpt, caplog):
        """Test that write_item_with_retry calls fsync when use_fsync=True"""
        fsync_call_count = {'count': 0}

        def mock_write_item_succeed(*args, **kwargs):
            return

        def mock_fsync(fileno):
            fsync_call_count['count'] += 1

        results_queue = mp.SimpleQueue()
        count_queue = mp.JoinableQueue()
        count_queue.put(None)

        mock_write_item_obj = mock.MagicMock()
        mock_write_item_obj.index = 'test_key_3'

        write_bucket = (
            tmp_path_dist_ckpt / 'test_file.pt',
            'storage_key',
            ([(mock_write_item_obj, b'test_data')], []),
        )

        with mock.patch(
            'megatron.core.dist_checkpointing.strategies.filesystem_async._write_item',
            side_effect=mock_write_item_succeed,
        ):
            with mock.patch('os.fsync', side_effect=mock_fsync):
                FileSystemWriterAsync.write_preloaded_data(
                    transform_list=[],
                    local_proc_idx=0,
                    write_bucket=write_bucket,
                    results_queue=results_queue,
                    count_queue=count_queue,
                    use_fsync=True,  # Enable fsync
                    max_item_retries=3,
                    item_retry_delay=1.0,
                )

        # Verify fsync was called (once per item + once at end)
        assert fsync_call_count['count'] == 2, "Should call fsync twice (per item + final)"

        # Verify successful result
        result = results_queue.get()
        proc_idx, results = result
        assert proc_idx == 0
        assert not isinstance(results, Exception)

    def test_write_item_retry_fsync_failure_logged(self, tmp_path_dist_ckpt, caplog):
        """Test that fsync failures are logged but don't fail the write"""

        def mock_write_item_succeed(*args, **kwargs):
            return None

        def mock_fsync_fail(fileno):
            raise OSError("fsync failed - disk full")

        results_queue = mp.SimpleQueue()
        count_queue = mp.JoinableQueue()
        count_queue.put(None)

        mock_write_item_obj = mock.MagicMock()
        mock_write_item_obj.index = 'test_key_4'

        write_bucket = (
            tmp_path_dist_ckpt / 'test_file.pt',
            'storage_key',
            ([(mock_write_item_obj, b'test_data')], []),
        )

        with caplog.at_level(logging.WARNING):
            with mock.patch(
                'megatron.core.dist_checkpointing.strategies.filesystem_async._write_item',
                side_effect=mock_write_item_succeed,
            ):
                with mock.patch('os.fsync', side_effect=mock_fsync_fail):
                    FileSystemWriterAsync.write_preloaded_data(
                        transform_list=[],
                        local_proc_idx=0,
                        write_bucket=write_bucket,
                        results_queue=results_queue,
                        count_queue=count_queue,
                        use_fsync=True,
                        use_msc=False,
                        max_item_retries=3,
                        item_retry_delay=1.0,
                    )

        # Verify write succeeded despite fsync failure
        result = results_queue.get()
        proc_idx, results = result
        assert proc_idx == 0
        assert not isinstance(results, Exception), "Write should succeed despite fsync failure"

        # Verify fsync failure was logged
        warning_logs = [record for record in caplog.records if record.levelname == 'WARNING']
        fsync_warnings = [log for log in warning_logs if 'fsync failed' in log.message]
        assert len(fsync_warnings) >= 1, "Should have warning about fsync failure"
        assert 'test_key_4' in fsync_warnings[0].message
        assert 'OSError' in fsync_warnings[0].message

    def test_write_item_retry_multiple_items(self, tmp_path_dist_ckpt, caplog):
        """Test that write_item_with_retry handles multiple items correctly"""
        call_counts = {'item_0': 0, 'item_1': 0, 'item_2': 0}

        def mock_write_item_mixed_failures(*args, **kwargs):
            # args[2] is write_item in the signature
            write_item = args[2]
            item_key = write_item.index
            call_counts[item_key] += 1

            # First item fails once, second succeeds, third fails twice
            if item_key == 'item_0' and call_counts[item_key] <= 1:
                raise ConnectionError(f"{item_key} transient error")
            elif item_key == 'item_2' and call_counts[item_key] <= 2:
                raise TimeoutError(f"{item_key} timeout")

            return None

        results_queue = mp.SimpleQueue()
        count_queue = mp.JoinableQueue()
        count_queue.put(None)

        # Create multiple write items
        mock_items = []
        for i in range(3):
            mock_item = mock.MagicMock()
            mock_item.index = f'item_{i}'
            mock_items.append((mock_item, f'data_{i}'.encode()))

        write_bucket = (
            tmp_path_dist_ckpt / 'test_file.pt',
            'storage_key',
            (mock_items, []),  # bytes_data with 3 items
        )

        with caplog.at_level(logging.WARNING):
            with mock.patch(
                'megatron.core.dist_checkpointing.strategies.filesystem_async._write_item',
                side_effect=mock_write_item_mixed_failures,
            ):
                FileSystemWriterAsync.write_preloaded_data(
                    transform_list=[],
                    local_proc_idx=0,
                    write_bucket=write_bucket,
                    results_queue=results_queue,
                    count_queue=count_queue,
                    use_fsync=False,
                    max_item_retries=3,
                    item_retry_delay=1.0,
                )

        # Verify call counts: item_0 fails once (2 attempts), item_1 succeeds (1 attempt), item_2 fails twice (3 attempts)
        assert call_counts['item_0'] == 2, "Item 0 should have 2 attempts"
        assert call_counts['item_1'] == 1, "Item 1 should have 1 attempt"
        assert call_counts['item_2'] == 3, "Item 2 should have 3 attempts"

        # Verify successful result
        result = results_queue.get()
        proc_idx, results = result
        assert proc_idx == 0
        assert not isinstance(results, Exception)
        assert len(results) == 3, "Should have 3 results for 3 items"

        # Verify warning logs for failed attempts
        warning_logs = [record for record in caplog.records if record.levelname == 'WARNING']
        item_0_warnings = [log for log in warning_logs if 'item_0' in log.message]
        item_2_warnings = [log for log in warning_logs if 'item_2' in log.message]
        assert len(item_0_warnings) == 1, "Item 0 should have 1 warning"
        assert len(item_2_warnings) == 2, "Item 2 should have 2 warnings"

    def test_write_item_with_tensor_data(self, tmp_path_dist_ckpt):
        """Test that write_item_with_retry works with tensor data"""
        call_count = {'count': 0}

        def mock_write_item_succeed(*args, **kwargs):
            call_count['count'] += 1
            return None

        results_queue = mp.SimpleQueue()
        count_queue = mp.JoinableQueue()
        count_queue.put(None)

        mock_write_item_obj = mock.MagicMock()
        mock_write_item_obj.index = 'test_tensor_0'

        # Create a CPU tensor
        test_tensor = torch.ones(3, 4)
        assert test_tensor.is_cpu, "Test tensor must be on CPU"

        write_bucket = (
            tmp_path_dist_ckpt / 'test_file.pt',
            'storage_key',
            ([], [(mock_write_item_obj, test_tensor)]),  # Empty bytes_data, one tensor
        )

        with mock.patch(
            'megatron.core.dist_checkpointing.strategies.filesystem_async._write_item',
            side_effect=mock_write_item_succeed,
        ):
            FileSystemWriterAsync.write_preloaded_data(
                transform_list=[],
                local_proc_idx=0,
                write_bucket=write_bucket,
                results_queue=results_queue,
                count_queue=count_queue,
                use_fsync=False,
                max_item_retries=3,
                item_retry_delay=1.0,
            )

        # Verify write was called
        assert call_count['count'] == 1, "Should write tensor once"

        # Verify successful result
        result = results_queue.get()
        proc_idx, results = result
        assert proc_idx == 0
        assert not isinstance(results, Exception)
        assert len(results) == 1
