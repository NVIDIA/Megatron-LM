# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import sys
from pathlib import Path
from unittest import mock

import pytest
import torch
from torch.distributed.checkpoint import CheckpointException

from megatron.core.dist_checkpointing import ShardedTensor, load, save
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue
from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync
from megatron.core.dist_checkpointing.strategies.nvrx import has_nvrx_async_support
from megatron.core.dist_checkpointing.strategies.state_dict_saver import (
    save_state_dict_async_finalize,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    TorchDistSaveShardedStrategy,
    get_async_strategy,
)
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
            async_request = save(
                sharded_state_dict, async_ckpt_dir, async_sharded_save=True, async_strategy="mcore"
            )
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

    @pytest.mark.parametrize('async_strategy', ["nvrx", "mcore"])
    def test_get_async_strategy(self, async_strategy):
        strategy, modules = get_async_strategy(async_strategy)

        assert len(modules) > 1
        assert strategy == async_strategy

        _, module = get_async_strategy(async_strategy, module="FileSystemWriterAsync")
        assert type(module) is not dict

    @pytest.mark.parametrize('async_strategy', ["nvrx", "mcore"])
    def test_get_async_strategy_no_nvrx_installed(self, async_strategy):
        with mock.patch.dict(
            'sys.modules', {'nvidia_resiliency_ext.checkpointing.async_ckpt.core': None}
        ):
            from megatron.core.dist_checkpointing.strategies.async_utils import (
                AsyncRequest as MCoreAsyncRequest,
            )

            if async_strategy == "nvrx":
                with pytest.raises(ModuleNotFoundError):
                    strategy, module = get_async_strategy(async_strategy, module="AsyncRequest")
            else:
                strategy, module = get_async_strategy(async_strategy, module="AsyncRequest")

                assert strategy == "mcore"
                assert module == MCoreAsyncRequest

    def test_get_async_strategy_missing_nvrx_cached_metadata_reader(self):
        with mock.patch.dict(
            'sys.modules',
            {
                'nvidia_resiliency_ext.checkpointing.async_ckpt.cached_metadata_filesystem_reader': None
            },
        ):
            with pytest.raises(ModuleNotFoundError):
                get_async_strategy("nvrx", module="CachedMetadataFileSystemReader")


_NVRX_SUBMODULES = [
    'nvidia_resiliency_ext.checkpointing.async_ckpt.core',
    'nvidia_resiliency_ext.checkpointing.async_ckpt.cached_metadata_filesystem_reader',
    'nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async',
    'nvidia_resiliency_ext.checkpointing.async_ckpt.state_dict_saver',
]


class TestHasNvrxAsyncSupport:
    """Tests for has_nvrx_async_support, focusing on the minimum-version assertion."""

    def _fake_modules(self):
        """MagicMock modules that satisfy every symbol and hasattr check in has_nvrx_async_support."""
        return {name: mock.MagicMock() for name in _NVRX_SUBMODULES}

    def test_version_check_passes(self):
        """Returns True when all NVRx symbols are present and version meets the minimum."""
        with (
            mock.patch(
                'megatron.core.dist_checkpointing.strategies.nvrx.import_module',
                side_effect=lambda name: self._fake_modules()[name],
            ),
            mock.patch(
                'megatron.core.dist_checkpointing.strategies.nvrx.is_nvrx_min_version',
                return_value=True,
            ),
        ):
            assert has_nvrx_async_support() is True

    def test_version_check_fails(self):
        """Raises AssertionError when all NVRx symbols are present but version is too old."""
        with (
            mock.patch(
                'megatron.core.dist_checkpointing.strategies.nvrx.import_module',
                side_effect=lambda name: self._fake_modules()[name],
            ),
            mock.patch(
                'megatron.core.dist_checkpointing.strategies.nvrx.is_nvrx_min_version',
                return_value=False,
            ),
        ):
            with pytest.raises(AssertionError, match="Minimum required nvidia-resiliency-ext"):
                has_nvrx_async_support()


class TestFileSystemWriterAsync:
    @staticmethod
    def _write_buckets(tensor):
        return [(Path("checkpoint"), "storage-key", ([], [(mock.sentinel.item, tensor)]))]

    def test_preload_cpu_tensors_does_not_synchronize_cuda(self):
        tensor = torch.ones(2)

        with mock.patch.object(torch.cuda, "synchronize") as synchronize:
            result = FileSystemWriterAsync.preload_tensors(self._write_buckets(tensor))

        synchronize.assert_not_called()
        assert result[0][2][1][0][1] is tensor

    def test_preload_cuda_tensors_synchronizes_cuda(self):
        tensor = mock.MagicMock()
        tensor.is_cuda = True
        tensor.to.return_value = torch.ones(2)

        with mock.patch.object(torch.cuda, "synchronize") as synchronize:
            FileSystemWriterAsync.preload_tensors(self._write_buckets(tensor))

        tensor.to.assert_called_once_with("cpu", non_blocking=True)
        synchronize.assert_called_once_with()


class TestSaveStateDictAsyncFinalize:
    @pytest.mark.parametrize("collective_device", ["cpu", "cuda"])
    def test_failure_status_uses_process_group_device(self, collective_device):
        storage_writer = mock.Mock()
        storage_writer.retrieve_write_results.return_value = [mock.sentinel.local_result]
        all_results = [mock.sentinel.all_results]
        dist_wrapper = mock.Mock(is_coordinator=True, coordinator_rank=0, group=mock.sentinel.group)
        dist_wrapper.gather_object.return_value = all_results
        failures_occurred = mock.MagicMock()
        failures_occurred.__bool__.return_value = False

        with (
            mock.patch(
                "megatron.core.dist_checkpointing.strategies.state_dict_saver._get_failure_dict",
                return_value={},
            ),
            mock.patch(
                "megatron.core.dist_checkpointing.strategies.state_dict_saver._get_object_coll_device",
                return_value=collective_device,
            ) as get_collective_device,
            mock.patch.object(torch, "tensor", return_value=failures_occurred) as tensor,
            mock.patch.object(torch.distributed, "get_rank", return_value=0),
            mock.patch.object(torch.distributed, "broadcast") as broadcast,
        ):
            save_state_dict_async_finalize(storage_writer, mock.sentinel.metadata, dist_wrapper)

        storage_writer.finish.assert_called_once_with(mock.sentinel.metadata, all_results)
        get_collective_device.assert_called_once_with(dist_wrapper.group)
        tensor.assert_called_once_with([0], dtype=torch.int, device=collective_device)
        broadcast.assert_called_once_with(
            failures_occurred, src=dist_wrapper.coordinator_rank, group=dist_wrapper.group
        )
