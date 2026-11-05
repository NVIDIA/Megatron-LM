# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Exercise the callbacks actually sent to DataLoader worker processes."""

import multiprocessing
import os
import signal
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.datasets.utils import Split
from megatron.training import global_vars
from megatron.training.datasets import data_samplers


class _WorkerDataset(torch.utils.data.Dataset):
    def __init__(self, split, variable_length, exit_signal_handler):
        self.split = split
        self.variable_length = variable_length
        self.exit_signal_handler = exit_signal_handler

    def __len__(self):
        return 4

    def __getitem__(self, index):
        worker = torch.utils.data.get_worker_info()
        assert worker is not None
        if multiprocessing.get_start_method() != 'fork':
            # The parent only patches the getters; fresh interpreters have no run config.
            assert global_vars._GLOBAL_ARGS is None
            assert global_vars._GLOBAL_RUN_CONFIG is None
        if self.exit_signal_handler:
            assert callable(signal.getsignal(signal.SIGUSR1))
            os.kill(os.getpid(), signal.SIGUSR1)
        else:
            assert signal.getsignal(signal.SIGUSR1) == signal.SIG_DFL
        length = index + 1 if self.variable_length else 1
        return {'tokens': torch.arange(length) + index, 'pid': os.getpid()}


@pytest.mark.parametrize('start_method', multiprocessing.get_all_start_methods())
@pytest.mark.parametrize(
    ('split', 'collate_path', 'exit_signal_handler'),
    [
        (Split.train, 'default', False),
        (Split.valid, 'default', True),
        (Split.test, 'default', False),
        (Split.train, 'varlen', True),
        (Split.valid, 'varlen', False),
        (Split.train, 'dynamic_cp', False),
        (Split.valid, 'packing', True),
        (Split.valid, 'sbhd', True),
        (Split.valid, 'full_validation', True),
    ],
)
def test_data_loader_workers(monkeypatch, start_method, split, collate_path, exit_signal_handler):
    variable_length = collate_path in ('varlen', 'dynamic_cp', 'packing')
    batch_size = 1 if collate_path == 'full_validation' else 2
    args = SimpleNamespace(
        dataloader_type='single',
        num_workers=1,
        use_varlen_dataset=collate_path in ('varlen', 'sbhd'),
        varlen_sbhd_validation=collate_path == 'sbhd',
        dynamic_context_parallel=collate_path == 'dynamic_cp',
        sequence_packing_scheduler='test' if collate_path == 'packing' else None,
    )
    cfg = SimpleNamespace(
        train=SimpleNamespace(
            micro_batch_size=2, exit_signal_handler=exit_signal_handler, exit_signal=signal.SIGUSR1
        ),
        validation=SimpleNamespace(
            eval_micro_batch_size=2, full_validation=collate_path == 'full_validation'
        ),
    )
    monkeypatch.setattr(data_samplers, 'get_args', lambda: args)
    monkeypatch.setattr(data_samplers, 'get_run_config', lambda: cfg)
    monkeypatch.setattr(data_samplers.mpu, 'get_data_parallel_rank', lambda: 0)
    monkeypatch.setattr(data_samplers.mpu, 'get_data_parallel_world_size', lambda: 1)
    dataset = _WorkerDataset(split, variable_length, exit_signal_handler)
    loader = data_samplers.build_pretraining_data_loader(dataset, consumed_samples=0)
    # Choose a context for this loader without changing the application's global default.
    loader.multiprocessing_context = multiprocessing.get_context(start_method)
    loader.timeout = 60
    iterator = iter(loader)
    try:
        for batch_index, batch in enumerate(iterator):
            indices = range(batch_size * batch_index, batch_size * (batch_index + 1))
            if variable_length:
                assert isinstance(batch, list)
                for sample, index in zip(batch, indices):
                    assert sample['tokens'].tolist() == list(range(index, 2 * index + 1))
                    assert sample['pid'] != os.getpid()
            else:
                assert batch['tokens'].tolist() == [[index] for index in indices]
                assert all(pid != os.getpid() for pid in batch['pid'].tolist())
        assert batch_index == len(dataset) // batch_size - 1
        # Persistent workers must also serve another iterator.
        assert len(list(loader)) == len(dataset) // batch_size
    finally:
        iterator._shutdown_workers()


def _check_worker_fds(queue):
    # Use real /proc entries and real closes, mapping one harmless FD to a GPU path.
    gpu_fd = os.open('/dev/null', os.O_RDONLY)
    other_fd = os.open('/dev/null', os.O_RDONLY)
    readlink = os.readlink

    def device_path(path):
        if path == f'/proc/self/fd/{gpu_fd}':
            return '/dev/nvidia0'
        return readlink(path)

    try:
        with patch.object(os, 'readlink', side_effect=device_path):
            data_samplers._worker_init_fn(0, exit_signal_handler=False, exit_signal=signal.SIGUSR1)
        with pytest.raises(OSError):
            os.fstat(gpu_fd)
        os.fstat(other_fd)
        queue.put('closed only the NVIDIA FD')
    finally:
        os.close(other_fd)


@pytest.mark.skipif(sys.platform != 'linux', reason='requires Linux /proc/self/fd')
@pytest.mark.parametrize('start_method', multiprocessing.get_all_start_methods())
def test_worker_nvidia_fd_cleanup(start_method):
    ctx = multiprocessing.get_context(start_method)
    queue = ctx.Queue()
    process = ctx.Process(target=_check_worker_fds, args=(queue,))
    process.start()
    try:
        assert queue.get(timeout=60) == 'closed only the NVIDIA FD'
        process.join(timeout=60)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join()
        queue.close()
        queue.join_thread()
