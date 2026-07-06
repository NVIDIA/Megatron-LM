# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""get_device() resolves the device from the backend bound to the process group."""

import os
import signal

import pytest
import torch

from megatron.training import dist_signal_handler
from tests.unit_tests.test_utilities import Utils


class _FakeBackend:
    """Stand-in for a c10d backend object, which exposes its name via `name()`."""

    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name


class _FakeProcessGroup:
    """Process group exposing only the per-device backend lookup `get_device` uses."""

    def __init__(self, device_backends):
        self._device_backends = device_backends

    def _get_backend(self, device):
        if device.type not in self._device_backends:
            raise RuntimeError(f"No backend type associated with device type {device.type}")
        return _FakeBackend(self._device_backends[device.type])


# (device backends registered on the process group, local_rank, expected device)
_BACKEND_CASES = [
    # Device-qualified world built by megatron/training/initialize.py for nccl2.
    ({'cpu': 'gloo', 'cuda': 'nccl2'}, None, 'cuda'),
    ({'cpu': 'gloo', 'cuda': 'nccl2'}, 1, 'cuda:1'),
    # Device-qualified world with stock nccl.
    ({'cpu': 'gloo', 'cuda': 'nccl'}, None, 'cuda'),
    ({'cpu': 'gloo', 'cuda': 'nccl'}, 0, 'cuda:0'),
    # Lazily initialized per-peer backend used for pipeline-parallel groups.
    ({'cpu': 'gloo', 'cuda': 'nccl-lazy'}, 3, 'cuda:3'),
    # Plain nccl world: no cpu backend is registered at all.
    ({'cuda': 'nccl'}, None, 'cuda'),
    ({'cuda': 'nccl'}, 2, 'cuda:2'),
    # Plain gloo world: gloo advertises cuda as well, but collectives run on cpu.
    ({'cpu': 'gloo', 'cuda': 'gloo'}, None, 'cpu'),
    ({'cpu': 'gloo', 'cuda': 'gloo'}, 1, 'cpu'),
]


@pytest.mark.parametrize("device_backends,local_rank,expected", _BACKEND_CASES)
def test_get_device_uses_default_group_backend(device_backends, local_rank, expected, mocker):
    mocker.patch.object(
        torch.distributed.distributed_c10d,
        "_get_default_group",
        return_value=_FakeProcessGroup(device_backends),
    )

    assert dist_signal_handler.get_device(local_rank) == torch.device(expected)


@pytest.mark.parametrize("device_backends,local_rank,expected", _BACKEND_CASES)
def test_get_device_uses_explicit_group_backend(device_backends, local_rank, expected, mocker):
    default_group = mocker.patch.object(torch.distributed.distributed_c10d, "_get_default_group")
    group = _FakeProcessGroup(device_backends)

    assert dist_signal_handler.get_device(local_rank, group) == torch.device(expected)
    default_group.assert_not_called()


def test_get_device_raises_without_cpu_or_cuda_backend(mocker):
    mocker.patch.object(torch.distributed, "get_backend", return_value="mpi")

    with pytest.raises(RuntimeError, match="Cannot pick a device"):
        dist_signal_handler.get_device(group=_FakeProcessGroup({}))


class TestDistSignalHandler:
    """Exercise the real path against the device-qualified world PG used by mcore."""

    def setup_method(self, method):
        Utils.initialize_distributed()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_get_device(self):
        assert dist_signal_handler.get_device() == torch.device('cuda')
        assert dist_signal_handler.get_device(local_rank=Utils.rank) == torch.device(
            f'cuda:{Utils.rank}'
        )

    def test_all_gather_item(self):
        gathered = dist_signal_handler.all_gather_item(Utils.rank, dtype=torch.int32)

        assert gathered == list(range(Utils.world_size))

    def test_signals_received(self):
        with dist_signal_handler.DistributedSignalHandler(signal.SIGTERM) as handler:
            if torch.distributed.get_rank() == 0:
                os.kill(os.getpid(), signal.SIGTERM)
            received = handler.signals_received()

        assert received == [1] + [0] * (Utils.world_size - 1)
