# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import signal

import torch

def get_world_size():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1
    return world_size


def _get_device_backend_name(device_type, group=None):
    """Name of the c10d backend registered for ``device_type``, or None if there is none.

    ``torch.distributed.get_backend()`` returns the configuration string the process
    group was created with, which may be device-qualified and compound (for example
    ``'cpu:gloo,cuda:nccl2'``), so it cannot be compared against a single backend name.
    Resolving the per-device backend object works for every world layout.

    Args:
        device_type: Device type to look up, e.g. ``'cpu'`` or ``'cuda'``.
        group: Process group to inspect. Defaults to the world group.

    Returns:
        The backend name (e.g. ``'gloo'``, ``'nccl'``, ``'nccl2'``), or None if no
        backend is registered for ``device_type``.
    """
    if group is None:
        group = torch.distributed.distributed_c10d._get_default_group()
    try:
        backend = group._get_backend(torch.device(device_type))
    except (RuntimeError, ValueError):
        # No backend is registered for this device type on this process group.
        return None
    name = getattr(backend, 'name', None)
    return name() if callable(name) else name


def get_device(local_rank=None, group=None):
    """Device that collectives issued for signal handling should run on.

    Args:
        local_rank: Local rank used to qualify the CUDA device. If None, an
            unqualified ``cuda`` device (the current device) is used.
        group: Process group the collective will run on. Defaults to the world group.

    Returns:
        The ``torch.device`` matching the backend the process group uses.

    Raises:
        RuntimeError: If the process group has no backend for cpu or cuda.
    """
    cuda_backend = _get_device_backend_name('cuda', group)
    # Gloo registers itself for cuda too, but its collectives are meant to run on cpu
    # tensors, so only a genuine device backend (nccl, nccl2, ...) selects cuda.
    if cuda_backend is not None and cuda_backend != 'gloo':
        if local_rank is None:
            return torch.device('cuda')
        return torch.device(f'cuda:{local_rank}')
    if _get_device_backend_name('cpu', group) is not None:
        return torch.device('cpu')
    raise RuntimeError(
        "Cannot pick a device for distributed signal handling: process group with "
        f"backend '{torch.distributed.get_backend(group)}' has neither a cpu nor a "
        "cuda backend."
    )


def all_gather_item(item, dtype, group=None, async_op=False, local_rank=None):
    if not torch.distributed.is_available() or \
       not torch.distributed.is_initialized():
        return [item]

    device = get_device(local_rank, group)

    if group is not None:
        group_size = group.size()
    else:
        group_size = get_world_size()

    tensor = torch.tensor([item], device=device, dtype=dtype)
    output_tensors = [
        torch.zeros(1, dtype=tensor.dtype, device=tensor.device)
        for _ in range(group_size)
    ]
    torch.distributed.all_gather(output_tensors, tensor, group, async_op)
    output = [elem.item() for elem in output_tensors]
    return output


class DistributedSignalHandler:
    def __init__(self, sig: signal.Signals = signal.SIGTERM):
        self.sig = sig

    def signals_received(self):
        all_received = all_gather_item(
            self._signal_received, dtype=torch.int32
        )
        return all_received

    def __enter__(self):
        self._signal_received = False
        self.released = False
        self.original_handler = signal.getsignal(self.sig)

        def handler(signum, frame):
            self._signal_received = True

        signal.signal(self.sig, handler)

        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):
        if self.released:
            return False

        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True
