# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import signal
from typing import List, Optional, Union

import torch

from megatron.core.device_utils import get_xla_model
from megatron.core.wrapped_process_group import WrappedProcessGroup

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None


def get_world_size():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1
    return world_size


def get_device(local_rank=None):

    if xm:
        return xm.xla_device()
    
    backend = torch.distributed.get_backend()
    if backend == 'nccl':
        if local_rank is None:
            device = torch.device('cuda')
        else:
            device = torch.device(f'cuda:{local_rank}')
    elif backend == 'gloo':
        device = torch.device('cpu')
    else:
        raise RuntimeError
    return device


def all_gather_item(item, dtype, group:Optional[WrappedProcessGroup]=None, async_op=False, local_rank=None):
    
    if not torch.distributed.is_available() or \
       not torch.distributed.is_initialized():
        return [item]

    device = get_device(local_rank)

    if group is not None:
        group_size = group.size()
    else:
        group_size = get_world_size()

    tensor = torch.tensor([item], device=device, dtype=dtype)
    xm = get_xla_model()
    if xm:
        groups = group.rank_groups if group else None
        output_tensors = list(xm.all_gather(tensor, groups=groups).split(tensor.size()[0]), pin_layout=False)
    else:
        output_tensors = [
            torch.zeros(1, dtype=tensor.dtype, device=tensor.device)
            for _ in range(group_size)
        ]
        torch.distributed.all_gather(output_tensors, tensor, group, async_op)
    output = [elem.item() for elem in output_tensors]
    return output


class DistributedSignalHandler:
    def __init__(self, sig=signal.SIGTERM):
        self.sig = sig

    def signals_received(self):
        all_received = all_gather_item(
            self._signal_received, dtype=torch.int32, 
            group=WrappedProcessGroup()
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
