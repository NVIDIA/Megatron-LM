# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.utils import is_torch_min_version


def noop_decorator(func):
    '''No-op decorator'''
    return func


jit_fuser = noop_decorator  # replaced by enable_jit_fuser() below


def enable_jit_fuser():
    '''Enable the JIT fuser'''
    global jit_fuser
    try:
        if is_torch_min_version("2.2.0a0"):
            jit_fuser = torch.compile
        else:
            # nvFuser is deprecated in PyTorch JIT starting from 2.2. Only read TorchScript
            # on old PyTorch: newer releases removed the public torch.jit.script API.
            jit_fuser = torch.jit.script
    except ImportError:

        jit_fuser = noop_decorator


def disable_jit_fuser():
    '''Disable the JIT fuser'''
    global jit_fuser
    jit_fuser = noop_decorator


enable_jit_fuser()
