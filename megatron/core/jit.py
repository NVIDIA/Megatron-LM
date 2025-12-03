# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import os
import torch

from megatron.core.utils import is_torch_min_version

jit_fuser = torch.jit.script
# nvFuser is deprecated in PyTorch JIT starting from 2.2

use_noop_decoator = os.environ.get('TORCHINDUCTOR_DISABLE', '0') == '1'

def noop_decorator(func):
    '''No-op decorator'''
    return func


def enable_jit_fuser():
    '''Enable the JIT fuser'''
    global jit_fuser

    if use_noop_decoator:
        jit_fuser = noop_decorator
    else:
        try:
            if is_torch_min_version("2.2.0a0"):
                jit_fuser = torch.compile
        except ImportError:
            jit_fuser = noop_decorator


def disable_jit_fuser():
    '''Disable the JIT fuser'''
    global jit_fuser
    jit_fuser = noop_decorator


enable_jit_fuser()
