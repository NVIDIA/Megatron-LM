# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.utils import is_torch_min_version

jit_fuser = torch.jit.script
# nvFuser is deprecated in PyTorch JIT starting from 2.2

try:
    if is_torch_min_version("2.2.0a0"):
        jit_fuser = torch.compile
except ImportError:

    def noop_decorator(func):
        return func

    jit_fuser = noop_decorator
