# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import os

import torch

from .utils import is_lazy_mode, is_real_cuda_device_available

TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])

jit_fuser = torch.jit.script
# nvFuser is deprecated in PyTorch JIT starting from 2.2
if (TORCH_MAJOR > 2) or (TORCH_MAJOR == 2 and TORCH_MINOR >= 2):
    jit_fuser = torch.compile

if not is_real_cuda_device_available() and is_lazy_mode():

    def dummy(func):
        return func

    jit_fuser = dummy
