# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def select_local_cuda_device():
    """Bind each torchrun process before an SSM test initializes CUDA or Triton."""
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
